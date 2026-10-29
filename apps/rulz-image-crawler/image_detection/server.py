"""FastAPI app: POST /classify, POST /expand_tag, GET /health. Bind to 127.0.0.1 only."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from .config import Settings
from .llm import ModelError, ModelServerDown
from .rubrics import TAG_ID_RE
from .service import DetectionService, ImageInput, Tag

log = logging.getLogger("image_detection.server")


class ImageIn(BaseModel):
    id: str = Field(min_length=1, max_length=512)
    data_b64: Optional[str] = None
    url: Optional[str] = None
    mime: Optional[str] = None

    @model_validator(mode="after")
    def _one_source(self):
        if not self.data_b64 and not self.url:
            raise ValueError("each image needs data_b64 or url")
        return self


class TagIn(BaseModel):
    id: str
    rubric: str = Field(min_length=1, max_length=2000)

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not TAG_ID_RE.match(v):
            raise ValueError("tag id must be snake_case: lowercase letters, digits, underscores (max 64)")
        return v


class ClassifyIn(BaseModel):
    images: list[ImageIn]
    tags: list[TagIn] = Field(min_length=1)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _unique(self):
        if len({i.id for i in self.images}) != len(self.images):
            raise ValueError("image ids must be unique within a request")
        if len({t.id for t in self.tags}) != len(self.tags):
            raise ValueError("tag ids must be unique within a request")
        return self


class ExpandIn(BaseModel):
    text: str = Field(min_length=1, max_length=300)


def create_app(settings: Settings | None = None, service: DetectionService | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or DetectionService(settings)
        try:
            yield
        finally:
            await app.state.service.aclose()

    app = FastAPI(title="Qwen3-VL Image Tag Matching", lifespan=lifespan)

    @app.exception_handler(ModelServerDown)
    async def _down(_: Request, exc: ModelServerDown):
        log.warning("model server down: %s", exc)
        return JSONResponse(status_code=503, content={"error": "model server down"})

    @app.post("/classify")
    async def classify(body: ClassifyIn, request: Request):
        if len(body.images) > settings.max_images_per_call:
            raise HTTPException(422, f"at most {settings.max_images_per_call} images per call")
        svc: DetectionService = request.app.state.service
        results = await svc.classify(
            [ImageInput(i.id, i.data_b64, i.url, i.mime) for i in body.images],
            [Tag(t.id, t.rubric) for t in body.tags],
            body.threshold,
        )
        return {"results": results}

    @app.post("/expand_tag")
    async def expand_tag(body: ExpandIn, request: Request):
        svc: DetectionService = request.app.state.service
        try:
            return await svc.expand_tag(body.text)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except ModelError as e:
            raise HTTPException(502, str(e))

    @app.get("/health")
    async def health(request: Request):
        return await request.app.state.service.health()

    return app


def main() -> None:
    import argparse

    import uvicorn

    p = argparse.ArgumentParser(description="Qwen3-VL image tag matching service")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--log-level", default="info")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
