"""Face detection (YuNet), face embeddings (SFace) and expressions (FER+) via OpenCV DNN."""
from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2  # noqa: E402
import numpy as np

from . import attributes, config
from .body import BodyAnalyzer


@dataclass
class FaceAttrs:
    expression: str | None = None
    expr_conf: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    pose: str | None = None       # filled in by body.BodyAnalyzer
    arms_up: bool = False


@dataclass
class DetectedFace:
    box: tuple[float, float, float, float]  # normalized x, y, w, h
    score: float
    embedding: np.ndarray                   # L2-normalized float32[128]
    attrs: FaceAttrs


class FaceEngine:
    """Not thread-safe: create one per thread."""

    def __init__(self, score_threshold: float = config.DETECT_SCORE):
        self.score_threshold = score_threshold
        self.detector = cv2.FaceDetectorYN.create(
            str(config.DETECTOR_MODEL), "", (320, 320), score_threshold, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(config.RECOGNIZER_MODEL), "")
        self.expression_net = None
        if config.EXPRESSION_MODEL.exists():  # optional: without it, expressions are simply not tagged
            try:
                self.expression_net = cv2.dnn.readNetFromONNX(str(config.EXPRESSION_MODEL))
            except cv2.error:
                pass
        self.body = None  # optional too: body pose (standing/sitting/...)
        if BodyAnalyzer.available():
            try:
                self.body = BodyAnalyzer()
            except cv2.error:
                pass

    def _detect_raw(self, bgr: np.ndarray, score_threshold: float | None = None) -> np.ndarray:
        h, w = bgr.shape[:2]
        self.detector.setInputSize((w, h))
        self.detector.setScoreThreshold(score_threshold or self.score_threshold)
        _, faces = self.detector.detect(bgr)
        return faces if faces is not None else np.empty((0, 15), np.float32)

    def expression(self, face_bgr: np.ndarray) -> tuple[str | None, float | None]:
        """Classify an (aligned) face crop with FER+."""
        if self.expression_net is None or face_bgr.size == 0:
            return None, None
        gray = cv2.resize(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32)
        self.expression_net.setInput(gray.reshape(1, 1, 64, 64))
        logits = self.expression_net.forward().reshape(-1)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        i = int(np.argmax(probs))
        return attributes.EXPRESSIONS[i], float(probs[i])

    def analyze_row(self, bgr: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, FaceAttrs]:
        """Embedding plus expression and head pose for one YuNet detection row."""
        aligned = self.recognizer.alignCrop(bgr, row)
        return _normalize(self.recognizer.feature(aligned)), self._attrs(aligned, row)

    def attributes_row(self, bgr: np.ndarray, row: np.ndarray) -> FaceAttrs:
        return self._attrs(self.recognizer.alignCrop(bgr, row), row)

    def _attrs(self, aligned: np.ndarray, row: np.ndarray) -> FaceAttrs:
        expr, conf = self.expression(aligned)
        yaw, pitch, roll = attributes.head_pose(row[4:14])
        return FaceAttrs(expr, conf, yaw, pitch, roll)

    def find_row(self, bgr: np.ndarray, box_norm) -> np.ndarray | None:
        """Re-detect and return the YuNet row that best matches a stored face box (IoU >= 0.3)."""
        H, W = bgr.shape[:2]
        bx, by, bw, bh = box_norm[0] * W, box_norm[1] * H, box_norm[2] * W, box_norm[3] * H
        best, best_iou = None, 0.3
        for row in self._detect_raw(bgr, score_threshold=0.5):
            x, y, w, h = (float(v) for v in row[:4])
            ix = max(0.0, min(x + w, bx + bw) - max(x, bx))
            iy = max(0.0, min(y + h, by + bh) - max(y, by))
            inter = ix * iy
            iou = inter / (w * h + bw * bh - inter + 1e-6)
            if iou > best_iou:
                best, best_iou = row, iou
        return best

    def detect(self, bgr: np.ndarray, min_face: int = config.MIN_FACE_PX) -> list[DetectedFace]:
        H, W = bgr.shape[:2]
        out = []
        for row in self._detect_raw(bgr):
            x, y, w, h = (float(v) for v in row[:4])
            if min(w, h) < min_face:
                continue
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(W, x + w), min(H, y + h)
            if x1 - x0 < min_face * 0.7 or y1 - y0 < min_face * 0.7:
                continue
            emb, attrs = self.analyze_row(bgr, row)
            out.append(DetectedFace((x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H), float(row[14]), emb, attrs))
        for face, body in zip(out, self.poses(bgr, [f.box for f in out])):
            if body is not None:
                face.attrs.pose, face.attrs.arms_up = body.pose, body.arms_up
        return out

    def poses(self, bgr: np.ndarray, boxes_norm: list) -> list:
        """Body pose for each face box (normalized x, y, w, h); None where it can't be read."""
        if self.body is None or not boxes_norm:
            return [None] * len(boxes_norm)
        H, W = bgr.shape[:2]
        return self.body.analyze(bgr, [(b[0] * W, b[1] * H, b[2] * W, b[3] * H) for b in boxes_norm])

    def embed_region(self, bgr: np.ndarray, box_norm) -> DetectedFace:
        """Embed a user-drawn face box (with expression, head direction and body pose)."""
        face = self._embed_region(bgr, box_norm)
        body = self.poses(bgr, [face.box])[0]
        if body is not None:
            face.attrs.pose, face.attrs.arms_up = body.pose, body.arms_up
        return face

    def _embed_region(self, bgr: np.ndarray, box_norm) -> DetectedFace:
        """Tries to find landmarks inside the box for proper alignment."""
        H, W = bgr.shape[:2]
        x, y, w, h = box_norm[0] * W, box_norm[1] * H, box_norm[2] * W, box_norm[3] * H
        pad = 0.35 * max(w, h)
        x0, y0 = int(max(0, x - pad)), int(max(0, y - pad))
        x1, y1 = int(min(W, x + w + pad)), int(min(H, y + h + pad))
        crop = bgr[y0:y1, x0:x1]
        if crop.size and min(crop.shape[:2]) >= 24:
            scale = 1.0
            if min(crop.shape[:2]) < 160:  # upscale tiny crops so the detector can see the face
                scale = 160 / min(crop.shape[:2])
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            rows = self._detect_raw(crop, score_threshold=0.5)
            if len(rows):
                # Pick the detection that best overlaps the drawn box.
                cx, cy = (x + w / 2 - x0) * scale, (y + h / 2 - y0) * scale
                best = min(rows, key=lambda r: (r[0] + r[2] / 2 - cx) ** 2 + (r[1] + r[3] / 2 - cy) ** 2)
                emb, attrs = self.analyze_row(crop, best)
                return DetectedFace(tuple(box_norm), float(best[14]), emb, attrs)
        # No landmarks found: embed the box directly without alignment.
        face = bgr[int(max(0, y)):int(min(H, y + h)), int(max(0, x)):int(min(W, x + w))]
        if face.size == 0:
            face = bgr
        face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_AREA)
        expr, conf = self.expression(face)
        return DetectedFace(tuple(box_norm), 0.0, _normalize(self.recognizer.feature(face)), FaceAttrs(expr, conf))


def _normalize(feat: np.ndarray) -> np.ndarray:
    v = np.asarray(feat, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v
