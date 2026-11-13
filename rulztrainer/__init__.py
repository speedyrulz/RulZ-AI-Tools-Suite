"""RulzTrainer: a simple diffusion LoRA trainer with a WebUI."""
import os as _os

__version__ = "3.14.0"

# No telemetry. onnxruntime's Linux builds (1.29 and later, loaded for InsightFace's face scoring) report a device ID
# and machine details to Microsoft unless ORT_DISABLE_TELEMETRY is set, and Hugging Face's libraries send usage pings
# unless HF_HUB_DISABLE_TELEMETRY is. Set before either loads, so every entry point and the jobs it starts inherit
# them; a value already set wins. (Found by Fizgig, 9.2.0.)
_os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Hugging Face downloads: hf_xet raises its number of parallel connections up to 64 while a file downloads, which
# saturates a home connection (everything else on it stalls). A fixed 8 stops that and still fills a fast link; set
# HF_XET_FIXED_DOWNLOAD_CONCURRENCY yourself for another value. (Found by OneTrainer, #1656.)
_os.environ.setdefault("HF_XET_FIXED_DOWNLOAD_CONCURRENCY", "8")


def onnxruntime_telemetry_off() -> None:
    """onnxruntime's Windows builds have their own switch instead of the variable: call it where onnxruntime loads.
    Never raises (no onnxruntime, or a build without the function)."""
    try:
        import onnxruntime

        onnxruntime.disable_telemetry_events()
    except Exception:  # noqa: BLE001
        pass
