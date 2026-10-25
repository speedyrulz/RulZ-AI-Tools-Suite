"""A demo "models folder" for RulZ LLM Suite screenshots: header-only GGUF files named after well-known open
models, with realistic metadata and (sparse, so no disk space is used) realistic file sizes.

They contain no weights: the app's library only reads the GGUF header, so the My Models page looks real, but
they can't be loaded (never press Load on them).

    python make_demo_gguf.py R:\\LLM
"""
from __future__ import annotations

import ctypes
import os
import struct
import sys
from ctypes import wintypes

GB = 1024 ** 3
# folder, file, arch, name, size label, ftype, ctx, blocks, embd, heads, kv heads, experts, size (GB)
MODELS = [
    ("google/gemma-3-27b-it-GGUF", "gemma-3-27b-it-Q4_K_M.gguf", "gemma3", "Gemma 3 27b It", "27B", 15, 131072, 62, 5376, 32, 16, 0, 15.6),
    ("google/gemma-3-12b-it-GGUF", "gemma-3-12b-it-Q5_K_M.gguf", "gemma3", "Gemma 3 12b It", "12B", 17, 131072, 48, 3840, 16, 8, 0, 7.9),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507-GGUF", "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", "qwen3moe", "Qwen3 30B A3B Instruct 2507", "30B-A3B", 15, 262144, 48, 2048, 32, 4, 128, 17.3),
    ("Qwen/Qwen3-14B-GGUF", "Qwen3-14B-Q5_K_M.gguf", "qwen3", "Qwen3 14B", "14B", 17, 40960, 40, 5120, 40, 8, 0, 9.8),
    ("Qwen/Qwen3-8B-GGUF", "Qwen3-8B-Q8_0.gguf", "qwen3", "Qwen3 8B", "8B", 7, 40960, 36, 4096, 32, 8, 0, 8.1),
    ("Qwen/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5-VL-7B-Instruct-Q6_K.gguf", "qwen2vl", "Qwen2.5 VL 7B Instruct", "7.6B", 18, 128000, 28, 3584, 28, 4, 0, 5.8),
    ("meta-llama/Llama-3.1-8B-Instruct-GGUF", "Llama-3.1-8B-Instruct-Q6_K.gguf", "llama", "Llama 3.1 8B Instruct", "8B", 18, 131072, 32, 4096, 32, 8, 0, 6.1),
    ("mistralai/Mistral-Small-3.2-24B-Instruct-2506-GGUF", "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf", "llama", "Mistral Small 3.2 24B Instruct 2506", "24B", 15, 131072, 40, 5120, 32, 8, 0, 13.3),
    ("openai/gpt-oss-20b-GGUF", "gpt-oss-20b-MXFP4.gguf", "gpt-oss", "Gpt Oss 20b", "21B-A3.6B", 38, 131072, 24, 2880, 64, 8, 32, 11.3),
    ("microsoft/Phi-4-mini-instruct-GGUF", "Phi-4-mini-instruct-Q8_0.gguf", "phi3", "Phi 4 Mini Instruct", "3.8B", 7, 131072, 32, 3072, 24, 8, 0, 3.8),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-GGUF", "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf", "qwen2", "DeepSeek R1 Distill Qwen 14B", "14B", 15, 131072, 48, 5120, 40, 8, 0, 8.4),
]
PROJECTORS = [  # vision projectors next to their models (shown as an eye icon)
    ("google/gemma-3-27b-it-GGUF", "mmproj-gemma-3-27b-it-F16.gguf", 0.8),
    ("google/gemma-3-12b-it-GGUF", "mmproj-gemma-3-12b-it-F16.gguf", 0.8),
    ("Qwen/Qwen2.5-VL-7B-Instruct-GGUF", "mmproj-Qwen2.5-VL-7B-Instruct-F16.gguf", 1.3),
]


def gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv(key: str, value) -> bytes:
    if isinstance(value, str):
        return gguf_str(key) + struct.pack("<I", 8) + gguf_str(value)
    return gguf_str(key) + struct.pack("<I", 4) + struct.pack("<I", value)


def header(pairs: dict) -> bytes:
    body = b"".join(kv(k, v) for k, v in pairs.items())
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(pairs)) + body


def make_sparse(path: str, data: bytes, size: int):
    with open(path, "wb") as f:
        f.write(data)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                wintypes.DWORD, wintypes.HANDLE]
    k32.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.LPVOID,
                                    wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong),
                                     wintypes.DWORD]
    k32.SetEndOfFile.argtypes = [wintypes.HANDLE]
    h = k32.CreateFileW(path, 0x40000000, 0, None, 3, 0x80, None)  # GENERIC_WRITE, OPEN_EXISTING
    if h == wintypes.HANDLE(-1).value:
        raise OSError(ctypes.get_last_error(), "CreateFileW")
    try:
        n = wintypes.DWORD()
        if not k32.DeviceIoControl(h, 0x900C4, None, 0, None, 0, ctypes.byref(n), None):  # FSCTL_SET_SPARSE
            raise OSError(ctypes.get_last_error(), "FSCTL_SET_SPARSE")
        # Extend with SetEndOfFile. (Not os.truncate: the C runtime extends files by writing zeros, which
        # would allocate the full size even on a sparse file.)
        if not (k32.SetFilePointerEx(h, size, None, 0) and k32.SetEndOfFile(h)):
            raise OSError(ctypes.get_last_error(), "SetEndOfFile")
    finally:
        k32.CloseHandle(h)
    # safety net: stop at once if the file really occupies its size on disk
    k32.GetCompressedFileSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    high = wintypes.DWORD()
    low = k32.GetCompressedFileSizeW(path, ctypes.byref(high))
    on_disk = (high.value << 32) + low
    if on_disk > 16 * 1024 * 1024:
        os.remove(path)
        raise SystemExit(f"{path}: not sparse ({on_disk} bytes on disk), aborted")


def main(root: str):
    for folder, fname, arch, name, label, ftype, ctx, blocks, embd, heads, kvh, experts, gb in MODELS:
        d = os.path.join(root, *folder.split("/"))
        os.makedirs(d, exist_ok=True)
        pairs = {"general.architecture": arch, "general.type": "model", "general.name": name,
                 "general.size_label": label, "general.file_type": ftype, f"{arch}.context_length": ctx,
                 f"{arch}.block_count": blocks, f"{arch}.embedding_length": embd,
                 f"{arch}.attention.head_count": heads, f"{arch}.attention.head_count_kv": kvh,
                 "tokenizer.chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}"}
        if experts:
            pairs[f"{arch}.expert_count"] = experts
            pairs[f"{arch}.expert_used_count"] = 8 if experts == 128 else 4
        make_sparse(os.path.join(d, fname), header(pairs), int(gb * GB))
        print(os.path.join(d, fname))
    for folder, fname, gb in PROJECTORS:
        d = os.path.join(root, *folder.split("/"))
        pairs = {"general.architecture": "clip", "general.type": "mmproj", "general.name": fname[:-5],
                 "general.file_type": 1}
        make_sparse(os.path.join(d, fname), header(pairs), int(gb * GB))
        print(os.path.join(d, fname))


if __name__ == "__main__":
    main(sys.argv[1])
