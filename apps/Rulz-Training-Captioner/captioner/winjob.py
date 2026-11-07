"""Tie child processes to this app's lifetime with a Windows job object.

llama-server holds gigabytes of VRAM and a port. Without this, a crash or a kill from Task Manager
leaves it running (atexit never runs), and the next start finds the port taken. Processes in a job
created with KILL_ON_JOB_CLOSE are ended by Windows when the last handle to the job closes, which
happens however this process exits.
"""
import ctypes
import os
import threading
from ctypes import wintypes

JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
PROCESS_TERMINATE = 0x0001
PROCESS_SET_QUOTA = 0x0100


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


_lock = threading.Lock()
_job = None
_k32 = None


def _kernel32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    k.SetInformationJobObject.restype = wintypes.BOOL
    k.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k.AssignProcessToJobObject.restype = wintypes.BOOL
    k.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    k.CloseHandle.restype = wintypes.BOOL
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    return k


def _create_job(k):
    job = k.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = _ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k.SetInformationJobObject(job, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION_CLASS,
                                     ctypes.byref(info), ctypes.sizeof(info)):
        err = ctypes.get_last_error()
        k.CloseHandle(job)
        raise ctypes.WinError(err)
    return job  # kept open for the life of the process on purpose


def adopt(pid):
    """End process `pid` when this app exits, however it exits. Returns False when that couldn't be
    arranged (not Windows, or the call failed); the process still runs, it just isn't tied to us."""
    global _job, _k32
    if os.name != "nt":
        return False
    try:
        with _lock:
            if _job is None:
                _k32 = _kernel32()
                _job = _create_job(_k32)
        h = _k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not h:
            return False
        try:
            return bool(_k32.AssignProcessToJobObject(_job, h))
        finally:
            _k32.CloseHandle(h)
    except OSError:
        return False
