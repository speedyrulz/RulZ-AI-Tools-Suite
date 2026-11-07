"""Tie child processes to this app's lifetime with Windows job objects.

llama-server holds gigabytes of VRAM and a port. Without this, a crash or a kill from Task Manager
leaves it running (atexit never runs), and the next start finds the port taken. Processes in a job
created with KILL_ON_JOB_CLOSE are ended by Windows when the last handle to the job closes, which
happens however this process exits.

Each server gets its own job, which also makes stopping it fast: terminating the job ends the server
and anything it started at once, where `taskkill /T` takes about 0.75 s per server.
"""
import ctypes
import os
import threading
import time
from ctypes import wintypes

JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS = 1
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


class _Accounting(ctypes.Structure):
    _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]


_lock = threading.Lock()
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
    k.TerminateJobObject.restype = wintypes.BOOL
    k.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    k.QueryInformationJobObject.restype = wintypes.BOOL
    k.QueryInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
                                            wintypes.LPVOID)
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
    return job


class Job:
    """A job holding one process and whatever it starts. Its handle stays open until close(), on
    purpose: closing it (or this app exiting, however it exits) ends the processes in it."""

    def __init__(self, k32, handle):
        self._k32, self._handle = k32, handle

    def terminate(self):
        """End every process in the job now. Returns False if that failed or the job is closed."""
        return bool(self._handle) and bool(self._k32.TerminateJobObject(self._handle, 1))

    def wait(self, timeout):
        """Wait until every process in the job has exited (terminate() returns before they have).
        Returns True once the job is empty."""
        info = _Accounting()
        deadline = time.monotonic() + timeout
        while self._handle and self._k32.QueryInformationJobObject(
                self._handle, JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION_CLASS, ctypes.byref(info),
                ctypes.sizeof(info), None):
            if info.ActiveProcesses == 0:
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return False

    def close(self):
        if self._handle:
            self._k32.CloseHandle(self._handle)
            self._handle = None


def adopt(pid):
    """Put process `pid` in a new Job, so it ends when this app exits, however it exits. Returns the
    Job, or None when that couldn't be arranged (not Windows, or the call failed); the process still
    runs then, it just isn't tied to us."""
    global _k32
    if os.name != "nt":
        return None
    try:
        with _lock:
            if _k32 is None:
                _k32 = _kernel32()
        job = _create_job(_k32)
        h = _k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        ok = bool(h) and bool(_k32.AssignProcessToJobObject(job, h))
        if h:
            _k32.CloseHandle(h)
        if not ok:
            _k32.CloseHandle(job)
            return None
        return Job(_k32, job)
    except OSError:
        return None
