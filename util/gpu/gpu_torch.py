import os
import torch
from contextlib import contextmanager

USE_GPU = os.getenv("USE_GPU", "true").lower() in ("1", "true", "yes")

_avail = USE_GPU and torch.cuda.is_available()
NUM_GPUS = torch.cuda.device_count() if _avail else 0

def gpu_enabled():
    #return _avail
    return False

def current_device(rank=0):
    if not _avail:
        raise RuntimeError("GPU requested but CUDA not available")
    return torch.device(f"cuda:{rank % NUM_GPUS}")

@contextmanager
def torch_device(rank=0):
    if _avail:
        with torch.cuda.device(current_device(rank)):
            yield
    else:
        yield