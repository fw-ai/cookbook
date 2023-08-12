import os
from typing import Optional
import torch
import torch.distributed as dist


class Env:
    def __init__(self):
        self._init_defaults()
        self._local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            self._device = torch.device(f"cuda:{self._local_rank}")
            torch.cuda.set_device(self._device)
            self._backend = "nccl"
        else:
            self._device = torch.device("cpu")
            self._backend = "gloo"
        dist.init_process_group(backend=self._backend)

    def _init_defaults(self) -> None:
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["MASTER_PORT"] = "30001"
            os.environ["MASTER_ADDR"] = "127.0.0.1"

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def backend(self) -> str:
        return self._backend

_env: Optional[Env] = None


def env() -> Env:
    global _env
    if _env is None:
        raise RuntimeError("call init_env() to initialize")
    return _env


def init_env() -> Env:
    global _env
    if _env is not None:
        return _env
    _env = Env()
    return _env
