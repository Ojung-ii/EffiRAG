from __future__ import annotations

import time


def process_rss_mb() -> float:
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        try:
            import resource

            # Linux ru_maxrss is KB.
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0


def reset_gpu_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        return


def gpu_peak_mb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    except Exception:
        return 0.0
    return 0.0


class Timer:
    def __init__(self) -> None:
        self._start = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
