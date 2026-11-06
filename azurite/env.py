"""Execution environment: memory cap, allocator reclaim, and the swap guard.

Three measured facts drive this module, all from `scripts/model_smoke.py` and the
probe that preceded it:

1. **`driver_allocated_memory()` is reservation, not usage.** On MPS the caching
   allocator keeps freed blocks. Measured on the BPE tiny rung: live memory
   stayed at **0.13 GiB** across three steps while reserved memory climbed
   3.64 -> 3.93 -> 4.52 GiB. A guard that compares reservation to a cap kills a
   healthy run; a guard that compares *live* memory to a cap does not.

2. **Reservation grows monotonically without reclaim.** A 32k-vocabulary MLM
   step reserves roughly 300 MB for logits and does not return it. In a
   multi-day run that is the difference between finishing and dying at hour 40,
   so `MemoryGuard.reclaim_if_needed()` is called between steps.

3. **`empty_cache()` does not return to zero.** After full reclaim the process
   still shows ~1.6 GiB reserved — Metal library and driver pools. The floor is
   real and is budgeted for rather than fought.

The swap guard is the one rule that is not negotiable: this project's measured
envelope was produced with **zero swap**, and a run that starts paging to SSD is
killed rather than tolerated.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field

import torch


def swap_used_mb() -> float:
    """System swap in use, in MB. Same reader as `scripts/local_bench.py`."""
    if sys.platform != "darwin":
        try:
            with open("/proc/meminfo") as fh:
                for line in fh:
                    if line.startswith("SwapTotal:"):
                        total = int(line.split()[1])
                    elif line.startswith("SwapFree:"):
                        free = int(line.split()[1])
                return (total - free) / 1024.0
        except OSError:
            return 0.0
    out = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, check=False
    ).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) if m else 0.0


def pageouts() -> int:
    """Cumulative page-out count, if the platform reports one."""
    try:
        out = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return 0
    m = re.search(r"Pageouts:\s+(\d+)", out)
    return int(m.group(1)) if m else 0


def configure_mps(cap_gib: float) -> dict:
    """Apply a hard MPS allocation cap. Returns what was actually applied."""
    if not torch.backends.mps.is_available():
        return {"mps": False, "cap_gib": cap_gib, "fraction": 1.0, "recommended_gib": None}
    cap_bytes = int(cap_gib * 2**30)
    recommended = torch.mps.recommended_max_memory()
    fraction = min(1.0, cap_bytes / recommended)
    torch.mps.set_per_process_memory_fraction(fraction)
    return {
        "mps": True,
        "cap_gib": cap_gib,
        "fraction": fraction,
        "recommended_gib": recommended / 2**30,
    }


def select_device() -> str:
    """Pick the training device from what is actually installed.

    CUDA is checked first, and that ordering is the whole point. Before this
    function existed, `pretrain.py` chose with `"mps" if applied["mps"] else
    "cpu"`. On a Linux CUDA box `torch.backends.mps.is_available()` is importable
    and simply false, so the ternary fell through to "cpu" and a machine with a
    102 GB RTX Pro 6000 trained on its CPU cores without raising anything. A
    silent downgrade is the worst possible failure here: the loss curve still
    looks plausible, only 100x slower.

    Reported by `scripts/pretrain.py` and recorded in `summary.json` so every
    number carries the device that produced it.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def amp_policy(requested: str, device: str | None = None) -> tuple[str, torch.dtype] | None:
    """Resolve an autocast request into (device_type, dtype), or None for off.

    Measured on the 403M rung (batch 8, seq 1024): 535 ms/step in fp32 versus
    223 ms/step under bf16 autocast, a 2.4x gain at identical geometry. The
    fp32 path never touches tensor cores, which is why the gap is so large.

    `auto` enables that gain on CUDA and deliberately does nothing on MPS and
    CPU. MPS autocast support depends on the torch build and has not been
    measured on this project; a default that can abort a laptop run is worse
    than a default that is merely slower. Both stay reachable explicitly.
    """
    dev = device or select_device()
    if requested == "off":
        return None
    if requested == "auto":
        if dev != "cuda":
            return None
        bf16 = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return ("cuda", bf16)
    if requested == "bf16":
        return (dev, torch.bfloat16)
    if requested == "fp16":
        return (dev, torch.float16)
    raise ValueError(f"unknown amp request {requested!r}")


def device_report() -> dict:
    """Device identity plus the memory figures the status cell reports."""
    device = select_device()
    info: dict = {"device": device}
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "name": torch.cuda.get_device_name(0),
                "total_gib": round(props.total_memory / 2**30, 2),
                "capability": f"sm_{props.major}{props.minor}",
            }
        )
    return info


@dataclass
class MemoryGuard:
    """Live-memory cap plus a reservation reclaim policy.

    `cap_gib` bounds *live* allocation. `reserve_ratio` triggers `empty_cache()`
    once reserved memory passes that fraction of the cap, which keeps the
    allocator from walking into the hard limit between checks.
    """

    cap_gib: float
    reserve_ratio: float = 0.75
    peak_live_gib: float = 0.0
    peak_reserved_gib: float = 0.0
    reclaims: int = 0
    swap_start_mb: float = field(default_factory=swap_used_mb)
    pageouts_start: int = field(default_factory=pageouts)

    def live_gib(self) -> float:
        """Allocated, not reserved. On CUDA that is `memory_allocated`."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 2**30
        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / 2**30
        return 0.0

    def reserved_gib(self) -> float:
        """What the caching allocator is holding, freed blocks included."""
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved() / 2**30
        if torch.backends.mps.is_available():
            return torch.mps.driver_allocated_memory() / 2**30
        return 0.0

    def check(self) -> None:
        """Raise if live memory exceeds the cap. Called every step.

        `cap_gib <= 0` means "no cap": the figures are still tracked and
        reported, which is what a 102 GB device wants. A 6 GiB default carried
        over from the laptop would have aborted the first CUDA step.
        """
        live = self.live_gib()
        reserved = self.reserved_gib()
        self.peak_live_gib = max(self.peak_live_gib, live)
        self.peak_reserved_gib = max(self.peak_reserved_gib, reserved)
        if self.cap_gib > 0 and live > self.cap_gib:
            raise MemoryError(
                f"live allocation {live:.2f} GiB exceeded cap {self.cap_gib:.2f} GiB "
                f"on {select_device()}"
            )

    def reclaim_if_needed(self) -> bool:
        """Drop cached blocks when reservation approaches the cap."""
        if self.cap_gib <= 0:
            return False
        if torch.cuda.is_available():
            if self.reserved_gib() > self.cap_gib * self.reserve_ratio:
                torch.cuda.empty_cache()
                self.reclaims += 1
                return True
            return False
        if not torch.backends.mps.is_available():
            return False
        if self.reserved_gib() > self.cap_gib * self.reserve_ratio:
            torch.mps.synchronize()
            torch.mps.empty_cache()
            self.reclaims += 1
            return True
        return False

    def swap_delta_mb(self) -> float:
        return round(swap_used_mb() - self.swap_start_mb, 1)

    def pageout_delta(self) -> int:
        return pageouts() - self.pageouts_start

    def assert_zero_swap(self) -> None:
        """The operating condition. A run that pages is a failed run."""
        delta = self.swap_delta_mb()
        pages = self.pageout_delta()
        if delta > 16 or pages > 128:
            raise RuntimeError(
                f"swap grew by {delta:.0f} MB and {pages} page-outs during this "
                "run; the measured envelope assumes zero swap"
            )

    def report(self) -> dict:
        out = {
            "device": select_device(),
            "cap_gib": self.cap_gib,
            "peak_live_gib": round(self.peak_live_gib, 3),
            "peak_reserved_gib": round(self.peak_reserved_gib, 3),
            "reclaims": self.reclaims,
            "swap_delta_mb": self.swap_delta_mb(),
            "pageout_delta": self.pageout_delta(),
        }
        if torch.cuda.is_available():
            # The allocator's own high-water mark, independent of how often we
            # sampled: this is the number a VRAM budget should be planned from.
            out["peak_cuda_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
        return out
