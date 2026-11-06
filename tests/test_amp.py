"""Autocast resolution: the measured gain is CUDA-only, so the policy must not
silently switch on hardware where it has never been measured."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from azurite.env import amp_policy  # noqa: E402


def test_off_is_off_everywhere() -> None:
    for device in ("cuda", "mps", "cpu"):
        assert amp_policy("off", device) is None


def test_auto_enables_bf16_on_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert amp_policy("auto", "cuda") == ("cuda", torch.bfloat16)


def test_auto_falls_back_to_fp16_without_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert amp_policy("auto", "cuda") == ("cuda", torch.float16)


def test_auto_stays_off_where_it_was_never_measured() -> None:
    # MPS autocast support varies by torch build; a laptop run must not break
    # on a default. Explicit requests still get through (next test).
    assert amp_policy("auto", "mps") is None
    assert amp_policy("auto", "cpu") is None


def test_explicit_dtype_reaches_the_requested_device() -> None:
    assert amp_policy("bf16", "mps") == ("mps", torch.bfloat16)
    assert amp_policy("fp16", "cuda") == ("cuda", torch.float16)
    assert amp_policy("fp16", "cpu") == ("cpu", torch.float16)


def test_unknown_request_is_a_programming_error() -> None:
    with pytest.raises(ValueError):
        amp_policy("fp8", "cuda")


def test_cpu_bf16_autocast_context_is_usable() -> None:
    # Guards the integration: the context manager must accept the resolved
    # pair on a device that has no tensor cores at all.
    dev, dtype = amp_policy("bf16", "cpu")
    a = torch.randn(8, 8)
    b = torch.randn(8, 8)
    with torch.autocast(device_type=dev, dtype=dtype, enabled=True):
        c = a @ b
    assert c.shape == (8, 8)
    assert torch.isfinite(c).all()
