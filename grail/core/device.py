"""Runtime device helpers shared by the reconstruction pipeline adapters."""

from __future__ import annotations

import torch


def musa_available() -> bool:
    """Return whether the installed torch_musa runtime exposes a usable device."""
    try:
        import torch_musa  # noqa: F401
    except Exception:
        return False
    try:
        return bool(torch.musa.is_available())
    except Exception:
        return False


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve ``auto`` with a MUSA-first policy and normalize device indices."""
    requested = str(device or "auto").lower()

    if requested == "auto":
        if musa_available():
            return torch.device("musa:0")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    if requested == "musa":
        requested = "musa:0"
    elif requested == "cuda":
        requested = "cuda:0"

    # Older configs commonly say cuda:0. In a MUSA-only installation this is
    # an accelerator request, not permission to silently run a heavy model on CPU.
    if requested.startswith("cuda") and not torch.cuda.is_available() and musa_available():
        index = requested.split(":", 1)[1] if ":" in requested else "0"
        requested = f"musa:{index}"

    return torch.device(requested)


def prepare_device(device: str | torch.device | None = "auto") -> torch.device:
    """Resolve a device, import its backend, and select the requested accelerator."""
    resolved = resolve_device(device)
    if resolved.type == "musa":
        import torch_musa  # noqa: F401

        if not torch.musa.is_available():
            raise RuntimeError(f"MUSA device requested but unavailable: {resolved}")
        torch.musa.set_device(resolved)
    elif resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {resolved}")
        torch.cuda.set_device(resolved)
    return resolved


def empty_cache(device: str | torch.device | None) -> None:
    """Release allocator cache for the selected accelerator."""
    resolved = resolve_device(device)
    if resolved.type == "musa" and musa_available():
        torch.musa.empty_cache()
    elif resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def require_musa(device: str | torch.device | None, component: str) -> torch.device:
    """Prepare and validate the MUSA-only migration path for a component."""
    resolved = prepare_device(device)
    if resolved.type != "musa":
        raise RuntimeError(
            f"{component} MUSA path requires a musa device, got {resolved}. "
            "Use --device musa:0 on the S5000 host."
        )
    return resolved
