#!/usr/bin/env python3
"""Minimal reproducer for the large MUSA GEMM seen in depth unprojection.

The Step 4 frame-0 human mask contains 65,970 valid depth pixels. PyTorch3D's
Transform3d turns those points into a [1, 65970, 4] x [1, 4, 4] bmm, matching
the failing mudmp launch grid. This script intentionally keeps that original
shape for torch_musa debugging; it is not part of normal GRAIL validation.

Run it in a disposable subprocess because the default size may raise a MUSA
illegal-address fault:

    MUSA_LAUNCH_BLOCKING=1 python scripts/repro_depth_unprojection_gemm_musa.py

Use ``--num_points 3000`` to exercise the bounded shape used by the GRAIL fix.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch_musa  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="musa:0")
    parser.add_argument("--num_points", type=int, default=65_970)
    parser.add_argument("--disable_tf32", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.musa.is_available():
        raise RuntimeError("MUSA is not available")

    device = torch.device(args.device)
    torch.musa.set_device(device)
    torch.backends.mudnn.allow_tf32 = not args.disable_tf32

    points_h = torch.ones((1, args.num_points, 4), dtype=torch.float32, device=device)
    transform = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)

    print(f"device={device}")
    print(f"MUSA_LAUNCH_BLOCKING={os.environ.get('MUSA_LAUNCH_BLOCKING')}")
    print(f"allow_tf32={torch.backends.mudnn.allow_tf32}")
    print(f"bmm={tuple(points_h.shape)} x {tuple(transform.shape)}")

    output = points_h.bmm(transform)
    torch.musa.synchronize()
    print(f"PASS shape={tuple(output.shape)} finite={bool(torch.isfinite(output).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
