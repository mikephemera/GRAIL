"""WiLoR hand-pose adapter for the GRAIL Step 1 MUSA path."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch

from grail.core.device import empty_cache, require_musa


_GRAIL_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WILOR_ROOT = _GRAIL_ROOT.parent / "WiLoR_musa"


def _setup_imports(wilor_root: str | Path | None) -> Path:
    root = Path(wilor_root or _DEFAULT_WILOR_ROOT).expanduser().resolve()
    package = root / "wilor_mini" / "__init__.py"
    if not package.is_file():
        raise FileNotFoundError(f"WiLoR source tree not found: {root}")

    loaded = sys.modules.get("wilor_mini")
    loaded_path = getattr(loaded, "__file__", None)
    if loaded_path and not str(Path(loaded_path).resolve()).startswith(str(root) + "/"):
        raise RuntimeError(
            f"wilor_mini is already imported from {loaded_path}, but Step 1 requested {root}. "
            "Run Step 1 in a fresh Python process."
        )

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def infer_hand_pose(
    video_path,
    device="auto",
    wilor_root=None,
    pretrained_dir=None,
    conf_threshold=0.5,
):
    """Run WiLoR for every video frame and return GRAIL-compatible MANO records."""
    import cv2

    resolved = require_musa(device, "WiLoR")
    root = _setup_imports(wilor_root)

    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    model_dir = Path(pretrained_dir).expanduser().resolve() if pretrained_dir else root / "wilor_mini"
    pipe = WiLorHandPose3dEstimationPipeline(
        device=resolved,
        dtype=torch.float16,
        verbose=False,
        wilor_pretrained_dir=str(model_dir),
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open WiLoR input video: {video_path}")

    outputs = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_output = pipe.predict(frame, hand_conf=conf_threshold)
            for hand in frame_output:
                # New WiLoR_musa emits this from the same detector invocation.
                # Keep compatibility with older model packages without running
                # the detector twice and risking a different hand ordering.
                hand.setdefault("bbox_conf", 1.0)
            outputs.append(frame_output)
    finally:
        cap.release()
        del pipe
        gc.collect()
        empty_cache(resolved)

    return outputs
