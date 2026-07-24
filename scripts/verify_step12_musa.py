#!/usr/bin/env python3
"""Preflight and artifact-contract verifier for GRAIL Step 1/2 on MUSA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


GRAIL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GRAIL_ROOT.parent
DEFAULT_ID = "ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="musa:0")
    parser.add_argument(
        "--genmo_root", type=Path, default=WORKSPACE_ROOT / "GENMO_musa"
    )
    parser.add_argument(
        "--genmo_asset_root", type=Path, default=GRAIL_ROOT / "imports" / "GEM-SMPL"
    )
    parser.add_argument("--genmo_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--genmo_text_encoder",
        type=Path,
        default=Path("/root/.cache/huggingface/hub/models--t5-3b"),
    )
    parser.add_argument(
        "--wilor_root", type=Path, default=WORKSPACE_ROOT / "WiLoR_musa"
    )
    parser.add_argument("--wilor_pretrained_dir", type=Path, default=None)
    parser.add_argument(
        "--moge_root", type=Path, default=GRAIL_ROOT / "imports" / "MoGe"
    )
    parser.add_argument(
        "--sam2_model_cache",
        type=Path,
        default=Path("/root/.cache/huggingface/hub/models--facebook--sam2-hiera-large"),
    )
    parser.add_argument(
        "--moge_model_cache",
        type=Path,
        default=Path(
            "/root/.cache/huggingface/hub/models--Ruicheng--moge-2-vitl-normal"
        ),
    )
    parser.add_argument(
        "--step1_npz",
        type=Path,
        default=GRAIL_ROOT / "results/generation/hmr_smplx" / f"{DEFAULT_ID}.npz",
    )
    parser.add_argument(
        "--reference_step1_npz",
        type=Path,
        default=None,
        help="Optional reference Step 1 artifact used for drift reporting.",
    )
    parser.add_argument(
        "--masks_npz",
        type=Path,
        default=GRAIL_ROOT
        / "results/generation/4dhoi_recon_cache/masks"
        / f"{DEFAULT_ID}.npz",
    )
    parser.add_argument(
        "--depth_pt",
        type=Path,
        default=GRAIL_ROOT
        / "results/generation/4dhoi_recon_cache/depth"
        / f"{DEFAULT_ID}.pt",
    )
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as handle:
        return handle.read(40).startswith(b"version https://git-lfs.github.com/spec")


def file_check(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "lfs_pointer": is_lfs_pointer(path),
        "size": path.stat().st_size if path.is_file() else None,
    }


def hf_cache_check(path: Path, required_names: tuple[str, ...]) -> dict[str, Any]:
    """Check a materialized Hugging Face cache without requiring network access."""

    snapshots = (
        sorted(p for p in path.glob("snapshots/*") if p.is_dir())
        if path.is_dir()
        else []
    )
    snapshot = path if path.is_dir() and (path / required_names[0]).is_file() else None
    if snapshot is None and snapshots:
        snapshot = snapshots[-1]
    files = {
        name: file_check(snapshot / name) if snapshot else file_check(path / name)
        for name in required_names
    }
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "snapshot": str(snapshot) if snapshot else None,
        "files": files,
        "status": "PASS"
        if path.is_dir()
        and snapshot
        and all(info["exists"] and not info["lfs_pointer"] for info in files.values())
        else "BLOCKED",
    }


def import_check(args: argparse.Namespace) -> dict[str, Any]:
    """Import only the GENMO inference graph and verify native T5 norm choice."""

    previous_root = os.environ.get("HMR4D_PROJECT_ROOT")
    os.environ["HMR4D_PROJECT_ROOT"] = str(args.genmo_asset_root)
    for entry in (args.genmo_root, args.genmo_root / "tools" / "demo"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    try:
        import hmr4d.model.genmo.genmo_demo  # noqa: F401
        import hmr4d.model.gvhmr.utils.endecoder  # noqa: F401
        from hmr4d.model.genmo.inference import load_data_dict, run_preprocess
        from hmr4d.model.genmo.genmo_demo import ensure_musa_t5_layer_norm

        norm = ensure_musa_t5_layer_norm()
        return {
            "status": "PASS",
            "inference_module": str(run_preprocess.__module__),
            "data_loader_module": str(load_data_dict.__module__),
            "t5_layer_norm": f"{norm.__module__}.{norm.__name__}",
            "training_registry_imported": "hmr4d.configs.store_gvhmr" in sys.modules,
            "optional_imports_loaded": {
                name: name in sys.modules
                for name in ("pytorch_lightning", "wis3d", "seaborn")
            },
        }
    except Exception as exc:  # noqa: BLE001 - report exact preflight blocker
        return {"status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if previous_root is None:
            os.environ.pop("HMR4D_PROJECT_ROOT", None)
        else:
            os.environ["HMR4D_PROJECT_ROOT"] = previous_root


def validate_step1(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "status": "MISSING"}
    if not path.is_file():
        return result
    if is_lfs_pointer(path):
        return {**result, "status": "LFS_POINTER"}

    with np.load(path, allow_pickle=True) as archive:
        if set(archive.files) != {"motion_global", "motion_incam"}:
            return {**result, "status": "FAIL", "keys": archive.files}
        global_motion = archive["motion_global"].item()
        incam_motion = archive["motion_incam"].item()

    required_global = {
        "poses",
        "betas",
        "trans",
        "left_hand_pose",
        "right_hand_pose",
        "predicted_body_height",
    }
    required_incam = required_global | {
        "vitpose",
        "hand_keypoints_2d",
        "foot_contact_probs",
    }
    missing = {
        "motion_global": sorted(required_global - set(global_motion)),
        "motion_incam": sorted(required_incam - set(incam_motion)),
    }
    poses = np.asarray(incam_motion.get("poses"))
    frames = int(poses.shape[0]) if poses.ndim == 2 else None
    shapes = {
        key: list(np.asarray(incam_motion[key]).shape)
        for key in (
            "poses",
            "trans",
            "left_hand_pose",
            "right_hand_pose",
            "vitpose",
            "hand_keypoints_2d",
            "foot_contact_probs",
        )
        if incam_motion.get(key) is not None
    }
    shape_ok = bool(
        frames
        and shapes.get("poses") == [frames, 165]
        and shapes.get("trans") == [frames, 3]
        and shapes.get("left_hand_pose") == [frames, 45]
        and shapes.get("right_hand_pose") == [frames, 45]
        and shapes.get("vitpose") == [frames, 17, 3]
        and shapes.get("hand_keypoints_2d") == [frames, 32, 3]
        and shapes.get("foot_contact_probs") == [frames, 4]
    )
    ok = not any(missing.values()) and shape_ok
    return {
        **result,
        "status": "PASS" if ok else "FAIL",
        "frames": frames,
        "missing": missing,
        "shapes": shapes,
    }


def compare_step1(path: Path, reference: Path | None) -> dict[str, Any] | None:
    if reference is None:
        return None
    result: dict[str, Any] = {
        "path": str(path),
        "reference": str(reference),
        "status": "BLOCKED",
    }
    if not path.is_file() or not reference.is_file():
        return result

    def load_step1(item: Path) -> dict[str, dict[str, Any]]:
        with np.load(item, allow_pickle=True) as archive:
            return {
                "motion_global": archive["motion_global"].item(),
                "motion_incam": archive["motion_incam"].item(),
            }

    def metric(current: Any, expected: Any) -> dict[str, Any]:
        current_array = np.asarray(current, dtype=np.float64)
        expected_array = np.asarray(expected, dtype=np.float64)
        if current_array.shape != expected_array.shape:
            return {
                "status": "SHAPE_MISMATCH",
                "shape": list(current_array.shape),
                "reference_shape": list(expected_array.shape),
            }
        diff = current_array - expected_array
        abs_diff = np.abs(diff)
        return {
            "status": "PASS"
            if np.allclose(current_array, expected_array, rtol=1e-4, atol=1e-5)
            else "WARN",
            "shape": list(current_array.shape),
            "max_abs": float(abs_diff.max()) if abs_diff.size else 0.0,
            "mean_abs": float(abs_diff.mean()) if abs_diff.size else 0.0,
            "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        }

    def rotation_metric(current: Any, expected: Any) -> dict[str, float]:
        """Axis-angle drift without treating equivalent 2-pi wraps as errors."""

        def as_quaternion(rotvec: Any) -> np.ndarray:
            rotvec = np.asarray(rotvec, dtype=np.float64).reshape(-1, 3)
            theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
            scale = np.divide(
                np.sin(theta / 2.0),
                theta,
                out=np.full_like(theta, 0.5),
                where=theta > 1e-12,
            )
            return np.concatenate([np.cos(theta / 2.0), rotvec * scale], axis=-1)

        current_quat = as_quaternion(current)
        expected_quat = as_quaternion(expected)
        cosine = np.clip(
            np.abs(np.sum(current_quat * expected_quat, axis=-1)), 0.0, 1.0
        )
        degrees = np.degrees(2.0 * np.arccos(cosine))
        return {
            "mean": float(degrees.mean()),
            "p95": float(np.percentile(degrees, 95)),
            "max": float(degrees.max()),
        }

    current = load_step1(path)
    expected = load_step1(reference)
    fields = {
        "motion_global": (
            "poses",
            "betas",
            "trans",
            "left_hand_pose",
            "right_hand_pose",
        ),
        "motion_incam": (
            "poses",
            "betas",
            "trans",
            "left_hand_pose",
            "right_hand_pose",
            "vitpose",
            "hand_keypoints_2d",
            "foot_contact_probs",
            "predicted_body_height",
        ),
    }
    metrics = {
        section: {
            field: metric(current[section][field], expected[section][field])
            for field in section_fields
            if field in current[section] and field in expected[section]
        }
        for section, section_fields in fields.items()
    }
    for section in ("motion_global", "motion_incam"):
        if "poses" in metrics[section]:
            metrics[section]["poses"]["geodesic_degrees"] = rotation_metric(
                current[section]["poses"], expected[section]["poses"]
            )
    statuses = [
        item["status"] for section in metrics.values() for item in section.values()
    ]
    return {
        **result,
        "status": "PASS"
        if statuses and all(status == "PASS" for status in statuses)
        else "WARN",
        "metrics": metrics,
    }


def validate_masks(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "status": "MISSING"}
    if not path.is_file():
        return result
    if is_lfs_pointer(path):
        return {**result, "status": "LFS_POINTER"}
    with np.load(path, allow_pickle=True) as archive:
        if "masks" not in archive.files:
            return {**result, "status": "FAIL", "keys": archive.files}
        masks = archive["masks"].item()
    frames = len(masks)
    object_ids_ok = all(0 in frame and 1 in frame for frame in masks.values())
    return {
        **result,
        "status": "PASS" if frames > 0 and object_ids_ok else "FAIL",
        "frames": frames,
        "object_ids_0_1": object_ids_ok,
    }


def validate_depth(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "status": "MISSING"}
    if not path.is_file():
        return result
    if is_lfs_pointer(path):
        return {**result, "status": "LFS_POINTER"}
    depths = torch.load(path, map_location="cpu", weights_only=False)
    tensors_ok = (
        isinstance(depths, (list, tuple))
        and bool(depths)
        and all(isinstance(depth, torch.Tensor) and depth.ndim == 2 for depth in depths)
    )
    finite = tensors_ok and all(bool(torch.isfinite(depth).all()) for depth in depths)
    return {
        **result,
        "status": "PASS" if tensors_ok and finite else "FAIL",
        "frames": len(depths) if isinstance(depths, (list, tuple)) else None,
        "first_shape": list(depths[0].shape) if tensors_ok else None,
        "cpu_portable": tensors_ok
        and all(depth.device.type == "cpu" for depth in depths),
    }


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(GRAIL_ROOT))
    from grail.core.device import require_musa

    device = require_musa(args.device, "GRAIL Step 1/2 verifier")
    genmo_checkpoint = args.genmo_checkpoint or (
        args.genmo_asset_root
        / "outputs/mocap_mixed_v1/genmo/genmo_lg_jukebox_jukebox_new/version_0/checkpoints/last.ckpt"
    )
    wilor_assets = args.wilor_pretrained_dir or args.wilor_root / "wilor_mini"

    sources = {
        "genmo": (args.genmo_root / "hmr4d/__init__.py").is_file(),
        "genmo_inference": (
            args.genmo_root / "hmr4d/model/genmo/inference.py"
        ).is_file(),
        "wilor": (args.wilor_root / "wilor_mini/__init__.py").is_file(),
        "sam2": importlib.util.find_spec("sam2") is not None,
        "moge": (args.moge_root / "moge/__init__.py").is_file(),
    }
    assets = {
        "genmo_checkpoint": file_check(genmo_checkpoint),
        "genmo_yolo": file_check(
            args.genmo_asset_root / "inputs/checkpoints/yolo/yolov8x.pt"
        ),
        "genmo_vitpose": file_check(
            args.genmo_asset_root
            / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
        ),
        "genmo_hmr2": file_check(
            args.genmo_asset_root / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
        ),
        "genmo_vimo": file_check(
            args.genmo_asset_root / "inputs/checkpoints/vimo/vimo_checkpoint.pth.tar"
        ),
        "genmo_smplx": file_check(
            args.genmo_asset_root
            / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"
        ),
        "wilor_model": file_check(wilor_assets / "pretrained_models/wilor_final.ckpt"),
        "wilor_detector": file_check(wilor_assets / "pretrained_models/detector.pt"),
        "wilor_mano": file_check(wilor_assets / "pretrained_models/MANO_RIGHT.pkl"),
        "wilor_mano_mean": file_check(
            wilor_assets / "pretrained_models/mano_mean_params.npz"
        ),
    }
    caches = {
        "genmo_t5": hf_cache_check(
            args.genmo_text_encoder,
            ("config.json", "model.safetensors", "spiece.model"),
        ),
        "sam2": hf_cache_check(args.sam2_model_cache, ("sam2_hiera_large.pt",)),
        "moge": hf_cache_check(args.moge_model_cache, ("model.pt",)),
    }
    imports = import_check(args)
    artifacts = {
        "step1": validate_step1(args.step1_npz),
        "step2_masks": validate_masks(args.masks_npz),
        "step2_depth": validate_depth(args.depth_pt),
    }
    comparison = compare_step1(args.step1_npz, args.reference_step1_npz)
    blockers = [name for name, ok in sources.items() if not ok]
    blockers.extend(
        name
        for name, info in assets.items()
        if not info["exists"] or info["lfs_pointer"]
    )
    blockers.extend(name for name, info in caches.items() if info["status"] != "PASS")
    if imports["status"] != "PASS":
        blockers.append("genmo_inference_import")
    blockers.extend(
        name for name, info in artifacts.items() if info["status"] != "PASS"
    )

    summary = {
        "runtime": {
            "torch": torch.__version__,
            "device": str(device),
            "device_name": torch.musa.get_device_name(device),
            "device_count": torch.musa.device_count(),
            "arch_list": list(torch.musa.get_arch_list()),
        },
        "sources": sources,
        "assets": assets,
        "caches": caches,
        "imports": imports,
        "artifacts": artifacts,
        "comparison": comparison,
        "blockers": blockers,
        "preflight": "PASS" if not blockers else "BLOCKED",
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
        )
    if args.strict and blockers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
