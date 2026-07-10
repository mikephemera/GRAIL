#!/usr/bin/env python3
"""Run and verify GRAIL Step 4 HOI optimization on a MUSA runtime."""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np


GRAIL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GRAIL_ROOT.parent
GEM_SMPL_ROOT = GRAIL_ROOT / "imports" / "GEM-SMPL"
DEFAULT_VIDEO_ID = "ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=GRAIL_ROOT / "configs/recon_4dhoi/manip_smplx.yaml")
    parser.add_argument("--video_id", type=str, default=DEFAULT_VIDEO_ID)
    parser.add_argument("--results_dir", type=Path, default=None)
    parser.add_argument("--foundation_pose_output_dir", type=str, default="generation/foundation_pose_output_musa")
    parser.add_argument("--output_dir", type=str, default="generation/4dhoi_recon_smplx_musa")
    parser.add_argument("--reference_hoi", type=Path, default=None)
    parser.add_argument("--pytorch3d_root", type=Path, default=WORKSPACE_ROOT / "pytorch3d_musa")
    parser.add_argument("--device", type=str, default="musa:0")
    parser.add_argument("--smoke_niter", type=int, default=2)
    parser.add_argument("--full_iters", action="store_true", default=False)
    parser.add_argument("--enable_vis", action="store_true", default=False)
    parser.add_argument("--skip_pre_eval", action="store_true", default=False)
    parser.add_argument("--skip_depth_pointcloud", action="store_true", default=False)
    parser.add_argument("--compare_only", action="store_true", default=False)
    parser.add_argument("--fail_on_diff", action="store_true", default=False)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--summary_json", type=Path, default=None)
    return parser.parse_args()


def _install_numpy_pickle_aliases() -> None:
    core = getattr(np, "_core", np.core)
    sys.modules.setdefault("numpy._core", core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    if hasattr(np.core, "_multiarray_umath"):
        sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)


def _prepend_paths(paths: list[Path]) -> None:
    ordered = [str(path.resolve()) for path in paths if path]
    for path in reversed(ordered):
        if path not in sys.path:
            sys.path.insert(0, path)


def _resolve_under_grail(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else GRAIL_ROOT / path


def _result_file(results_dir: Path, rel_dir: str, video_id: str, suffix: str = "") -> Path:
    root = results_dir if results_dir.is_absolute() else GRAIL_ROOT / results_dir
    return root / rel_dir / video_id / suffix


def _is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("rb") as f:
            header = f.read(128)
    except OSError:
        return False
    return header.startswith(b"version https://git-lfs.github.com/spec/v1")


def _require_materialized(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    if _is_lfs_pointer(path):
        raise RuntimeError(f"{label} is still a Git LFS pointer: {path}")


def _load_pickle(path: Path) -> Any:
    _install_numpy_pickle_aliases()
    with path.open("rb") as f:
        return pickle.load(f)


def _to_array(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(value)


def _finite_array(value: Any) -> bool:
    arr = _to_array(value)
    return np.issubdtype(arr.dtype, np.number) and bool(np.isfinite(arr).all())


def _rotation_errors_deg(ref_r: np.ndarray, new_r: np.ndarray) -> np.ndarray:
    rel = np.matmul(new_r, np.linalg.inv(ref_r))
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def _numeric_metrics(ref: Any, new: Any) -> dict[str, Any]:
    ref_arr = _to_array(ref).astype(np.float64)
    new_arr = _to_array(new).astype(np.float64)
    result: dict[str, Any] = {
        "shape_match": tuple(ref_arr.shape) == tuple(new_arr.shape),
        "ref_shape": tuple(ref_arr.shape),
        "new_shape": tuple(new_arr.shape),
    }
    if not result["shape_match"]:
        return result
    diff = np.abs(ref_arr - new_arr)
    result.update(
        {
            "max_abs": float(diff.max(initial=0.0)),
            "mean_abs": float(diff.mean()) if diff.size else 0.0,
            "allclose": bool(np.allclose(ref_arr, new_arr, rtol=0.0, atol=0.0)),
        }
    )
    return result


def _summarize_hoi(path: Path, reference: Path | None, rtol: float, atol: float) -> dict[str, Any]:
    data = _load_pickle(path)
    human = data.get("human_data", {})
    obj = data.get("obj_data", {})

    required_keys = {
        "top": ["human_data", "obj_data", "meta", "eval_data"],
        "human_data": ["poses", "trans"],
        "obj_data": ["obj_R", "obj_t", "obj_scale"],
    }
    missing = []
    for key in required_keys["top"]:
        if key not in data:
            missing.append(key)
    for key in required_keys["human_data"]:
        if key not in human:
            missing.append(f"human_data.{key}")
    for key in required_keys["obj_data"]:
        if key not in obj:
            missing.append(f"obj_data.{key}")

    human_poses = human.get("poses")
    human_trans = human.get("trans")
    obj_r = obj.get("obj_R")
    obj_t = obj.get("obj_t")
    frame_count = int(_to_array(obj_t).shape[0]) if obj_t is not None else 0

    finite = True
    for value in (human_poses, human_trans, obj_r, obj_t):
        finite = finite and value is not None and _finite_array(value)

    summary: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "missing_keys": missing,
        "frame_count": frame_count,
        "human_poses_shape": tuple(_to_array(human_poses).shape) if human_poses is not None else None,
        "human_trans_shape": tuple(_to_array(human_trans).shape) if human_trans is not None else None,
        "obj_R_shape": tuple(_to_array(obj_r).shape) if obj_r is not None else None,
        "obj_t_shape": tuple(_to_array(obj_t).shape) if obj_t is not None else None,
        "finite": finite,
    }

    contract_ok = (
        path.is_file()
        and not missing
        and finite
        and summary["human_poses_shape"] is not None
        and summary["human_trans_shape"] == (frame_count, 3)
        and summary["obj_R_shape"] == (frame_count, 3, 3)
        and summary["obj_t_shape"] == (frame_count, 3)
    )
    summary["contract_ok"] = contract_ok

    if reference is not None and reference.is_file() and not _is_lfs_pointer(reference):
        ref = _load_pickle(reference)
        ref_human = ref.get("human_data", {})
        ref_obj = ref.get("obj_data", {})
        drift = {
            "human_poses": _numeric_metrics(ref_human.get("poses"), human_poses),
            "human_trans": _numeric_metrics(ref_human.get("trans"), human_trans),
            "obj_t": _numeric_metrics(ref_obj.get("obj_t"), obj_t),
            "obj_R": _numeric_metrics(ref_obj.get("obj_R"), obj_r),
        }
        if drift["obj_R"]["shape_match"]:
            drift["obj_R"]["rot_error_deg"] = {
                "max": float(_rotation_errors_deg(_to_array(ref_obj["obj_R"]), _to_array(obj_r)).max(initial=0.0)),
                "mean": float(_rotation_errors_deg(_to_array(ref_obj["obj_R"]), _to_array(obj_r)).mean()),
            }
        try:
            from scripts.verify_module_output import compare_values
        except ModuleNotFoundError:
            from verify_module_output import compare_values

        summary["generic_compare"] = compare_values(ref, data, rtol=rtol, atol=atol)
        summary["drift"] = drift
    return summary


def _print_import_status(status: dict[str, str]) -> None:
    print("\n[Preflight] imports")
    for name, value in status.items():
        print(f"  {name}: {value}")


def _check_imports(compare_only: bool) -> dict[str, str]:
    checks = ["torch", "pytorch3d", "pytorch3d._C"]
    if not compare_only:
        checks.extend(["hmr4d", "smplx", "trimesh", "cv2"])
    status = {}
    for name in checks:
        try:
            __import__(name)
            status[name] = "ok"
        except Exception as exc:
            status[name] = f"ERROR: {exc}"
    try:
        from pytorch3d.ops import knn_points  # noqa: F401
        from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings  # noqa: F401

        status["pytorch3d.step4_surface"] = "ok"
    except Exception as exc:
        status["pytorch3d.step4_surface"] = f"ERROR: {exc}"
    return status


def _setup_runtime(args: argparse.Namespace, compare_only: bool) -> str:
    _prepend_paths([args.pytorch3d_root, GEM_SMPL_ROOT, GRAIL_ROOT])
    os.environ.setdefault("TORCH_MUSA_ARCH_LIST", "31")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    status = _check_imports(compare_only=compare_only)
    _print_import_status(status)
    failed = [name for name, value in status.items() if value.startswith("ERROR")]
    if failed and not compare_only:
        raise RuntimeError(f"Step 4 runtime imports failed: {', '.join(failed)}")

    if compare_only:
        return args.device

    import torch

    device = str(args.device)
    if device.startswith("musa"):
        import torch_musa  # noqa: F401

        if not torch.musa.is_available():
            raise RuntimeError("MUSA is not available for GRAIL Step 4")
        torch.musa.set_device(torch.device(device))
        print(f"\n[Preflight] MUSA available: {torch.musa.is_available()}")
        print(f"[Preflight] MUSA device count: {torch.musa.device_count()}")
        try:
            print(f"[Preflight] MUSA arch list: {torch.musa.get_arch_list()}")
        except Exception:
            pass
    return device


def _build_step4_paths(args: argparse.Namespace, cfg_flat: dict[str, Any]) -> dict[str, Path]:
    results_dir = args.results_dir or Path(cfg_flat.get("results_dir", "results"))
    results_root = results_dir if results_dir.is_absolute() else GRAIL_ROOT / results_dir
    video_id = args.video_id
    dataset, category = video_id.split("/")[:2]

    mesh_candidates = sorted(glob(str(results_root / "generation/mesh" / dataset / category / "*.obj")))
    if not mesh_candidates:
        raise FileNotFoundError(f"No object mesh found under {results_root / 'generation/mesh' / dataset / category}")

    reference_hoi = args.reference_hoi
    if reference_hoi is None:
        reference_hoi = _result_file(
            results_dir,
            "generation/4dhoi_recon_smplx",
            video_id,
            "hoi_data.pkl",
        )
    else:
        reference_hoi = _resolve_under_grail(reference_hoi)

    output_seq_dir = _result_file(results_dir, args.output_dir, video_id)
    return {
        "results_dir": results_root,
        "hmr_file": results_root / cfg_flat.get("hmr_dir", "generation/hmr_smplx") / f"{video_id}.npz",
        "video_file": results_root / cfg_flat.get("video_dir", "generation/videos_kling") / f"{video_id}.mp4",
        "mesh": Path(mesh_candidates[0]),
        "obj_pose_file": _result_file(
            results_dir,
            args.foundation_pose_output_dir,
            video_id,
            "pose_estimation_output/poses_in_cam.pkl",
        ),
        "render_cfg_file": _result_file(
            results_dir,
            args.foundation_pose_output_dir,
            video_id,
            "first_frame_pose.pickle",
        ),
        "masks_cache": results_root / cfg_flat.get("recon_cache_dir", "generation/4dhoi_recon_cache") / "masks" / f"{video_id}.npz",
        "depth_cache": results_root / cfg_flat.get("recon_cache_dir", "generation/4dhoi_recon_cache") / "depth" / f"{video_id}.pt",
        "contact_cache": results_root / cfg_flat.get("recon_cache_dir", "generation/4dhoi_recon_cache") / "contact_labels" / f"{video_id}.json",
        "cache_dir": results_root / cfg_flat.get("recon_cache_dir", "generation/4dhoi_recon_cache"),
        "output_seq_dir": output_seq_dir,
        "output_hoi": output_seq_dir / "hoi_data.pkl",
        "reference_hoi": reference_hoi,
    }


def _check_inputs(paths: dict[str, Path], compare_only: bool) -> None:
    if compare_only:
        _require_materialized(paths["output_hoi"], "new Step 4 hoi_data.pkl")
        _require_materialized(paths["reference_hoi"], "reference Step 4 hoi_data.pkl")
        return

    for key in [
        "hmr_file",
        "video_file",
        "mesh",
        "obj_pose_file",
        "render_cfg_file",
        "masks_cache",
        "depth_cache",
    ]:
        _require_materialized(paths[key], key)
    if _is_lfs_pointer(paths["reference_hoi"]):
        raise RuntimeError(f"reference Step 4 hoi_data.pkl is still a Git LFS pointer: {paths['reference_hoi']}")

    print("\n[Preflight] inputs")
    for key in ["hmr_file", "video_file", "mesh", "obj_pose_file", "render_cfg_file", "masks_cache", "depth_cache"]:
        print(f"  {key}: {paths[key]}")
    print(f"  contact_cache: {paths['contact_cache']} ({'present' if paths['contact_cache'].is_file() else 'missing; contact detection may call external API'})")


def _make_opt_cfg(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    opt_cfg = copy.deepcopy(cfg["optimization"])
    opt_cfg["human_model"] = copy.deepcopy(cfg["human_model"])
    if args.skip_pre_eval:
        opt_cfg["skip_pre_eval"] = True
    opt_cfg.setdefault("vis_cfg", {})
    if not args.enable_vis:
        opt_cfg["vis_cfg"]["enable"] = False
        opt_cfg["vis_cfg"]["vis_init"] = False
        opt_cfg["vis_cfg"]["render_video"] = False
        opt_cfg["vis_cfg"]["vis_html"] = False
    if not args.full_iters:
        for stage_cfg in opt_cfg.get("opt_stage_specs", {}).values():
            stage_cfg["niter"] = args.smoke_niter
    if args.skip_depth_pointcloud:
        for stage_cfg in opt_cfg.get("opt_stage_specs", {}).values():
            stage_cfg.get("loss_cfg", {}).pop("depth_pointcloud", None)
    return opt_cfg


def _run_step4(paths: dict[str, Path], cfg: dict[str, Any], args: argparse.Namespace, device: str) -> Path:
    from grail.core.io import save_hoi_data
    from grail.optimization.hoi_optimizer import HOIOptimizer

    opt_cfg = _make_opt_cfg(cfg, args)
    paths["output_seq_dir"].mkdir(parents=True, exist_ok=True)

    optimizer = HOIOptimizer(
        exp_name=args.video_id,
        cfg=opt_cfg,
        cache_dir=str(paths["cache_dir"]),
        output_dir=str(paths["output_seq_dir"]),
        device=device,
    )
    data = optimizer.init_data(
        str(paths["video_file"]),
        str(paths["hmr_file"]),
        str(paths["mesh"]),
        str(paths["obj_pose_file"]),
        str(paths["render_cfg_file"]),
    )
    hoi_data = optimizer.optimize(data=data)
    save_hoi_data(hoi_data, str(paths["output_hoi"]))
    return paths["output_hoi"]


def _print_summary(summary: dict[str, Any], reference: Path) -> bool:
    print("\n" + "=" * 70)
    print("  GRAIL Step 4 MUSA artifact contract")
    print("=" * 70)
    print(f"  output: {summary['path']}")
    print(f"  reference: {reference}")
    print(f"  frames: {summary['frame_count']}")
    print(f"  human poses: {summary['human_poses_shape']}")
    print(f"  human trans: {summary['human_trans_shape']}")
    print(f"  obj_R: {summary['obj_R_shape']}")
    print(f"  obj_t: {summary['obj_t_shape']}")
    print(f"  finite: {'yes' if summary['finite'] else 'no'}")
    if summary["missing_keys"]:
        print(f"  missing keys: {summary['missing_keys']}")
    print(f"  artifact contract: {'PASS' if summary['contract_ok'] else 'FAIL'}")

    if "drift" in summary:
        print("\n" + "=" * 70)
        print("  Drift report against reference")
        print("=" * 70)
        for name, metrics in summary["drift"].items():
            print(f"  {name}: shape_match={metrics.get('shape_match')}")
            if metrics.get("shape_match"):
                print(
                    f"    max_abs={metrics.get('max_abs', 0.0):.6g}, "
                    f"mean_abs={metrics.get('mean_abs', 0.0):.6g}"
                )
                if name == "obj_R" and "rot_error_deg" in metrics:
                    rot = metrics["rot_error_deg"]
                    print(f"    rot_error_deg max={rot['max']:.6g}, mean={rot['mean']:.6g}")
        generic = summary.get("generic_compare", {})
        print(f"  generic allclose: {'PASS' if generic.get('match') else 'WARN/DIFF'}")

    print("=" * 70 + "\n")
    return bool(summary["contract_ok"])


def main() -> int:
    args = parse_args()
    args.config = _resolve_under_grail(args.config)

    _prepend_paths([args.pytorch3d_root, GRAIL_ROOT])
    from grail.core.config import load_recon_config
    from grail.core.types import parse_recon_config

    cfg, cfg_flat = load_recon_config(str(args.config))
    cfg = parse_recon_config(cfg)
    paths = _build_step4_paths(args, cfg_flat)

    _check_inputs(paths, compare_only=args.compare_only)
    device = _setup_runtime(args, compare_only=args.compare_only)

    if not args.compare_only:
        print("\n[Run] Step 4 HOI optimization")
        print(f"  device: {device}")
        print(f"  output: {paths['output_seq_dir']}")
        if not args.full_iters:
            print(f"  smoke_niter per stage: {args.smoke_niter}")
        if args.skip_pre_eval:
            print("  pre_eval: skipped")
        if args.skip_depth_pointcloud:
            print("  depth_pointcloud loss: skipped")
        _run_step4(paths, cfg, args, device)

    summary = _summarize_hoi(paths["output_hoi"], paths["reference_hoi"], args.rtol, args.atol)
    ok = _print_summary(summary, paths["reference_hoi"])

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary JSON saved to: {args.summary_json}")

    if not ok:
        return 1
    if args.fail_on_diff and not summary.get("generic_compare", {}).get("match", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
