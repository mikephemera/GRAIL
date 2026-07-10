#!/usr/bin/env python3
"""Run and verify GRAIL Step 5 filtering/post-processing for MUSA Step 4 output."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
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
    parser.add_argument("--source_output_dir", type=str, default="generation/4dhoi_recon_smplx_musa")
    parser.add_argument("--valid_output_dir", type=str, default="generation/4dhoi_recon_smplx_musa_valid")
    parser.add_argument("--reference_valid_hoi", type=Path, default=None)
    parser.add_argument("--foundation_pose_output_dir", type=str, default="generation/foundation_pose_output_musa")
    parser.add_argument("--pytorch3d_root", type=Path, default=WORKSPACE_ROOT / "pytorch3d_musa")
    parser.add_argument("--filter_device", type=str, default="cpu")
    parser.add_argument("--mask_stride", type=int, default=8)
    parser.add_argument("--mask_max_frames", type=int, default=None)
    parser.add_argument("--full_mask_check", action="store_true", default=False)
    parser.add_argument("--simplify_mesh", action="store_true", default=False)
    parser.add_argument("--enable_visualization", action="store_true", default=False)
    parser.add_argument("--visualization_device", type=str, default="cpu")
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


def _results_root(results_dir: Path) -> Path:
    return results_dir if results_dir.is_absolute() else GRAIL_ROOT / results_dir


def _is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("rb") as f:
            return f.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


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


def _numeric_metrics(ref: Any, new: Any, rtol: float, atol: float) -> dict[str, Any]:
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
            "allclose": bool(np.allclose(ref_arr, new_arr, rtol=rtol, atol=atol)),
        }
    )
    return result


def _summarize_hoi(path: Path, reference: Path | None, rtol: float, atol: float) -> dict[str, Any]:
    data = _load_pickle(path)
    human = data.get("human_data", {})
    obj = data.get("obj_data", {})

    missing = []
    for key in ["human_data", "obj_data", "meta", "eval_data"]:
        if key not in data:
            missing.append(key)
    for key in ["poses", "trans"]:
        if key not in human:
            missing.append(f"human_data.{key}")
    for key in ["obj_R", "obj_t", "obj_scale"]:
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
        "has_object_path": "object_path" in data,
        "eval_data": data.get("eval_data", {}),
    }

    summary["contract_ok"] = (
        path.is_file()
        and not missing
        and finite
        and summary["human_poses_shape"] is not None
        and summary["human_trans_shape"] == (frame_count, 3)
        and summary["obj_R_shape"] == (frame_count, 3, 3)
        and summary["obj_t_shape"] == (frame_count, 3)
    )

    if reference is not None and reference.is_file() and not _is_lfs_pointer(reference):
        ref = _load_pickle(reference)
        ref_human = ref.get("human_data", {})
        ref_obj = ref.get("obj_data", {})
        drift = {
            "human_poses": _numeric_metrics(ref_human.get("poses"), human_poses, rtol, atol),
            "human_trans": _numeric_metrics(ref_human.get("trans"), human_trans, rtol, atol),
            "obj_t": _numeric_metrics(ref_obj.get("obj_t"), obj_t, rtol, atol),
            "obj_R": _numeric_metrics(ref_obj.get("obj_R"), obj_r, rtol, atol),
        }
        if drift["obj_R"].get("shape_match"):
            rot_errors = _rotation_errors_deg(_to_array(ref_obj["obj_R"]), _to_array(obj_r))
            drift["obj_R"]["rot_error_deg"] = {
                "max": float(rot_errors.max(initial=0.0)),
                "mean": float(rot_errors.mean()),
            }
        summary["drift"] = drift
        summary["generic_compare"] = {
            "match": bool(all(metrics.get("allclose", False) for metrics in drift.values()))
        }
    return summary


def _check_imports(enable_visualization: bool) -> dict[str, str]:
    checks = ["torch", "pytorch3d", "pytorch3d._C", "trimesh", "cv2", "smplx"]
    if enable_visualization:
        checks.append("scenepic")
    status = {}
    for name in checks:
        try:
            __import__(name)
            status[name] = "ok"
        except Exception as exc:
            status[name] = f"ERROR: {exc}"
    return status


def _setup_runtime(args: argparse.Namespace) -> dict[str, str]:
    _prepend_paths([args.pytorch3d_root, GEM_SMPL_ROOT, GRAIL_ROOT])
    os.environ.setdefault("TORCH_MUSA_ARCH_LIST", "31")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    return _check_imports(enable_visualization=args.enable_visualization)


def _build_paths(args: argparse.Namespace, cfg_flat: dict[str, Any]) -> dict[str, Path]:
    results_dir = args.results_dir or Path(cfg_flat.get("results_dir", "results"))
    results_root = _results_root(results_dir)
    video_id = args.video_id
    dataset, category = video_id.split("/")[:2]

    mesh_dir = results_root / "generation/mesh" / dataset / category
    mesh_candidates = sorted(glob(str(mesh_dir / "*.obj")))
    if not mesh_candidates:
        raise FileNotFoundError(f"No object mesh found under {mesh_dir}")

    reference_valid_hoi = args.reference_valid_hoi
    if reference_valid_hoi is None:
        reference_valid_hoi = (
            results_root
            / "generation/4dhoi_recon_smplx_valid"
            / video_id
            / "hoi_data/hoi_data.pkl"
        )
    else:
        reference_valid_hoi = _resolve_under_grail(reference_valid_hoi)

    source_seq_dir = results_root / args.source_output_dir / video_id
    valid_seq_dir = results_root / args.valid_output_dir / video_id
    return {
        "results_root": results_root,
        "source_seq_dir": source_seq_dir,
        "source_hoi": source_seq_dir / "hoi_data.pkl",
        "valid_seq_dir": valid_seq_dir,
        "valid_hoi_dir": valid_seq_dir / "hoi_data",
        "valid_hoi": valid_seq_dir / "hoi_data/hoi_data.pkl",
        "valid_mesh_dir": valid_seq_dir / "mesh_data",
        "valid_result_dir": valid_seq_dir / "result_vis",
        "reference_valid_hoi": reference_valid_hoi,
        "obj_path": Path(mesh_candidates[0]),
        "src_mesh_dir": mesh_dir,
        "video_file": results_root / cfg_flat.get("video_dir", "generation/videos_kling") / f"{video_id}.mp4",
        "cam_file": (
            results_root
            / cfg_flat.get("hmr_cache_dir", "generation/hmr_smplx_cache")
            / video_id
            / "preprocess/slam_results.pt"
        ),
        "obj_pose_file": (
            results_root
            / args.foundation_pose_output_dir
            / video_id
            / "pose_estimation_output/poses_in_cam.pkl"
        ),
        "render_cfg_file": (
            results_root / args.foundation_pose_output_dir / video_id / "first_frame_pose.pickle"
        ),
        "masks_cache_file": (
            results_root
            / cfg_flat.get("recon_cache_dir", "generation/4dhoi_recon_cache")
            / "masks"
            / f"{video_id}.npz"
        ),
    }


def _check_inputs(paths: dict[str, Path], compare_only: bool) -> None:
    if compare_only:
        _require_materialized(paths["valid_hoi"], "Step 5 MUSA valid hoi_data.pkl")
        return

    for key, label in [
        ("source_hoi", "Step 4 MUSA hoi_data.pkl"),
        ("obj_path", "object mesh"),
        ("obj_pose_file", "object pose file"),
        ("render_cfg_file", "render config file"),
        ("masks_cache_file", "masks cache"),
    ]:
        _require_materialized(paths[key], label)
    if paths["reference_valid_hoi"].exists():
        _require_materialized(paths["reference_valid_hoi"], "reference Step 5 valid hoi_data.pkl")


def _copy_mesh_data(src_mesh_dir: Path, dst_mesh_dir: Path) -> None:
    dst_mesh_dir.mkdir(parents=True, exist_ok=True)
    for item in src_mesh_dir.iterdir():
        dst = dst_mesh_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, dst)


def _select_mask_frame_indices(frame_num: int, mask_frame_stride: int = 1, mask_max_frames: int | None = None) -> list[int]:
    if frame_num <= 0:
        return []
    if mask_max_frames is not None and 0 < mask_max_frames < frame_num:
        return sorted(
            set(np.linspace(0, frame_num - 1, int(mask_max_frames)).round().astype(int).tolist())
        )

    stride = max(1, int(mask_frame_stride or 1))
    frame_indices = list(range(0, frame_num, stride))
    if frame_indices[-1] != frame_num - 1:
        frame_indices.append(frame_num - 1)
    return frame_indices


def _prepare_data_for_validation_smoke(
    hoi_data: dict[str, Any],
    human_model_cfg: dict[str, Any],
    *,
    device: str,
    simplify_mesh: bool,
):
    from grail.core.io import load_init_rendering_data
    from grail.models.human_model import create_human_model
    from grail.rendering.camera import (
        cam_pose_blender_to_opencv,
        cam_pose_opencv_to_pytorch3d,
        get_camera,
    )
    from grail.rendering.renderer import RendererType, create_renderer
    from grail.visualization.utils.vis_utils import prep_visualizer_input

    _, _, _obj_scale, blender_cam_R, blender_cam_t, render_config, additional_data = (
        load_init_rendering_data(
            hoi_data["meta"]["render_config_file"],
            to_tensor=True,
            with_human_data=True,
            device=device,
        )
    )
    frame_height, frame_width, focal_length = render_config
    opencv_cam_R, opencv_cam_t = cam_pose_blender_to_opencv(blender_cam_R, blender_cam_t)
    cam_R, cam_t = cam_pose_opencv_to_pytorch3d(opencv_cam_R, opencv_cam_t)
    cameras = get_camera(cam_R, cam_t, focal_length, (frame_height, frame_width), device=device)

    masks_cache_file = hoi_data["meta"]["masks_cache_file"]
    if not os.path.exists(masks_cache_file):
        raise FileNotFoundError(f"Masks cache not found: {masks_cache_file}")
    masks = np.load(masks_cache_file, allow_pickle=True)["masks"].item()

    mask_renderer = create_renderer(
        cameras,
        (frame_height, frame_width),
        renderer_type=RendererType.HARD_PHONG,
        neutral_light=True,
        background_color=[0, 0, 0],
        device=device,
    )

    hoi_data["object_path"] = hoi_data["meta"]["obj_path"]
    human_model = create_human_model(human_model_cfg, device=device)
    motion_seq = prep_visualizer_input(
        hoi_data,
        human_model=human_model,
        normalize_trans=False,
        to_numpy=False,
        simplify_mesh=simplify_mesh,
        device=device,
    )

    data = {
        "masks": masks,
        "obj_data": {
            "verts_seq": motion_seq["obj_seq"]["vertices_transformed"],
            "obj_faces": motion_seq["obj_seq"]["faces"],
        },
        "human_data": {
            "verts_seq": motion_seq["human_seq"]["vertices"],
            "joints_seq": motion_seq["human_seq"]["joints_pos"],
            "human_faces": motion_seq["human_seq"]["triangles"],
        },
        "scene_data": hoi_data.get("scene_data", None),
    }
    human_R = additional_data.get("human_R", None)
    human_t = additional_data.get("human_t", None)
    data["human_data"]["first_frame_pose"] = (
        {"R": human_R, "t": human_t} if human_R is not None and human_t is not None else None
    )
    return data, cameras, mask_renderer


def _check_object_mask_smoke(
    data: dict[str, Any],
    cameras,
    mask_renderer,
    *,
    device: str,
    frame_indices: list[int],
    tol: float,
    total_tol: float,
    logger,
) -> tuple[bool, int | None]:
    import torch
    import torch.nn.functional as F

    from grail.rendering.renderer import render_frame
    from grail.rendering.textures import create_colored_meshes

    obj_verts_seq = data["obj_data"]["verts_seq"]
    obj_faces = data["obj_data"]["obj_faces"]
    obj_colors = torch.tensor([0.0, 0.0, 1.0], device=device)
    if not frame_indices:
        if logger:
            logger.info("No object-mask frames selected; skipping object mask check")
        return True, None

    total_err = 0
    for i in frame_indices:
        obj_mesh = create_colored_meshes(obj_verts_seq[i], obj_faces, obj_colors, device=device)
        _, pred_obj_mask = render_frame(obj_mesh, cameras, mask_renderer, require_grad=False)
        pred_obj_mask = (pred_obj_mask > 0.1).float()

        gt_obj_mask = torch.from_numpy(data["masks"][i][0]).to(device).squeeze(0).float()
        if gt_obj_mask.shape != pred_obj_mask.shape:
            gt_obj_mask = gt_obj_mask.unsqueeze(0).unsqueeze(0)
            gt_obj_mask = F.interpolate(
                gt_obj_mask, size=pred_obj_mask.shape, mode="bilinear", align_corners=False
            )
            gt_obj_mask = gt_obj_mask.squeeze(0).squeeze(0)

        err = ((1 - pred_obj_mask) * gt_obj_mask).sum() / (pred_obj_mask.sum() + 100)
        total_err += err
        if err > tol:
            if logger:
                logger.info(
                    f"Not valid: object mask not aligned at frame {i} / "
                    f"{len(obj_verts_seq)}: {err} > {tol}"
                )
            return False, i

    total_err /= len(frame_indices)
    if total_err > total_tol:
        if logger:
            logger.info(f"Not valid: total object mask difference is too large: {total_err} > {total_tol}")
        return False, None
    if logger:
        logger.info(f"Valid: total object mask difference is within threshold: {total_err} < {total_tol}")
        logger.info("Valid: object mask aligned on selected frames!")
    return True, None


def _filter_hoi_result_for_verify(
    result_camera,
    hoi_data: dict[str, Any],
    cfg: dict[str, Any],
    *,
    device: str,
    logger,
    mask_frame_stride: int,
    mask_max_frames: int | None,
    simplify_mesh: bool,
) -> tuple[bool, dict[str, Any]]:
    from grail.postprocessing.filter import (
        check_camera_translation,
        check_eval_data,
        check_init_penetration,
        check_initial_human_position,
        check_object_initially_on_table,
        check_object_is_static,
        check_static_human,
        truncate_hoi_data,
    )

    human_model_cfg = cfg["human_model"]
    eval_cfg = cfg["eval"]
    filtering_cfg = cfg.get("filtering", {})
    camera_trans_thr = filtering_cfg.get("camera_trans_thr", 0.1)
    object_mask_tol = filtering_cfg.get("object_mask_tol", 0.5)
    total_mask_tol = filtering_cfg.get("total_mask_tol", 0.3)
    human_static_thr = filtering_cfg.get("human_static_thr", 0.01)
    min_frames = filtering_cfg.get("min_frames", None)

    valid = True if result_camera is None else check_camera_translation(
        result_camera, camera_trans_thr, logger=logger
    )
    if not valid:
        return False, hoi_data

    data, cameras, mask_renderer = _prepare_data_for_validation_smoke(
        hoi_data, human_model_cfg, device=device, simplify_mesh=simplify_mesh
    )

    if data["human_data"]["first_frame_pose"] is not None:
        valid = check_initial_human_position(data, logger=logger)
        if not valid:
            return False, hoi_data

    mask_frame_indices = _select_mask_frame_indices(
        len(data["obj_data"]["verts_seq"]),
        mask_frame_stride=mask_frame_stride,
        mask_max_frames=mask_max_frames,
    )
    if len(mask_frame_indices) != len(data["obj_data"]["verts_seq"]) and logger:
        logger.info(
            f"Object mask validation using {len(mask_frame_indices)} sampled frames "
            f"out of {len(data['obj_data']['verts_seq'])}"
        )
    valid, failed_frame_idx = _check_object_mask_smoke(
        data,
        cameras,
        mask_renderer,
        device=device,
        frame_indices=mask_frame_indices,
        tol=object_mask_tol,
        total_tol=total_mask_tol,
        logger=logger,
    )
    if not valid:
        if min_frames is not None and failed_frame_idx is not None and failed_frame_idx >= min_frames:
            if logger:
                logger.info(f"Truncating data to {failed_frame_idx} frames (min_frames={min_frames})")
            hoi_data = truncate_hoi_data(hoi_data, failed_frame_idx)
        else:
            return False, hoi_data

    if not check_eval_data(hoi_data["eval_data"], eval_cfg, logger=logger):
        return False, hoi_data
    if not check_init_penetration(data, device=device, logger=logger):
        return False, hoi_data
    if not check_static_human(data, device=device, threshold=human_static_thr, logger=logger):
        return False, hoi_data
    if filtering_cfg.get("check_object_initially_on_table", True):
        if not check_object_initially_on_table(data, device=device, logger=logger):
            return False, hoi_data

    filter_object_motion = filtering_cfg.get("filter_object_motion", "all")
    if filter_object_motion != "all":
        object_static_thr = filtering_cfg.get("object_static_thr", 0.02)
        is_static, motion_range = check_object_is_static(hoi_data, threshold=object_static_thr)
        motion_label = "static" if is_static else "dynamic"
        if logger:
            logger.info(
                f"Object motion: {motion_label} (range={motion_range:.4f}m, "
                f"thr={object_static_thr})"
            )
        if filter_object_motion == "static_only" and not is_static:
            if logger:
                logger.info("Rejected: object is dynamic but static_only filter is active")
            return False, hoi_data
        if filter_object_motion == "dynamic_only" and is_static:
            if logger:
                logger.info("Rejected: object is static but dynamic_only filter is active")
            return False, hoi_data

    if logger:
        logger.info("Valid HOI result!")
    return True, hoi_data


def _run_step5(paths: dict[str, Path], cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from grail.core.io import load_hoi_data, save_hoi_data
    from grail.core.logging import create_logger
    from grail.postprocessing.postprocess import post_process_hoi_result

    _install_numpy_pickle_aliases()
    hoi_data = load_hoi_data(str(paths["source_hoi"]))
    hoi_data["meta"].update(
        {
            "obj_path": str(paths["obj_path"]),
            "obj_pose_file": str(paths["obj_pose_file"]),
            "render_config_file": str(paths["render_cfg_file"]),
            "masks_cache_file": str(paths["masks_cache_file"]),
        }
    )

    result_camera = torch.load(paths["cam_file"]) if paths["cam_file"].exists() else None
    filter_cfg = {
        "human_model": cfg["human_model"],
        "eval": cfg["optimization"].get("eval", {}),
        "filtering": cfg["filtering"],
    }
    post_cfg = {
        "human_model": cfg["human_model"],
        "post_processing": cfg["post_processing"],
    }

    log_dir = paths["valid_seq_dir"] / "step5_filter_log"
    logger = create_logger(str(log_dir))
    mask_stride = 1 if args.full_mask_check else args.mask_stride
    mask_max_frames = None if args.full_mask_check else args.mask_max_frames

    valid_hoi, filtered = _filter_hoi_result_for_verify(
        result_camera,
        hoi_data,
        filter_cfg,
        device=args.filter_device,
        logger=logger,
        mask_frame_stride=mask_stride,
        mask_max_frames=mask_max_frames,
        simplify_mesh=args.simplify_mesh,
    )

    run_summary: dict[str, Any] = {
        "valid": bool(valid_hoi),
        "filter_device": args.filter_device,
        "mask_stride": mask_stride,
        "mask_max_frames": mask_max_frames,
        "full_mask_check": bool(args.full_mask_check),
        "simplify_mesh": bool(args.simplify_mesh),
        "log_dir": str(log_dir),
    }
    if not valid_hoi:
        return run_summary

    paths["valid_hoi_dir"].mkdir(parents=True, exist_ok=True)
    paths["valid_mesh_dir"].mkdir(parents=True, exist_ok=True)
    processed = post_process_hoi_result(filtered, str(paths["valid_mesh_dir"]), post_cfg)
    processed["object_path"] = str(paths["obj_path"])
    save_hoi_data(processed, str(paths["valid_hoi"]))
    _copy_mesh_data(paths["src_mesh_dir"], paths["valid_mesh_dir"])

    if args.enable_visualization:
        from grail.models.human_model import create_human_model
        from grail.visualization.scenepic import ScenepicVisualizer
        from grail.visualization.utils.vis_utils import prep_visualizer_input

        paths["valid_result_dir"].mkdir(parents=True, exist_ok=True)
        input_copy = paths["valid_result_dir"] / "input.mp4"
        if paths["video_file"].exists():
            shutil.copy2(paths["video_file"], input_copy)
        human_model = create_human_model(cfg["human_model"], device=args.visualization_device)
        vis_input = prep_visualizer_input(
            processed,
            human_model=human_model,
            device=args.visualization_device,
        )
        ScenepicVisualizer().vis_scene(
            vis_input,
            str(paths["valid_result_dir"] / "recon_result.html"),
            window_size=(400, 400),
            fps=16,
        )
        run_summary["visualization"] = "written"
    else:
        run_summary["visualization"] = "skipped"

    return run_summary


def _print_summary(summary: dict[str, Any], reference: Path) -> bool:
    artifact = summary.get("artifact", {})
    run = summary.get("run", {})

    print("\n" + "=" * 70)
    print("  GRAIL Step 5 MUSA artifact contract")
    print("=" * 70)
    print(f"  output: {artifact.get('path')}")
    print(f"  reference: {reference}")
    if run:
        print(f"  filter valid: {run.get('valid')}")
        print(
            f"  filter: device={run.get('filter_device')}, "
            f"mask_stride={run.get('mask_stride')}, "
            f"mask_max_frames={run.get('mask_max_frames')}, "
            f"simplify_mesh={run.get('simplify_mesh')}"
        )
    print(f"  frames: {artifact.get('frame_count')}")
    print(f"  human poses: {artifact.get('human_poses_shape')}")
    print(f"  human trans: {artifact.get('human_trans_shape')}")
    print(f"  obj_R: {artifact.get('obj_R_shape')}")
    print(f"  obj_t: {artifact.get('obj_t_shape')}")
    print(f"  finite: {'yes' if artifact.get('finite') else 'no'}")
    print(f"  object_path: {'yes' if artifact.get('has_object_path') else 'no'}")
    if artifact.get("missing_keys"):
        print(f"  missing keys: {artifact.get('missing_keys')}")
    print(f"  artifact contract: {'PASS' if artifact.get('contract_ok') else 'FAIL'}")

    if "drift" in artifact:
        print("\n" + "=" * 70)
        print("  Drift report against reference valid output")
        print("=" * 70)
        for name, metrics in artifact["drift"].items():
            print(f"  {name}: shape_match={metrics.get('shape_match')}")
            if metrics.get("shape_match"):
                print(
                    f"    max_abs={metrics.get('max_abs', 0.0):.6g}, "
                    f"mean_abs={metrics.get('mean_abs', 0.0):.6g}"
                )
                if name == "obj_R" and "rot_error_deg" in metrics:
                    rot = metrics["rot_error_deg"]
                    print(f"    rot_error_deg max={rot['max']:.6g}, mean={rot['mean']:.6g}")
        generic = artifact.get("generic_compare", {})
        print(f"  generic allclose: {'PASS' if generic.get('match') else 'WARN/DIFF'}")

    print("=" * 70 + "\n")
    return bool(artifact.get("contract_ok")) and bool(run.get("valid", True))


def main() -> int:
    args = parse_args()
    args.config = _resolve_under_grail(args.config)

    import_status = _setup_runtime(args)
    failed = [name for name, value in import_status.items() if value.startswith("ERROR")]
    print("\n[Preflight] imports")
    for name, value in import_status.items():
        print(f"  {name}: {value}")
    if failed and not args.compare_only:
        raise RuntimeError(f"Step 5 runtime imports failed: {', '.join(failed)}")

    from grail.core.config import load_recon_config
    from grail.core.types import parse_recon_config

    cfg, cfg_flat = load_recon_config(str(args.config))
    cfg = parse_recon_config(cfg)
    paths = _build_paths(args, cfg_flat)
    _check_inputs(paths, compare_only=args.compare_only)

    print("\n[Preflight] inputs")
    for key in [
        "source_hoi",
        "obj_path",
        "obj_pose_file",
        "render_cfg_file",
        "masks_cache_file",
        "valid_hoi",
    ]:
        print(f"  {key}: {paths[key]}")

    run_summary: dict[str, Any] = {}
    if not args.compare_only:
        print("\n[Run] Step 5 filter/post-process")
        print(f"  source: {paths['source_hoi']}")
        print(f"  output: {paths['valid_seq_dir']}")
        run_summary = _run_step5(paths, cfg, args)

    summary: dict[str, Any] = {
        "inputs": {key: str(value) for key, value in paths.items()},
        "imports": import_status,
        "run": run_summary,
    }
    if paths["valid_hoi"].is_file():
        summary["artifact"] = _summarize_hoi(
            paths["valid_hoi"],
            paths["reference_valid_hoi"],
            args.rtol,
            args.atol,
        )
    else:
        summary["artifact"] = {
            "path": str(paths["valid_hoi"]),
            "exists": False,
            "contract_ok": False,
            "missing_keys": ["valid_hoi"],
        }

    ok = _print_summary(summary, paths["reference_valid_hoi"])

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_json.open("w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Summary JSON saved to: {args.summary_json}")

    if not ok:
        return 1
    if args.fail_on_diff and not summary["artifact"].get("generic_compare", {}).get("match", True):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
