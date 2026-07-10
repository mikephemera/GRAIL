#!/usr/bin/env python3
"""Render MP4 visualizations from a GRAIL Step 5 MUSA valid artifact."""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from glob import glob
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm


GRAIL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GRAIL_ROOT.parent
GEM_SMPL_ROOT = GRAIL_ROOT / "imports" / "GEM-SMPL"
DEFAULT_VIDEO_ID = "ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=GRAIL_ROOT / "configs/recon_4dhoi/manip_smplx.yaml")
    parser.add_argument("--video_id", type=str, default=DEFAULT_VIDEO_ID)
    parser.add_argument("--results_dir", type=Path, default=None)
    parser.add_argument("--valid_output_dir", type=str, default="generation/4dhoi_recon_smplx_musa_valid")
    parser.add_argument("--foundation_pose_output_dir", type=str, default="generation/foundation_pose_output_musa")
    parser.add_argument("--pytorch3d_root", type=Path, default=WORKSPACE_ROOT / "pytorch3d_musa")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--skip_top_view", action="store_true", default=False)
    parser.add_argument("--skip_comparison", action="store_true", default=False)
    parser.add_argument("--no_simplify_mesh", action="store_true", default=False)
    parser.add_argument("--no_contact_colors", action="store_true", default=False)
    return parser.parse_args()


def _install_numpy_pickle_aliases() -> None:
    core = getattr(np, "_core", np.core)
    sys.modules.setdefault("numpy._core", core)
    sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
    if hasattr(np.core, "_multiarray_umath"):
        sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)


def _prepend_paths(paths: Iterable[Path]) -> None:
    for path in reversed([str(path.resolve()) for path in paths if path]):
        if path not in sys.path:
            sys.path.insert(0, path)


def _resolve_under_grail(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else GRAIL_ROOT / path


def _results_root(results_dir: Path) -> Path:
    return results_dir if results_dir.is_absolute() else GRAIL_ROOT / results_dir


def _load_pickle(path: Path):
    _install_numpy_pickle_aliases()
    with path.open("rb") as f:
        return pickle.load(f)


def _is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as f:
        return f.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    if _is_lfs_pointer(path):
        raise RuntimeError(f"{label} is still a Git LFS pointer: {path}")


def _fps_from_video(video_path: Path, default: float = 24.0) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return default
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 0 else default


def _read_background_frames(video_path: Path, count: int, height: int, width: int) -> list[np.ndarray]:
    frame_dir = video_path.parent / "frames" / video_path.stem
    frame_paths = sorted(glob(str(frame_dir / "*.jpg")))
    frames: list[np.ndarray] = []

    if len(frame_paths) >= count:
        for frame_path in frame_paths[:count]:
            frame_bgr = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_rgb.shape[:2] != (height, width):
                frame_rgb = cv2.resize(frame_rgb, (width, height))
            frames.append(frame_rgb)

    if len(frames) >= count:
        return frames[:count]

    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [np.full((height, width, 3), 255, dtype=np.uint8) for _ in range(count)]

    while len(frames) < count:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[:2] != (height, width):
            frame_rgb = cv2.resize(frame_rgb, (width, height))
        frames.append(frame_rgb)
    cap.release()

    if not frames:
        frames = [np.full((height, width, 3), 255, dtype=np.uint8)]
    while len(frames) < count:
        frames.append(frames[-1].copy())
    return frames


def _open_writer(path: Path, fps: float, height: int, width: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    return writer


def _write_rgb(writer: cv2.VideoWriter, frame_rgb: np.ndarray) -> None:
    writer.write(cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR))


def _contact_colors(
    human_verts: torch.Tensor,
    obj_verts: torch.Tensor,
    left_hand_indices: list[int] | None,
    right_hand_indices: list[int] | None,
    human_color: torch.Tensor,
    obj_color: torch.Tensor,
    threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    red = torch.tensor([1.0, 0.0, 0.0], device=human_verts.device)
    human_colors = human_color.unsqueeze(0).repeat(human_verts.shape[0], 1)
    obj_colors = obj_color.unsqueeze(0).repeat(obj_verts.shape[0], 1)

    for indices in (left_hand_indices, right_hand_indices):
        if not indices:
            continue
        hand_idx = torch.tensor(indices, device=human_verts.device, dtype=torch.long)
        hand_verts = human_verts[hand_idx]
        distances = torch.norm(obj_verts.unsqueeze(1) - hand_verts.unsqueeze(0), dim=-1)

        obj_colors[distances.min(dim=1).values < threshold] = red
        hand_contact = distances.min(dim=0).values < threshold
        human_colors[hand_idx[hand_contact]] = red

    return human_colors, obj_colors


def _build_mesh(
    frame_idx: int,
    motion_seq: dict,
    human_color: torch.Tensor,
    obj_color: torch.Tensor,
    *,
    device: str,
    vis_contact: bool,
    left_hand_indices: list[int] | None,
    right_hand_indices: list[int] | None,
):
    from pytorch3d.structures import join_meshes_as_scene

    from grail.rendering.textures import create_colored_meshes, create_mesh_with_vertex_colors

    human = motion_seq["human_seq"]
    obj = motion_seq["obj_seq"]
    human_verts = human["vertices"][frame_idx]
    obj_verts = obj["vertices_transformed"][frame_idx]

    if vis_contact and (left_hand_indices or right_hand_indices):
        human_colors, obj_colors = _contact_colors(
            human_verts,
            obj_verts,
            left_hand_indices,
            right_hand_indices,
            human_color,
            obj_color,
        )
        human_mesh = create_mesh_with_vertex_colors(
            human_verts, human["triangles"], human_colors, device=device
        )
        obj_mesh = create_mesh_with_vertex_colors(obj_verts, obj["triangles"], obj_colors, device=device)
    else:
        human_mesh = create_colored_meshes(human_verts, human["triangles"], human_color, device=device)
        obj_mesh = create_colored_meshes(obj_verts, obj["triangles"], obj_color, device=device)

    meshes = [human_mesh, obj_mesh]
    for key, value in motion_seq.items():
        if key.startswith("static_") and key.endswith("_seq"):
            meshes.append(
                create_colored_meshes(
                    value["vertices"], value["triangles"], [0.7, 0.7, 0.7], device=device
                )
            )
    return join_meshes_as_scene(meshes)


def _annotate(frame: np.ndarray, frame_idx: int, inter_start_idx: int | None, inter_end_idx: int | None) -> np.ndarray:
    frame = frame.copy()
    cv2.putText(
        frame,
        f"{frame_idx:03d}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    if frame_idx == inter_start_idx or (inter_end_idx is not None and frame_idx == inter_end_idx - 1):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (255, 0, 0), 8)
    return frame


def _render_video(
    output_path: Path,
    frame_indices: list[int],
    motion_seq: dict,
    camera,
    renderer,
    backgrounds: list[np.ndarray] | None,
    fps: float,
    height: int,
    width: int,
    *,
    device: str,
    vis_contact: bool,
    left_hand_indices: list[int] | None,
    right_hand_indices: list[int] | None,
    inter_start_idx: int | None,
    inter_end_idx: int | None,
    label: str,
) -> None:
    from grail.rendering.renderer import render_frame

    writer = _open_writer(output_path, fps, height, width)
    human_color = torch.tensor([0.8, 0.6, 0.4], device=device)
    obj_color = torch.tensor([0.4, 0.6, 0.8], device=device)
    white = np.full((height, width, 3), 255, dtype=np.uint8)

    try:
        for out_idx, frame_idx in enumerate(tqdm(frame_indices, desc=label)):
            mesh = _build_mesh(
                frame_idx,
                motion_seq,
                human_color,
                obj_color,
                device=device,
                vis_contact=vis_contact,
                left_hand_indices=left_hand_indices,
                right_hand_indices=right_hand_indices,
            )
            image, alpha = render_frame(mesh, camera, renderer)
            rgb = (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            mask = (alpha.detach().cpu().numpy() > 0.5)[..., None].astype(np.float32)
            bg = backgrounds[out_idx] if backgrounds is not None else white
            frame = (rgb.astype(np.float32) * mask + bg.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
            _write_rgb(writer, _annotate(frame, frame_idx, inter_start_idx, inter_end_idx))
    finally:
        writer.release()


def _make_top_camera(motion_seq: dict, focal_length: float, image_size: tuple[int, int], device: str):
    from grail.rendering.camera import get_camera
    from grail.rendering.renderer import create_renderer

    human_verts = motion_seq["human_seq"]["vertices"][0]
    obj_verts = motion_seq["obj_seq"]["vertices_transformed"][0]
    all_verts = torch.cat([human_verts, obj_verts], dim=0)
    center = (all_verts.min(dim=0).values + all_verts.max(dim=0).values) / 2
    distance = (all_verts.max(dim=0).values - all_verts.min(dim=0).values).max().item() * 2.5
    distance = max(distance, 1.0)

    cam_r = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32, device=device)
    cam_t = center + torch.tensor([0.0, 0.0, distance], dtype=torch.float32, device=device)
    camera = get_camera(cam_r, cam_t, focal_length, image_size, device=device)
    return camera, create_renderer(camera, image_size, device=device)


def _concat_videos(video_paths: list[Path], output_path: Path, fps: float) -> None:
    caps = [cv2.VideoCapture(str(path)) for path in video_paths if path.is_file()]
    caps = [cap for cap in caps if cap.isOpened()]
    if len(caps) < 2:
        for cap in caps:
            cap.release()
        return

    try:
        ref_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        ref_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        vertical_input = ref_h > ref_w
        frame_count = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps)
        if vertical_input:
            widths = []
            for cap in caps:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                widths.append(int(w * (ref_h / max(h, 1))) // 2 * 2)
            out_h, out_w = ref_h, sum(widths)
        else:
            heights = []
            for cap in caps:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                heights.append(int(h * (ref_w / max(w, 1))) // 2 * 2)
            out_h, out_w = sum(heights), ref_w

        writer = _open_writer(output_path, fps, out_h, out_w)
        try:
            for _ in tqdm(range(frame_count), desc="comparison"):
                frames = []
                for cap in caps:
                    ok, frame = cap.read()
                    if not ok:
                        return
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if vertical_input:
                        h, w = frame.shape[:2]
                        new_w = int(w * (ref_h / max(h, 1))) // 2 * 2
                        frame = cv2.resize(frame, (new_w, ref_h))
                    else:
                        h, w = frame.shape[:2]
                        new_h = int(h * (ref_w / max(w, 1))) // 2 * 2
                        frame = cv2.resize(frame, (ref_w, new_h))
                    frames.append(frame)
                combined = np.hstack(frames) if vertical_input else np.vstack(frames)
                _write_rgb(writer, combined)
        finally:
            writer.release()
    finally:
        for cap in caps:
            cap.release()


def main() -> int:
    args = parse_args()
    _prepend_paths([args.pytorch3d_root, GEM_SMPL_ROOT, GRAIL_ROOT])
    os.environ.setdefault("TORCH_MUSA_ARCH_LIST", "31")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

    from grail.core.config import load_recon_config
    from grail.core.io import load_hoi_data, load_init_rendering_data
    from grail.core.types import parse_recon_config
    from grail.models.human_model import create_human_model
    from grail.rendering.camera import (
        cam_pose_blender_to_opencv,
        cam_pose_opencv_to_pytorch3d,
        get_camera,
    )
    from grail.rendering.renderer import create_renderer
    from grail.visualization.utils.vis_utils import prep_visualizer_input

    args.config = _resolve_under_grail(args.config)
    cfg, cfg_flat = load_recon_config(str(args.config))
    cfg = parse_recon_config(cfg)

    results_dir = args.results_dir or Path(cfg_flat.get("results_dir", "results"))
    results_root = _results_root(results_dir)
    video_id = args.video_id
    dataset, category = video_id.split("/")[:2]

    valid_seq_dir = results_root / args.valid_output_dir / video_id
    result_dir = valid_seq_dir / "result_vis"
    hoi_path = valid_seq_dir / "hoi_data/hoi_data.pkl"
    render_cfg_file = results_root / args.foundation_pose_output_dir / video_id / "first_frame_pose.pickle"
    video_file = results_root / cfg_flat.get("video_dir", "generation/videos_kling") / f"{video_id}.mp4"
    mesh_candidates = sorted(glob(str(results_root / "generation/mesh" / dataset / category / "*.obj")))

    _require_file(hoi_path, "Step 5 MUSA valid hoi_data.pkl")
    _require_file(render_cfg_file, "render config file")
    _require_file(video_file, "input video")
    if not mesh_candidates:
        raise FileNotFoundError(f"No object mesh found under {results_root / 'generation/mesh' / dataset / category}")

    hoi_data = load_hoi_data(str(hoi_path))
    hoi_data["object_path"] = str(hoi_data.get("object_path") or mesh_candidates[0])

    _, _, _, blender_cam_r, blender_cam_t, render_config, _ = load_init_rendering_data(
        str(render_cfg_file),
        to_tensor=True,
        with_human_data=True,
        device=args.device,
    )
    if render_config is None:
        raise RuntimeError(f"Missing frame_height/frame_width/focal_length in {render_cfg_file}")
    frame_height, frame_width, focal_length = render_config
    frame_height = int(frame_height)
    frame_width = int(frame_width)
    focal_length = float(focal_length)

    opencv_cam_r, opencv_cam_t = cam_pose_blender_to_opencv(blender_cam_r, blender_cam_t)
    cam_r, cam_t = cam_pose_opencv_to_pytorch3d(opencv_cam_r, opencv_cam_t)
    camera = get_camera(cam_r, cam_t, focal_length, (frame_height, frame_width), device=args.device)
    renderer = create_renderer(camera, (frame_height, frame_width), device=args.device)

    human_model = create_human_model(cfg["human_model"], device=args.device)
    motion_seq = prep_visualizer_input(
        hoi_data,
        human_model=human_model,
        normalize_trans=False,
        to_numpy=False,
        simplify_mesh=not args.no_simplify_mesh,
        device=args.device,
    )

    total_frames = int(motion_seq["human_seq"]["vertices"].shape[0])
    frame_indices = list(range(0, total_frames, max(1, args.frame_stride)))
    if args.max_frames is not None:
        frame_indices = frame_indices[: max(0, args.max_frames)]
    if not frame_indices:
        raise RuntimeError("No frames selected for rendering")

    fps = float(args.fps if args.fps is not None else _fps_from_video(video_file))
    result_dir.mkdir(parents=True, exist_ok=True)
    input_copy = result_dir / "input.mp4"
    if not input_copy.exists() or input_copy.stat().st_size == 0:
        shutil.copy2(video_file, input_copy)

    print("Rendering Step 5 MUSA visualizations")
    print(f"  hoi: {hoi_path}")
    print(f"  output_dir: {result_dir}")
    print(f"  device: {args.device}")
    print(f"  frames: {len(frame_indices)} / {total_frames}")
    print(f"  image_size: {frame_width}x{frame_height}, fps={fps:g}")

    backgrounds = _read_background_frames(video_file, len(frame_indices), frame_height, frame_width)
    left_hand_indices = right_hand_indices = None
    if not args.no_contact_colors:
        left_hand_indices = human_model.get_segment_indices(["L_Hand"])
        right_hand_indices = human_model.get_segment_indices(["R_Hand"])

    inter_start_idx = hoi_data.get("meta", {}).get("inter_start_idx")
    inter_end_idx = hoi_data.get("meta", {}).get("inter_end_idx")

    result_video = result_dir / "recon_result.mp4"
    _render_video(
        result_video,
        frame_indices,
        motion_seq,
        camera,
        renderer,
        backgrounds,
        fps,
        frame_height,
        frame_width,
        device=args.device,
        vis_contact=not args.no_contact_colors,
        left_hand_indices=left_hand_indices,
        right_hand_indices=right_hand_indices,
        inter_start_idx=inter_start_idx,
        inter_end_idx=inter_end_idx,
        label="camera view",
    )

    top_video = result_dir / "recon_result_top_view.mp4"
    if not args.skip_top_view:
        top_camera, top_renderer = _make_top_camera(
            motion_seq, focal_length, (frame_height, frame_width), args.device
        )
        _render_video(
            top_video,
            frame_indices,
            motion_seq,
            top_camera,
            top_renderer,
            None,
            fps,
            frame_height,
            frame_width,
            device=args.device,
            vis_contact=not args.no_contact_colors,
            left_hand_indices=left_hand_indices,
            right_hand_indices=right_hand_indices,
            inter_start_idx=inter_start_idx,
            inter_end_idx=inter_end_idx,
            label="top view",
        )

    if not args.skip_comparison:
        comparison_inputs = [input_copy, result_video]
        if top_video.is_file():
            comparison_inputs.append(top_video)
        _concat_videos(comparison_inputs, result_dir / "recon_comparison.mp4", fps)

    print("Done")
    print(f"  result: {result_video}")
    if top_video.is_file():
        print(f"  top_view: {top_video}")
    comparison = result_dir / "recon_comparison.mp4"
    if comparison.is_file():
        print(f"  comparison: {comparison}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
