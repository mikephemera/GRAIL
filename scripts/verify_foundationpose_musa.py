#!/usr/bin/env python3
"""Run FoundationPose on MUSA and compare its GRAIL-facing outputs.

The default sample is the cordless-drill GRAIL Step 3 artifact set. The script
does not require CUDA and does not try to make the MUSA result bit-identical to
the CUDA golden output; it verifies that tracking runs, writes the downstream
`poses_in_cam.pkl` contract, and reports pose/image drift.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np


GRAIL_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GRAIL_ROOT.parent
DEFAULT_SCENE = "kid_indoor2-manipulation_rand00001"
DEFAULT_OBJECT = "cordless_drill"
DEFAULT_CATEGORY = "ComAsset"
DEFAULT_REF_ROOT = (
    GRAIL_ROOT
    / "results/generation/foundation_pose_output"
    / DEFAULT_CATEGORY
    / DEFAULT_OBJECT
    / DEFAULT_SCENE
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--foundationpose_root", type=Path, default=WORKSPACE_ROOT / "FoundationPose")
  parser.add_argument("--nvdiffrast_root", type=Path, default=WORKSPACE_ROOT / "nvdiffrast_musa")
  parser.add_argument("--pytorch3d_root", type=Path, default=WORKSPACE_ROOT / "pytorch3d_musa")
  parser.add_argument("--mesh", type=Path, default=GRAIL_ROOT / "results/generation/mesh/ComAsset/cordless_drill/model.obj")
  parser.add_argument("--input_dir", type=Path, default=GRAIL_ROOT / "results/generation/foundation_pose/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001")
  parser.add_argument("--source_rgb_dir", type=Path, default=DEFAULT_REF_ROOT / "rgb")
  parser.add_argument("--video", type=Path, default=GRAIL_ROOT / "results/generation/videos_kling/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001.mp4")
  parser.add_argument("--output_dir", type=Path, default=Path("/tmp/foundationpose_musa_verify/kid_indoor2-manipulation_rand00001"))
  parser.add_argument("--reference_pkl", type=Path, default=DEFAULT_REF_ROOT / "pose_estimation_output/poses_in_cam.pkl")
  parser.add_argument("--reference_track_vis", type=Path, default=DEFAULT_REF_ROOT / "pose_estimation_output/debug/track_vis")
  parser.add_argument("--summary_json", type=Path, default=None)
  parser.add_argument("--musa_device", type=int, default=0)
  parser.add_argument("--max_frames", type=int, default=None)
  parser.add_argument("--track_refine_iter", type=int, default=2)
  parser.add_argument("--debug", type=int, default=1)
  parser.add_argument("--smooth_window", type=int, default=9)
  parser.add_argument("--smooth_polyorder", type=int, default=3)
  parser.add_argument("--is_static", action="store_true", default=False)
  parser.add_argument("--compare_only", action="store_true", default=False)
  parser.add_argument("--build_mycpp", action="store_true", default=False)
  parser.add_argument("--no_build_mycpp", dest="build_mycpp", action="store_false")
  parser.add_argument("--require_mycpp", action="store_true", default=False)
  parser.add_argument("--fail_on_diff", action="store_true", default=False)
  parser.add_argument("--rtol", type=float, default=1e-3)
  parser.add_argument("--atol", type=float, default=1e-5)
  parser.add_argument("--pose_trans_warn", type=float, default=0.02)
  parser.add_argument("--pose_rot_warn", type=float, default=1.0)
  parser.add_argument("--image_mean_warn", type=float, default=20.0)
  return parser.parse_args()


def setup_paths(args: argparse.Namespace) -> None:
  os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
  os.environ.setdefault("TORCH_MUSA_ARCH_LIST", "31")
  os.environ["FOUNDATIONPOSE_DEVICE"] = f"musa:{args.musa_device}"
  for path in (args.pytorch3d_root, args.nvdiffrast_root, args.foundationpose_root, GRAIL_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
      sys.path.insert(0, path_str)


def ensure_mycpp(args: argparse.Namespace) -> bool:
  build_dir = args.foundationpose_root / "mycpp" / "build"
  if str(build_dir) not in sys.path:
    sys.path.insert(0, str(build_dir))

  def has_cluster_poses() -> bool:
    sys.modules.pop("mycpp", None)
    try:
      import mycpp  # type: ignore
      return hasattr(mycpp, "cluster_poses")
    except Exception:
      return False

  if has_cluster_poses():
    return True
  if not args.build_mycpp:
    if args.require_mycpp:
      raise RuntimeError("FoundationPose mycpp extension is missing cluster_poses")
    logging.warning("mycpp.cluster_poses missing; estimator will use the Python fallback rotation grid")
    return False

  logging.info("Building FoundationPose mycpp extension")
  build_dir.mkdir(parents=True, exist_ok=True)
  cmake_cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"]
  try:
    import pybind11
    cmake_cmd.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
  except Exception:
    pass

  try:
    subprocess.run(cmake_cmd, cwd=build_dir, check=True)
    subprocess.run(["cmake", "--build", ".", f"-j{os.cpu_count() or 1}"], cwd=build_dir, check=True)
  except Exception as exc:
    if args.require_mycpp:
      raise
    logging.warning("mycpp build failed; continuing with fallback rotation grid: %s", exc)
    return False

  ok = has_cluster_poses()
  if not ok and args.require_mycpp:
    raise RuntimeError("mycpp build completed but cluster_poses is still unavailable")
  if not ok:
    logging.warning("mycpp.cluster_poses still unavailable; continuing with fallback rotation grid")
  return ok


def preflight(args: argparse.Namespace) -> dict[str, Any]:
  import torch
  import torch_musa  # noqa: F401

  if not torch.musa.is_available():
    raise RuntimeError("MUSA is not available. Run this outside the restricted sandbox on the S5000 host.")
  torch.musa.set_device(args.musa_device)

  import nvdiffrast.torch as dr
  import pytorch3d
  import pytorch3d._C as p3d_c

  info = {
      "torch": torch.__version__,
      "torch_musa": getattr(torch_musa, "__version__", "unknown"),
      "musa_available": bool(torch.musa.is_available()),
      "musa_device_count": int(torch.musa.device_count()),
      "musa_arch_list": list(torch.musa.get_arch_list()),
      "musa_device": torch.musa.get_device_name(args.musa_device),
      "musa_capability": tuple(torch.musa.get_device_capability(args.musa_device)),
      "nvdiffrast": str(Path(dr.__file__).resolve()),
      "has_rasterize_musa_context": bool(hasattr(dr, "RasterizeMusaContext")),
      "pytorch3d": str(Path(pytorch3d.__file__).resolve()),
      "pytorch3d_c": str(Path(p3d_c.__file__).resolve()),
  }
  logging.info("MUSA device: %s arch=%s count=%s", info["musa_device"], info["musa_arch_list"], info["musa_device_count"])
  logging.info("nvdiffrast: %s", info["nvdiffrast"])
  logging.info("pytorch3d: %s", info["pytorch3d"])
  return info


def is_readable_image(path: Path) -> bool:
  if not path.is_file():
    return False
  image = cv2.imread(str(path), cv2.IMREAD_COLOR)
  return image is not None and image.size > 0


def prepare_output_dir(args: argparse.Namespace) -> None:
  rgb_dir = args.output_dir / "rgb"
  rgb_dir.mkdir(parents=True, exist_ok=True)
  shutil.copy2(args.input_dir / "cam_K.txt", args.output_dir / "cam_K.txt")
  shutil.copy2(args.input_dir / "first_frame_pose.pickle", args.output_dir / "first_frame_pose.pickle")

  existing = sorted(rgb_dir.glob("*.png"))
  if existing and is_readable_image(existing[0]):
    logging.info("Using %d existing output RGB frames", len(existing))
    return

  for old in existing:
    old.unlink()

  source_frames = sorted(args.source_rgb_dir.glob("*.png")) if args.source_rgb_dir.is_dir() else []
  if source_frames and is_readable_image(source_frames[0]):
    if args.max_frames is not None:
      source_frames = source_frames[: args.max_frames]
    for idx, src in enumerate(source_frames):
      shutil.copy2(src, rgb_dir / f"{idx:06d}.png")
    logging.info("Copied %d RGB frames from %s", len(source_frames), args.source_rgb_dir)
    return

  if not args.video.is_file():
    raise FileNotFoundError(f"No readable RGB source and video is missing: {args.video}")

  cap = cv2.VideoCapture(str(args.video))
  if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {args.video}")
  frame_idx = 0
  while True:
    ok, frame = cap.read()
    if not ok:
      break
    cv2.imwrite(str(rgb_dir / f"{frame_idx:06d}.png"), frame)
    frame_idx += 1
    if args.max_frames is not None and frame_idx >= args.max_frames:
      break
  cap.release()
  logging.info("Extracted %d RGB frames from %s", frame_idx, args.video)


def blender_to_opencv_convention(world_to_camera_blender: np.ndarray) -> np.ndarray:
  conversion_matrix = np.array([
      [1, 0, 0, 0],
      [0, -1, 0, 0],
      [0, 0, -1, 0],
      [0, 0, 0, 1],
  ])
  return conversion_matrix @ world_to_camera_blender


def transform_object_to_camera_frame(world_to_camera_matrix: np.ndarray, object_position: np.ndarray, object_rotation: np.ndarray):
  object_matrix = np.eye(4)
  object_matrix[:3, :3] = object_rotation
  object_matrix[:3, 3] = object_position.reshape(-1)
  camspace_object_matrix = world_to_camera_matrix @ object_matrix
  return camspace_object_matrix[:3, :3], camspace_object_matrix[:3, 3]


def smooth_if_possible(pose_list: list[np.ndarray], window_length: int, polyorder: int):
  from grail.pose_est.utils import smooth_pose_matrices

  if len(pose_list) < 5:
    return pose_list
  window_length = max(1, int(window_length))
  window = min(window_length, len(pose_list) if len(pose_list) % 2 == 1 else len(pose_list) - 1)
  if window % 2 == 0:
    window -= 1
  if window < 5:
    return pose_list
  return smooth_pose_matrices(pose_list, window_length=window, polyorder=min(int(polyorder), window - 2))


def run_foundationpose_tracking(args: argparse.Namespace) -> list[np.ndarray]:
  import torch
  import trimesh
  import nvdiffrast.torch as dr
  from datareader import YcbineoatReader
  from estimater import FoundationPose, draw_posed_3d_box, draw_xyz_axis, set_logging_format, set_seed
  from learning.training.predict_pose_refine import PoseRefinePredictor
  from learning.training.predict_score import ScorePredictor
  from grail.core.io import load_init_rendering_data, save_object_pose_data
  from grail.core.video import compile_images_to_video
  from grail.rendering.camera import world_to_camera_matrix
  from device_utils import empty_cache

  device = torch.device(f"musa:{args.musa_device}")
  torch.musa.set_device(device)
  set_logging_format()
  set_seed(0)

  mesh = trimesh.load(args.mesh)
  if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump(concatenate=True)

  first_frame_file = args.input_dir / "first_frame_pose.pickle"
  obj_R, obj_t, obj_scale, cam_R, cam_t, _ = load_init_rendering_data(first_frame_file)
  world_to_camera_blender = world_to_camera_matrix(torch.from_numpy(cam_R).float(), torch.from_numpy(cam_t).float()).numpy()
  world_to_camera_opencv = blender_to_opencv_convention(world_to_camera_blender)
  obj_R_camspace, obj_t_camspace = transform_object_to_camera_frame(world_to_camera_opencv, obj_t, obj_R)

  ob_in_cam = np.eye(4)
  ob_in_cam[:3, :3] = obj_R_camspace
  ob_in_cam[:3, 3] = obj_t_camspace
  mesh.apply_scale(obj_scale)

  debug_dir = args.output_dir / "pose_estimation_output" / "debug"
  debug_dir.mkdir(parents=True, exist_ok=True)
  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

  logging.info("Initializing FoundationPose on %s", device)
  scorer = ScorePredictor(device=device)
  refiner = PoseRefinePredictor(device=device)
  glctx = dr.RasterizeCudaContext(device)
  est = FoundationPose(
      model_pts=mesh.vertices,
      model_normals=mesh.vertex_normals,
      mesh=mesh,
      scorer=scorer,
      refiner=refiner,
      debug_dir=str(debug_dir),
      debug=args.debug,
      glctx=glctx,
      device=device,
  )

  reader = YcbineoatReader(video_dir=str(args.output_dir), shorter_side=None, zfar=np.inf)
  num_frames = len(reader.color_files)
  if args.max_frames is not None:
    num_frames = min(num_frames, args.max_frames)

  pose_list: list[np.ndarray] = []
  t_start = time.time()
  for i in range(num_frames):
    color = reader.get_color(i)
    h, w = color.shape[:2]
    depth = np.zeros((h, w), dtype=float)
    if args.is_static:
      pose = ob_in_cam.copy()
    elif i == 0:
      pose = ob_in_cam.copy()
      ob_in_cam_adjusted = ob_in_cam.copy()
      ob_in_cam_adjusted[:3, 3] += ob_in_cam_adjusted[:3, :3] @ est.model_center
      est.pose_last = torch.as_tensor(ob_in_cam_adjusted, dtype=torch.float, device=device)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)
    pose_list.append(pose)
    if (i + 1) % 10 == 0 or i == num_frames - 1:
      elapsed = time.time() - t_start
      logging.info("Frame %d/%d (%.2f fps)", i + 1, num_frames, (i + 1) / max(elapsed, 1e-6))

  pose_list = smooth_if_possible(pose_list, window_length=args.smooth_window, polyorder=args.smooth_polyorder)

  if args.debug >= 1:
    track_vis_dir = debug_dir / "track_vis"
    ob_in_cam_dir = debug_dir / "ob_in_cam"
    track_vis_dir.mkdir(parents=True, exist_ok=True)
    ob_in_cam_dir.mkdir(parents=True, exist_ok=True)
    for i, pose in enumerate(pose_list):
      color = reader.get_color(i)
      center_pose = pose @ np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      imageio.imwrite(track_vis_dir / f"{reader.id_strs[i]}.png", vis)
      np.savetxt(ob_in_cam_dir / f"{reader.id_strs[i]}.txt", pose.reshape(4, 4))
    if args.debug >= 2:
      compile_images_to_video(track_vis_dir, args.output_dir / "pose_estimation_output/pose_estimation_tracking.mp4", fps=24, image_pattern="*.png")

  output_pkl = args.output_dir / "pose_estimation_output" / "poses_in_cam.pkl"
  output_pkl.parent.mkdir(parents=True, exist_ok=True)
  save_object_pose_data(pose_list, output_pkl)
  empty_cache(device)
  logging.info("Saved %d poses to %s", len(pose_list), output_pkl)
  return pose_list


def install_numpy_pickle_aliases() -> None:
  import numpy.core as np_core
  import numpy.core.numeric as np_numeric

  sys.modules.setdefault("numpy._core", np_core)
  sys.modules.setdefault("numpy._core.numeric", np_numeric)


def load_pickle(path: Path):
  install_numpy_pickle_aliases()
  with path.open("rb") as f:
    return pickle.load(f)


def pose_error(ref_pose: np.ndarray, new_pose: np.ndarray) -> tuple[float, float]:
  trans_error = float(np.linalg.norm(new_pose[:3, 3] - ref_pose[:3, 3]))
  r_diff = new_pose[:3, :3] @ ref_pose[:3, :3].T
  trace = np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0)
  rot_error_deg = float(np.degrees(np.arccos(trace)))
  return trans_error, rot_error_deg


def compare_poses(
    reference_pkl: Path,
    output_pkl: Path,
    max_frames: int | None,
    trans_warn: float,
    rot_warn_deg: float,
) -> dict[str, Any]:
  ref_poses = load_pickle(reference_pkl)
  new_poses = load_pickle(output_pkl)
  if max_frames is not None:
    ref_poses = ref_poses[:max_frames]
    new_poses = new_poses[:max_frames]
  result: dict[str, Any] = {
      "reference": str(reference_pkl),
      "output": str(output_pkl),
      "num_reference": len(ref_poses),
      "num_output": len(new_poses),
      "match_frame_count": len(ref_poses) == len(new_poses),
  }
  if len(ref_poses) != len(new_poses):
    return result

  trans_errors = []
  rot_errors = []
  failed_frames = []
  finite = True
  valid_last_row = True
  for frame_idx, (ref_pose, new_pose) in enumerate(zip(ref_poses, new_poses)):
    new_pose = np.asarray(new_pose)
    finite = finite and bool(np.isfinite(new_pose).all())
    valid_last_row = valid_last_row and bool(np.allclose(new_pose[3], np.array([0, 0, 0, 1]), atol=1e-5))
    trans, rot = pose_error(np.asarray(ref_pose), new_pose)
    trans_errors.append(trans)
    rot_errors.append(rot)
    if trans > trans_warn or rot > rot_warn_deg:
      failed_frames.append({
          "frame": frame_idx,
          "trans_error_m": float(trans),
          "rot_error_deg": float(rot),
      })

  trans_arr = np.asarray(trans_errors)
  rot_arr = np.asarray(rot_errors)
  result.update({
      "match": len(failed_frames) == 0,
      "finite": finite,
      "valid_last_row": valid_last_row,
      "max_trans_error_m": float(trans_arr.max(initial=0.0)),
      "mean_trans_error_m": float(trans_arr.mean() if len(trans_arr) else 0.0),
      "median_trans_error_m": float(np.median(trans_arr) if len(trans_arr) else 0.0),
      "p95_trans_error_m": float(np.percentile(trans_arr, 95) if len(trans_arr) else 0.0),
      "max_rot_error_deg": float(rot_arr.max(initial=0.0)),
      "mean_rot_error_deg": float(rot_arr.mean() if len(rot_arr) else 0.0),
      "median_rot_error_deg": float(np.median(rot_arr) if len(rot_arr) else 0.0),
      "p95_rot_error_deg": float(np.percentile(rot_arr, 95) if len(rot_arr) else 0.0),
      "num_failed_frames": len(failed_frames),
      "failed_frames": failed_frames[:20],
      "failed_frames_truncated": len(failed_frames) > 20,
      "trans_errors": trans_arr.tolist(),
      "rot_errors": rot_arr.tolist(),
  })
  return result


def compare_track_vis(
    reference_dir: Path,
    output_dir: Path,
    max_frames: int | None,
    diff_dir: Path,
    mean_warn: float,
) -> dict[str, Any]:
  ref_files = sorted(reference_dir.glob("*.png"))
  out_files = sorted(output_dir.glob("*.png"))
  if max_frames is not None:
    ref_files = ref_files[:max_frames]
    out_files = out_files[:max_frames]
  result: dict[str, Any] = {
      "reference": str(reference_dir),
      "output": str(output_dir),
      "num_reference": len(ref_files),
      "num_output": len(out_files),
      "match_frame_count": len(ref_files) == len(out_files),
  }
  if not ref_files or not out_files or len(ref_files) != len(out_files):
    return result

  per_frame = []
  unreadable = []
  for ref_path, out_path in zip(ref_files, out_files):
    ref = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
    out = cv2.imread(str(out_path), cv2.IMREAD_COLOR)
    if ref is None or out is None or ref.shape != out.shape:
      item = {"frame": ref_path.stem, "readable": False}
      per_frame.append(item)
      unreadable.append(item)
      continue
    diff = np.abs(ref.astype(np.float32) - out.astype(np.float32))
    per_frame.append({
        "frame": ref_path.stem,
        "readable": True,
        "mean_abs_error": float(diff.mean()),
        "max_abs_error": float(diff.max()),
  })

  readable = [item for item in per_frame if item.get("readable")]
  worst = sorted(readable, key=lambda item: item["mean_abs_error"], reverse=True)[:5]
  warn_frames = [item for item in readable if item["mean_abs_error"] > mean_warn]
  written_diff_dir = None
  for item in worst:
    if item["mean_abs_error"] <= 0:
      continue
    diff_dir.mkdir(parents=True, exist_ok=True)
    written_diff_dir = str(diff_dir)
    ref = cv2.imread(str(reference_dir / f"{item['frame']}.png"), cv2.IMREAD_COLOR)
    out = cv2.imread(str(output_dir / f"{item['frame']}.png"), cv2.IMREAD_COLOR)
    diff = np.abs(ref.astype(np.float32) - out.astype(np.float32)).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(diff_dir / f"{item['frame']}_diff.png"), diff)

  means = np.asarray([item["mean_abs_error"] for item in readable], dtype=np.float64)
  maxes = np.asarray([item["max_abs_error"] for item in readable], dtype=np.float64)
  result.update({
      "match": len(warn_frames) == 0 and not unreadable,
      "num_readable_pairs": len(readable),
      "num_unreadable_or_shape_mismatch": len(unreadable),
      "mean_abs_error": float(means.mean() if len(means) else 0.0),
      "median_abs_error": float(np.median(means) if len(means) else 0.0),
      "max_frame_mean_abs_error": float(means.max(initial=0.0)),
      "max_abs_error": float(maxes.max(initial=0.0)),
      "num_warn_frames": len(warn_frames),
      "warn_frames": warn_frames[:20],
      "warn_frames_truncated": len(warn_frames) > 20,
      "worst_frames": worst,
      "diff_dir": written_diff_dir,
  })
  return result


def compare_generic_values(reference_pkl: Path, output_pkl: Path, max_frames: int | None, rtol: float, atol: float):
  ref_poses = load_pickle(reference_pkl)
  new_poses = load_pickle(output_pkl)
  if max_frames is not None:
    ref_poses = ref_poses[:max_frames]
    new_poses = new_poses[:max_frames]
  try:
    from scripts.verify_module_output import compare_values
  except ModuleNotFoundError:
    from verify_module_output import compare_values
  return compare_values(ref_poses, new_poses, rtol=rtol, atol=atol)


def print_report(summary: dict[str, Any], args: argparse.Namespace) -> bool:
  pose = summary["pose_compare"]
  image = summary.get("track_vis_compare")
  generic = summary.get("generic_compare")
  artifacts_ok = (
      pose.get("match_frame_count")
      and pose.get("finite")
      and pose.get("valid_last_row")
      and Path(summary["output_pkl"]).is_file()
  )
  pose_warn = (
      pose.get("max_trans_error_m", 0.0) > args.pose_trans_warn
      or pose.get("max_rot_error_deg", 0.0) > args.pose_rot_warn
  )
  image_warn = image is not None and image.get("mean_abs_error", 0.0) > args.image_mean_warn

  print(f"\n{'=' * 70}")
  print("  FoundationPose MUSA runtime")
  print(f"{'=' * 70}")
  print(f"  device: {summary['preflight']['musa_device']}")
  print(f"  arch:   {summary['preflight']['musa_arch_list']}")
  print(f"  count:  {summary['preflight']['musa_device_count']}")
  print(f"  output: {summary['output_dir']}")
  print(f"  pkl:    {summary['output_pkl']}")
  print(f"  mycpp.cluster_poses: {'available' if summary.get('mycpp_cluster_poses') else 'fallback/unavailable'}")

  print(f"\n{'=' * 70}")
  print("  [Check 1] Generic structured comparison")
  print(f"  tolerances: rtol={args.rtol}, atol={args.atol}")
  print(f"  note: informational for MUSA; use pose/image checks below for pass/fail")
  print(f"{'=' * 70}")
  if generic is None:
    print("  - skipped: reference/output poses were not both available")
  elif generic.get("match"):
    print("  ✓ Generic comparison: ALL VALUES MATCH")
  else:
    print("  ✗ Generic comparison: DIFFERENCES DETECTED")
    try:
      try:
        from scripts.verify_module_output import print_report as print_value_report
      except ModuleNotFoundError:
        from verify_module_output import print_report as print_value_report
      print_value_report(generic, key_path="poses", indent=2)
    except Exception as exc:
      print(f"  unable to print structured diff details: {exc}")

  print(f"\n{'=' * 70}")
  print("  [Check 2] Per-frame pose errors")
  print(f"  warning thresholds: trans={args.pose_trans_warn}m ({args.pose_trans_warn * 1000:.0f}mm), rot={args.pose_rot_warn}°")
  print(f"{'=' * 70}")
  if not pose.get("match_frame_count"):
    print(f"  ✗ Frame count mismatch: {pose.get('num_reference')} reference vs {pose.get('num_output')} output")
  else:
    print(f"  Frames: {pose.get('num_output')}")
    print("  Translation error (mm):")
    print(f"    max    = {pose.get('max_trans_error_m', 0.0) * 1000:.2f} mm")
    print(f"    p95    = {pose.get('p95_trans_error_m', 0.0) * 1000:.2f} mm")
    print(f"    mean   = {pose.get('mean_trans_error_m', 0.0) * 1000:.2f} mm")
    print(f"    median = {pose.get('median_trans_error_m', 0.0) * 1000:.2f} mm")
    print("  Rotation error (deg):")
    print(f"    max    = {pose.get('max_rot_error_deg', 0.0):.4f}°")
    print(f"    p95    = {pose.get('p95_rot_error_deg', 0.0):.4f}°")
    print(f"    mean   = {pose.get('mean_rot_error_deg', 0.0):.4f}°")
    print(f"    median = {pose.get('median_rot_error_deg', 0.0):.4f}°")
    if pose.get("match"):
      print(f"\n  ✓ All {pose.get('num_output')} frames within warning thresholds")
    else:
      print(f"\n  ✗ {pose.get('num_failed_frames', 0)}/{pose.get('num_output')} frames exceed warning thresholds")
      for ff in pose.get("failed_frames", []):
        print(
            f"    Frame {ff['frame']:04d}: "
            f"trans={ff['trans_error_m'] * 1000:.2f}mm, "
            f"rot={ff['rot_error_deg']:.4f}°"
        )
      if pose.get("failed_frames_truncated"):
        print("    ... (list truncated)")

  print(f"\n{'=' * 70}")
  print("  [Check 3] Track-vis image differences")
  print(f"  warning threshold: mean_abs_error={args.image_mean_warn}")
  print(f"{'=' * 70}")
  if image is None:
    print("  - skipped: reference or output track_vis directory is missing")
  elif not image.get("match_frame_count"):
    print(f"  ✗ Frame count mismatch: {image.get('num_reference')} reference vs {image.get('num_output')} output")
  else:
    print(f"  Frames: {image.get('num_output')}")
    print(f"  Readable pairs: {image.get('num_readable_pairs')} / {image.get('num_output')}")
    print(f"  Unreadable/shape mismatch: {image.get('num_unreadable_or_shape_mismatch')}")
    print("  Mean absolute error:")
    print(f"    mean       = {image.get('mean_abs_error', 0.0):.3f}")
    print(f"    median     = {image.get('median_abs_error', 0.0):.3f}")
    print(f"    worst mean = {image.get('max_frame_mean_abs_error', 0.0):.3f}")
    print(f"    max pixel  = {image.get('max_abs_error', 0.0):.1f}")
    if image.get("diff_dir"):
      print(f"  Diff images: {image.get('diff_dir')}")
    if image.get("match"):
      print(f"\n  ✓ All readable frames within image warning threshold")
    else:
      print(f"\n  ✗ {image.get('num_warn_frames', 0)}/{image.get('num_output')} frames exceed image warning threshold")
      for ff in image.get("warn_frames", []):
        print(
            f"    Frame {ff['frame']}: "
            f"mean_abs={ff['mean_abs_error']:.3f}, "
            f"max_abs={ff['max_abs_error']:.1f}"
        )
      if image.get("warn_frames_truncated"):
        print("    ... (list truncated)")
      if image.get("worst_frames"):
        print("  Worst frames:")
        for ff in image.get("worst_frames", []):
          print(
              f"    Frame {ff['frame']}: "
              f"mean_abs={ff['mean_abs_error']:.3f}, "
              f"max_abs={ff['max_abs_error']:.1f}"
          )

  print(f"\n{'=' * 70}")
  print("  [Check 4] Artifact contract and final summary")
  print(f"{'=' * 70}")
  print(f"  finite poses: {'yes' if pose.get('finite') else 'no'}")
  print(f"  valid last row: {'yes' if pose.get('valid_last_row') else 'no'}")
  print(f"  output pkl exists: {'yes' if Path(summary['output_pkl']).is_file() else 'no'}")
  print(f"  artifact contract: {'PASS' if artifacts_ok else 'FAIL'}")
  if pose_warn:
    print("  pose drift: WARN (above reporting threshold)")
  else:
    print("  pose drift: OK")
  if image_warn:
    print("  image drift: WARN (above reporting threshold)")
  elif image is not None:
    print("  image drift: OK")

  print(f"\n{'=' * 70}")
  if artifacts_ok and not (args.fail_on_diff and (pose_warn or image_warn)):
    print("  ✓ VERIFICATION PASSED — downstream artifact contract is valid")
    if pose_warn or image_warn:
      print("  ⚠ Differences were reported above; strict CUDA parity is not required by default")
  else:
    print("  ✗ VERIFICATION FAILED — see details above")
  print(f"{'=' * 70}\n")

  if not artifacts_ok:
    return False
  if args.fail_on_diff and (pose_warn or image_warn):
    return False
  return True


def main() -> int:
  args = parse_args()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
  install_numpy_pickle_aliases()
  setup_paths(args)
  mycpp_ok = ensure_mycpp(args)
  preflight_info = preflight(args)

  if not args.compare_only:
    prepare_output_dir(args)
    run_foundationpose_tracking(args)

  output_pkl = args.output_dir / "pose_estimation_output" / "poses_in_cam.pkl"
  generic_summary = compare_generic_values(args.reference_pkl, output_pkl, args.max_frames, args.rtol, args.atol)
  pose_summary = compare_poses(args.reference_pkl, output_pkl, args.max_frames, args.pose_trans_warn, args.pose_rot_warn)
  track_vis_dir = args.output_dir / "pose_estimation_output" / "debug" / "track_vis"
  image_summary = None
  if args.reference_track_vis.is_dir() and track_vis_dir.is_dir():
    image_summary = compare_track_vis(args.reference_track_vis, track_vis_dir, args.max_frames, args.output_dir / "pose_estimation_output" / "debug" / "track_vis_diff", args.image_mean_warn)

  summary = {
      "preflight": preflight_info,
      "mycpp_cluster_poses": mycpp_ok,
      "track_refine_iter": args.track_refine_iter,
      "temporal_smoothing": {
          "window_length": args.smooth_window,
          "polyorder": args.smooth_polyorder,
      },
      "output_dir": str(args.output_dir),
      "output_pkl": str(output_pkl),
      "generic_compare": generic_summary,
      "pose_compare": pose_summary,
      "track_vis_compare": image_summary,
  }
  summary_json = args.summary_json or args.output_dir / "foundationpose_musa_summary.json"
  summary_json.parent.mkdir(parents=True, exist_ok=True)
  summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  logging.info("Wrote summary JSON: %s", summary_json)
  return 0 if print_report(summary, args) else 1


if __name__ == "__main__":
  raise SystemExit(main())
