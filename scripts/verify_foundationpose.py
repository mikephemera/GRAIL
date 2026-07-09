#!/usr/bin/env python3
"""
FoundationPose verification script for the recon_4dhoi pipeline.

Runs FoundationPose 6-DoF object tracking in-process and verifies the output
against a golden reference (poses_in_cam.pkl). Supports two modes:

  Mode A (run + verify): Run FoundationPose tracking, then compare against reference.
  Mode B (verify only):    Compare existing output against reference without re-running.

Usage:
    # Mode A: Run FoundationPose and verify against golden reference
    python scripts/verify_foundationpose.py \
        --mesh results/generation/mesh/ComAsset/cordless_drill/model.obj \
        --input_dir results/generation/foundation_pose/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001 \
        --video results/generation/videos_kling/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001.mp4 \
        --reference_pkl results/generation/foundation_pose_output/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001/pose_estimation_output/poses_in_cam.pkl

    # Mode B: Verify existing output only
    python scripts/verify_foundationpose.py \
        --output_dir results/generation/foundation_pose_output/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001 \
        --reference_pkl results/generation/foundation_pose_output/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001/pose_estimation_output/poses_in_cam.pkl \
        --compare_only

    # Self-test: run on existing data and verify output matches itself
    python scripts/verify_foundationpose.py \
        --mesh results/generation/mesh/ComAsset/cordless_drill/model.obj \
        --input_dir results/generation/foundation_pose/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001 \
        --output_dir /tmp/fp_self_test

Requirements:
    - grail conda environment (or equivalent with nvdiffrast, pytorch3d, warp-lang)
    - FoundationPose weights downloaded (scripts/setup/download_checkpoints.sh)
    - Python 3.10+
"""

import argparse
import gc
import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — replicate grail/adapters/foundation_pose.py setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_WORKSPACE_ROOT = os.path.dirname(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)


def _has_foundationpose_source(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "estimater.py"))


def _resolve_foundationpose_root(path: str) -> str:
    candidates = [
        os.path.abspath(os.path.expanduser(path)),
        os.path.join(_PROJECT_ROOT, "imports", "FoundationPose"),
    ]
    for candidate in candidates:
        if _has_foundationpose_source(candidate):
            return candidate

    sys.exit(
        "FoundationPose not found at "
        + " or ".join(candidates)
        + "; expected estimater.py"
    )


def _setup_paths(foundationpose_root: str) -> None:
    """Ensure all required modules are importable."""
    foundation_pose_dir = _resolve_foundationpose_root(foundationpose_root)
    if foundation_pose_dir not in sys.path:
        sys.path.insert(0, foundation_pose_dir)

    weights = [
        "weights/2023-10-28-18-33-37/model_best.pth",
        "weights/2024-01-11-20-02-45/model_best.pth",
    ]
    for w in weights:
        wpath = os.path.join(foundation_pose_dir, w)
        if not os.path.exists(wpath):
            sys.exit(f"Weight file missing: {wpath}. Run: bash scripts/setup/download_checkpoints.sh")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify FoundationPose object tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Paths
    parser.add_argument(
        "--foundationpose_root",
        type=str,
        default=os.path.join(_WORKSPACE_ROOT, "FoundationPose_musa"),
        help="Path to the FoundationPose checkout",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="results/generation/mesh/ComAsset/cordless_drill/model.obj",
        help="Path to object mesh (.obj) [required for Mode A]",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory with first_frame_pose.pickle and cam_K.txt [required for Mode A]",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write tracking results. Default: {input_dir}_verify_output",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Input video (required if output_dir/rgb/ does not already exist)",
    )
    parser.add_argument(
        "--reference_pkl",
        type=str,
        default=(
            "results/generation/foundation_pose_output/ComAsset/cordless_drill/"
            "kid_indoor2-manipulation_rand00001/pose_estimation_output/poses_in_cam.pkl"
        ),
        help="Path to golden poses_in_cam.pkl for comparison",
    )
    # Mode control
    parser.add_argument(
        "--compare_only",
        action="store_true",
        default=False,
        help="Skip tracking, only compare existing output_dir against reference (Mode B)",
    )
    # Tracking parameters
    parser.add_argument(
        "--track_refine_iter",
        type=int,
        default=2,
        help="Refinement iterations per frame (default: 2)",
    )
    parser.add_argument(
        "--is_static",
        action="store_true",
        default=False,
        help="If set, reuse first-frame pose for all frames (skip tracking)",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=1,
        help="Debug level: 0=none, 1=vis images, 2=vis images+video (default: 1)",
    )
    # Tolerance
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for float comparison (default: 1e-3)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for float comparison (default: 1e-5)",
    )
    parser.add_argument(
        "--pose_trans_atol",
        type=float,
        default=0.02,
        help="Max allowed translation error in meters (default: 0.02 = 2cm)",
    )
    parser.add_argument(
        "--pose_rot_atol",
        type=float,
        default=1.0,
        help="Max allowed rotation error in degrees (default: 1.0)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Coordinate conversion (from grail/adapters/foundation_pose.py)
# ---------------------------------------------------------------------------

def blender_to_opencv_convention(world_to_camera_blender: np.ndarray) -> np.ndarray:
    """Convert camera matrix from Blender to OpenCV convention."""
    conversion_matrix = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ])
    return conversion_matrix @ world_to_camera_blender


def transform_object_to_camera_frame(
    world_to_camera_matrix: np.ndarray,
    object_position: np.ndarray,
    object_rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform object pose from world frame to camera frame."""
    object_matrix = np.eye(4)
    object_matrix[:3, :3] = object_rotation
    object_matrix[:3, 3] = object_position.reshape(-1)
    camspace_object_matrix = world_to_camera_matrix @ object_matrix
    return camspace_object_matrix[:3, :3], camspace_object_matrix[:3, 3]


# ---------------------------------------------------------------------------
# Phase 1: Run FoundationPose tracking
# ---------------------------------------------------------------------------

def prepare_output_dir(
    output_dir: str, input_dir: str, video_path: Optional[str] = None
) -> None:
    """Prepare output directory with required files.

    Copies cam_K.txt and ensures rgb/*.png frames exist (extracted from video if needed).
    """
    import cv2

    os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)

    # Copy cam_K.txt if not already present
    src_k = os.path.join(input_dir, "cam_K.txt")
    dst_k = os.path.join(output_dir, "cam_K.txt")
    if not os.path.exists(dst_k):
        if os.path.exists(src_k):
            import shutil
            shutil.copy2(src_k, dst_k)
            logging.info("Copied cam_K.txt to output dir")
        else:
            raise FileNotFoundError(f"cam_K.txt not found at {src_k}")

    # Check if frames already exist
    existing_frames = sorted(
        f for f in os.listdir(os.path.join(output_dir, "rgb"))
        if f.endswith(".png")
    )
    if existing_frames:
        logging.info(f"Using {len(existing_frames)} existing frames in output_dir/rgb/")
        return

    # Extract frames from video
    if not video_path:
        raise FileNotFoundError(
            "No frames found in output_dir/rgb/ and no --video provided. "
            "Either pre-populate output_dir/rgb/*.png or pass --video."
        )
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    logging.info(f"Extracting frames from video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = os.path.join(output_dir, "rgb", f"{frame_idx:06d}.png")
        cv2.imwrite(out_path, frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            logging.info(f"  Extracted {frame_idx} frames...")
    cap.release()
    logging.info(f"Extracted {frame_idx} frames total")


def run_foundationpose_tracking(
    mesh_file: str,
    input_dir: str,
    output_dir: str,
    track_refine_iter: int = 2,
    is_static: bool = False,
    debug: int = 1,
    debug_dir: Optional[str] = None,
) -> List[np.ndarray]:
    """Run FoundationPose 6-DoF object tracking in-process.

    Returns:
        List of (4,4) numpy arrays — camera-frame object poses for each frame.
    """
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    from datareader import YcbineoatReader
    from estimater import FoundationPose, set_logging_format, set_seed
    from learning.training.predict_score import ScorePredictor
    from learning.training.predict_pose_refine import PoseRefinePredictor

    from grail.core.io import load_init_rendering_data, save_object_pose_data
    from grail.core.video import compile_images_to_video
    from grail.pose_est.utils import smooth_pose_matrices
    from grail.rendering.camera import world_to_camera_matrix

    # ------------------------------------------------------------------
    # 1. Load mesh
    # ------------------------------------------------------------------
    logging.info(f"Loading mesh: {mesh_file}")
    mesh = trimesh.load(mesh_file)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    logging.info(
        f"  vertices: {mesh.vertices.shape}, faces: {mesh.faces.shape}"
    )

    # ------------------------------------------------------------------
    # 2. Load first-frame pose and convert to camera frame
    # ------------------------------------------------------------------
    first_frame_file = os.path.join(input_dir, "first_frame_pose.pickle")
    logging.info(f"Loading first-frame pose: {first_frame_file}")
    obj_R, obj_t, obj_scale, cam_R, cam_t, render_config = load_init_rendering_data(
        first_frame_file
    )

    world_to_camera_blender = world_to_camera_matrix(
        torch.from_numpy(cam_R).float(), torch.from_numpy(cam_t).float()
    ).numpy()
    world_to_camera_opencv = blender_to_opencv_convention(world_to_camera_blender)
    obj_R_camspace, obj_t_camspace = transform_object_to_camera_frame(
        world_to_camera_opencv, obj_t, obj_R
    )

    ob_in_cam = np.eye(4)
    ob_in_cam[:3, :3] = obj_R_camspace
    ob_in_cam[:3, 3] = obj_t_camspace
    mesh.apply_scale(obj_scale)
    logging.info("  First-frame pose converted to camera frame")

    # ------------------------------------------------------------------
    # 3. Initialize FoundationPose
    # ------------------------------------------------------------------
    set_logging_format()
    set_seed(0)

    if debug_dir is None:
        debug_dir = os.path.join(output_dir, "pose_estimation_output", "debug")
    os.makedirs(debug_dir, exist_ok=True)

    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    logging.info("Initializing FoundationPose (loading scorer + refiner + nvdiffrast)...")
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir,
        debug=debug,
        glctx=glctx,
    )
    logging.info("  FoundationPose initialized")

    # ------------------------------------------------------------------
    # 4. Create reader and run tracking
    # ------------------------------------------------------------------
    reader = YcbineoatReader(video_dir=output_dir, shorter_side=None, zfar=np.inf)
    logging.info(f"Tracking {len(reader.color_files)} frames...")

    pose_list = []
    t_start = time.time()
    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        H, W = color.shape[:2]
        depth = np.zeros((H, W), dtype=float)

        if is_static:
            pose = ob_in_cam.copy()
        else:
            if i == 0:
                pose = ob_in_cam.copy()
                ob_in_cam_adjusted = ob_in_cam.copy()
                ob_in_cam_adjusted[:3, 3] += (
                    ob_in_cam_adjusted[:3, :3] @ est.model_center
                )
                est.pose_last = torch.as_tensor(
                    ob_in_cam_adjusted, dtype=torch.float, device="cuda"
                )
            else:
                pose = est.track_one(
                    rgb=color,
                    depth=depth,
                    K=reader.K,
                    iteration=track_refine_iter,
                )

        pose_list.append(pose)

        if (i + 1) % 30 == 0 or i == len(reader.color_files) - 1:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed if elapsed > 0 else 0
            logging.info(f"  Frame {i + 1}/{len(reader.color_files)} ({fps:.1f} fps)")

    elapsed = time.time() - t_start
    logging.info(f"Tracking complete: {len(pose_list)} frames in {elapsed:.1f}s")

    # ------------------------------------------------------------------
    # 5. Smooth and save
    # ------------------------------------------------------------------
    logging.info("Applying Savitzky-Golay temporal smoothing...")
    pose_list = smooth_pose_matrices(pose_list, window_length=9, polyorder=3)

    # Visualization (debug >= 1)
    if debug >= 1:
        from estimater import draw_posed_3d_box, draw_xyz_axis
        import imageio

        track_vis_dir = os.path.join(debug_dir, "track_vis")
        ob_in_cam_dir = os.path.join(debug_dir, "ob_in_cam")
        os.makedirs(track_vis_dir, exist_ok=True)
        os.makedirs(ob_in_cam_dir, exist_ok=True)

        for i in range(len(pose_list)):
            color = reader.get_color(i)
            center_pose = pose_list[i] @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(
                color,
                ob_in_cam=center_pose,
                scale=0.1,
                K=reader.K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            imageio.imwrite(f"{track_vis_dir}/{reader.id_strs[i]}.png", vis)
            np.savetxt(
                f"{ob_in_cam_dir}/{reader.id_strs[i]}.txt",
                pose_list[i].reshape(4, 4),
            )

        if debug >= 2:
            video_output_path = os.path.join(
                output_dir, "pose_estimation_output", "pose_estimation_tracking.mp4"
            )
            os.makedirs(os.path.dirname(video_output_path), exist_ok=True)
            compile_images_to_video(
                image_dir=track_vis_dir,
                output_video_path=video_output_path,
                fps=24,
                image_pattern="*.png",
            )
            logging.info(f"Visualization video: {video_output_path}")

    # Save final output
    poses_output_file = os.path.join(
        output_dir, "pose_estimation_output", "poses_in_cam.pkl"
    )
    os.makedirs(os.path.dirname(poses_output_file), exist_ok=True)
    save_object_pose_data(pose_list, poses_output_file)
    logging.info(f"Saved {len(pose_list)} poses → {poses_output_file}")

    return pose_list


# ---------------------------------------------------------------------------
# Phase 2: Compare against golden reference
# ---------------------------------------------------------------------------

def compute_pose_errors(
    ref_pose: np.ndarray, new_pose: np.ndarray,
) -> Tuple[float, float]:
    """Compute per-frame translation error (meters) and rotation error (degrees).

    Args:
        ref_pose: (4, 4) reference pose matrix
        new_pose: (4, 4) new pose matrix

    Returns:
        (trans_error_meters, rot_error_degrees)
    """
    # Translation error (Euclidean distance in meters)
    trans_error = float(np.linalg.norm(new_pose[:3, 3] - ref_pose[:3, 3]))

    # Rotation error (geodesic distance in degrees)
    R_diff = new_pose[:3, :3] @ ref_pose[:3, :3].T
    # Clamp trace to valid arccos range
    trace = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    rot_error_rad = float(np.arccos(trace))
    rot_error_deg = float(np.degrees(rot_error_rad))

    return trans_error, rot_error_deg


def compare_poses(
    ref_poses: List[np.ndarray],
    new_poses: List[np.ndarray],
    trans_atol: float = 0.02,
    rot_atol_deg: float = 1.0,
) -> Dict:
    """Compare two pose lists with per-frame error statistics.

    Returns:
        dict with keys: match, trans_errors, rot_errors, max_trans, mean_trans,
                        max_rot, mean_rot, num_frames, failed_frames
    """
    if len(ref_poses) != len(new_poses):
        return {
            "match": False,
            "error": f"Frame count mismatch: {len(ref_poses)} vs {len(new_poses)}",
        }

    num_frames = len(ref_poses)
    trans_errors = np.zeros(num_frames)
    rot_errors = np.zeros(num_frames)
    failed_frames = []

    for i in range(num_frames):
        trans_errors[i], rot_errors[i] = compute_pose_errors(ref_poses[i], new_poses[i])
        if trans_errors[i] > trans_atol or rot_errors[i] > rot_atol_deg:
            failed_frames.append({
                "frame": i,
                "trans_error_m": float(trans_errors[i]),
                "rot_error_deg": float(rot_errors[i]),
            })

    all_match = len(failed_frames) == 0

    return {
        "match": all_match,
        "num_frames": num_frames,
        "max_trans_error_m": float(np.max(trans_errors)),
        "mean_trans_error_m": float(np.mean(trans_errors)),
        "median_trans_error_m": float(np.median(trans_errors)),
        "max_rot_error_deg": float(np.max(rot_errors)),
        "mean_rot_error_deg": float(np.mean(rot_errors)),
        "median_rot_error_deg": float(np.median(rot_errors)),
        "num_failed_frames": len(failed_frames),
        "failed_frames": failed_frames[:20],  # truncate long lists
        "failed_frames_truncated": len(failed_frames) > 20,
        "trans_errors": trans_errors.tolist(),
        "rot_errors": rot_errors.tolist(),
    }


def verify_poses_pkl(
    reference_pkl: str,
    output_pkl: str,
    trans_atol: float = 0.02,
    rot_atol_deg: float = 1.0,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """Full verification: load, compare, report.

    Returns:
        True if all checks pass.
    """
    # Load both files
    logging.info(f"Loading reference: {reference_pkl}")
    with open(reference_pkl, "rb") as f:
        ref_poses = pickle.load(f)
    logging.info(f"  Reference: {len(ref_poses)} frames")

    logging.info(f"Loading new output: {output_pkl}")
    with open(output_pkl, "rb") as f:
        new_poses = pickle.load(f)
    logging.info(f"  New output: {len(new_poses)} frames")

    all_pass = True

    # --- Check 1: Generic structured comparison ---
    print(f"\n{'=' * 70}")
    print("  [Check 1] Generic structured comparison")
    print(f"  tolerances: rtol={rtol}, atol={atol}")
    print(f"{'=' * 70}")

    from scripts.verify_module_output import compare_values, print_report

    gen_result = compare_values(ref_poses, new_poses, rtol=rtol, atol=atol)
    if gen_result["match"]:
        print("  ✓ Generic comparison: ALL VALUES MATCH")
    else:
        print("  ✗ Generic comparison: DIFFERENCES DETECTED")
        print_report(gen_result, key_path="poses", indent=2)
        all_pass = False

    # --- Check 2: Per-frame pose errors ---
    print(f"\n{'=' * 70}")
    print("  [Check 2] Per-frame pose errors")
    print(f"  tolerances: trans={trans_atol}m ({trans_atol*1000:.0f}mm), rot={rot_atol_deg}°")
    print(f"{'=' * 70}")

    pose_result = compare_poses(ref_poses, new_poses, trans_atol, rot_atol_deg)

    print(f"  Frames: {pose_result['num_frames']}")
    print(f"  Translation error (mm):")
    print(f"    max    = {pose_result['max_trans_error_m']*1000:.2f} mm")
    print(f"    mean   = {pose_result['mean_trans_error_m']*1000:.2f} mm")
    print(f"    median = {pose_result['median_trans_error_m']*1000:.2f} mm")
    print(f"  Rotation error (deg):")
    print(f"    max    = {pose_result['max_rot_error_deg']:.4f}°")
    print(f"    mean   = {pose_result['mean_rot_error_deg']:.4f}°")
    print(f"    median = {pose_result['median_rot_error_deg']:.4f}°")

    if pose_result["match"]:
        print(f"\n  ✓ All {pose_result['num_frames']} frames within tolerance")
    else:
        print(
            f"\n  ✗ {pose_result['num_failed_frames']}/{pose_result['num_frames']} "
            "frames exceed tolerance"
        )
        for ff in pose_result["failed_frames"]:
            print(
                f"    Frame {ff['frame']:04d}: "
                f"trans={ff['trans_error_m']*1000:.2f}mm, "
                f"rot={ff['rot_error_deg']:.4f}°"
            )
        if pose_result.get("failed_frames_truncated"):
            print("    ... (list truncated)")
        all_pass = False

    # --- Final summary ---
    print(f"\n{'=' * 70}")
    if all_pass:
        print("  ✓ VERIFICATION PASSED — all checks match")
        print(f"{'=' * 70}")
    else:
        print("  ✗ VERIFICATION FAILED — see details above")
        print(f"{'=' * 70}")

    return all_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.compare_only:
        _setup_paths(args.foundationpose_root)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.input_dir:
        output_dir = f"{args.input_dir}_verify_output"
    elif args.compare_only:
        # In compare_only mode, output_dir must be the dir containing pose_estimation_output/
        logging.error("--compare_only requires --output_dir")
        sys.exit(1)
    else:
        logging.error("Either --output_dir or --input_dir must be provided")
        sys.exit(1)

    all_pass = True

    # ------------------------------------------------------------------
    # Phase 1: Run tracking (skip if --compare_only)
    # ------------------------------------------------------------------
    if not args.compare_only:
        logging.info("=" * 60)
        logging.info("  Phase 1: Running FoundationPose tracking")
        logging.info("=" * 60)
        logging.info(f"  Input dir:  {args.input_dir}")
        logging.info(f"  Output dir: {output_dir}")

        # Prepare output directory (copy cam_K.txt, extract frames if needed)
        prepare_output_dir(output_dir, args.input_dir, video_path=args.video)

        # Run tracking
        try:
            pose_list = run_foundationpose_tracking(
                mesh_file=args.mesh,
                input_dir=args.input_dir,
                output_dir=output_dir,
                track_refine_iter=args.track_refine_iter,
                is_static=args.is_static,
                debug=args.debug,
            )
        except Exception as e:
            logging.error(f"Tracking failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Cleanup GPU memory
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Phase 2: Verify against golden reference
    # ------------------------------------------------------------------
    output_pkl = os.path.join(
        output_dir, "pose_estimation_output", "poses_in_cam.pkl"
    )
    reference_pkl = args.reference_pkl

    if not os.path.exists(reference_pkl):
        logging.warning(f"Reference not found: {reference_pkl}. Skipping verification.")
        logging.info("Run with --compare_only when reference is available.")
        sys.exit(0)

    if not os.path.exists(output_pkl):
        logging.error(f"Output poses not found: {output_pkl}")
        logging.error("Tracking may have failed, or use --output_dir to specify correct path.")
        sys.exit(1)

    logging.info("")
    logging.info("=" * 60)
    logging.info("  Phase 2: Verifying against golden reference")
    logging.info("=" * 60)

    all_pass = verify_poses_pkl(
        reference_pkl=reference_pkl,
        output_pkl=output_pkl,
        trans_atol=args.pose_trans_atol,
        rot_atol_deg=args.pose_rot_atol,
        rtol=args.rtol,
        atol=args.atol,
    )

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
