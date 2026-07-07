#!/usr/bin/env python3
"""
Standalone WiLoR verification script for Step 1 of recon_4dhoi.

Reuses existing GEM-SMPL cache (gem_smpl_pred.pt) and compares hand pose
outputs against a reference .npz from a prior CUDA run.

Two modes:
  (a) Run WiLoR from scratch and compare against reference — verifies that
      the current WiLoR installation produces identical hand poses.
  (b) Accept external hand poses (--external_mano_preds) — validates a
      replacement WiLoR implementation against the reference.

Usage:
    # Mode A: compare current WiLoR against reference
    python scripts/verify_step1_wilor.py \
        --video_path results/generation/videos_kling/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001.mp4 \
        --cache_dir results/generation/hmr_smplx_cache/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001 \
        --reference_npz results/generation/hmr_smplx/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001.npz

    # Mode B: validate external hand poses
    python scripts/verify_step1_wilor.py \
        --video_path ... \
        --cache_dir ... \
        --reference_npz ... \
        --external_mano_preds my_hand_poses.pkl
"""

import argparse
import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helper: locate grail package
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def load_gem_smpl_cache(cache_dir: str) -> dict:
    """Load GEM-SMPL body-only predictions from cache.

    Returns dict with keys:
        smpl_params_global, smpl_params_incam, vitpose, foot_contact_probs
    """
    cache_path = os.path.join(cache_dir, "gem_smpl_pred.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"GEM-SMPL cache not found: {cache_path}")
    return torch.load(cache_path, map_location="cpu")


def load_reference_npz(npz_path: str) -> dict:
    """Load reference .npz produced by a prior full Step 1 run.

    Returns dict with keys: motion_global, motion_incam
    """
    data = np.load(npz_path, allow_pickle=True)
    return {"motion_global": data["motion_global"].item(), "motion_incam": data["motion_incam"].item()}


def load_external_mano_preds(path: str) -> List[List[dict]]:
    """Load externally produced WiLoR mano predictions.

    Expected format: List[List[dict]] where outer list = frames,
    inner list = detected hands per frame (0-2 hands).
    Each dict should have keys: is_right, bbox_conf, wilor_preds
    """
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Core: reconstruct motion from GEM-SMPL cache + WiLoR predictions
# ---------------------------------------------------------------------------


def reconstruct_motion_from_cache(
    gem_smpl_cache: dict,
    mano_preds: List[List[dict]],
    smplx_model_path: str,
) -> dict:
    """Reconstruct full motion_global/motion_incam from GEM-SMPL cache + WiLoR.

    This replicates the logic in run_human_pose_est_smplx() (human_pose.py),
    but skips the GEM-SMPL inference step since we load it from cache.

    Args:
        gem_smpl_cache: dict from load_gem_smpl_cache()
        mano_preds: per-frame WiLoR hand predictions
        smplx_model_path: path to SMPLX_NEUTRAL.npz

    Returns:
        dict with keys: motion_global, motion_incam
    """
    from grail.adapters.wilor import infer_hand_pose
    from grail.core.torch_utils import tensor_to
    from grail.pose_est.human_pose import (
        fuse_smplx_mano_predictions,
        interp_smplx_mano_predictions,
    )

    smplx_global = gem_smpl_cache["smpl_params_global"]
    smplx_incam = gem_smpl_cache["smpl_params_incam"]

    # Load relaxed hand poses as fallback
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "grail", "constants", "smplx_handposes.npz"
    )
    with np.load(data_path, allow_pickle=True) as data:
        hand_poses = data["hand_poses"].item()
        left_hand_pose_relaxed, right_hand_pose_relaxed = hand_poses["relaxed"]

    frame_num = smplx_global["body_pose"].shape[0]
    if frame_num != len(mano_preds):
        raise ValueError(
            f"Frame number mismatch: GEM-SMPL={frame_num}, WiLoR={len(mano_preds)}"
        )

    # Initialize fused containers
    fused_global = {
        "body_pose": smplx_global["body_pose"].clone(),
        "global_orient": smplx_global["global_orient"].clone(),
        "transl": smplx_global["transl"].clone(),
        "betas": smplx_global["betas"].clone(),
        "left_hand_pose": torch.zeros((frame_num, 15, 3)),
        "right_hand_pose": torch.zeros((frame_num, 15, 3)),
    }
    fused_incam = {
        "body_pose": smplx_incam["body_pose"].clone(),
        "global_orient": smplx_incam["global_orient"].clone(),
        "transl": smplx_incam["transl"].clone(),
        "betas": smplx_incam["betas"].clone(),
        "left_hand_pose": torch.zeros((frame_num, 15, 3)),
        "right_hand_pose": torch.zeros((frame_num, 15, 3)),
    }

    hand_keypoints_2d = torch.zeros((frame_num, 32, 3))

    # Per-frame fusion
    for i in range(frame_num):
        frame_global = {k: v[i : i + 1] for k, v in fused_global.items()}
        frame_incam = {k: v[i : i + 1] for k, v in fused_incam.items()}

        frame_mano_preds = mano_preds[i]
        frame_vitpose = (
            gem_smpl_cache["vitpose"][i]
            if gem_smpl_cache.get("vitpose") is not None
            else None
        )

        # Fuse WiLoR hand poses into SMPL-X body
        fused_frame_incam, hand_kp_frame = fuse_smplx_mano_predictions(
            frame_incam,
            frame_mano_preds,
            left_hand_pose_relaxed,
            right_hand_pose_relaxed,
            body_keypoints_2d=frame_vitpose,
        )

        # Write back fused frame data
        for key in ("body_pose", "left_hand_pose", "right_hand_pose"):
            fused_incam[key][i : i + 1] = fused_frame_incam[key]
            fused_global[key][i : i + 1] = fused_frame_incam[key]

        hand_keypoints_2d[i : i + 1] = hand_kp_frame

    # Interpolate missing hand frames
    fused_global = interp_smplx_mano_predictions(fused_global)
    fused_incam = interp_smplx_mano_predictions(fused_incam)

    # ── Build motion_incam dict (the one Step 4 consumes) ──
    body_pose = fused_incam["body_pose"].reshape(-1, 63)
    betas = fused_incam["betas"].reshape(-1, 10)
    global_orient = fused_incam["global_orient"].reshape(-1, 3)
    trans = fused_incam["transl"].reshape(-1, 3)
    left_hand_pose = fused_incam["left_hand_pose"].reshape(-1, 45)
    right_hand_pose = fused_incam["right_hand_pose"].reshape(-1, 45)

    incam_poses = torch.zeros((frame_num, 165))
    incam_poses[:, :3] = global_orient
    incam_poses[:, 3 : 3 + 63] = body_pose
    incam_poses[:, 3 + 63 : 3 + 63 + 45] = left_hand_pose
    incam_poses[:, 3 + 63 + 45 : 3 + 63 + 45 + 45] = right_hand_pose

    # Handle foot_contact_probs shape: (1, L, 4) → (L, 4)
    foot_contact_probs = gem_smpl_cache.get("foot_contact_probs", None)
    if foot_contact_probs is not None and foot_contact_probs.dim() == 3:
        foot_contact_probs = foot_contact_probs.squeeze(0)

    motion_incam = dict(
        poses=incam_poses,
        betas=betas[0, :10],
        trans=trans,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        mocap_frame_rate=30,
        gender="neutral",
        vitpose=gem_smpl_cache["vitpose"],
        hand_keypoints_2d=hand_keypoints_2d,
        foot_contact_probs=foot_contact_probs,
    )

    # ── Build motion_global dict ──
    body_pose_g = fused_global["body_pose"].reshape(-1, 63)
    betas_g = fused_global["betas"].reshape(-1, 10)
    global_orient_g = fused_global["global_orient"].reshape(-1, 3)
    trans_g = fused_global["transl"].reshape(-1, 3)
    left_hand_pose_g = fused_global["left_hand_pose"].reshape(-1, 45)
    right_hand_pose_g = fused_global["right_hand_pose"].reshape(-1, 45)

    global_poses = torch.zeros((frame_num, 165))
    global_poses[:, :3] = global_orient_g
    global_poses[:, 3 : 3 + 63] = body_pose_g
    global_poses[:, 3 + 63 : 3 + 63 + 45] = left_hand_pose_g
    global_poses[:, 3 + 63 + 45 : 3 + 63 + 45 + 45] = right_hand_pose_g

    motion_global = dict(
        poses=global_poses,
        betas=betas_g[0, :10],
        trans=trans_g,
        left_hand_pose=left_hand_pose_g,
        right_hand_pose=right_hand_pose_g,
        mocap_frame_rate=30,
        gender="neutral",
    )

    # ── Compute predicted_body_height (hard dependency for Step 4) ──
    from grail.models.smplx_model import get_tpose_human_height, setup_smplx_model

    smplx_model_dir = os.path.dirname(os.path.dirname(smplx_model_path))
    smplx_model = setup_smplx_model(
        model_path=smplx_model_dir, flat_hand_mean=True, device="cpu"
    )
    predicted_body_height = float(
        get_tpose_human_height(smplx_model, betas=betas[0, :10].cpu(), device="cpu")
    )

    motion_incam["model"] = "smplx"
    motion_incam["predicted_body_height"] = predicted_body_height
    motion_global["model"] = "smplx"
    motion_global["predicted_body_height"] = predicted_body_height

    return {"motion_global": motion_global, "motion_incam": motion_incam}


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------


def _to_numpy(v):
    """Convert tensor or ndarray to numpy for comparison."""
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return v
    return np.array(v)


def compare_motion_dicts(
    ref: dict,
    new: dict,
    label: str = "motion",
    rtol: float = 1e-3,
    atol: float = 1e-5,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Compare two motion dicts field by field.

    Returns:
        dict: {field_name: {"max_abs_diff": float, "mean_abs_diff": float,
                            "match": bool, "shape": tuple}}
    """
    results = {}
    all_keys = sorted(set(ref.keys()) | set(new.keys()))

    for key in all_keys:
        in_ref = key in ref
        in_new = key in new

        if not in_ref:
            results[key] = {"match": False, "error": "missing from reference"}
            continue
        if not in_new:
            results[key] = {"match": False, "error": "missing from new"}
            continue

        ref_val = ref[key]
        new_val = new[key]

        # Handle scalar values (str, int, float)
        if isinstance(ref_val, (str, int, float, type(None))):
            if ref_val == new_val:
                results[key] = {"match": True, "value": ref_val}
            else:
                results[key] = {
                    "match": False,
                    "error": f"value mismatch: {ref_val} vs {new_val}",
                }
            continue

        # Handle tensor/ndarray
        try:
            ref_arr = _to_numpy(ref_val)
            new_arr = _to_numpy(new_val)
        except Exception as e:
            results[key] = {"match": False, "error": f"conversion error: {e}"}
            continue

        if ref_arr.shape != new_arr.shape:
            results[key] = {
                "match": False,
                "error": f"shape mismatch: {ref_arr.shape} vs {new_arr.shape}",
            }
            continue

        abs_diff = np.abs(ref_arr - new_arr)
        max_abs = float(np.max(abs_diff))
        mean_abs = float(np.mean(abs_diff))
        is_close = bool(np.allclose(ref_arr, new_arr, rtol=rtol, atol=atol))

        results[key] = {
            "match": is_close,
            "shape": ref_arr.shape,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
        }

    return results


def print_comparison_report(results: dict, label: str, rtol: float, atol: float):
    """Pretty-print comparison results."""
    print(f"\n{'=' * 70}")
    print(f"  Comparison Report: {label}")
    print(f"  rtol={rtol}, atol={atol}")
    print(f"{'=' * 70}")

    all_match = True
    for key, info in sorted(results.items()):
        if info.get("match"):
            if "value" in info:
                print(f"  ✓ {key}: {info['value']}")
            else:
                print(
                    f"  ✓ {key}: shape={info.get('shape', '?')}, "
                    f"max_diff={info.get('max_abs_diff', 0):.2e}, "
                    f"mean_diff={info.get('mean_abs_diff', 0):.2e}"
                )
        else:
            all_match = False
            error = info.get("error", f"max_diff={info.get('max_abs_diff', '?'):.2e}")
            print(f"  ✗ {key}: {error}")

    status = "ALL MATCH" if all_match else "MISMATCHES FOUND"
    print(f"\n  → {status}")
    return all_match


# ---------------------------------------------------------------------------
# Save utility
# ---------------------------------------------------------------------------


def save_motion_npz(motion: dict, output_path: str):
    """Save motion_global + motion_incam to .npz (compatible with Step 4)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        motion_global=motion["motion_global"],
        motion_incam=motion["motion_incam"],
    )
    print(f"Saved motion to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Verify WiLoR hand pose against reference Step 1 output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input .mp4 video"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Path to GEM-SMPL cache directory (contains gem_smpl_pred.pt)",
    )
    parser.add_argument(
        "--reference_npz",
        type=str,
        required=True,
        help="Path to reference .npz from prior CUDA Step 1 run",
    )
    parser.add_argument(
        "--external_mano_preds",
        type=str,
        default=None,
        help="Path to external WiLoR mano_preds .pkl (if not provided, runs WiLoR from scratch)",
    )
    parser.add_argument(
        "--smplx_model_path",
        type=str,
        default="imports/GEM-SMPL/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
        help="Path to SMPLX_NEUTRAL.npz",
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        default=None,
        help="If set, save reconstructed motion to this .npz path",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-3, help="Relative tolerance for float comparison"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-5, help="Absolute tolerance for float comparison"
    )
    parser.add_argument(
        "--save_external_mano_preds",
        type=str,
        default=None,
        help="If set, save current WiLoR mano_preds to this .pkl (for later use as --external_mano_preds)",
    )
    args = parser.parse_args()

    # Resolve project-relative paths
    smplx_path = args.smplx_model_path
    if not os.path.isabs(smplx_path):
        smplx_path = os.path.join(_PROJECT_ROOT, smplx_path)

    # ── 1. Load GEM-SMPL cache ──
    print(f"[1/4] Loading GEM-SMPL cache from: {args.cache_dir}")
    gem_smpl_cache = load_gem_smpl_cache(args.cache_dir)
    print(f"      Frames: {gem_smpl_cache['smpl_params_incam']['body_pose'].shape[0]}")

    # ── 2. Get WiLoR predictions ──
    if args.external_mano_preds:
        print(f"[2/4] Loading external WiLoR predictions from: {args.external_mano_preds}")
        mano_preds = load_external_mano_preds(args.external_mano_preds)
    else:
        print(f"[2/4] Running WiLoR on: {args.video_path}")
        from grail.adapters.wilor import infer_hand_pose

        mano_preds = infer_hand_pose(args.video_path)

        if args.save_external_mano_preds:
            with open(args.save_external_mano_preds, "wb") as f:
                pickle.dump(mano_preds, f)
            print(f"      Saved mano_preds to: {args.save_external_mano_preds}")

    print(f"      Frames with hand detections: {sum(1 for m in mano_preds if m)}")

    # ── 3. Reconstruct full motion ──
    print(f"[3/4] Reconstructing motion (fuse + interpolate + body height)...")
    motion = reconstruct_motion_from_cache(
        gem_smpl_cache=gem_smpl_cache,
        mano_preds=mano_preds,
        smplx_model_path=smplx_path,
    )

    # ── 4. Compare against reference ──
    print(f"[4/4] Comparing against reference: {args.reference_npz}")
    reference = load_reference_npz(args.reference_npz)

    all_match = True
    all_match &= print_comparison_report(
        compare_motion_dicts(
            reference["motion_incam"],
            motion["motion_incam"],
            label="motion_incam",
            rtol=args.rtol,
            atol=args.atol,
        ),
        label="motion_incam",
        rtol=args.rtol,
        atol=args.atol,
    )
    all_match &= print_comparison_report(
        compare_motion_dicts(
            reference["motion_global"],
            motion["motion_global"],
            label="motion_global",
            rtol=args.rtol,
            atol=args.atol,
        ),
        label="motion_global",
        rtol=args.rtol,
        atol=args.atol,
    )

    # ── Optional: save output ──
    if args.output_npz:
        save_motion_npz(motion, args.output_npz)

    # ── Summary ──
    print(f"\n{'=' * 70}")
    if all_match:
        print("  ✓ VERIFIED: Hand pose output matches reference.")
        print("    The WiLoR replacement produces identical results.")
    else:
        print("  ✗ DIFFERENCES DETECTED: See details above.")
        print("    Review per-field max_abs_diff to assess impact on downstream steps.")
    print(f"{'=' * 70}")

    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
