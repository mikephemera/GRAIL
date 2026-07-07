#!/usr/bin/env python3
"""
Generic module output comparison framework for recon_4dhoi pipeline.

Supports comparing intermediate outputs from different pipeline runs or
different implementations (e.g., CUDA vs. non-CUDA replacements).

Supported formats:
  - .npz    — numpy compressed archive         (Step 1 hmr, Step 2 masks)
  - .pkl    — pickle-serialized Python objects  (Step 3 poses, Step 4/5 hoi_data)
  - .pt     — PyTorch saved tensors/dicts       (Step 2 depth)
  - .npy    — numpy array file                  (various)

Usage:
    # Compare two .npz files
    python scripts/verify_module_output.py \
        --ref results/generation/hmr_smplx/ComAsset/cordless_drill/kid_indoor2-manipulation_rand00001.npz \
        --new output/new_hmr/kid_indoor2-manipulation_rand00001.npz

    # Compare two .pkl files with custom tolerances
    python scripts/verify_module_output.py \
        --ref results/generation/4dhoi_recon_smplx/.../hoi_data.pkl \
        --new output/new_recon/hoi_data.pkl \
        --rtol 1e-2 --atol 1e-4

    # Compare two directories of module outputs
    python scripts/verify_module_output.py \
        --ref results/generation/4dhoi_recon_cache/masks/ \
        --new output/new_masks/ \
        --glob "*.npz"
"""

import argparse
import os
import pickle
import sys
from glob import glob
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Loaders for different file formats
# ---------------------------------------------------------------------------


def load_file(path: str) -> Any:
    """Auto-detect format and load a file."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return dict(np.load(path, allow_pickle=True))
    elif ext == ".npy":
        return np.load(path, allow_pickle=True)
    elif ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    elif ext == ".pt":
        import torch

        return torch.load(path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------


def _to_comparable(obj: Any) -> Any:
    """Convert an object to a numpy-comparable form.

    - 0-d numpy object arrays → .item() (e.g., dicts stored in .npz)
    - torch.Tensor → numpy.ndarray
    - numpy arrays with dtype=object → list of items
    - Scalars → as-is
    """
    # Unwrap 0-d numpy object arrays (common in .npz files)
    if isinstance(obj, np.ndarray) and obj.ndim == 0 and obj.dtype == np.dtype("O"):
        return _to_comparable(obj.item())

    # Unwrap higher-dim object arrays
    if isinstance(obj, np.ndarray) and obj.dtype == np.dtype("O"):
        return [_to_comparable(v) for v in obj.flat]

    try:
        import torch

        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
    except ImportError:
        pass

    return obj


def _flatten_keys(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dicts into dotted key paths, converting tensors to numpy."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(_flatten_keys(v, key))
            elif isinstance(v, (list, tuple)):
                # For lists, keep as-is but flatten if list of dicts
                result[key] = _to_comparable(v)
            else:
                result[key] = _to_comparable(v)
        return result
    return {prefix: _to_comparable(obj)}


def compare_values(
    ref: Any,
    new: Any,
    key: str = "",
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Compare two values, returning a structured diff report.

    Args:
        ref: Reference value (ground truth from CUDA run)
        new: New value (from replacement module)
        key: Key path for reporting
        rtol: Relative tolerance for np.allclose
        atol: Absolute tolerance for np.allclose

    Returns:
        dict with keys: match (bool), error (str|None), details (dict)
    """
    result: Dict[str, Any] = {"match": True, "error": None, "details": {}}

    # Handle None
    if ref is None and new is None:
        return result
    if ref is None or new is None:
        result["match"] = False
        result["error"] = f"One is None: ref={ref is None}, new={new is None}"
        return result

    # Handle scalars (str, int, float, bool)
    if isinstance(ref, (str, int, float, bool)):
        if ref != new:
            result["match"] = False
            result["error"] = f"Value mismatch: {ref} != {new}"
        return result

    # Convert to numpy for tensor comparison
    ref_cmp = _to_comparable(ref)
    new_cmp = _to_comparable(new)

    # Handle numpy arrays
    if isinstance(ref_cmp, np.ndarray) and isinstance(new_cmp, np.ndarray):
        if ref_cmp.shape != new_cmp.shape:
            result["match"] = False
            result["error"] = f"Shape mismatch: {ref_cmp.shape} vs {new_cmp.shape}"
            return result

        if ref_cmp.dtype != new_cmp.dtype:
            result["details"]["dtype_warning"] = (
                f"dtype differs: {ref_cmp.dtype} vs {new_cmp.dtype}"
            )
            # Promote to common dtype for comparison
            try:
                ref_cmp = ref_cmp.astype(np.float64)
                new_cmp = new_cmp.astype(np.float64)
            except (ValueError, TypeError):
                pass

        # For boolean/uint arrays, use exact match
        if ref_cmp.dtype == np.dtype(bool) or ref_cmp.dtype.kind in ("u", "i"):
            is_close = bool(np.array_equal(ref_cmp, new_cmp))
        else:
            is_close = bool(np.allclose(ref_cmp, new_cmp, rtol=rtol, atol=atol))

        if not is_close:
            abs_diff = np.abs(ref_cmp.astype(np.float64) - new_cmp.astype(np.float64))
            result["match"] = False
            result["details"].update(
                {
                    "shape": ref_cmp.shape,
                    "max_abs_diff": float(np.max(abs_diff)),
                    "mean_abs_diff": float(np.mean(abs_diff)),
                    "pct_beyond_atol": float(
                        np.mean(abs_diff > atol) * 100
                    ),
                }
            )
        else:
            result["details"]["shape"] = ref_cmp.shape

        return result

    # Handle dicts
    if isinstance(ref, dict) and isinstance(new, dict):
        flat_ref = _flatten_keys(ref)
        flat_new = _flatten_keys(new)

        all_keys = sorted(set(flat_ref.keys()) | set(flat_new.keys()), key=str)
        for k in all_keys:
            if k not in flat_ref:
                result["details"][k] = {"match": False, "error": "missing from reference"}
                result["match"] = False
            elif k not in flat_new:
                result["details"][k] = {"match": False, "error": "missing from new"}
                result["match"] = False
            else:
                sub = compare_values(flat_ref[k], flat_new[k], key=k, rtol=rtol, atol=atol)
                if not sub["match"]:
                    result["match"] = False
                    result["details"][k] = sub

        return result

    # Handle lists
    if isinstance(ref, (list, tuple)) and isinstance(new, (list, tuple)):
        if len(ref) != len(new):
            result["match"] = False
            result["error"] = f"Length mismatch: {len(ref)} vs {len(new)}"
            return result

        list_details = {}
        for i, (r_item, n_item) in enumerate(zip(ref, new)):
            sub = compare_values(r_item, n_item, key=f"[{i}]", rtol=rtol, atol=atol)
            if not sub["match"]:
                result["match"] = False
                list_details[f"[{i}]"] = sub

        if list_details:
            result["details"]["list_items"] = list_details
        result["details"]["length"] = len(ref)
        return result

    # Fallback: type mismatch
    if type(ref) != type(new):
        result["match"] = False
        result["error"] = f"Type mismatch: {type(ref).__name__} vs {type(new).__name__}"
        return result

    # Last resort: direct equality
    if ref != new:
        result["match"] = False
        result["error"] = f"Direct comparison failed for {type(ref).__name__}"
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(
    result: Dict[str, Any],
    key_path: str = "",
    indent: int = 0,
    max_depth: int = 4,
    show_matching: bool = False,
):
    """Pretty-print a comparison result tree."""
    prefix = "  " * indent

    if result.get("match") and not show_matching:
        return

    if "error" in result and result["error"]:
        print(f"{prefix}✗ {key_path}: {result['error']}")
        return

    # If this result itself has numeric comparison data (leaf-level mismatch)
    details = result.get("details", {})
    if "max_abs_diff" in details:
        shape = details.get("shape", "")
        max_diff = details.get("max_abs_diff", 0)
        mean_diff = details.get("mean_abs_diff", 0)
        pct = details.get("pct_beyond_atol", 0)
        print(
            f"{prefix}✗ {key_path}: shape={shape}, "
            f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
            f"pct_beyond={pct:.1f}%"
        )
        return

    if result.get("match"):
        shape = details.get("shape", "")
        print(f"{prefix}✓ {key_path}: shape={shape}")
        return

    # Has sub-details — recurse
    for k, v in sorted(details.items()):
        if k == "list_items":
            if indent >= max_depth:
                print(f"{prefix}  [... {len(v)} list items with differences ...]")
                continue
            for lk, lv in sorted(v.items()):
                full_key = f"{key_path}{lk}" if key_path else lk
                print_report(lv, key_path=full_key, indent=indent + 1, max_depth=max_depth)
        elif isinstance(v, dict) and ("match" in v or "error" in v):
            full_key = f"{key_path}.{k}" if key_path else k
            print_report(v, key_path=full_key, indent=indent, max_depth=max_depth)
        elif isinstance(v, dict):
            # Direct numeric details inside a nested key
            shape = v.get("shape", "")
            max_diff = v.get("max_abs_diff", 0)
            mean_diff = v.get("mean_abs_diff", 0)
            pct = v.get("pct_beyond_atol", 0)
            full_key = f"{key_path}.{k}" if key_path else k
            print(
                f"{prefix}✗ {full_key}: shape={shape}, "
                f"max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, "
                f"pct_beyond={pct:.1f}%"
            )


def print_summary(
    result: Dict[str, Any], label: str, rtol: float, atol: float
) -> bool:
    """Print a comparison summary header and return overall match status."""
    print(f"\n{'=' * 70}")
    print(f"  Comparison: {label}")
    print(f"  tolerances: rtol={rtol}, atol={atol}")
    print(f"{'=' * 70}")

    if result.get("match"):
        print("  ✓ ALL FIELDS MATCH")
        return True
    else:
        print_report(result)
        print(f"\n  ✗ DIFFERENCES DETECTED")
        return False


# ---------------------------------------------------------------------------
# Multi-file comparison
# ---------------------------------------------------------------------------


def compare_files(
    ref_path: str,
    new_path: str,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Dict[str, Any]:
    """Compare two files of the same format."""
    ref_data = load_file(ref_path)
    new_data = load_file(new_path)

    label = os.path.basename(ref_path)
    result = compare_values(ref_data, new_data, rtol=rtol, atol=atol)
    return result


def compare_directories(
    ref_dir: str,
    new_dir: str,
    pattern: str = "*",
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Dict[str, Dict[str, Any]]:
    """Compare all matching files in two directories.

    Returns:
        dict: {filename: comparison_result}
    """
    ref_files = sorted(glob(os.path.join(ref_dir, pattern)))
    results = {}

    for ref_path in ref_files:
        fname = os.path.basename(ref_path)
        new_path = os.path.join(new_dir, fname)

        if not os.path.exists(new_path):
            results[fname] = {"match": False, "error": f"Missing from new directory: {new_path}"}
            continue

        try:
            results[fname] = compare_files(ref_path, new_path, rtol=rtol, atol=atol)
        except Exception as e:
            results[fname] = {"match": False, "error": str(e)}

    # Check for files only in new dir
    new_files = set(os.path.basename(p) for p in glob(os.path.join(new_dir, pattern)))
    ref_files_set = set(os.path.basename(p) for p in ref_files)
    for fname in sorted(new_files - ref_files_set):
        results[fname] = {"match": False, "error": "Only in new directory (not in reference)"}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare pipeline module outputs between reference and new runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ref", type=str, required=True, help="Reference file or directory (CUDA run)"
    )
    parser.add_argument(
        "--new", type=str, required=True, help="New file or directory (replacement module)"
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*",
        help="Glob pattern when comparing directories (default: *)",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-3, help="Relative tolerance (default: 1e-3)"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-5, help="Absolute tolerance (default: 1e-5)"
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="If set, write comparison results as JSON to this path",
    )
    args = parser.parse_args()

    all_match = True

    if os.path.isdir(args.ref) and os.path.isdir(args.new):
        # Directory comparison
        print(f"Comparing directories:")
        print(f"  ref: {args.ref}")
        print(f"  new: {args.new}")
        print(f"  pattern: {args.glob}")

        results = compare_directories(
            args.ref, args.new, pattern=args.glob, rtol=args.rtol, atol=args.atol
        )

        for fname, result in sorted(results.items()):
            match = print_summary(result, label=fname, rtol=args.rtol, atol=args.atol)
            if not match:
                all_match = False

        # Directory-level summary
        total = len(results)
        matched = sum(1 for r in results.values() if r.get("match", False))
        print(f"\n  Directory summary: {matched}/{total} files match")

    else:
        # Single file comparison
        print(f"Comparing files:")
        print(f"  ref: {args.ref}")
        print(f"  new: {args.new}")

        result = compare_files(args.ref, args.new, rtol=args.rtol, atol=args.atol)
        label = os.path.basename(args.ref)
        all_match = print_summary(result, label=label, rtol=args.rtol, atol=args.atol)

    # Optional JSON export
    if args.json:
        import json

        def _make_serializable(obj):
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(args.json, "w") as f:
            json.dump(_make_serializable(result), f, indent=2, default=str)
        print(f"  JSON report saved to: {args.json}")

    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
