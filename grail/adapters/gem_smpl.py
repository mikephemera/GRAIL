"""
GEM-SMPL human pose estimation adapter (grail branch).

Provides `infer_human_pose()` using the hmr4d package from the GEM-SMPL
grail branch (GENMO architecture with SMPL-X body model).

Reuses `demo_slam.py` from GEM-SMPL for preprocessing and data loading.
"""

import gc
import os
import sys
import time
import types
from pathlib import Path

import torch

from grail.core.device import empty_cache, require_musa


_GRAIL_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GENMO_ROOT = _GRAIL_ROOT.parent / "GENMO_musa"


def _setup_imports(genmo_root):
    """Add GEM-SMPL demo to sys.path and mock problematic modules."""
    gem_smpl_root = os.path.abspath(genmo_root)
    demo_dir = os.path.join(gem_smpl_root, "tools", "demo")
    if not os.path.isfile(os.path.join(gem_smpl_root, "hmr4d", "__init__.py")):
        raise FileNotFoundError(f"GENMO source tree not found: {gem_smpl_root}")
    if not os.path.isfile(os.path.join(demo_dir, "demo_slam.py")):
        raise FileNotFoundError(f"GENMO demo entrypoint not found: {demo_dir}/demo_slam.py")

    loaded_hmr4d = sys.modules.get("hmr4d")
    loaded_path = getattr(loaded_hmr4d, "__file__", None)
    if loaded_path and not os.path.abspath(loaded_path).startswith(gem_smpl_root + os.sep):
        raise RuntimeError(
            f"hmr4d is already imported from {loaded_path}, but Step 1 requested {gem_smpl_root}. "
            "Run Step 1 in a fresh Python process."
        )
    if gem_smpl_root not in sys.path:
        sys.path.insert(0, gem_smpl_root)
    if demo_dir not in sys.path:
        sys.path.insert(0, demo_dir)

    # Mock modules that demo_slam.py imports at module level but we don't need
    # (visualization, rendering, remote sync utilities)
    if "motiondiff.utils.vis_scenepic" not in sys.modules:
        mock = types.ModuleType("motiondiff.utils.vis_scenepic")
        mock.ScenepicVisualizer = lambda *a, **kw: None
        sys.modules["motiondiff.utils.vis_scenepic"] = mock

    if "motiondiff.utils.tools" not in sys.modules:

        class _CatchAllMock(types.ModuleType):
            """Returns a no-op for any attribute access (Timer, wandb_run_exists, etc.)."""

            def __getattr__(self, name):
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)
                return lambda *a, **kw: None

        sys.modules["motiondiff.utils.tools"] = _CatchAllMock("motiondiff.utils.tools")

    if "hmr4d.utils.vis.o3d_render" not in sys.modules:
        mock = types.ModuleType("hmr4d.utils.vis.o3d_render")
        mock.Settings = type("Settings", (), {"Transparency": 0, "LIT": 1})
        mock.create_meshes = lambda *a, **kw: None
        mock.get_ground = lambda *a, **kw: None
        sys.modules["hmr4d.utils.vis.o3d_render"] = mock


def _reject_lfs_pointer(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open("rb") as handle:
        if handle.read(40).startswith(b"version https://git-lfs.github.com/spec"):
            raise RuntimeError(f"{label} is a Git LFS pointer, not materialized data: {path}")


def infer_human_pose(
    video_path,
    cache_dir,
    is_static_cam=False,
    verbose=False,
    device="auto",
    genmo_root=None,
    asset_root=None,
    checkpoint_path=None,
):
    """
    Run GEM-SMPL (hmr4d) human pose estimation on a video.

    Returns a dict compatible with run_human_pose_est.py:
        {
            "smpl_params_global": {body_pose, global_orient, transl, betas},
            "smpl_params_incam":  {body_pose, global_orient, transl, betas},
            "vitpose": (L, 17, 3),
            "foot_contact_probs": (L, 4) or None,
        }
    """
    device = require_musa(device, "GENMO")
    genmo_root = Path(genmo_root or _DEFAULT_GENMO_ROOT).expanduser().resolve()
    asset_root = Path(asset_root or genmo_root).expanduser().resolve()
    if not (asset_root / "inputs" / "checkpoints").is_dir():
        raise FileNotFoundError(f"GENMO checkpoint bundle not found under asset root: {asset_root}")
    os.environ["HMR4D_PROJECT_ROOT"] = str(asset_root)
    # Install the GENMO source path before checking the cache.  The cached
    # inference fast path must still leave the SMPL-X height post-processing
    # able to import hmr4d body-model utilities.
    _setup_imports(genmo_root)
    output_dir = os.path.abspath(cache_dir)
    output_root_abs = os.path.dirname(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Check for cached results
    cached_results = os.path.join(output_dir, "gem_smpl_pred.pt")
    if os.path.exists(cached_results):
        _reject_lfs_pointer(cached_results, "GEM-SMPL cache")
        print(f"[GEM-SMPL] Loading cached results from {cached_results}")
        return torch.load(cached_results, map_location="cpu", weights_only=False)

    t0 = time.time()
    video_path_abs = os.path.abspath(video_path)

    # Setup imports and run directly
    import cv2
    import hmr4d.model.genmo.genmo_demo  # noqa: F401 — registers genmo_demo model config
    import hmr4d.model.gvhmr.utils.endecoder  # noqa: F401 — registers inference endecoder config
    import hydra
    from hmr4d.model.genmo.inference import load_data_dict, run_preprocess
    from hmr4d.utils.net_utils import detach_to_cpu
    from hmr4d.utils.pylogger import Log
    from hmr4d.utils.video_io_utils import get_video_lwh
    from hydra import compose, initialize_config_module

    # Build Hydra config (replicates parse_args_to_cfg without argparse)
    video_path_obj = Path(video_path_abs)
    assert video_path_obj.exists(), f"Video not found at {video_path_obj}"
    length, width, height = get_video_lwh(video_path_obj)
    orig_fps = cv2.VideoCapture(str(video_path_obj)).get(cv2.CAP_PROP_FPS)
    Log.info(f"[GEM-SMPL] Input: {video_path_obj}, (L, W, H) = ({length}, {width}, {height})")

    overrides = [
        f"video_name={video_path_obj.stem}",
        f"static_cam={is_static_cam}",
        f"verbose={verbose}",
        f"output_root={output_root_abs}",
    ]
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        cfg = compose(config_name="demo_genmo", overrides=overrides)

    # Override ckpt_path to absolute path so it doesn't depend on CWD
    gem_smpl_root = str(genmo_root)
    if checkpoint_path:
        cfg.ckpt_path = str(Path(checkpoint_path).expanduser().resolve())
    else:
        cfg.ckpt_path = os.path.join(str(asset_root), cfg.ckpt_path)
    _reject_lfs_pointer(cfg.ckpt_path, "GENMO checkpoint")

    paths = cfg.paths
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy input video to expected location (demo convention)
    from hmr4d.utils.video_io_utils import get_video_reader, get_writer

    if (
        not Path(cfg.video_path).exists()
        or get_video_lwh(video_path_obj)[0] != get_video_lwh(cfg.video_path)[0]
    ):
        from tqdm import tqdm

        reader = get_video_reader(video_path_obj)
        writer = get_writer(cfg.video_path, fps=30, crf=23)
        for img in tqdm(reader, total=length, desc="[GEM-SMPL] Copy video"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    # GEM-SMPL contains historical relative checkpoint paths. Run with the
    # materialized asset bundle as CWD while importing code from GENMO_musa.
    prev_cwd = os.getcwd()
    os.chdir(asset_root)
    try:
        # Preprocess (bbx tracking, vitpose, vit features, VIMO, optionally DROID-SLAM)
        run_preprocess(cfg, orig_fps, device=str(device))

        # Load preprocessed data
        data = load_data_dict(cfg)

        # Run HMR4D inference
        if not Path(paths.hmr4d_results).exists():
            Log.info("[GEM-SMPL] Running HMR4D prediction")
            model = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(cfg.ckpt_path)
            model = model.eval().to(device)
            pred = model.predict(data, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
            torch.save(pred, paths.hmr4d_results)
            Log.info(f"[GEM-SMPL] Saved HMR4D results to {paths.hmr4d_results}")

            # Free GPU memory: the HMR4D model (~7.6 GiB on 10 GiB cards) must
            # be released before downstream steps (SAM2, FoundationPose) run.
            del model
            gc.collect()
            empty_cache(device)
        else:
            Log.info(f"[GEM-SMPL] Loading cached HMR4D results from {paths.hmr4d_results}")
            _reject_lfs_pointer(paths.hmr4d_results, "HMR4D cache")
            pred = torch.load(paths.hmr4d_results, map_location="cpu", weights_only=False)
    finally:
        os.chdir(prev_cwd)

    # Load vitpose
    vitpose = None
    if os.path.exists(paths.vitpose):
        _reject_lfs_pointer(paths.vitpose, "ViTPose cache")
        vitpose = torch.load(paths.vitpose, map_location="cpu", weights_only=False)
        if isinstance(vitpose, tuple):
            vitpose = vitpose[0]

    # Remap to expected output format
    result = {
        "smpl_params_global": pred.get("smpl_params_global", {}),
        "smpl_params_incam": pred.get("smpl_params_incam", {}),
        "vitpose": pred.get("vitpose", vitpose),
        "foot_contact_probs": pred.get("foot_contact_probs", None),
    }

    # Extract foot contact probs if not already in pred
    if result["foot_contact_probs"] is None:
        if "net_outputs" in pred and "model_output" in pred.get("net_outputs", {}):
            model_output = pred["net_outputs"]["model_output"]
            if "static_conf_logits" in model_output:
                static_conf_logits = model_output["static_conf_logits"]
                result["foot_contact_probs"] = torch.sigmoid(static_conf_logits[:, :, :4])

    # Ensure CPU tensors
    for key in ["smpl_params_global", "smpl_params_incam"]:
        if isinstance(result[key], dict):
            result[key] = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in result[key].items()
            }

    # Cache results
    torch.save(result, cached_results)
    print(f"[GEM-SMPL] Results cached to {cached_results}")
    print(f"[GEM-SMPL] Total time: {time.time() - t0:.1f}s")

    return result
