"""Typed configuration dataclasses for GRAIL pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _from_dict(cls, d: dict):
    """Create a dataclass instance from a dict, ignoring unknown keys."""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in d.items() if k in known})


# ============================================================================
# 4D HOI Reconstruction Config
# ============================================================================


@dataclass
class ReconPaths:
    """Directory paths for the reconstruction pipeline."""

    results_dir: str = "results"
    video_dir: str = "generation/videos_kling"
    hmr_dir: str = "generation/hmr"
    hmr_cache_dir: str = "generation/hmr4d_cache"
    foundation_pose_dir: str = "generation/foundation_pose"
    foundation_pose_output_dir: str = "generation/foundation_pose_output"
    recon_cache_dir: str = "generation/4dhoi_recon_cache"
    depth_gt_dir: str = "generation/depth_maps"
    output_dir: str = "generation/4dhoi_recon"
    vis_scenepic_dir: str = "generation/vis_html"


@dataclass
class HumanModelConfig:
    """Body model configuration."""

    body_model: str = "smplx"  # smplx | soma | g1_smplx
    static_cam: bool = True
    smplx_model_path: str = ""
    smplx_shape_params_path: str = ""
    soma_model_path: str = ""
    soma_shape_params_path: str = ""
    g1_smplx_params_path: str = ""
    g1_smplx_tpose_mesh_path: str = ""
    # MUSA Step 1/2 runtime sources.  They remain optional so older configs
    # keep their historical defaults while the new pipeline can be explicit.
    genmo_root: str = ""
    genmo_asset_root: str = ""
    genmo_checkpoint: str | None = None
    wilor_root: str = ""
    wilor_pretrained_dir: str | None = None
    sam2_model_id: str = "facebook/sam2-hiera-large"
    moge_root: str = ""
    moge_model_id: str = "Ruicheng/moge-2-vitl-normal"


@dataclass
class ObjPoseTrackingConfig:
    """FoundationPose object tracking settings."""

    foundation_pose_debug: int = 2
    crop_image: bool = False
    interpolation_factor: int = 1
    foundationpose_root: str = "/workspace/FoundationPose_musa"
    nvdiffrast_root: str = "/workspace/nvdiffrast_musa"
    pytorch3d_root: str = "/workspace/pytorch3d_musa"
    track_refine_iter: int = 2
    smooth_window: int = 9
    smooth_polyorder: int = 3
    require_mycpp: bool = False


@dataclass
class FilteringConfig:
    """Quality filtering thresholds for reconstruction results."""

    camera_trans_thr: float = 0.1
    object_mask_tol: float = 0.5
    total_mask_tol: float = 0.3
    human_static_thr: float = 0.01
    min_frames: int | None = None
    filter_object_motion: str = "all"  # all | static_only | dynamic_only
    object_static_thr: float = 0.02
    check_object_initially_on_table: bool = True


@dataclass
class PostProcessingConfig:
    """Post-processing settings for valid results."""

    center_at_object: bool = True
    compute_contact_points_nframes_before_inter_start: int = 48
    save_joints: bool = False


@dataclass
class VisualizationConfig:
    """Visualization settings for the optimizer."""

    vis_init: bool = True
    enable: bool = True
    render_video: bool = True
    extra_views: list[str] = field(default_factory=lambda: ["top"])
    export_mesh: bool = False
    vis_html: bool = True
    vis_contact: bool = True


@dataclass
class PipelineFlags:
    """Step skip flags and runtime toggles."""

    skip_step1: bool = False
    skip_step2: bool = False
    skip_step3: bool = False
    skip_step4: bool = False
    skip_step5: bool = False
    skip_step6: bool = False
    skip_done: bool = False
    verbose: bool = False
    vis_valid_only: bool = False


# ============================================================================
# 2D HOI Generation Config
# ============================================================================


@dataclass
class GenPaths:
    """Directory paths for the 2D generation pipeline."""

    prompts_dir: str = "generation/prompts"
    initial_state_dir: str = "generation/initial_states"
    obj_scale_dir: str = "generation/obj_scales"
    render_dir: str = "generation/asset_renders"
    camera_dir: str = "generation/cameras"
    foundation_pose_input_dir: str = "generation/foundation_pose"
    depth_save_dir: str = "generation/depth_maps"
    video_output_dir: str = "generation/videos_kling"


@dataclass
class SimulationConfig:
    """Physics simulation settings."""

    drop_height: float = 0.1
    settling_time: float = 5.0
    initial_rotation_perturbation: float = 5.0
    seed: int = 42
    save_usd: bool = False
    use_initial_state: bool = True


@dataclass
class RenderingConfig:
    """Blender rendering settings."""

    samples: int = 128
    width: int = 1280
    height: int = 720
    num_rand_scenes: list[int] = field(default_factory=lambda: [3])
    gpu: bool = True
    no_rand_seed: bool = False
    render_start_end: bool = False


@dataclass
class VideoGenConfig:
    """Video generation API settings."""

    model_api: str = "kling-ai"  # kling-ai | fal-ai
    num_videos: int = 1
    num_video_segments: int = 1
    duration: int = 5
    kling_mode: str = "pro"
    skip_prompt_refinement: bool = False
    base_prompt: str = ""
    video_max_retries: int = 3
    video_retry_wait: int = 10


# ============================================================================
# Config parsing helpers
# ============================================================================


def parse_recon_config(cfg: dict) -> dict:
    """Validate and parse a raw recon config dict.

    Returns the same dict structure but with typed sub-configs validated.
    Raises KeyError/TypeError early if required sections are missing.
    """
    required = ["paths", "human_model", "optimization"]
    for key in required:
        if key not in cfg:
            raise KeyError(f"Missing required config section: '{key}'")

    # Validate typed sections (catches wrong types/missing fields early)
    _from_dict(ReconPaths, cfg.get("paths", {}))
    _from_dict(HumanModelConfig, cfg.get("human_model", {}))
    _from_dict(ObjPoseTrackingConfig, cfg.get("obj_pose_tracking", {}))
    _from_dict(FilteringConfig, cfg.get("filtering", {}))
    _from_dict(PostProcessingConfig, cfg.get("post_processing", {}))
    _from_dict(PipelineFlags, cfg.get("pipeline", {}))

    return cfg
