"""Adapter for the public ``general_motion_retargeting`` (GMR) package.

GRAIL's retargeting pipeline diverges from public GMR in a few places. Rather
than editing files inside the submodule (which forces ``imports/GMR`` to a
"dirty" working tree and complicates submodule bumps), we monkey-patch the
public package at import time. Importing this module is enough to activate
the patches; downstream code does:

    from grail.adapters.gmr import GMR, ROBOT_XML_DICT, ...

and gets the patched GMR transparently.

Patches applied:

1. ``GeneralMotionRetargeting.__init__`` — public GMR multiplies
   ``ik_config["human_scale_table"][k]`` by a per-instance ``ratio``. GRAIL's
   SMPL-X inputs already provide the correct ratio embedded in the SMPL-X
   betas, so we want identity (1.0). After the public init runs, every value
   in ``self.human_scale_table`` is reset to 1.0.

2. ``general_motion_retargeting.utils.smpl.load_smplx_file`` — public GMR
   only knows how to load ``.npz`` SMPL-X dumps. GRAIL's
   ``grail.pipelines.recon_4dhoi`` writes ``.pkl`` files with SMPL-X data wrapped
   under a ``human_data`` key. The replacement function transparently
   handles both ``.pkl`` (GRAIL) and ``.npz`` (public) formats; it also
   (a) truncates ``betas`` to the first 10 dims (public GMR keeps all 16),
   (b) zeroes out root translation before body-model evaluation. Public
   ``.npz`` callers see no change.

3. ``GeneralMotionRetargeting`` quaternion handling — some public GMR releases
   call SciPy's newer ``scalar_first=`` API. The stageD runtime can carry
   SciPy 1.10, so the methods that consume GRAIL's ``[w, x, y, z]`` quaternions
   reorder explicitly.
"""

from __future__ import annotations

import logging
import pickle

import general_motion_retargeting as _gmr
import general_motion_retargeting.utils.smpl as _gmr_smpl
import mink as _mink
import numpy as np
from scipy.spatial.transform import Rotation as _Rotation
import torch

_logger = logging.getLogger(__name__)


def _rotation_from_quat_wxyz(quat):
    quat_xyzw = np.asarray(quat)[..., [1, 2, 3, 0]]
    return _Rotation.from_quat(quat_xyzw)


def _quat_wxyz_from_rotation(rot):
    return rot.as_quat()[..., [3, 0, 1, 2]]


# ---------------------------------------------------------------------------
# Patch 1: human_scale_table -> identity (override public per-joint ratio).
# ---------------------------------------------------------------------------
_orig_gmr_init = _gmr.GeneralMotionRetargeting.__init__


def _patched_gmr_init(self, *args, **kwargs):
    _orig_gmr_init(self, *args, **kwargs)
    if hasattr(self, "human_scale_table"):
        for key in list(self.human_scale_table.keys()):
            self.human_scale_table[key] = 1.0


_gmr.GeneralMotionRetargeting.__init__ = _patched_gmr_init


# ---------------------------------------------------------------------------
# Patch 2: SciPy 1.10-compatible quaternion handling in GMR's main IK path.
# ---------------------------------------------------------------------------
def _patched_setup_retarget_configuration(self):
    self.configuration = _mink.Configuration(self.model)

    self.tasks1 = []
    self.tasks2 = []

    for frame_name, entry in self.ik_match_table1.items():
        body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
        if pos_weight != 0 or rot_weight != 0:
            task = _mink.FrameTask(
                frame_name=frame_name,
                frame_type="body",
                position_cost=pos_weight,
                orientation_cost=rot_weight,
                lm_damping=1,
            )
            self.human_body_to_task1[body_name] = task
            self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
            self.rot_offsets1[body_name] = _rotation_from_quat_wxyz(rot_offset)
            self.tasks1.append(task)
            self.task_errors1[task] = []

    for frame_name, entry in self.ik_match_table2.items():
        body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
        if pos_weight != 0 or rot_weight != 0:
            task = _mink.FrameTask(
                frame_name=frame_name,
                frame_type="body",
                position_cost=pos_weight,
                orientation_cost=rot_weight,
                lm_damping=1,
            )
            self.human_body_to_task2[body_name] = task
            self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
            self.rot_offsets2[body_name] = _rotation_from_quat_wxyz(rot_offset)
            self.tasks2.append(task)
            self.task_errors2[task] = []


def _patched_offset_human_data(self, human_data, pos_offsets, rot_offsets):
    offset_human_data = {}
    for body_name in human_data.keys():
        pos, quat = human_data[body_name]
        offset_human_data[body_name] = [pos, quat]
        updated_rot = _rotation_from_quat_wxyz(quat) * rot_offsets[body_name]
        updated_quat = _quat_wxyz_from_rotation(updated_rot)
        offset_human_data[body_name][1] = updated_quat

        local_offset = pos_offsets[body_name]
        global_pos_offset = _rotation_from_quat_wxyz(updated_quat).apply(local_offset)
        offset_human_data[body_name][0] = pos + global_pos_offset

    return offset_human_data


_gmr.GeneralMotionRetargeting.setup_retarget_configuration = _patched_setup_retarget_configuration
_gmr.GeneralMotionRetargeting.offset_human_data = _patched_offset_human_data


# ---------------------------------------------------------------------------
# Patch 3: smpl.load_smplx_file — accept GRAIL .pkl + truncated betas.
# ---------------------------------------------------------------------------
import smplx as _smplx  # noqa: E402  (imported here so the patch is self-contained)


def _grail_load_smplx_file(smplx_file, smplx_body_model_path):
    """Drop-in replacement that supports GRAIL ``.pkl`` SMPL-X dumps.

    For ``.npz`` inputs the behavior matches public GMR except that we use
    the first 10 betas (public uses all 16) and zero out root translation
    before body-model forward — both of which GRAIL relies on.
    """
    if smplx_file.endswith(".pkl"):
        with open(smplx_file, "rb") as f:
            pkl_data = pickle.load(f)["human_data"]
        gender = "neutral"
        smplx_data = {
            "pose_body": pkl_data["poses"][..., 3:66],
            "root_orient": pkl_data["poses"][..., :3],
            "betas": pkl_data["betas"],
            "trans": pkl_data["trans"],
            "mocap_frame_rate": torch.tensor(30),
        }
        scale = torch.tensor(pkl_data.get("scale", 1.0))
    else:
        smplx_data = np.load(smplx_file, allow_pickle=True)
        gender = str(smplx_data["gender"])
        scale = torch.tensor(1.0)

    body_model = _smplx.create(
        smplx_body_model_path,
        "smplx",
        gender=gender,
        use_pca=False,
    )
    num_frames = smplx_data["pose_body"].shape[0]
    transl = (
        smplx_data["trans"].copy()
        if hasattr(smplx_data["trans"], "copy")
        else np.array(smplx_data["trans"]).copy()
    )
    transl[..., :3] = 0.0

    smplx_output = body_model(
        betas=torch.tensor(smplx_data["betas"][..., :10]).float().view(1, -1),
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(transl).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float() * 10,
        right_hand_pose=torch.zeros(num_frames, 45).float() * 10,
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        return_full_pose=True,
    )
    smplx_output.vertices *= scale
    smplx_output.joints *= scale

    if len(smplx_data["betas"].shape) == 1:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0]
    else:
        human_height = 1.66 + 0.1 * smplx_data["betas"][0, 0]

    return smplx_data, body_model, smplx_output, human_height


_gmr_smpl.load_smplx_file = _grail_load_smplx_file

_logger.debug(
    "Applied GRAIL GMR runtime patches "
    "(scale=1.0 + SciPy 1.10 quaternions + .pkl SMPL-X loader)."
)


# ---------------------------------------------------------------------------
# Public re-exports — callers do `from grail.adapters.gmr import GMR, ...`.
# ---------------------------------------------------------------------------
from general_motion_retargeting import (  # noqa: E402
    IK_CONFIG_DICT,
    ROBOT_BASE_DICT,
    ROBOT_XML_DICT,
    VIEWER_CAM_DISTANCE_DICT,
    GeneralMotionRetargeting as GMR,
    RobotMotionViewer,
)
from general_motion_retargeting.robot_motion_viewer import draw_frame  # noqa: E402

__all__ = [
    "GMR",
    "IK_CONFIG_DICT",
    "ROBOT_BASE_DICT",
    "ROBOT_XML_DICT",
    "VIEWER_CAM_DISTANCE_DICT",
    "RobotMotionViewer",
    "draw_frame",
]
