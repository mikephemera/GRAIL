import math
import os
import sys
import gc

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from grail.core.device import empty_cache, require_musa


def est_depth(
    image_list,
    device="auto",
    intrinsics=None,
    gt_depth_first_frame=None,
    moge_root=None,
    model_id="Ruicheng/moge-2-vitl-normal",
):
    """Estimate per-frame metric depth with MoGe-2.

    Args:
        image_list: List of image paths.
        device: Device to run on.
        intrinsics: Optional camera intrinsics (3, 3) numpy array or path to
            intrinsics file. Same intrinsics will be used for all images
            (scaled appropriately if images are different sizes).
        gt_depth_first_frame: Optional ground truth depth for the first frame
            (H, W) in meters. Currently unused; kept for API symmetry.
    """
    device = require_musa(device, "MoGe")
    if not image_list:
        raise ValueError("MoGe requires at least one input image")

    moge_dir = os.path.abspath(
        moge_root
        or os.path.join(os.path.dirname(__file__), "..", "..", "..", "imports", "MoGe")
    )
    if not os.path.isfile(os.path.join(moge_dir, "moge", "__init__.py")):
        raise FileNotFoundError(
            f"MoGe source tree not found: {moge_dir}. Initialize the GRAIL MoGe submodule "
            "or pass --moge_root to a materialized checkout."
        )
    if moge_dir not in sys.path:
        sys.path.insert(0, moge_dir)

    from moge.model.v2 import MoGeModel

    print("Loading MoGe-2 model (Ruicheng/moge-2-vitl-normal)...")
    model = MoGeModel.from_pretrained(model_id).to(device).eval()

    original_image_size = cv2.imread(image_list[0]).shape[:2]

    # Convert camera intrinsics to horizontal FOV in degrees
    fov_x = None
    if intrinsics is not None:
        if isinstance(intrinsics, str):
            K = np.loadtxt(intrinsics)
        else:
            K = np.array(intrinsics)
        fx = K[0, 0]
        W = original_image_size[1]
        fov_x = 2 * math.atan(W / (2 * fx)) * 180.0 / math.pi
        print(f"Using known intrinsics: fx={fx:.2f}, image_width={W}, fov_x={fov_x:.2f} degrees")
    else:
        print("No intrinsics provided, MoGe will recover focal length internally")

    depth_list = []
    for image_path in tqdm(image_list, desc="Estimating depth (MoGe)"):
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read MoGe input image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_tensor = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(
            2, 0, 1
        )

        with torch.no_grad():
            output = model.infer(image_tensor, fov_x=fov_x, resolution_level=9, use_fp16=True)

        depth = output["depth"]  # (H, W), metric meters

        # Replace invalid pixels (inf) with 0
        depth = torch.where(torch.isinf(depth), torch.zeros_like(depth), depth)

        # Resize to original image size if needed
        if depth.shape != tuple(original_image_size):
            depth = (
                F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=original_image_size,
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
            )

        depth_list.append(depth)

    del model
    gc.collect()
    empty_cache(device)
    print(f"MoGe: produced {len(depth_list)} metric depth frames")

    return depth_list
