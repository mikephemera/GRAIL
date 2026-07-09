import os
import shutil
from glob import glob

import cv2
import imageio
import numpy as np
import requests
from tqdm import tqdm


def _normalize_video_frame(image):
    if hasattr(image, "cpu"):
        image = image.cpu().numpy()

    if len(image.shape) > 2:
        image = image.squeeze()

    if len(image.shape) == 2:
        if image.dtype == bool or image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        if image.dtype == bool or image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.shape[2] == 1:
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    return np.ascontiguousarray(image)


def _write_video_with_cv2(frames, output_video_path, fps=30, desc="Writing video"):
    frames = [_normalize_video_frame(frame) for frame in frames]
    if not frames:
        print("No images provided to save")
        return False

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        return False

    for frame in tqdm(frames, desc=desc):
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return os.path.exists(output_video_path)


def concat_videos(input_video, result_video, topview_video, output_path):
    """
    Concatenate input, result, and top-view videos into a single comparison video.
    Vertical input -> horizontal stack; horizontal input -> vertical stack.
    Uses imageio for H.264 output compatible with VSCode.
    """
    video_paths = [v for v in [input_video, result_video, topview_video] if os.path.exists(v)]
    if len(video_paths) < 2:
        return False

    readers = []
    for vp in video_paths:
        try:
            readers.append(imageio.get_reader(vp))
        except Exception:
            for r in readers:
                r.close()
            return False

    meta = readers[0].get_meta_data()
    fps = meta.get("fps", 30.0)
    ref_h, ref_w = readers[0].get_data(0).shape[:2]
    vertical_input = ref_h > ref_w

    min_frames = min(r.count_frames() for r in readers)

    with imageio.get_writer(output_path, fps=fps) as writer:
        for i in range(min_frames):
            frames = [r.get_data(i) for r in readers]
            resized = []
            if vertical_input:
                for frame in frames:
                    fh, fw = frame.shape[:2]
                    scale = ref_h / fh
                    new_w = int(fw * scale) // 2 * 2
                    resized.append(cv2.resize(frame, (new_w, ref_h)))
                combined = np.hstack(resized)
            else:
                for frame in frames:
                    fh, fw = frame.shape[:2]
                    scale = ref_w / fw
                    new_h = int(fh * scale) // 2 * 2
                    resized.append(cv2.resize(frame, (ref_w, new_h)))
                combined = np.vstack(resized)
            writer.append_data(combined)

    for r in readers:
        r.close()
    return os.path.exists(output_path)


def save_images_to_video(images, output_video_path, fps=16, desc="Saving video"):
    """
    Save a sequence of images to a video file

    Args:
        images (list or generator): Sequence of images as numpy arrays (H, W) or (H, W, C)
        output_video_path (str): Path for output video
        fps (int): Frames per second (default: 16)
        desc (str): Description for progress bar
    """
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Convert to list if it's a generator to get length
    if not isinstance(images, (list, tuple)):
        images = list(images)

    if not images:
        print("No images provided to save")
        return

    print(f"Saving {len(images)} images to video: {output_video_path}")

    try:
        with imageio.get_writer(output_video_path, fps=fps) as writer:
            for image in tqdm(images, desc=desc):
                writer.append_data(_normalize_video_frame(image))
    except Exception as exc:
        print(f"imageio video writer failed ({exc}); falling back to OpenCV")
        if not _write_video_with_cv2(images, output_video_path, fps=fps, desc=desc):
            raise

    print(f"Video saved: {output_video_path}")


def compile_images_to_video(image_dir, output_video_path, fps=30, image_pattern="*.jpg"):
    """
    Compile rendered images into a video file

    Args:
        image_dir (str): Directory containing rendered images
        output_video_path (str): Path for output video
        fps (int): Frames per second
    """
    image_files = sorted(glob(os.path.join(image_dir, image_pattern)))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Compiling {len(image_files)} images into video: {output_video_path}")

    try:
        with imageio.get_writer(output_video_path, fps=fps) as writer:
            for image_file in tqdm(image_files, desc="Compiling video"):
                image = imageio.imread(image_file)
                writer.append_data(_normalize_video_frame(image))
    except Exception as exc:
        print(f"imageio video writer failed ({exc}); falling back to OpenCV")
        frames = [imageio.imread(image_file) for image_file in image_files]
        if not _write_video_with_cv2(frames, output_video_path, fps=fps, desc="Compiling video"):
            raise

    print(f"Video saved: {output_video_path}")


def get_video_fps_and_frame_count(video_path):
    """
    Get video FPS and total frame count using OpenCV.

    Args:
        video_path (str): Path to video file

    Returns:
        tuple: (fps, frame_count)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return fps, frame_count


def resize_video(input_path, target_width, target_height, output_path=None):
    """Resize a video to target dimensions.

    When *output_path* is ``None`` the file is resized **in-place** (via a
    temporary file).  If already at the target size the file is returned
    unchanged.

    Args:
        input_path: Path to the video file.
        target_width: Target width in pixels.
        target_height: Target height in pixels.
        output_path: Optional separate output path.  Defaults to overwriting
            *input_path*.

    Returns:
        Path to the (possibly resized) video file.
    """
    reader = imageio.get_reader(input_path, format="ffmpeg")
    meta = reader.get_meta_data()
    first = reader.get_data(0)
    cur_h, cur_w = first.shape[:2]
    fps = meta.get("fps", 30)

    if cur_w == target_width and cur_h == target_height:
        reader.close()
        return input_path

    from PIL import Image as _PILImage

    dest = output_path or input_path
    tmp = f"{os.path.splitext(dest)[0]}_tmp{os.path.splitext(dest)[1]}"

    writer = imageio.get_writer(
        tmp,
        format="ffmpeg",
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    for frame in reader:
        img = _PILImage.fromarray(frame).resize((target_width, target_height), _PILImage.LANCZOS)
        writer.append_data(np.array(img))
    reader.close()
    writer.close()

    os.replace(tmp, dest)
    return dest


def crop_video_from_start(input_video_path, output_video_path, start_seconds):
    """
    Crop video starting from start_seconds to the end using moviepy.

    Args:
        input_video_path (str): Path to input video
        output_video_path (str): Path to output cropped video
        start_seconds (float): Number of seconds to crop from the beginning
    """
    try:
        from moviepy.editor import VideoFileClip

        # Load video
        clip = VideoFileClip(input_video_path)

        # Crop from start_seconds to end
        cropped_clip = clip.subclip(start_seconds)

        # Write cropped video
        cropped_clip.write_videofile(
            output_video_path, codec="libx264", preset="slow", bitrate="5000k", audio=True
        )

        # Clean up
        clip.close()
        cropped_clip.close()

        print(f"Cropped video saved to: {output_video_path}")

    except Exception as e:
        print(f"Error cropping video {input_video_path}: {e}")
        raise


def trim_video(input_video_path, output_video_path, start_seconds):
    """
    Trim video starting from start_seconds to the end (alias for crop_video_from_start)

    Args:
        input_video_path (str): Path to input video
        output_video_path (str): Path to output trimmed video
        start_seconds (float): Number of seconds to trim from the beginning
    """
    crop_video_from_start(input_video_path, output_video_path, start_seconds)


def extract_frames_from_video(video_file, output_dir, frame_prefix="", image_format="jpg"):
    """
    Extract all frames from video and save as JPG images

    Args:
        video_file (str): Path to input video file
        output_dir (str): Directory to save extracted frames
        frame_prefix (str): Prefix for frame filenames (optional)

    Returns:
        int: Number of frames extracted
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}")
        return 0

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Extracting {total_frames} frames from {video_file}")
    print(f"Saving frames to: {output_dir}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Format frame filename with zero-padding
        if frame_prefix:
            frame_filename = f"{frame_prefix}{frame_count:06d}.{image_format}"
        else:
            frame_filename = f"{frame_count:06d}.{image_format}"

        frame_path = os.path.join(output_dir, frame_filename)

        # Save frame as JPG
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Successfully extracted {frame_count} frames")

    return frame_count


def extract_frames_from_cropped_video(video_file, output_dir, start_frame=0):
    """
    Extract frames from cropped video and save as JPG images.

    Args:
        video_file (str): Path to cropped video file
        output_dir (str): Directory to save extracted frames
        start_frame (int): Starting frame number for naming (usually frames skipped from original)

    Returns:
        int: Number of frames extracted
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}")
        return 0

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Extracting {total_frames} frames from cropped video")
    print(f"Saving frames to: {output_dir}")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Frame filename with zero-padding (continuing from where original left off)
        frame_filename = f"{(start_frame + frame_count):06d}.jpg"
        frame_path = os.path.join(output_dir, frame_filename)

        # Save frame as JPG
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Successfully extracted {frame_count} frames")

    return frame_count


def download_video(url, output_dir, filename):
    """Download a video from a URL.

    Args:
        url: Video URL.
        output_dir: Directory to save into.
        filename: Output filename (`.mp4` appended if missing).

    Returns:
        Path to the downloaded file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not filename.endswith(".mp4"):
        filename += ".mp4"
    output_path = os.path.join(output_dir, filename)

    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return output_path


def is_valid_video(video_path):
    """Return True if *video_path* can be opened and has at least one frame."""
    try:
        reader = imageio.get_reader(video_path, format="ffmpeg")
        meta = reader.get_meta_data()
        reader.close()
        return meta.get("nframes", 0) != 0
    except Exception:
        return False


def extract_last_frame(video_path, output_path=None):
    """Extract the last frame from a video and save it as PNG.

    Args:
        video_path: Path to the video.
        output_path: Where to save; defaults to ``<video>_lastframe.png``.

    Returns:
        Path to the saved frame image.
    """
    from PIL import Image as _PILImage

    reader = imageio.get_reader(video_path, format="ffmpeg")
    last_frame = None
    for frame in reader:
        last_frame = frame
    reader.close()

    if last_frame is None:
        raise ValueError(f"No frames in {video_path}")

    if output_path is None:
        output_path = f"{os.path.splitext(video_path)[0]}_lastframe.png"

    _PILImage.fromarray(last_frame).save(output_path)
    return output_path


def concatenate_videos(video_paths, output_path):
    """Concatenate multiple videos into one file.

    Frames are resized to match the first video's dimensions when needed.

    Args:
        video_paths: Ordered list of video file paths.
        output_path: Destination for the concatenated video.

    Returns:
        *output_path*.
    """
    if not video_paths:
        raise ValueError("No video paths provided")

    from PIL import Image as _PILImage

    first_reader = imageio.get_reader(video_paths[0], format="ffmpeg")
    fps = first_reader.get_meta_data().get("fps", 30)
    first_frame = first_reader.get_data(0)
    height, width = first_frame.shape[:2]
    first_reader.close()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=1,
    )

    for vp in video_paths:
        reader = imageio.get_reader(vp, format="ffmpeg")
        for frame in reader:
            if frame.shape[:2] != (height, width):
                frame = np.array(
                    _PILImage.fromarray(frame).resize((width, height), _PILImage.LANCZOS)
                )
            writer.append_data(frame)
        reader.close()

    writer.close()
    return output_path


def crop_visualization_images(vis_dir, frames_to_remove, output_vis_dir):
    """
    Crop visualization images by removing the first N frames.

    Args:
        vis_dir (str): Directory containing original visualization images
        frames_to_remove (int): Number of frames to remove from the beginning
        output_vis_dir (str): Directory to save cropped visualization images
    """
    if not os.path.exists(vis_dir):
        print(f"Warning: Visualization directory not found: {vis_dir}")
        return

    # Get all JPG files and sort them
    vis_files = sorted(glob(os.path.join(vis_dir, "*.jpg")))

    if not vis_files:
        print(f"No visualization images found in {vis_dir}")
        return

    print(f"Found {len(vis_files)} visualization images")

    if frames_to_remove >= len(vis_files):
        print(
            f"Warning: Trying to remove {frames_to_remove} images but only {len(vis_files)} available"
        )
        return

    # Create output directory
    os.makedirs(output_vis_dir, exist_ok=True)

    # Copy images starting from frames_to_remove
    cropped_files = vis_files[frames_to_remove:]

    for i, src_file in enumerate(cropped_files):
        # Create new filename with sequential numbering
        src_filename = os.path.basename(src_file)

        # If the original filename has a specific format, preserve the extension but renumber
        if src_filename.endswith(".jpg"):
            new_filename = f"{i:06d}.jpg"
        else:
            new_filename = f"{i:06d}_{src_filename}"

        dst_file = os.path.join(output_vis_dir, new_filename)

        # Copy the file
        shutil.copy2(src_file, dst_file)

    print(f"Copied {len(cropped_files)} visualization images to {output_vis_dir}")
