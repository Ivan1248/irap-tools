import cv2
import av
import numpy as np
import math
from tqdm import tqdm
import typing as T
from functools import partial
import warnings
from scipy.spatial.transform import Rotation

DEFAULT_FISHEYE_RADIUS_FACTOR = 0.94


def get_fisheye_to_perspective_converter(
    input_size: tuple,
    fov: tuple,
    fisheye_radius_factor: float,
    output_size: tuple = None,
    antialias: str = "none",
):
    (map_x, map_y), computed_output_size = create_fisheye_to_perspective_map(
        input_size=input_size,
        output_size=output_size,
        fov=fov,
        fisheye_radius_factor=fisheye_radius_factor,
    )

    return partial(
        remap_frame,
        map=(map_x, map_y),
        out_size=computed_output_size if output_size is None else output_size,
        antialias=antialias,
    )


def get_aspect_ratio_from_fov(fov, projection: T.Literal["perspective", "equirectangular"] = "perspective"):
    if projection == "perspective":
        return math.tan(math.radians(fov[0] / 2)) / math.tan(math.radians(fov[1] / 2))
    elif projection == "equirectangular":
        return 2.0  # todo


def get_perspective_output_size(
    input_size,
    fov,
    fisheye_radius_factor,
    projection: T.Literal["perspective", "equirectangular"] = "perspective",
):
    if projection == "perspective":
        # Fisheye effective focal length in pixels (radius of fisheye circle)
        fisheye_radius = min(input_size[0] / 2, input_size[1] / 2) * fisheye_radius_factor
        # Perspective output size so that center scale matches fisheye center
        output_width = 2 * fisheye_radius * fov[0] / 360
        output_height = 2 * fisheye_radius * fov[1] / 360
        return int(round(output_width)), int(round(output_height))
    elif projection == "equirectangular":
        return input_size[1] * 2, input_size[1]  # Standard equirectangular aspect ratio


def get_fov_v(aspect_ratio, fov_h):
    """Calculates the vertical field of view (FOV) given an aspect ratio and horizontal FOV.

    Args:
        aspect_ratio (float): Aspect ratio (width / height).
        fov_h (float): Horizontal field of view in degrees.

    Returns:
        float: Vertical field of view in degrees.
    """
    return 2 * math.degrees(math.atan(math.tan(math.radians(fov_h / 2)) / aspect_ratio))


def get_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Returns a 3x3 rotation matrix for yaw (Z), pitch (Y), and roll (X) in degrees.

    ROS REP-103 compliant rotation conventions:
    - Body frame: x forward, y left, z up (right-handed)
    - Fixed axis rotations: roll about X, pitch about Y, yaw about Z
    - Angles provided in degrees in the order [yaw, pitch, roll]
    - Uses scipy's `Rotation.from_euler("zyx", [yaw, pitch, roll], degrees=True)`
    - Yaw increases counter-clockwise when viewed from above (positive Z)

    Args:
        yaw (float): Rotation around Z axis in degrees (counter-clockwise positive).
        pitch (float): Rotation around Y axis in degrees (nose up positive).
        roll (float): Rotation around X axis in degrees (right wing down positive).

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    return Rotation.from_euler("zyx", [yaw, pitch, roll], degrees=True).as_matrix()


def create_fisheye_to_perspective_map(
    input_size,
    output_size=None,
    fov=(120.0, 120.0),
    yaw_pitch_roll=(0.0, 0.0, 0.0),
    fisheye_center=None,
    fisheye_radius_factor=DEFAULT_FISHEYE_RADIUS_FACTOR,
):
    """Creates tables for remapping an equidistant fisheye image to a perspective projection with arbitrary camera orientation.

    Args:
        input_size (tuple): (width, height) of the input fisheye image.
        output_size (tuple, optional): (width, height) of the output perspective image. If None, it will be computed.
        fov (tuple): (horizontal_fov, vertical_fov) in degrees for the output perspective view.
        yaw_pitch_roll (tuple): (yaw, pitch, roll) in degrees.
        fisheye_center (tuple, optional): (x, y) coordinates of the fisheye circle center in the input image. If None, the image center is used.
        fisheye_radius_factor (float): Fraction of min(center_x, center_y) to use as the fisheye radius.

    Returns:
        tuple: (map_x, map_y), output_size
            map_x (np.ndarray): X coordinates for remapping.
            map_y (np.ndarray): Y coordinates for remapping.
            output_size (tuple): (width, height) of the output image.
    """
    input_width, input_height = input_size
    fov_h, fov_v = fov

    if output_size is None:
        output_size = get_perspective_output_size(input_size, fov, fisheye_radius_factor)
    output_width, output_height = output_size

    if fisheye_center is None:
        fisheye_center = (input_width / 2, input_height / 2)

    # Fisheye projection parameters
    fisheye_cx, fisheye_cy = fisheye_center
    fisheye_radius = min(fisheye_cx, fisheye_cy) * fisheye_radius_factor

    # Convert FOV to focal length equivalent for perspective projection
    focal_length_x = output_width / (2 * math.tan(math.radians(fov_h / 2)))
    focal_length_y = output_height / (2 * math.tan(math.radians(fov_v / 2)))
    if not np.isclose(focal_length_x, focal_length_y, rtol=2.0 / min(output_size)):
        warnings.warn("The output size does not match the aspect ratio of the FOV.")

    # Perspective projection center
    perspective_cx = output_width / 2
    perspective_cy = output_height / 2

    # Output image coordinates normalized to [-1 .. 1]
    y, x = np.indices((output_height, output_width), dtype=np.float32)
    x_norm = (x - perspective_cx) / focal_length_x
    y_norm = (y - perspective_cy) / focal_length_y

    # 3D unit vectors pointing at each pixel
    rays = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    # Rotate rays
    R = get_rotation_matrix(*yaw_pitch_roll)
    rays_rot = np.einsum("...j,ij->...i", rays, R)  # shape: (H, W, 3)
    ray_x_rot, ray_y_rot, ray_z_rot = rays_rot[..., 0], rays_rot[..., 1], rays_rot[..., 2]

    # Angle from optical axis (Z-axis)
    theta = np.arccos(np.clip(ray_z_rot, -1, 1))  # shape: (H, W)
    # Azimuth
    phi = np.arctan2(ray_y_rot, ray_x_rot)

    # Convert to fisheye image coordinates
    r = fisheye_radius / (np.pi / 2) * theta  # equidistant fisheye: proportional to the angle
    fisheye_x = fisheye_cx + r * np.cos(phi)
    fisheye_y = fisheye_cy + r * np.sin(phi)

    # Only map rays in front hemisphere and within input bounds
    valid = (theta >= 0) & (theta < np.pi / 2)
    valid &= (fisheye_x >= 0) & (fisheye_x < input_width) & (fisheye_y >= 0) & (fisheye_y < input_height)

    map_x, map_y = [np.zeros((output_height, output_width), dtype=np.float32) for _ in range(2)]
    map_x[valid] = fisheye_x[valid].astype(np.float32)
    map_y[valid] = fisheye_y[valid].astype(np.float32)

    return (map_x, map_y), output_size


def create_dual_fisheye_to_equirectangular_map(
    input_size=None,
    output_size=None,
    output_aspect_ratio=2.0,
    fov=(180.0, 180.0),
    fisheye_center=None,
    fisheye_radius_factor=DEFAULT_FISHEYE_RADIUS_FACTOR,
    front_fisheye_position="left",
):
    """Creates mapping tables for dual fisheye to equirectangular projection, with flexible parameters and NumPy vectorization.

    Args:
        input_size (tuple): (width, height) of the input image (side-by-side fisheye).
        output_size (tuple, optional): (width, height) of the output image. If None, computed from input_size and aspect ratio.
        output_aspect_ratio (float): Output aspect ratio (2.0 = standard equirectangular).
        fov (tuple): (horizontal_fov, vertical_fov) in degrees for each fisheye (default: 180, 180).
        fisheye_center (tuple, optional): (x, y) center of each fisheye (relative to each half). If None, uses center of each half.
        fisheye_radius_factor (float): Fisheye radius as fraction of available space.
        front_fisheye_position (str): 'right'/'left' for side_by_side, 'top'/'bottom' for top_bottom.

    Returns:
        tuple: (map_x, map_y), output_size
            map_x (np.ndarray): X coordinates for remapping.
            map_y (np.ndarray): Y coordinates for remapping.
            output_size (tuple): (width, height) of the output image.
    """
    if input_size is None:
        raise ValueError("input_size must be provided as (width, height)")
    input_width, input_height = input_size
    fisheye_width = input_width // 2
    fisheye_height = input_height

    # Compute output size if not provided
    if output_size is None:
        output_width = input_width
        output_height = int(output_width / output_aspect_ratio)
        output_size = (output_width, output_height)
    else:
        output_width, output_height = output_size

    # Fisheye parameters
    if fisheye_center is None:
        center_x = fisheye_width / 2
        center_y = fisheye_height / 2
    else:
        center_x, center_y = fisheye_center
    radius = min(center_x, center_y) * fisheye_radius_factor

    fov_h, fov_v = fov
    max_theta = np.radians(fov_h / 2)

    # Vectorized grid of output pixel coordinates
    y, x = np.indices((output_height, output_width), dtype=np.float32)
    lon = (x / output_width) * 2 * np.pi - np.pi
    lat = (y / output_height) * np.pi - np.pi / 2

    # 3D coordinates on unit sphere
    x3d = np.cos(lat) * np.cos(lon)
    y3d = np.sin(lat)
    z3d = np.cos(lat) * np.sin(lon)

    # Determine which fisheye to use (front or back)
    use_front = z3d >= 0
    if front_fisheye_position == "right":
        front_offset_x = fisheye_width
        back_offset_x = 0
    else:
        front_offset_x = 0
        back_offset_x = fisheye_width
    offset_x = np.where(use_front, front_offset_x, back_offset_x)
    offset_y = 0

    # Flip coordinates for back camera
    x3d_back = np.where(use_front, x3d, -x3d)
    y3d_back = y3d
    z3d_back = np.where(use_front, z3d, -z3d)

    r3d = np.sqrt(x3d_back**2 + y3d_back**2)
    theta = np.arctan2(r3d, np.abs(z3d_back))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = radius * theta / max_theta
        fisheye_x = center_x + np.where(r3d > 1e-8, rho * (x3d_back / r3d), 0)
        fisheye_y = center_y + np.where(r3d > 1e-8, rho * (y3d_back / r3d), 0)
        fisheye_x = np.where(r3d <= 1e-8, center_x, fisheye_x)
        fisheye_y = np.where(r3d <= 1e-8, center_y, fisheye_y)

    final_x = fisheye_x + offset_x
    final_y = fisheye_y + offset_y

    # Mask for valid coordinates
    valid = (
        (np.abs(z3d_back) > 1e-6) & (final_x >= 0) & (final_x < input_width) & (final_y >= 0) & (final_y < input_height)
    )

    map_x = np.zeros((output_height, output_width), dtype=np.float32)
    map_y = np.zeros((output_height, output_width), dtype=np.float32)
    map_x[valid] = final_x[valid].astype(np.float32)
    map_y[valid] = final_y[valid].astype(np.float32)

    return (map_x, map_y), output_size


def remap_frame(frame, map, out_size, interpolation=cv2.INTER_LINEAR, antialias="auto"):
    """Remaps a frame using the provided mapping tables, with optional antialiasing.

    Args:
        frame (np.ndarray): Input image/frame.
        map (tuple): (map_x, map_y) mapping tables.
        out_size (tuple): (width, height) of the output image.
        interpolation (int): OpenCV interpolation flag.
        antialias (str|int): Antialiasing mode ('auto', 'none', 'gaussian', or integer upsampling factor).

    Returns:
        np.ndarray: Remapped output image.
    """
    assert antialias in ["auto", "none", "gaussian", 2, 4, 8, 16]

    # Apply smoothing before downsampling if output size is smaller than input
    if antialias in ["auto", "gaussian"] and (out_size[0] < frame.shape[1] or out_size[1] < frame.shape[0]):
        import scipy.ndimage

        scale_x = frame.shape[1] / out_size[0]
        scale_y = frame.shape[0] / out_size[1]
        sigma = (max(0.01, scale_x / 2.0), max(0.01, scale_y / 2.0))
        smoothed_frame = scipy.ndimage.gaussian_filter(frame, sigma=sigma + ((0,) if frame.ndim == 3 else ()))
    elif isinstance(antialias, int) and (out_size[0] < frame.shape[1] or out_size[1] < frame.shape[0]):
        # Upscale the map and output size by smoothing factor if specified
        upscale = antialias
        up_out_size = (out_size[0] * upscale, out_size[1] * upscale)
        up_map_x = cv2.resize(map[0], up_out_size, interpolation=cv2.INTER_LINEAR)
        up_map_y = cv2.resize(map[1], up_out_size, interpolation=cv2.INTER_LINEAR)

        # Remap at higher resolution
        upsampled = cv2.remap(
            frame,
            up_map_x,
            up_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        # Downsample to target size
        return cv2.resize(upsampled, out_size, interpolation=interpolation)
    else:
        smoothed_frame = frame
    return cv2.remap(smoothed_frame, *map, interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT)


def get_frame_indices(start_time, end_time, frames, fps, total_frames):
    start_frame = int(start_time * fps)
    stop_frame = int(end_time * fps) if end_time is not None else total_frames
    stop_frame = min(max(stop_frame, start_frame), total_frames)

    if frames is None:
        frame_indices = range(start_frame, stop_frame)
    elif isinstance(frames, slice):
        s, e, st = frames.indices(stop_frame - start_frame)
        frame_indices = range(s + start_frame, e + start_frame, st)
    else:
        frame_indices = np.asarray(frames) + start_frame
        if frame_indices[-1] >= stop_frame:
            raise ValueError(f"Frame index {frame_indices[-1]} exceeds final frame ({stop_frame - 1}).")
    return frame_indices


class VideoReader:
    """Abstract base class for video readers."""

    def get_frame(self, frame_index: int) -> np.ndarray:
        """Get a frame by index."""
        raise NotImplementedError()

    def close(self):
        """Close the video reader and release resources."""
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class PyAVVideoReader(VideoReader):
    def __init__(
        self,
        container: av.container.input.InputContainer,
        stream_index: int = 1,
    ):
        self.container = container
        video_streams = [s for s in container.streams if s.type == "video"]
        self.stream = video_streams[stream_index]

        self.fps = float(self.stream.average_rate)
        self.num_frames = self.stream.frames
        self.frame_size = (self.stream.width, self.stream.height)

        self.decoder_index = -1
        self.keyframe_index = 0
        self.keyframe_interval = 1
        self.frame_iter = None

    def _seek(self, frame_index: int):
        timestamp = int(frame_index / self.fps / self.stream.time_base)
        self.container.seek(timestamp, any_frame=False, backward=True, stream=self.stream)
        self.frame_iter = self.container.decode(self.stream)

    def get_frame(self, frame_index):
        if (
            self.frame_iter is None
            or frame_index > self.keyframe_index + self.keyframe_interval
            or frame_index < self.decoder_index
        ):
            self._seek(frame_index)
            self.decoder_index = -1

        while self.decoder_index < frame_index:
            frame = next(self.frame_iter)
            self.decoder_index = int(frame.pts * self.stream.time_base * self.fps)
            if frame.key_frame:
                self.keyframe_interval = max(self.keyframe_interval, self.decoder_index - self.keyframe_index + 1)
                self.keyframe_index = self.decoder_index
        self.keyframe_interval = max(self.keyframe_interval, self.decoder_index - self.keyframe_index + 1)

        assert self.decoder_index == frame_index
        return frame.to_ndarray(format="bgr24")

    def close(self):
        self.container.close()


class OpenCVVideoReader(VideoReader):
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

        self.prev_frame_index = None

    def get_frame(self, frame_index: int) -> np.ndarray:
        if self.prev_frame_index is None or frame_index - 1 != self.prev_frame_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        self.prev_frame_index = frame_index
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame {frame_index}.")
        return frame

    def close(self) -> None:
        self.cap.release()


def iterate_video_frames(
    video_reader: VideoReader,
    start_time: int = 0.0,
    end_time: int | None = None,
    frames: list[int] | slice | None = None,
    pbar_f: T.Callable | None = None,
) -> T.Generator[tuple[int, np.ndarray], None, None]:
    """Generator that yields (frame_index, frame) from a PyAV container object (or similar),
    supporting flexible frame selection, stream selection, and progress reporting.

    Args:
        video_reader (VideoReader): Video reader object that implements the get_frame method.
        start_time (float, optional): Start time in seconds from which to begin yielding frames.
            Defaults to 0.
        end_time (float, optional): End time in seconds at which to stop yielding frames. If None,
            continues to the end of the video. Defaults to None.
        frames (list[int] | slice | None, optional): Frame indices or slice to yield from start
            time. If None, yields all frames between start_time and end_time.
        pbar_f (callable, optional): Factory function (e.g., tqdm) to wrap the frame index iterator
            for progress reporting. If None, no progress bar is used.

    Yields:
        tuple: (frame_index (int), frame (np.ndarray))
            frame_index: Index of the frame in the video (starting from 0).
            frame: Frame as a NumPy ndarray in BGR24 format.

    Raises:
        ValueError: If no video streams are found in the container, or if requested frame indices are out of bounds.
    """
    frame_indices = get_frame_indices(start_time, end_time, frames, video_reader.fps, video_reader.num_frames)
    if pbar_f is not None:
        frame_indices = pbar_f(frame_indices)
    for i in frame_indices:
        yield i, video_reader.get_frame(i)


def convert_video(
    input_path: str,
    output_path: str,
    frame_converter_f: T.Callable = get_fisheye_to_perspective_converter,
    frame_iter_args: dict = dict(start_time=0, end_time=None, frames=None),
    stream_index: int = 0,
    video_writer_f=partial(cv2.VideoWriter, fourcc=cv2.VideoWriter_fourcc(*"mp4v")),
    use_opencv_reader: bool = False,
):
    """
    Converts a video using a frame converter function and saves the result.

    Args:
        input_path (str): Path to input video file.
        output_path (str): Path to output video file.
        frame_converter_f (Callable): Function that returns a frame converter (should accept input_size as kwarg).
        frame_iter_args (dict): Arguments for frame iteration (start_time, end_time, frames).
        stream_index (int): Index of the video stream to use for PyAV (default: 0).
        video_writer_f (Callable): Function to create a cv2.VideoWriter instance (default: mp4v codec).
        use_opencv_reader (bool): If True, use OpenCV for video reading; if False, use PyAV.
    """
    if use_opencv_reader:
        if stream_index != 0:
            raise ValueError("OpenCV reader does not support stream_index, use PyAV instead.")
        reader = OpenCVVideoReader(cv2.VideoCapture(input_path))
    else:
        reader = PyAVVideoReader(av.open(input_path), stream_index)
    with reader:
        input_size = reader.frame_size
        print(f"Input: {input_size[0]}x{input_size[1]}, {reader.fps} FPS, stream {stream_index}")
        convert_frame = frame_converter_f(input_size=input_size)
        frame_iter = iterate_video_frames(
            reader,
            **frame_iter_args,
            pbar_f=partial(tqdm, desc="Processing frames"),
        )
        writer = None
        for frame_index, frame in frame_iter:
            out_frame = convert_frame(frame)

            if writer is None:
                out_frame_size = (out_frame.shape[1], out_frame.shape[0])
                writer = video_writer_f(output_path, fps=reader.fps, frameSize=out_frame_size)
                print(f"Output: {out_frame_size[0]}x{out_frame_size[1]}, {reader.fps} FPS")

            writer.write(out_frame)

        if writer is not None:
            writer.release()
            print(f"Conversion complete: {output_path}")


def undistort_fisheye(image: np.ndarray, camera_matrix: np.ndarray, distortion_coeffs: np.ndarray) -> np.ndarray:
    """
    Undistort a fisheye image using camera calibration parameters

    Args:
        image: Input fisheye image (BGR format)
        camera_matrix: 3x3 camera intrinsic matrix
        distortion_coeffs: Fisheye distortion coefficients (4-element array)

    Returns:
        Undistorted image
    """
    height, width = image.shape[:2]

    # Create undistortion maps
    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        camera_matrix, distortion_coeffs, np.eye(3), camera_matrix, (width, height), cv2.CV_32FC1
    )

    # Apply undistortion
    undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    return undistorted


# Example usage
if __name__ == "__main__":
    fisheye_radius_factor = 0.94

    # Single fisheye to perspective with custom aspect ratio and FOV
    convert_video(
        "input/20231223_unit1_58_59_836/VID_20241219_105926_00_145.insv",
        "output/20231223_unit1_58_59_836/VID_20241219_105926_00_145_perspective_hw.mp4",
        frame_converter_f=partial(
            get_fisheye_to_perspective_converter,
            fov_h=127,  # GoPro Hero 4 medium: 127°
            fov_v=None,
            yaw_pitch_roll=(0.0, 0.0, 0.0),
            output_size=(384, 288),
            fisheye_radius_factor=fisheye_radius_factor,
        ),
        use_opencv_reader=True,
        frame_iter_args=dict(start_time=0, end_time=None, frames=list(range(0, 1000, 1))),
    )

    # Single fisheye to perspective with custom aspect ratio and FOV
    convert_video(
        "input/20231223_unit1_58_59_836/VID_20241219_105926_00_145.insv",
        "output/20231223_unit1_58_59_836/VID_20241219_105926_00_145_perspective.mp4",
        frame_converter_f=partial(
            get_fisheye_to_perspective_converter,
            fov_h=90,  # GoPro Hero 4 medium: 127°
            fov_v=None,
            output_aspect_ratio=16 / 9,  # TODO
            fisheye_radius_factor=fisheye_radius_factor,
        ),
        frame_iter_args=dict(end_time=10),
    )

    # Dual fisheye to equirectangular 360°
    convert_video(
        "input/20231223_unit1_58_59_836/LRV_20241219_105926_01_145.lrv",
        "output/20231223_unit1_58_59_836/LRV_20241219_105926_01_145_equirectangular.mp4",
        frame_converter_f=partial(get_fisheye_to_perspective_converter, mode="equirectangular"),
    )
