from pathlib import Path
import argparse
from functools import partial
from typing import Callable, Dict

from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .metadata import extract_metadata, get_motion_data
from .data import IMUData, FrameOrientationData


# Orientation computation


def apply_gravity_correction(
    current_rot: Rotation, acc: np.ndarray, gravity_vec: np.ndarray, correction_gain: float
) -> Rotation:
    """
    Applies a complementary filter style gravity correction to the orientation.
    Blends the gyroscope-based orientation with the orientation suggested by the accelerometer.
    """

    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    acc_norm = np.linalg.norm(acc)
    if acc_norm > 1e-2:
        g_body = normalize(acc)
        g_est = current_rot.apply(gravity_vec)

        # Calculate the rotation needed to align g_est to g_body
        correction_axis = np.cross(g_est, g_body)
        axis_norm = np.linalg.norm(correction_axis)
        if axis_norm > 1e-6:
            correction_axis = correction_axis / axis_norm
            correction_angle = np.arcsin(np.clip(axis_norm, -1.0, 1.0))
            # Complementary filter: blend gyro and accel
            correction_rot = Rotation.from_rotvec(
                correction_gain * correction_angle * correction_axis
            )
            current_rot = correction_rot * current_rot
    return current_rot


class BasicOrientationIntegrator:
    def __init__(self, apply_grav_corr: bool = False, grav_corr_gain: float = 0.01):
        self.grav_corr_gain = grav_corr_gain
        self.gravity_vec = np.array([0, 0, -1]) if apply_grav_corr else None

    def updateIMU(self, rot: Rotation, angvel: np.ndarray, accel: np.ndarray, dt: float):
        angvel_norm = np.linalg.norm(angvel)
        if angvel_norm > 1e-8 and dt > 0:
            axis = angvel / angvel_norm
            angle = angvel_norm * dt
            delta_rot = Rotation.from_rotvec(axis * angle)
        else:
            delta_rot = Rotation.identity()
        next_rot = rot * delta_rot

        if self.gravity_vec is not None:
            next_rot = apply_gravity_correction(
                next_rot, accel, self.gravity_vec, self.grav_corr_gain
            )

        return next_rot


def compute_frame_orientations(
    imu_data: IMUData,
    frame_times: np.ndarray,
    integrator_f=partial(BasicOrientationIntegrator, apply_grav_corr=False),
    pbar_f: Callable = partial(tqdm, desc=f"Estimating frame orientations"),
) -> Dict[str, np.ndarray]:
    # Extract timestamps (in seconds) minus start time for numerical stability
    times = imu_data.time - imu_data.time[0]
    angvels = imu_data.angular_velocity
    accels = imu_data.accelerometer

    integrator = integrator_f()  # TODO try Magwick or Mahony from ahrs

    # Initial orientation identity
    rotations = [Rotation.identity()]

    for i in pbar_f(range(1, len(times))):
        rotations.append(
            integrator.updateIMU(rotations[-1], angvels[i], accels[i], dt=times[i] - times[i - 1])
        )

    rotations = Rotation.from_quat([r.as_quat() for r in rotations])

    # Interpolate orientation to frame timestamps
    slerp = Slerp(times, rotations)
    frame_orientations = slerp(frame_times - frame_times[0])

    ypr_deg = frame_orientations.as_euler("zyx", degrees=True)  # shape (frames, 3)
    return FrameOrientationData(time=frame_times, yaw_pitch_roll=ypr_deg)


def estimate_orientations(
    video_file: str, apply_gravity_corr: bool = False
) -> FrameOrientationData:
    metadata = extract_metadata(video_file)
    motion_data = get_motion_data(metadata)
    return compute_frame_orientations(
        motion_data["imu_data"],
        motion_data["frames"].time,
        integrator_f=partial(
            BasicOrientationIntegrator, apply_grav_corr=apply_gravity_corr, grav_corr_gain=0.01
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract gyroscope data from INSV files")
    parser.add_argument("input_file", help="Input INSV file path")
    parser.add_argument(
        "-m", "--metadata", action="store_true", help="Print file metadata using exiftool"
    )
    args = parser.parse_args()

    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        exit(1)

    print("\nExtracting metadata with exiftool:")
    metadata = extract_metadata(args.input_file)
    for k, v in metadata["main"].items():
        print(f"{k}: {v}")

    motion_data = get_motion_data(metadata)

    orientation_data = compute_frame_orientations(
        motion_data["imu_data"],
        motion_data["frames"].time,
        integrator_f=partial(
            BasicOrientationIntegrator, apply_grav_corr=False, grav_corr_gain=0.01
        ),
    )
