from pathlib import Path
import argparse
import shutil
import subprocess
import re
import logging
import pickle
import hashlib
from typing import Any, List, Dict, Optional
from datetime import datetime, timezone

import numpy as np
from tqdm import tqdm
import telemetry_parser

from .data import IMUData, FrameData
from ..gps import GPSTrack

logger = logging.getLogger(__name__)
DEFAULT_CACHE_DIR = Path(".cache/insta")


def parse_metadata_value(value: str) -> Any:
    """Parses a metadata value string into its appropriate type.
    Handles integers, floats, and lists of numbers.
    """

    def parse_float_list(value: str) -> List[float]:
        return [float(x) for x in value.strip().split() if x]

    parsers = [float, parse_float_list, str.strip]
    for parser in parsers:
        try:
            return parser(value)
        except ValueError:
            pass


def extract_telemetry(filepath: str) -> Dict[str, Any]:
    """Extract metadata using telemetry-parser package.

    This provides the same data structure as the exiftool method but uses
    the official telemetry-parser package for better accuracy and consistency.

    Args:
        filepath: Path to the video file

    Returns:
        Dictionary with metadata in the same format as exiftool output

    Raises:
        Exception: If telemetry-parser fails to extract data
    """
    # Create parser instance
    tp = telemetry_parser.Parser(filepath)
    logger.info(f"Using telemetry-parser: Camera={tp.camera}, Model={tp.model}")

    telemetry = tp.telemetry()
    """
    tp.telemetry()

    Default:
        Metadata:
            'serial_number': 'IBMLA2407767R3'
            'camera_type': 'Insta360 X4'
            'fw_version': 'v1.4.8_build1'
            'offset': [2.0, 2905.65, 3989.5, 3006.75, 0.371, 0.209, 90.52, 2891.65, 11997.71, 3013.87, -0.412, 0.198, 89.639, 16000.0, 6000.0, 1095.0]
            'total_time': 174
            'dimension': {'x': 1920, 'y': 1920}
            'frame_rate': 24
            'first_frame_timestamp': 51714761
            'rolling_shutter_time': 8.957462310791016
            'gyro_timestamp': 1.6
            'gyro_calib': {'numbers': [-0.0029296875, -0.0068359375, -0.001953125, 0.0031957933080950826, 0.010652644360316942, 0.005326322180158471], 'unix_timestamp': 0}
            'first_gps_timestamp': 1733824363906
            'fov': 40.0:
            'gyro_type': 'InsdevImuType40609'
            'offset_v2': [2.0, 2897.52, 3989.48, 3004.11, 0.276, 0.29, 90.535, 0.0, 0.0, 0.0, 1.0, -0.20175603, 0.34253067, -0.12655279, 16000.0, 6000.0, 71.0, 2883.42, 11997.0, 3011.52, -0.409, 0.3, 89.625, -0.001606, -0.000255, -0.030557, 1.0, -0.19953983, 0.34139037, -0.12675293, 16000.0, 6000.0, 71.0, 132096.0]
            'offset_v3': [2.0, 1.94817, 4623.81, 4623.4, 3987.5, 3012.7, 0.329, 0.379, 90.493, 0.0, 0.0, 0.0, 0.38634348, 1.30296624, -3.93261027, -0.00147663, 0.00045841, 16000.0, 6000.0, 71.0, 1.94817, 4608.23, 4606.91, 11995.24, 3019.82, -0.199, 0.408, 89.661, -0.000292, -7.3e-05, -0.03132, 0.3855471, 1.32885373, -4.07654667, -0.00139833, 0.00011861, 16000.0, 6000.0, 71.0, 197632.0]
            'original_offset_v2': [2.0, 2897.52, 3989.48, 3004.11, 0.276, 0.29, 90.535, 0.0, 0.0, 0.0, 1.0, -0.20175603, 0.34253067, -0.12655279, 16000.0, 6000.0, 71.0, 2883.42, 11997.0, 3011.52, -0.409, 0.3, 89.625, -0.001606, -0.000255, -0.030557, 1.0, -0.19953983, 0.34139037, -0.12675293, 16000.0, 6000.0, 71.0, 132096.0]
            'original_offset_v3': [2.0, 1.94817, 4623.81, 4623.4, 3987.5, 3012.7, 0.329, 0.379, 90.493, 0.0, 0.0, 0.0, 0.38634348, 1.30296624, -3.93261027, -0.00147663, 0.00045841, 16000.0, 6000.0, 71.0, 1.94817, 4608.23, 4606.91, 11995.24, 3019.82, -0.199, 0.408, 89.661, -0.000292, -7.3e-05, -0.03132, 0.3855471, 1.32885373, -4.07654667, -0.00139833, 0.00011861, 16000.0, 6000.0, 71.0, 197632.0]
            'gyro_cfg_info': {'acc_range': 32, 'gyro_range': 2000}
            ...
        Exposure:
            Data:
                [0]: {'t': -0.3333139999999985, 'v': 0.0029557745438069105}
                ...
        Lens:
            Data:
                'assymetrical': True
                'fisheye_params':
                    'camera_matrix': [[1541.2700000000002, 0.0, 478.5], [0.0, 1541.1333333333332, 964.064], [0.0, 0.0, 1.0]]
                    'distortion_coeffs': [0.38634348, 1.30296624, -3.93261027, -0.00147663, 0.00045841, 1.94817]
                    'sync_settings': {'initial_offset': 0, 'initial_offset_inv': False, 'search_size': 0.3, 'max_sync_points': 5, 'every_nth_frame': 1, 'time_per_syncpoint': 0.5, 'do_autosync': False}
        GPS:
            Data:
                [0]: {'is_acquired': True, 'unix_timestamp': 1733824361.996, 'lat': 10.087909999999999, 'lon': 105.84957166666666, 'speed': 4.715885210037231, 'track': 315.44000244140625, 'altitude': 10.2}
                ...
        Gyroscope:
            Data:
                [0]: {'t': -0.22370199999999893, 'x': -27.69771452906017, 'y': -265.0637305487238, 'z': 179.93069598448238}
                ...
            'Unit': 'deg/s'
            'Scale': 16.384
            'Orientation': 'Xyz'
        Accelerometer:
            Data:
                [0]: {'x': 0.0, 'y': 0.0, 'z': 0.0}
                ...
                'Unit': 'g'
                'Scale': 1024.0
                'Orientation': 'Xyz'
    """
    frame_records = _extract_frame_records(tp)
    # Get normalized IMU data (units deg/s, m/s^2) with raw orientation
    imu_records = tp.normalized_imu(orientation="XYZ")
    # contains timestamp (in seconds), lat, lon, speed, track, altitude
    gps_records = telemetry[0]["GPS"]["Data"] if hasattr(telemetry[0], "GPS") else None

    logger.info(f"Extracted {len(imu_records)} IMU samples using telemetry-parser")

    metadata = telemetry[0]["Default"]["Metadata"]
    metadata.pop("total_time")  # inaccurate

    return {
        "metadata": metadata,
        "lens": telemetry[0]["Lens"]["Data"],
        "imu_records": imu_records,
        "gps_records": gps_records,
        "frame_records": frame_records,
        "imu_orientation": telemetry[0]["Gyroscope"]["Orientation"],
    }


def _extract_frame_records(tp: telemetry_parser.Parser) -> List[Dict[str, Any]]:
    """Extract frame records from telemetry-parser."""
    frame_data = tp.telemetry()[0]["Exposure"]["Data"]
    return [{"timestamp_ms": frame["t"] * 1000, "exposure_time": frame["v"]} for frame in frame_data]


def _run_exiftool(filepath: str) -> str:
    """Run exiftool on the given file and return the output.

    Args:
        filepath: Path to the video file

    Returns:
        Exiftool output as string

    Raises:
        ValueError: If exiftool is not found or execution fails
    """
    exiftool_path = shutil.which("exiftool")
    if not exiftool_path:
        raise ValueError("exiftool not found in PATH")

    try:
        completed = subprocess.run(
            [exiftool_path, "-api", "largefilesupport=1", "-ee", "-G3", "-m", filepath],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        return completed.stdout
    except subprocess.CalledProcessError as e:
        raise ValueError(f"exiftool execution failed with return code {e.returncode}: {e}")
    except Exception as e:
        raise ValueError(f"Error running exiftool: {e}") from e


def _parse_exiftool_output(out: str) -> Dict[str, Any]:
    """Parse exiftool output into structured metadata dictionary.

    Args:
        out: Raw exiftool output string

    Returns:
        Dictionary with main metadata, IMU records, GPS records, and frame records
    """
    metadata: Dict[str, List[tuple]] = {}

    # Parse all metadata entries
    for m in tqdm(re.finditer(r"\[(.*?)\]\s+([^\:]+?)\s*:\s*(.*)", out)):
        cat, key, val = m.groups()
        metadata.setdefault(cat, []).append((key, parse_metadata_value(val)))

    # Extract main metadata
    if "Main" not in metadata:
        raise ValueError("No main metadata found in exiftool output")
    main = dict(metadata.pop("Main"))

    # Extract timecoded data (Doc1, Doc2, etc.)
    timecoded = [metadata[f"Doc{i}"] for i in range(1, len(metadata) + 1)]

    # Categorize records
    imu_records = []
    gps_records = []
    frame_records = []

    for x in timecoded:
        data = dict(x)
        if "Accelerometer" in data:
            imu_records.append(data)
        elif "Exposure Time" in data:
            frame_records.append(data)
        elif "GPS Date/Time" in data:
            gps_records.append(data)
        else:
            logger.warning(f"Unexpected entry type: {list(data.keys())}")

    return {
        "main": main,
        "imu_records": imu_records,
        "gps_records": gps_records,
        "frame_records": frame_records,
    }


def extract_metadata(filepath: str, use_telemetry_parser: bool = True) -> Dict[str, Any]:
    """Extracts metadata from the INSV file using telemetry-parser (default) or exiftool.

    Args:
        filepath: Path to the INSV file
        use_telemetry_parser: If True, use telemetry-parser package (recommended).
                             If False, use exiftool.

    Returns:
        Dictionary with general metadata, IMU records, GPS records, and frame records.

    Raises:
        FileNotFoundError: If the input file doesn't exist
        ValueError: If both extraction methods fail
    """
    src = Path(filepath)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")

    # Try telemetry-parser first if requested
    if use_telemetry_parser:
        logger.info("Using telemetry-parser for metadata extraction")
        return extract_telemetry(filepath)

    # Use exiftool
    assert False, "exiftool produces different metadata than telemetry-parser"
    logger.info("Using exiftool for metadata extraction")
    out = _run_exiftool(filepath)
    return _parse_exiftool_output(out)


class MetadataCache:
    """Configurable pickle-based cache for extracted metadata.

    cache_dir controls where entries are stored:
      - Path-like value: central directory (.cache/insta by default) using sha1 of abs path
      - None: sidecar next to the source file (<file>.meta.pkl)
    """

    def __init__(self, cache_dir: Optional[Path | str] = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir: Optional[Path] = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir

    def _cache_path(self, src: Path) -> Path:
        if self.cache_dir is None:
            return src.with_suffix(src.suffix + ".meta.pkl")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha1(str(src.resolve()).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.pkl"

    @staticmethod
    def _current_key(src: Path) -> Optional[Dict[str, int]]:
        try:
            st = src.stat()
            return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        except FileNotFoundError:
            return None

    def load(self, filepath: str | Path) -> Optional[Dict[str, Any]]:
        src = Path(filepath)
        key = self._current_key(src)
        if key is None:
            return None
        cp = self._cache_path(src)
        if not cp.exists():
            return None
        try:
            with cp.open("rb") as f:
                payload = pickle.load(f)
            cached_key = payload.get("_cache_key", {})
            if cached_key == key:
                return payload.get("metadata", {})
        except Exception:
            logger.debug("Ignoring corrupt cache: %s", cp)
        return None

    def save(self, filepath: str | Path, metadata: Dict[str, Any]) -> None:
        src = Path(filepath)
        key = self._current_key(src)
        if key is None:
            return
        cp = self._cache_path(src)
        try:
            with cp.open("wb") as f:
                pickle.dump({"_cache_key": key, "metadata": metadata}, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            logger.debug("Failed to write cache: %s", cp)


def cached(func, cache_dir=DEFAULT_CACHE_DIR):
    def wrapper(filepath, use_telemetry_parser=True):
        cache = MetadataCache(cache_dir)
        result = cache.load(filepath)
        if result is not None:
            return result
        data = func(filepath, use_telemetry_parser)
        cache.save(filepath, data)
        return data

    return wrapper


def extract_metadata_cached(
    filepath: str | Path, cache_dir=DEFAULT_CACHE_DIR, use_telemetry_parser: bool = True
) -> Dict[str, Any]:
    return cached(extract_metadata, cache_dir)(filepath, use_telemetry_parser)


def compute_sampling_rate(times: np.ndarray) -> float:
    """Computes the average sampling rate from a list of timestamps.

    Assumes that each timestamp corresponds to the beginning (or end) of one frame.
    """
    return (len(times) - 1) / (times[-1] - times[0])


def parse_exiftool_gps_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse GPS metadata dictionary into structured Python types.
    Each field has a documented format (regex or strptime format).
    """

    def _parse_timestamp(ts: str) -> datetime:
        # EXIF-style
        try:
            dt = datetime.strptime(ts, "%Y:%m:%d %H:%M:%S.%fZ")
        except ValueError:
            # sometimes there are no milliseconds
            dt = datetime.strptime(ts, "%Y:%m:%d %H:%M:%SZ")
        return dt.replace(tzinfo=timezone.utc).timestamp()  # TODO: check if this is correct

    def _dms_to_dd(dms_str: str) -> float:
        match = re.match(r"(-?\d+) deg (-?\d+)' (-?[\d.]+)\" ([NSEW])", dms_str)
        if not match:
            raise ValueError(f"Invalid DMS format: {dms_str}")
        deg, minutes, seconds, direction = match.groups()
        dd = int(deg) + int(minutes) / 60 + float(seconds) / 3600
        if direction in ["S", "W"]:
            dd *= -1
        return dd

    def _parse_altitude(s: str) -> float:
        match = re.match(r"(-?[\d.]+) m", s)
        return float(match.group(1))

    parsers = {
        "GPS Date/Time": _parse_timestamp,
        "GPS Latitude": _dms_to_dd,
        "GPS Longitude": _dms_to_dd,
        "GPS Altitude": _parse_altitude,
        "GPS Speed": float,
        "GPS Track": float,
        "Warning": str,  # It says "Unrecognized INSV GPS format", but there is data
    }

    return {k: parsers[k](v) for k, v in data.items()}


def remap_imu_coordinates(angular_velocity, accelerometer, imu_orientation: str = "XYZ"):
    letter_to_index_dict = {"X": 0, "Y": 1, "Z": 2}

    def letter_to_sign(a):
        return 1 if a.isupper() else -1

    indexes = list(map(letter_to_index_dict.get, imu_orientation.upper()))
    signs = list(map(letter_to_sign, imu_orientation))
    return angular_velocity[:, indexes] * signs, accelerometer[:, indexes] * signs


def get_motion_data(metadata: str | Path | Dict[str, Any], imu_orientation: str = None) -> Dict[str, Any]:
    if isinstance(metadata, str | Path):
        metadata = extract_metadata_cached(metadata)

    timecode_unit = 1 / 1000.0
    frame_rate = metadata["metadata"]["frame_rate"]

    imu_records = metadata["imu_records"]
    raw_angular_velocity = np.array([rec["gyro"] for rec in imu_records]) * np.pi / 180.0  # deg to rad
    raw_accelerometer = np.array([rec["accl"] for rec in imu_records])

    if imu_orientation is None:
        imu_orientation = metadata["metadata"]["imu_orientation"]

    angular_velocity, accelerometer = remap_imu_coordinates(raw_angular_velocity, raw_accelerometer, imu_orientation)

    imu_data = IMUData(
        time=np.array([rec["timestamp_ms"] for rec in imu_records]) * timecode_unit,
        angular_velocity=angular_velocity,
        accelerometer=accelerometer,
    )
    num_imu_records = len(imu_data.time)
    measured_imu_sampling_rate = compute_sampling_rate(imu_data.time)
    duration_imu = num_imu_records / measured_imu_sampling_rate

    frame_records = metadata["frame_records"]
    frame_data = FrameData(
        time=np.array([rec["timestamp_ms"] for rec in frame_records]) * timecode_unit,
        exposure_time=np.array([rec["exposure_time"] for rec in frame_records]),
    )
    num_frames = len(frame_data.time)
    measured_frame_rate = compute_sampling_rate(frame_data.time)
    duration_frames = num_frames / measured_frame_rate

    frame_rate_factor = frame_rate / measured_frame_rate

    gps_records = metadata["gps_records"]
    gps_data = (
        None
        if gps_records is None
        else GPSTrack(
            time=np.array([rec["unix_timestamp"] for rec in gps_records]),
            lat=np.array([rec["lat"] for rec in gps_records]),
            lon=np.array([rec["lon"] for rec in gps_records]),
            speed=np.array([rec["speed"] for rec in gps_records]),
            track=np.array([rec["track"] for rec in gps_records]),
            altitude=np.array([rec["altitude"] for rec in gps_records]),
        )
    )

    # Ranges for diagnostics
    gyro = np.asarray(imu_data.angular_velocity, dtype=float)
    acc = np.asarray(imu_data.accelerometer, dtype=float)
    gmin = gyro.min(axis=0) if gyro.size else np.array([np.nan, np.nan, np.nan])
    gmax = gyro.max(axis=0) if gyro.size else np.array([np.nan, np.nan, np.nan])
    amin = acc.min(axis=0) if acc.size else np.array([np.nan, np.nan, np.nan])
    amax = acc.max(axis=0) if acc.size else np.array([np.nan, np.nan, np.nan])

    print(
        f"Frame rate from metadata: {frame_rate:.5f} Hz, "
        + f"\nFrame rate from records: {measured_frame_rate:.5f} Hz, "
        + f"\nIMU sampling rate: {measured_imu_sampling_rate:.5f} Hz"
        + f"\nIMU sampling rate (corrected): {measured_imu_sampling_rate * frame_rate_factor:.5f} Hz, "
        + f"\nDuration from metadata: {metadata['metadata']['total_time']} "
        + f"\nDuration (frames): {duration_frames:.3f} s"
        + f"\nDuration (24 Hz frame rate): {num_frames / frame_rate:.3f} s"
        + f"\nDuration (IMU): {duration_imu:.3f} s, "
        + f"\nDuration (IMU, from timecodes): {imu_data.time[-1] - imu_data.time[0]:.3f} s, "
        + f"\nDuration (IMU, corrected): {duration_imu / frame_rate_factor:.3f} s, "
        + f"\nFrame - IMU start delta: {frame_data.time[0] - imu_data.time[0]}, "
        + f"\nFrame - IMU end delta: {frame_data.time[-1] - imu_data.time[-1]}, "
        + f"\nIMU time range: [{float(imu_data.time[0]):.3f}, {float(imu_data.time[-1]):.3f}] s"
        + f"\nFrame time range: [{float(frame_data.time[0]):.3f}, {float(frame_data.time[-1]):.3f}] s"
        + "\nGyro ranges (x,y,z): "
        + f"[{gmin[0]:.3f},{gmax[0]:.3f}], [{gmin[1]:.3f},{gmax[1]:.3f}], [{gmin[2]:.3f},{gmax[2]:.3f}]"
        + f"\nGyro average: {np.mean(imu_data.angular_velocity, axis=0)}"
        + f"\nGyro integral: {np.mean(imu_data.angular_velocity, axis=0) * duration_imu}"
        + "\nAccel ranges (x,y,z): "
        + f"[{amin[0]:.3f},{amax[0]:.3f}], [{amin[1]:.3f},{amax[1]:.3f}], [{amin[2]:.3f},{amax[2]:.3f}]"
        + f"\nAccel average: {np.mean(imu_data.accelerometer, axis=0)}"
        + f"\nAccel integral: {np.mean(imu_data.accelerometer, axis=0) * duration_imu}"
        + ""
        if gps_data is None
        else (
            f"\nGPS time range: [{float(gps_data.time[0]):.3f}, {float(gps_data.time[-1]):.3f}] s"
            + f"\nGPS latitude range: [{gps_data.lat.min():.3f}, {gps_data.lat.max():.3f}]"
            + f"\nGPS longitude range: [{gps_data.lon.min():.3f}, {gps_data.lon.max():.3f}]"
            + f"\nGPS speed range: [{gps_data.speed.min():.3f}, {gps_data.speed.max():.3f}] m/s"
            + f"\nGPS track range: [{gps_data.track.min():.3f}, {gps_data.track.max():.3f}] deg"
            + f"\nGPS altitude range: [{gps_data.altitude.min():.3f}, {gps_data.altitude.max():.3f}] m"
        )
    )

    assert np.isclose(measured_frame_rate, frame_rate, atol=1e-3, rtol=1e-4)

    return dict(
        frame_rate=measured_frame_rate,
        imu_data=imu_data,
        frames=frame_data,
        gps_data=gps_data,
        imu_sampling_rate=measured_imu_sampling_rate,
        duration_frames=duration_frames,
        duration_imu=duration_imu,
        declared_frame_rate=frame_rate,
        frame_rate_factor=frame_rate_factor,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract gyroscope data from INSV files")
    parser.add_argument("input_file", help="Input INSV file path")
    parser.add_argument(
        "--use-exiftool",
        action="store_true",
        help="Force use of exiftool instead of telemetry-parser",
    )
    args = parser.parse_args()

    use_telemetry_parser = not args.use_exiftool
    method_name = "telemetry-parser" if use_telemetry_parser else "exiftool"

    print(f"\nExtracting metadata with {method_name}:")
    try:
        metadata = extract_metadata(args.input_file, use_telemetry_parser=use_telemetry_parser)
        for k, v in metadata["main"].items():
            print(f"{k}: {v}")

        motion_data = get_motion_data(metadata)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit(1)
