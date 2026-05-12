"""
ENU (East-North-Up) coordinate system utilities.

ENU coordinates are local tangent plane coordinates where:
- X: East (positive eastward)
- Y: North (positive northward)
- Z: Up (positive upward)

This follows ROS REP-103 geographic frame convention.
"""

import numpy as np
from typing import Tuple

# Earth radius in meters
EARTH_RADIUS = 6371000.0


def latlon_to_enu(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float,
    lon0: float,
    altitude: np.ndarray | None = None,
    altitude0: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon to local ENU coordinates following ROS REP-103.

    Args:
        lat: Latitude array in degrees
        lon: Longitude array in degrees
        lat0: Reference latitude in degrees
        lon0: Reference longitude in degrees

    Returns:
        Tuple of (east, north) coordinates in meters

    Note:
        Uses equirectangular projection approximation around the reference point.
        This is accurate for small distances (< 100km) from the reference point.
    """
    # Equirectangular approximation around origin
    R = EARTH_RADIUS
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    e = (lon_rad - lon0_rad) * np.cos(lat0_rad) * R  # East
    n = (lat_rad - lat0_rad) * R  # North
    u = altitude - altitude0 if altitude is not None else None  # Up

    return (e, n) if u is None else (e, n, u)


def enu_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    lat0: float,
    lon0: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert ENU coordinates back to lat/lon.

    Args:
        x: East coordinates in meters
        y: North coordinates in meters
        lat0: Reference latitude in degrees
        lon0: Reference longitude in degrees

    Returns:
        Tuple of (latitude, longitude) in degrees
    """
    R = EARTH_RADIUS
    lat0_rad = np.deg2rad(lat0)
    lon0_rad = np.deg2rad(lon0)

    lat_rad = y / R + lat0_rad
    lon_rad = x / (R * np.cos(lat0_rad)) + lon0_rad

    lat = np.rad2deg(lat_rad)
    lon = np.rad2deg(lon_rad)

    return lat, lon


def compute_course(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute course over ground following ROS REP-103.

    Yaw is zero when pointing east and increases counter-clockwise.
    This differs from traditional compass bearing (zero north, clockwise).

    Args:
        x: East coordinates in meters
        y: North coordinates in meters

    Returns:
        Course angles in radians
    """
    # Course over ground in radians
    dx = np.diff(x)
    dy = np.diff(y)
    heading = np.arctan2(dy,
                         dx)  # atan2(North, East) → 0 at East, CCW positive
    # Repeat last value to keep same length as positions
    return np.concatenate([heading, heading[-1:]])


def compute_speeds(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute speed magnitude from position data.

    Args:
        t: Time array in seconds
        x: East coordinates in meters
        y: North coordinates in meters

    Returns:
        Speed values in m/s
    """
    dt = np.diff(t)
    dt[dt <= 0] = np.nan  # avoid division by zero
    dx = np.diff(x)
    dy = np.diff(y)
    v = np.sqrt(dx**2 + dy**2) / dt
    # Simple forward fill for NaNs
    for i in range(1, len(v)):
        if not np.isfinite(v[i]):
            v[i] = v[i - 1]
    # Repeat last value to keep same length
    return np.concatenate([v, [v[-1]]])
