"""
Spherical coordinate system utilities for geographic calculations.

This module provides functions for calculations on Earth's surface using
spherical geometry approximations. These functions work with latitude/longitude
coordinates and assume a spherical Earth model.

lat is positive north, lon is positive east, in degrees.
altitude is positive up, in meters.
"""

import math
import numpy as np
from typing import List, Tuple

from .enu import EARTH_RADIUS


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    sphere_radius: float = EARTH_RADIUS,
) -> float:
    """
    Calculate the great circle distance between two points on the Earth.

    Args:
        lat1: Latitude of the first point in degrees
        lon1: Longitude of the first point in degrees
        lat2: Latitude of the second point in degrees
        lon2: Longitude of the second point in degrees
        sphere_radius: Radius of the sphere in meters (default: Earth radius)

    Returns:
        Distance between the two points in meters

    References:
        - https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return c * sphere_radius


def distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    ele1: float = 0.0,
    ele2: float = 0.0,
    sphere_radius: float = EARTH_RADIUS,
) -> float:
    """
    Calculate approximate distance between two points on the Earth, considering altitude.

    Args:
        lat1: Latitude of the first point in degrees
        lon1: Longitude of the first point in degrees
        lat2: Latitude of the second point in degrees
        lon2: Longitude of the second point in degrees
        ele1: Elevation of the first point in meters
        ele2: Elevation of the second point in meters
        sphere_radius: Radius of the sphere in meters

    Returns:
        Total distance including altitude in meters
    """
    horizontal_distance = haversine_distance(lat1, lon1, lat2, lon2, sphere_radius)
    return math.sqrt(horizontal_distance**2 + (ele2 - ele1) ** 2)


def compute_initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute initial bearing (forward azimuth) from point 1 to point 2.

    Args:
        lat1: Latitude of the first point in degrees
        lon1: Longitude of the first point in degrees
        lat2: Latitude of the second point in degrees
        lon2: Longitude of the second point in degrees

    Returns:
        Initial bearing in degrees [0, 360), where 0 is North, 90 is East, etc.

    Note:
        Based on spherical Earth model. For accurate results over long distances,
        consider using more sophisticated geodesic calculations.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)
    
    y = math.sin(d_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    brng = math.degrees(math.atan2(y, x))
    
    return (brng + 360.0) % 360.0


def compute_segment_bearings(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """
    Compute initial bearings for each segment between consecutive points.

    Args:
        latitudes: Array of latitudes in degrees
        longitudes: Array of longitudes in degrees

    Returns:
        Array of bearings in degrees. For N points, returns N-1 bearings.
        The last bearing duplicates the second-to-last for convenience.

    Note:
        For N track points there are N-1 segment bearings; for convenience 
        we return an array of length N where the last element duplicates 
        the last segment bearing.
    """
    if len(latitudes) < 2:
        return np.array([0.0])
    
    bearings = [
        compute_initial_bearing(lat1, lon1, lat2, lon2)
        for lat1, lon1, lat2, lon2 in zip(
            latitudes[:-1], longitudes[:-1],
            latitudes[1:], longitudes[1:]
        )
    ]
    
    # Duplicate last to match number of points
    bearings.append(bearings[-1])
    return np.array(bearings, dtype=float)


def calculate_distances_from_start(
    latitudes: np.ndarray, 
    longitudes: np.ndarray,
    sphere_radius: float = EARTH_RADIUS
) -> np.ndarray:
    """
    Calculate cumulative path distances from the starting point.

    Args:
        latitudes: Array of latitudes in degrees
        longitudes: Array of longitudes in degrees
        sphere_radius: Radius of the sphere in meters

    Returns:
        Array of cumulative distances in meters from the starting point
    """
    if len(latitudes) < 2:
        return np.array([0.0])
    
    distances = [0.0]
    prev_lat, prev_lon = latitudes[0], longitudes[0]
    
    for curr_lat, curr_lon in zip(latitudes[1:], longitudes[1:]):
        segment_distance = haversine_distance(
            prev_lat, prev_lon, curr_lat, curr_lon, sphere_radius
        )
        distances.append(distances[-1] + segment_distance)
        prev_lat, prev_lon = curr_lat, curr_lon
    
    return np.array(distances)
