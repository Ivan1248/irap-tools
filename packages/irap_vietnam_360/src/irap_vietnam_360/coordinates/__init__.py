"""
Coordinate systems package for irap_vietnam_360.

This package provides utilities for working with different coordinate systems:
- ENU (East-North-Up): Local tangent plane coordinates
- Spherical: Geographic coordinates on Earth's surface
- Transformations: Converting between coordinate systems
"""

from .enu import latlon_to_enu, enu_to_latlon
