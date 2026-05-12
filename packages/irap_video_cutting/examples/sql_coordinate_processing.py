"""
Example: SQL Coordinate Processing for WebGIS Cutting

This demonstrates how WGS84 GPS coordinates (lon/lat in degrees)
are converted to EPSG:3857 Web Mercator coordinates (x/y in meters)
for the PostGIS database INSERT statement.
"""

import math
import json

# Earth's semi-major axis (WGS84 radius)
R = 6378137.0

def wgs84_to_web_mercator(lon: float, lat: float) -> tuple:
    """Convert WGS84 (degrees) to Web Mercator EPSG:3857 (meters)."""
    x = math.radians(lon) * R
    y = math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)) * R
    return x, y


# ============================================================================
# EXAMPLE 1: Simple conversion
# ============================================================================
print("=" * 70)
print("EXAMPLE 1: Single Point Conversion")
print("=" * 70)

lon, lat = 105.9239067, 21.3655649
x, y = wgs84_to_web_mercator(lon, lat)

print(f"\nInput (WGS84 degrees):")
print(f"  Longitude: {lon}°")
print(f"  Latitude:  {lat}°")

print(f"\nOutput (Web Mercator meters):")
print(f"  X (Easting):  {x:.6f} m")
print(f"  Y (Northing): {y:.6f} m")

print(f"\nFormulas:")
print(f"  x = lon_radians × R = {math.radians(lon):.8f} × {R}")
print(f"    = {x:.6f} m")
print(f"  y = ln(tan(π/4 + lat_radians/2)) × R")
print(f"    = ln(tan({math.pi/4 + math.radians(lat)/2:.8f})) × {R}")
print(f"    = {y:.6f} m")


# ============================================================================
# EXAMPLE 2: Full segment with interpolation
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 2: Segment Boundary with Interpolation")
print("=" * 70)

# Load actual data from the test
original = json.load(open('input_webgis/VID_20250104_085135_00_008_20251122224908.json'))
points = [p for p in original if 'time' in p]

# Get points around the 20s boundary
boundary_time = 20.0
before_points = [p for p in points if p['time'] < boundary_time]
after_points = [p for p in points if p['time'] > boundary_time]

if before_points and after_points:
    last_before = before_points[-1]
    first_after = after_points[0]
    
    print(f"\nSegment 1 ends at: time={last_before['time']}s")
    print(f"  WGS84: lon={last_before['coordinates'][0]}, lat={last_before['coordinates'][1]}")
    
    print(f"Segment 2 starts at: time={first_after['time']}s")
    print(f"  WGS84: lon={first_after['coordinates'][0]}, lat={first_after['coordinates'][1]}")
    
    # Linear interpolation at 20s boundary
    t0, t1 = last_before['time'], first_after['time']
    lon0, lat0 = last_before['coordinates']
    lon1, lat1 = first_after['coordinates']
    
    # Interpolate
    alpha = (boundary_time - t0) / (t1 - t0)
    lon_interp = lon0 + alpha * (lon1 - lon0)
    lat_interp = lat0 + alpha * (lat1 - lat0)
    
    print(f"\nInterpolation at {boundary_time}s:")
    print(f"  α = ({boundary_time} - {t0}) / ({t1} - {t0}) = {alpha:.6f}")
    print(f"  lon = {lon0} + {alpha:.6f} × ({lon1} - {lon0})")
    print(f"      = {lon_interp:.10f}")
    print(f"  lat = {lat0} + {alpha:.6f} × ({lat1} - {lat0})")
    print(f"      = {lat_interp:.10f}")
    
    x_interp, y_interp = wgs84_to_web_mercator(lon_interp, lat_interp)
    print(f"\nConverted to Web Mercator:")
    print(f"  X = {x_interp:.6f} m")
    print(f"  Y = {y_interp:.6f} m")


# ============================================================================
# EXAMPLE 3: Build actual SQL LINESTRING
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 3: Building SQL LINESTRING from Multiple Points")
print("=" * 70)

# Use first 5 points from segment 1
cut1 = json.load(open('output_webgis_test/VID_20250104_085135_00_008_20251122224908_1.json'))
cut1_points = [p for p in cut1 if 'time' in p][:5]

print(f"\nConverting {len(cut1_points)} GPS points to SQL:")

linestring_coords = []
for i, point in enumerate(cut1_points):
    lon, lat = point['coordinates']
    x, y = wgs84_to_web_mercator(lon, lat)
    linestring_coords.append(f"{x} {y}")
    
    print(f"\nPoint {i+1}:")
    print(f"  Time: {point['time']}s")
    print(f"  WGS84: [{lon}, {lat}]")
    print(f"  Web Mercator: {x:.2f} {y:.2f}")

linestring = ",".join(linestring_coords)
print(f"\nResulting LINESTRING SQL fragment:")
print(f"  LINESTRING({linestring})")

print(f"\nFull INSERT statement:")
filename = "VID_20250104_085135_00_008_20251122224908_1"
sql = (
    f"INSERT INTO eurorap.vi_video_gps_logs"
    f"(filename,recording_start,uploaded_by,gis_map_id,the_geom)\n"
    f"VALUES ('{filename}','2025-01-04 01:51:35','','PQPcRP9hLSBJQLMxpio668',\n"
    f"ST_GeomFromText('LINESTRING({linestring})',3857));"
)
print(sql)


# ============================================================================
# EXAMPLE 4: Verify against actual output
# ============================================================================
print("\n" + "=" * 70)
print("EXAMPLE 4: Verification Against Actual Output")
print("=" * 70)

with open('output_webgis_test/VID_20250104_085135_00_008_20251122224908_1.sql') as f:
    actual_sql = f.read()

# Extract first coordinate pair from actual SQL
import re
match = re.search(r"LINESTRING\(([^)]+)\)", actual_sql)
if match:
    linestring_content = match.group(1)
    coords = linestring_content.split(',')
    first_coord = coords[0].strip()
    last_coord = coords[-1].strip()
    
    print(f"\nActual SQL from output_webgis_test/")
    print(f"  First coordinate: {first_coord}")
    print(f"  Last coordinate:  {last_coord}")
    print(f"  Total coordinates: {len(coords)}")
    
    print(f"\nOur computed first coordinate: {linestring_coords[0]}")
    print(f"Match: {'✓ YES' if linestring_coords[0] == first_coord else '✗ NO'}")
