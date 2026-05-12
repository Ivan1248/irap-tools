"""
Parse and write the three WebGIS sidecar formats:
  .json    – video-local GPS timestamps + WGS84 coordinates
  .geojson – GeoJSON FeatureCollection / LineString (WGS84)
  .sql     – PostgreSQL INSERT with PostGIS LINESTRING in EPSG:3857
"""

import json
import math
import re
from dataclasses import dataclass, field
from typing import List, Tuple

# WGS84 semi-major axis, used as the Earth radius for EPSG:3857 (Web Mercator).
_R = 6378137.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class GpsPoint:
    time_s: float
    lon: float
    lat: float


@dataclass
class WebgisTrack:
    timeoffset: float
    points: List[GpsPoint] = field(default_factory=list)

    @property
    def times_s(self) -> List[float]:
        return [p.time_s for p in self.points]

    @property
    def lons(self) -> List[float]:
        return [p.lon for p in self.points]

    @property
    def lats(self) -> List[float]:
        return [p.lat for p in self.points]


@dataclass(frozen=True)
class SqlMeta:
    """Fields extracted from the INSERT statement (excluding the_geom)."""
    table: str
    filename: str
    recording_start: str
    uploaded_by: str
    gis_map_id: str


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------

def _wgs84_to_web_mercator(lon: float, lat: float) -> Tuple[float, float]:
    """Project WGS84 (lon, lat) degrees to EPSG:3857 (x, y) metres."""
    x = math.radians(lon) * _R
    y = math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0)) * _R
    return x, y


# ---------------------------------------------------------------------------
# JSON (.json)
# ---------------------------------------------------------------------------

def parse_json(path: str) -> WebgisTrack:
    """
    Parse a WebGIS JSON file.

    Format:
        [{"timeoffset": 0.0},
         {"time": 0.0, "coordinates": [lon, lat]},
         {"time": 1.7, "coordinates": [lon, lat]}, ...]
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    timeoffset = 0.0
    points: List[GpsPoint] = []

    for entry in data:
        if "timeoffset" in entry:
            timeoffset = float(entry["timeoffset"])
        elif "time" in entry and "coordinates" in entry:
            lon, lat = entry["coordinates"]
            points.append(GpsPoint(
                time_s=float(entry["time"]),
                lon=float(lon),
                lat=float(lat),
            ))

    return WebgisTrack(timeoffset=timeoffset, points=points)


def save_json(track: WebgisTrack, path: str) -> None:
    """Write a WebgisTrack back to the .json format."""
    data: List[dict] = [{"timeoffset": track.timeoffset}]
    for p in track.points:
        data.append({"time": p.time_s, "coordinates": [p.lon, p.lat]})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# SQL (.sql)
# ---------------------------------------------------------------------------

def parse_sql_meta(path: str) -> SqlMeta:
    """
    Extract the five scalar fields from the INSERT statement.
    The the_geom field is ignored; it will be recomputed from the .json data.

    Expected form:
        INSERT INTO schema.table(filename,recording_start,uploaded_by,gis_map_id,the_geom)
        VALUES ('...','...','...','...',ST_GeomFromText('LINESTRING(...)',3857));
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    m_table = re.search(r"INSERT\s+INTO\s+(\S+)\s*\(", text, re.IGNORECASE)
    table = m_table.group(1) if m_table else "eurorap.vi_video_gps_logs"

    m_values = re.search(
        r"VALUES\s*\(\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'\s*,\s*'([^']*)'",
        text,
        re.IGNORECASE,
    )
    if not m_values:
        raise ValueError(f"Could not parse VALUES clause in {path!r}")

    filename, recording_start, uploaded_by, gis_map_id = m_values.groups()
    return SqlMeta(
        table=table,
        filename=filename,
        recording_start=recording_start,
        uploaded_by=uploaded_by,
        gis_map_id=gis_map_id,
    )


def save_sql(meta: SqlMeta, coords_wgs84: List[Tuple[float, float]], path: str) -> None:
    """
    Write a new INSERT statement, projecting WGS84 coords to EPSG:3857.

    coords_wgs84: list of (lon, lat) pairs in WGS84 degrees.
    """
    projected = [_wgs84_to_web_mercator(lon, lat) for lon, lat in coords_wgs84]
    linestring = ",".join(f"{x} {y}" for x, y in projected)
    sql = (
        f"INSERT INTO {meta.table}"
        f"(filename,recording_start,uploaded_by,gis_map_id,the_geom)"
        f" VALUES ('{meta.filename}','{meta.recording_start}',"
        f"'{meta.uploaded_by}','{meta.gis_map_id}',"
        f"ST_GeomFromText('LINESTRING({linestring})',3857));\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(sql)


# ---------------------------------------------------------------------------
# GeoJSON (.geojson)
# ---------------------------------------------------------------------------

def save_geojson(
    filename_mp4: str,
    coords_wgs84: List[Tuple[float, float]],
    path: str,
) -> None:
    """
    Write a GeoJSON FeatureCollection with a single LineString feature.

    filename_mp4: e.g. "VID_xxx_1.mp4" – stored in the FILENAME property.
    coords_wgs84: list of (lon, lat) pairs in WGS84 degrees.
    """
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"FILENAME": filename_mp4},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon, lat] for lon, lat in coords_wgs84],
                },
            }
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, separators=(",", ":"))
