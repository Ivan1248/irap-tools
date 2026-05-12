"""
Cut WebGIS sidecar files (.json, .geojson, .sql) into segments.
"""

import os
from typing import List, Optional

import numpy as np

from irap_video_cutting.shared.models import Segment
from irap_video_cutting.webgis.webgis_io import GpsPoint, SqlMeta, WebgisTrack, save_geojson, save_json, save_sql


def cut_webgis_segments(
    track: WebgisTrack,
    sql_meta: SqlMeta,
    segments: List[Segment],
    output_dir: str,
    base_name: str,
    snapped_starts_s: Optional[List[Optional[float]]] = None,
) -> None:
    """
    Cut JSON, GeoJSON and SQL sidecar files into one set per segment.

    GPS points at the exact cut boundaries are synthesised by linear
    interpolation so that every output segment starts and ends at a precisely
    known coordinate.  Timestamps in the output .json are re-zeroed so that
    the first point of every segment is at time 0.
    """
    os.makedirs(output_dir, exist_ok=True)

    times = np.array(track.times_s, dtype=float)
    lons = np.array(track.lons, dtype=float)
    lats = np.array(track.lats, dtype=float)

    for idx, seg in enumerate(segments):
        actual_start_s: float = (
            snapped_starts_s[idx]
            if (snapped_starts_s and snapped_starts_s[idx] is not None)
            else seg.start_s
        )
        actual_end_s: float = seg.end_s

        lon_start = float(np.interp(actual_start_s, times, lons))
        lat_start = float(np.interp(actual_start_s, times, lats))
        lon_end = float(np.interp(actual_end_s, times, lons))
        lat_end = float(np.interp(actual_end_s, times, lats))

        seg_times: List[float] = [actual_start_s]
        seg_lons: List[float] = [lon_start]
        seg_lats: List[float] = [lat_start]

        for p in track.points:
            if actual_start_s < p.time_s < actual_end_s:
                seg_times.append(p.time_s)
                seg_lons.append(p.lon)
                seg_lats.append(p.lat)

        seg_times.append(actual_end_s)
        seg_lons.append(lon_end)
        seg_lats.append(lat_end)

        seg_times_zeroed = [t - actual_start_s for t in seg_times]

        seg_points = [
            GpsPoint(time_s=t, lon=lo, lat=la)
            for t, lo, la in zip(seg_times_zeroed, seg_lons, seg_lats)
        ]
        seg_track = WebgisTrack(timeoffset=0.0, points=seg_points)
        coords_wgs84 = [(p.lon, p.lat) for p in seg_points]

        output_stem = f"{base_name}_{idx + 1}"

        save_json(seg_track, os.path.join(output_dir, f"{output_stem}.json"))

        save_geojson(
            f"{output_stem}.mp4",
            coords_wgs84,
            os.path.join(output_dir, f"{output_stem}.geojson"),
        )

        seg_sql_meta = SqlMeta(
            table=sql_meta.table,
            filename=output_stem,
            recording_start=sql_meta.recording_start,
            uploaded_by=sql_meta.uploaded_by,
            gis_map_id=sql_meta.gis_map_id,
        )
        save_sql(seg_sql_meta, coords_wgs84, os.path.join(output_dir, f"{output_stem}.sql"))
