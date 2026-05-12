"""
Shared prepare/execute logic for WebGIS video cutting.

Both the CLI and GUI entry points delegate to these functions, which
contain the pure business logic and raise on error.
"""

import os

from irap_video_cutting.shared.ffmpeg_cut import ffmpeg_cut_segments
from irap_video_cutting.shared.manual_cuts import cuts_to_segments
from irap_video_cutting.shared.models import Segment
from irap_video_cutting.webgis.webgis_cut import cut_webgis_segments
from irap_video_cutting.webgis.webgis_io import WebgisTrack, parse_json, parse_sql_meta


def prepare_cut(
    video_path: str,
    cuts: list[float],
) -> tuple[float, list[Segment], WebgisTrack]:
    """Parse companion JSON, validate GPS points, build segments. Raises on error."""
    json_path = os.path.splitext(video_path)[0] + ".json"
    track = parse_json(json_path)
    if not track.points:
        raise ValueError(f"No GPS points in {json_path!r}")
    duration_s = track.points[-1].time_s
    segments = cuts_to_segments(cuts, duration_s)
    return duration_s, segments, track


def execute_cut(
    stem: str,
    video_path: str,
    segments: list[Segment],
    snapped_starts: list[float | None],
    out_dir: str,
    track: WebgisTrack,
) -> None:
    """Cut the MP4 and all WebGIS sidecar files. Raises on error."""
    ffmpeg_cut_segments(
        video_path=video_path,
        segments=segments,
        output_dir=out_dir,
        base_name=stem,
        snapped_starts_s=snapped_starts,
        movflags_faststart=True,
        strip_audio=True,
    )

    sql_path = os.path.splitext(video_path)[0] + ".sql"
    sql_meta = parse_sql_meta(sql_path)

    cut_webgis_segments(
        track=track,
        sql_meta=sql_meta,
        segments=segments,
        output_dir=out_dir,
        base_name=stem,
        snapped_starts_s=snapped_starts,
    )
