"""
Shared prepare/execute logic for manual (GPX) video cutting.

Both the CLI and GUI entry points delegate to these functions, which
contain the pure business logic and raise on error.
"""

import os

from irap_video_cutting.gpx.gpx_cut import cut_gpx_segments
from irap_video_cutting.gpx.gpx_io import parse_gpx
from irap_video_cutting.gpx.video_meta import VideoMeta, compute_time_alignment, extract_video_meta
from irap_video_cutting.shared.ffmpeg_cut import ffmpeg_cut_segments
from irap_video_cutting.shared.manual_cuts import cuts_to_segments
from irap_video_cutting.shared.models import Segment


def prepare_cut(
    video_path: str,
    cuts: list[float],
) -> tuple[float, list[Segment], VideoMeta]:
    """Extract video metadata, build segments. Raises on ffprobe failure."""
    meta = extract_video_meta(video_path)
    segments = cuts_to_segments(cut_times_s=cuts, duration_s=meta.duration_s)
    return meta.duration_s, segments, meta


def execute_cut(
    stem: str,
    video_path: str,
    segments: list[Segment],
    snapped_starts: list[float | None],
    out_dir: str,
    meta: VideoMeta,
) -> bool:
    """
    Cut the MP4 and companion GPX file.

    Returns ``True`` if a GPX sidecar was found and cut, ``False`` if no GPX
    file existed (the MP4 cut still succeeds).  Raises on error.
    """
    ffmpeg_cut_segments(
        video_path=video_path,
        segments=segments,
        output_dir=out_dir,
        base_name=stem,
        original_creation_time_s_epoch=meta.creation_time_s_epoch,
        snapped_starts_s=snapped_starts,
    )

    gpx_path = os.path.splitext(video_path)[0] + ".gpx"
    if not os.path.isfile(gpx_path):
        return False

    alignment = compute_time_alignment(parse_gpx(gpx_path), meta)
    cut_gpx_segments(
        gpx_path=gpx_path,
        segments=segments,
        alignment=alignment,
        output_dir=out_dir,
        base_name=stem,
        snapped_starts_s=snapped_starts,
    )
    return True
