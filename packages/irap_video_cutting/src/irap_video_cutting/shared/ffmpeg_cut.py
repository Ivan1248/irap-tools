"""
Shared ffmpeg helpers for stream-copy video cutting.

Both gpx and webgis use this module.  The two callers differ in
two optional flags:

  - movflags_faststart  (webgis: True  — puts moov atom first for HTML5 playback)
  - strip_audio         (webgis: True  — WebGIS viewer does not use audio)
  - original_creation_time_s_epoch  (gpx: float — embeds creation_time metadata)
"""

import datetime as _dt
import os
import subprocess
from typing import Iterable, List, Optional

from irap_video_cutting.shared.models import Segment


def build_ffmpeg_cmd(
    video_path: str,
    segment: Segment,
    output_path: str,
    creation_time: str | None = None,
    movflags_faststart: bool = False,
    strip_audio: bool = False,
) -> List[str]:
    """Construct an ffmpeg stream-copy command to cut a single segment."""
    start = max(0.0, segment.start_s)
    duration = max(0.0, segment.end_s - segment.start_s)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
    ]
    if movflags_faststart:
        cmd += ["-movflags", "faststart"]
    cmd += ["-c", "copy"]
    if strip_audio:
        cmd.append("-an")
    if creation_time:
        cmd += ["-metadata", f"creation_time={creation_time}"]
    cmd.append(output_path)
    return cmd


def ffmpeg_cut_segments(
    video_path: str,
    segments: Iterable[Segment],
    output_dir: str,
    base_name: str = "segment",
    snapped_starts_s: Optional[List[Optional[float]]] = None,
    original_creation_time_s_epoch: Optional[float] = None,
    movflags_faststart: bool = False,
    strip_audio: bool = False,
) -> None:
    """Cut multiple segments from a video, writing one MP4 per segment."""
    os.makedirs(output_dir, exist_ok=True)

    for idx, seg in enumerate(segments, start=1):
        output_path = os.path.join(output_dir, f"{base_name}_{idx}.mp4")

        snapped = snapped_starts_s[idx - 1] if snapped_starts_s else None
        start_offset = snapped if snapped is not None else seg.start_s
        ffmpeg_seg = Segment(start_s=start_offset, end_s=seg.end_s)

        creation_time_str = None
        if original_creation_time_s_epoch is not None:
            seg_start_epoch = original_creation_time_s_epoch + start_offset
            dt = _dt.datetime.fromtimestamp(seg_start_epoch, tz=_dt.timezone.utc)
            creation_time_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        cmd = build_ffmpeg_cmd(
            video_path, ffmpeg_seg, output_path,
            creation_time=creation_time_str,
            movflags_faststart=movflags_faststart,
            strip_audio=strip_audio,
        )
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {output_path}: {proc.stderr.strip()}")
