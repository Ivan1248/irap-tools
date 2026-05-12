import datetime as _dt
import json
import subprocess
from dataclasses import dataclass

from irap_video_cutting.gpx.gpx_io import Track


@dataclass(frozen=True)
class VideoMeta:
    duration_s: float
    creation_time_s_epoch: float | None


@dataclass(frozen=True)
class TimeAlignment:
    offset_s: float


def _parse_creation_time(raw: str | None) -> float | None:
    if not raw:
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = _dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_dt.timezone.utc)
    return dt.timestamp()


def extract_video_meta(path: str) -> VideoMeta:
    """
    Extract duration and creation time (if available) using ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-print_format", "json",
        "-show_entries", "format=duration:format_tags=creation_time",
        path,
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.strip()}")

    data = json.loads(proc.stdout or "{}")
    fmt = data.get("format", {})
    duration_s = float(fmt.get("duration", 0.0) or 0.0)
    tags = fmt.get("tags")
    creation_raw = tags.get("creation_time") if isinstance(tags, dict) else None
    creation_time_s_epoch = _parse_creation_time(creation_raw)
    return VideoMeta(duration_s=duration_s, creation_time_s_epoch=creation_time_s_epoch)


def compute_time_alignment(track: Track, meta: VideoMeta) -> TimeAlignment:
    times = track.times_s
    if not times or meta.duration_s <= 0.0:
        return TimeAlignment(offset_s=0.0)
    anchor_time_s = (
        meta.creation_time_s_epoch
        if meta.creation_time_s_epoch is not None
        else times[0]
    )
    return TimeAlignment(offset_s=-anchor_time_s)
