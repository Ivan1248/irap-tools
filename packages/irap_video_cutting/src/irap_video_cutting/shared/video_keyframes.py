"""
Detect keyframe (I-frame) timestamps in a video file using ffprobe.

Used to snap segment start times to the nearest preceding keyframe so that
stream-copy ffmpeg cuts start cleanly without a black-frame artefact.
"""

import bisect
import json
import subprocess
import sys
from typing import List, Optional


def get_keyframe_timestamps(video_path: str) -> List[float]:
    """Return a sorted list of keyframe (I-frame) timestamps in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v",
        "-skip_frame", "nokey",
        "-show_entries", "frame=pkt_pts_time",
        "-of", "json",
        video_path,
    ]
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {proc.stderr.strip()}")

    data = json.loads(proc.stdout or "{}")
    frames = data.get("frames", [])
    timestamps = [float(f["pkt_pts_time"]) for f in frames if "pkt_pts_time" in f]
    return sorted(timestamps)


def snap_to_previous_keyframe(
    requested_time_s: float, keyframe_timestamps: List[float]
) -> float:
    """Return the timestamp of the nearest keyframe at or before requested_time_s."""
    if not keyframe_timestamps:
        return requested_time_s
    idx = bisect.bisect_right(keyframe_timestamps, requested_time_s) - 1
    return keyframe_timestamps[idx] if idx >= 0 else requested_time_s


def snap_segments_to_keyframes(
    video_path: str,
    segments: list,
) -> List[Optional[float]]:
    """
    Return snapped start times for each segment, or ``None`` per segment on failure.

    On success each entry is the nearest preceding keyframe timestamp for the
    segment's requested start time.  If keyframe detection fails the function
    prints a warning to stderr and returns a list of ``None`` values so callers
    can fall back to the original requested times.
    """
    try:
        keyframes = get_keyframe_timestamps(video_path)
        return [snap_to_previous_keyframe(seg.start_s, keyframes) for seg in segments]
    except Exception as e:  # noqa: BLE001
        print(
            f"Keyframe detection failed for {video_path!r}: {e}. Falling back to requested times.",
            file=sys.stderr,
        )
        return [None] * len(segments)
