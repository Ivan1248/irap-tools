import sys
from typing import Dict, List

from irap_video_cutting.shared.models import Segment


def parse_timecode_to_seconds(text: str) -> float:
    """
    Parse a timecode string into seconds.

    Supported formats:
    - ss
    - mm:ss
    - hh:mm:ss
    """
    parts = text.strip().split(":")
    if not parts or any(not p.isdigit() for p in parts):
        raise ValueError(f"Invalid timecode: {text!r}")

    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = map(int, parts)
        return float(minutes * 60 + seconds)
    if len(parts) == 3:
        hours, minutes, seconds = map(int, parts)
        return float(hours * 3600 + minutes * 60 + seconds)

    raise ValueError(f"Unsupported timecode format: {text!r}")


def parse_pasted_table(text: str) -> Dict[str, List[float]]:
    """
    Parse pasted rows of the form:

        videoStem  mm:ss  mm:ss ...

    Columns may be separated by any whitespace (spaces, tabs, or mixed).
    Returns a mapping from video stem to a sorted list of unique cut times (seconds).
    Prints warnings to stderr for lines with invalid timecode format.
    """
    mapping: Dict[str, List[float]] = {}

    for line_num, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        stem = tokens[0]
        time_tokens = tokens[1:]
        if not time_tokens:
            print(f"Warning: Line {line_num} has stem '{stem}' but no timestamps", file=sys.stderr)
            continue

        cuts = []
        for tok in time_tokens:
            try:
                cuts.append(parse_timecode_to_seconds(tok))
            except ValueError as e:
                print(f"Warning: Line {line_num}, timestamp '{tok}': {e}", file=sys.stderr)

        if cuts:
            mapping[stem] = sorted(set(mapping.get(stem, []) + cuts))

    return mapping


def cuts_to_segments(
    cut_times_s: List[float],
    duration_s: float,
) -> List[Segment]:
    """
    Convert a list of manual cut times into [start, end) segments.
    """
    if duration_s <= 0.0:
        return []
    boundaries = sorted({0.0, duration_s, *(t for t in cut_times_s if 0.0 < t < duration_s)})
    return [Segment(start_s=s, end_s=e) for s, e in zip(boundaries, boundaries[1:])]
