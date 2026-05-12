"""
Shared CLI pipeline loop for video cutting entry points.

Both ``cut_manual`` and ``cut_webgis`` follow the same structure:
  index → loop stems → get duration → dry-run check → snap keyframes →
  compute out_dir → cut video + sidecars.

Callers supply two callbacks to handle the parts that differ:

  get_duration_and_context(stem, video_path, cuts)
      Return ``(duration_s, segments, ctx)`` or ``None`` to skip this video.
      *ctx* is opaque — it is forwarded unchanged to *cut_all*.

  cut_all(stem, video_path, segments, snapped_starts, out_dir, ctx)
      Cut the MP4 and any sidecar files.  Return ``True`` on success,
      ``False`` if an error occurred (already printed to stderr).
"""

import argparse
import os
import shutil
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

from irap_video_cutting.shared.models import Segment
from irap_video_cutting.shared.stem_index import resolve_stems
from irap_video_cutting.shared.video_keyframes import snap_segments_to_keyframes


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add the --input-dir / --output-dir / --list-file / --dry-run arguments."""
    p.add_argument("--input-dir", required=True, help="Directory containing the input files.")
    p.add_argument("--output-dir", required=True, help="Directory where cut segments will be written.")
    p.add_argument(
        "--list-file",
        help="Path to a text file with lines: videoStem <whitespace> mm:ss [mm:ss ...]. "
        "If omitted, the list is read from stdin.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print segments without writing files.")
    p.add_argument("--clear-output", action="store_true", help="Clear the output directory before processing.")


def read_cut_list(list_file: Optional[str]) -> str:
    """Read the cut list from *list_file* or stdin."""
    if list_file:
        with open(list_file, "r", encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def run_pipeline(
    input_dir: str,
    output_dir: str,
    stem_index: Dict[str, str],
    mapping: Dict[str, List[float]],
    dry_run: bool,
    get_duration_and_context: Callable[
        [str, str, List[float]], Optional[Tuple[float, List[Segment], Any]]
    ],
    cut_all: Callable[
        [str, str, List[Segment], List[Optional[float]], str, Any], bool
    ],
    clear_output: bool = False,
    tqdm_desc: str = "Cutting",
) -> int:
    """
    Process every stem in *mapping*, returning 0 on full success or 1 if any
    video produced an error.
    """
    from tqdm import tqdm

    if clear_output and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    resolved, unresolved = resolve_stems(stem_index, mapping)
    for prefix in unresolved:
        print(f"No file found for stem {prefix!r} in {input_dir!r}", file=sys.stderr)
    had_error = bool(unresolved)

    for stem, video_path, cuts in tqdm(resolved, desc=tqdm_desc, unit="video"):
        result = get_duration_and_context(stem, video_path, cuts)
        if result is None:
            had_error = True
            continue
        duration_s, segments, ctx = result

        if not segments:
            print(f"No segments for {stem!r} (duration {duration_s:.3f}s)", file=sys.stderr)
            continue

        if dry_run:
            for seg in segments:
                print(f"{stem}\t{seg.start_s:.3f}\t{seg.end_s:.3f}")
            continue

        snapped_starts = snap_segments_to_keyframes(video_path, segments)

        rel_dir = os.path.dirname(os.path.relpath(video_path, input_dir))
        out_dir = os.path.join(output_dir, rel_dir)

        if not cut_all(stem, video_path, segments, snapped_starts, out_dir, ctx):
            had_error = True

    return 1 if had_error else 0
