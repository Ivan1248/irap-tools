"""
CLI entry point: cut WebGIS video sets (MP4 + JSON + GeoJSON + SQL) at
timestamps specified in a text cut list (same format as cut_manual).
"""

import argparse
import sys
from typing import Any, List, Optional

from irap_video_cutting.shared.cli_runner import add_common_args, read_cut_list, run_pipeline
from irap_video_cutting.shared.manual_cuts import parse_pasted_table
from irap_video_cutting.shared.models import Segment
from irap_video_cutting.shared.stem_index import build_stem_index
from irap_video_cutting.webgis.pipeline import execute_cut, prepare_cut


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cut WebGIS video sets (MP4+JSON+GeoJSON+SQL) using cut times from a text list."
    )
    add_common_args(p)
    return p.parse_args(argv)


def _get_duration_and_context(
    stem: str, video_path: str, cuts: List[float]
) -> Optional[tuple]:
    try:
        return prepare_cut(video_path, cuts)
    except Exception as e:  # noqa: BLE001
        print(f"Prepare failed for {stem!r}: {e}", file=sys.stderr)
        return None


def _cut_all(
    stem: str,
    video_path: str,
    segments: List[Segment],
    snapped_starts: List[Optional[float]],
    out_dir: str,
    ctx: Any,
) -> bool:
    try:
        execute_cut(stem, video_path, segments, snapped_starts, out_dir, ctx)
    except Exception as e:  # noqa: BLE001
        print(f"Cut failed for {stem!r}: {e}", file=sys.stderr)
        return False
    return True


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    table_text = read_cut_list(args.list_file)
    mapping = parse_pasted_table(table_text)
    if not mapping:
        print("No valid rows found in cut list.", file=sys.stderr)
        return 1

    try:
        stem_index = build_stem_index(args.input_dir, extensions={".mp4"})
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    return run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        stem_index=stem_index,
        mapping=mapping,
        dry_run=args.dry_run,
        get_duration_and_context=_get_duration_and_context,
        cut_all=_cut_all,
        clear_output=args.clear_output,
        tqdm_desc="Cutting",
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
