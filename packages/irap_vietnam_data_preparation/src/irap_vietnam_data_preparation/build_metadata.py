"""Build the IRAP-Vietnam metadata directory from parsed coding tables.

Usage:
    python irap_vietnam_data_preparation/build_metadata.py <data_dir>

Reads (under <data_dir>):
    _raw/attribute_metadata.json
    _work/rows.parquet                    (from parse_coding_tables.py)
    images/<video_dir>/...                (nested dir from extract_images.py)

Writes (into <data_dir>/ directly):
    segment_id_to_data_paths_rel.json
    segment_id_to_road_data.json
    road_id_to_segment_id_sequence.json
    attribute_metadata.json               (copy of _raw/attribute_metadata.json)

Writes (into <data_dir>/_work/):
    build_report.json                     (counts, warnings)

Rows are matched to images **by seg_id**: every ``*.png`` under ``images/`` is
indexed by the integer in its ``_seg<N>.png`` suffix, and each parquet row is
joined to its seg_id. If exactly one image has that seg_id, it is used. Rows
with no matching image are dropped. The image's parent folder name is compared
to the row's ``section``; mismatches are recorded (``prefix_mismatch``) but
not dropped — they happen because the coding-table "Section" cell and the
RAR-side video-folder name are independently authored.

Within each section, the sequence is sorted by Distance ascending. The
adjacency invariant (distance step == 0.02 km between consecutive
segments) is checked and violations reported in the build_report.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
import typing as T

import pandas as pd

import layout


SEGMENT_LENGTH_KM = 0.02

# Captures the integer N in any "..._segN.png" filename.
SEG_ID_FROM_FILENAME_RE = re.compile(r"_seg(\d+)\.png$", re.IGNORECASE)


def index_images_by_seg_id(images_dir: Path) -> dict[str, list[Path]]:
    """Walk ``images_dir`` recursively, indexing ``*.png`` files by seg_id.

    Keys are decimal seg_id strings (matching the parquet's ``seg_id`` column).
    Values are paths relative to ``images_dir.parent`` so they begin with
    ``images/<video_dir>/<basename>.png``.
    """
    index: dict[str, list[Path]] = defaultdict(list)
    data_root = images_dir.parent
    for path in images_dir.rglob("*.png"):
        m = SEG_ID_FROM_FILENAME_RE.search(path.name)
        if m is None:
            continue
        seg_id = str(int(m.group(1)))  # normalize away leading zeros
        index[seg_id].append(path.relative_to(data_root))
    return index


def match_rows_to_images(
    df: pd.DataFrame,
    seg_id_to_image_paths: dict[str, list[Path]],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, T.Any]]:
    """Match each parquet row to an image by seg_id.

    Returns ``(kept_df, seg_id_to_relpath, stats)`` where ``stats`` carries
    per-row counters and example lists for the build report.

    Resolution policy when a seg_id matches >1 image:
        1. Prefer paths whose parent folder name equals ``row.section``.
        2. Otherwise prefer paths whose parent folder name is a prefix of
           ``row.section`` (or vice versa).
        3. Otherwise pick the lexicographically smallest path and flag as
           ``ambiguous_image_match``.
    """

    kept_indices: list[int] = []
    seg_to_path: dict[str, str] = {}
    no_image_examples: list[str] = []
    prefix_mismatch_examples: list[dict] = []
    ambiguous_examples: list[dict] = []
    num_no_image = 0
    num_prefix_mismatch = 0
    num_ambiguous = 0

    for idx, row in df.iterrows():
        seg_id = str(row["seg_id"])
        section = str(row["section"])
        candidates = seg_id_to_image_paths.get(seg_id, [])
        if not candidates:
            num_no_image += 1
            if len(no_image_examples) < 20:
                no_image_examples.append(seg_id)
            continue

        if len(candidates) == 1:
            chosen = candidates[0]
        else:
            exact = [p for p in candidates if p.parent.name == section]
            if len(exact) == 1:
                chosen = exact[0]
            elif exact:
                chosen = sorted(exact)[0]
                num_ambiguous += 1
                if len(ambiguous_examples) < 20:
                    ambiguous_examples.append({
                        "seg_id": seg_id, "section": section,
                        "candidates": [str(p) for p in candidates],
                    })
            else:
                related = [
                    p for p in candidates
                    if p.parent.name.startswith(section) or section.startswith(p.parent.name)
                ]
                if len(related) == 1:
                    chosen = related[0]
                else:
                    chosen = sorted(candidates)[0]
                    num_ambiguous += 1
                    if len(ambiguous_examples) < 20:
                        ambiguous_examples.append({
                            "seg_id": seg_id, "section": section,
                            "candidates": [str(p) for p in candidates],
                        })

        prefix = chosen.parent.name
        if prefix != section:
            num_prefix_mismatch += 1
            if len(prefix_mismatch_examples) < 20:
                prefix_mismatch_examples.append({
                    "seg_id": seg_id,
                    "section_in_table": section,
                    "prefix_in_image": prefix,
                })

        seg_to_path[seg_id] = chosen.as_posix()
        kept_indices.append(idx)

    kept_df = df.loc[kept_indices].copy()
    stats = {
        "num_rows_no_image": num_no_image,
        "num_rows_prefix_mismatch": num_prefix_mismatch,
        "num_rows_ambiguous_image_match": num_ambiguous,
        "no_image_examples": no_image_examples,
        "prefix_mismatch_examples": prefix_mismatch_examples,
        "ambiguous_image_examples": ambiguous_examples,
    }
    return kept_df, seg_to_path, stats


def build_segment_id_to_data_paths_rel(
    seg_to_path: T.Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """Map ``seg_id -> {"rgb": "<rel/path/to/image.png>"}``."""
    return {sid: {"rgb": rel} for sid, rel in seg_to_path.items()}


def build_segment_id_to_road_data(
    df: pd.DataFrame, attribute_names: T.Sequence[str],
) -> dict[str, dict]:
    """Build ``segment_id -> {required_attributes, ..., comments, ...}``."""
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        sid = row["seg_id"]
        out[sid] = {
            "section": row["section"],
            "distance_km": float(row["distance"]),
            "length_km": float(row["length"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "comments": row["comments"],
            "required_attributes": {a: (None if pd.isna(row[a]) else int(row[a]))
                                    for a in attribute_names},
        }
    return out


def build_road_sequences_and_validate(
    df: pd.DataFrame,
) -> tuple[dict[str, list[str]], list[dict]]:
    """Group by section, sort by distance, validate adjacency.

    The expected distance between consecutive segments is
    ``SEGMENT_LENGTH_KM`` (0.02 km). Returns ``(road_to_seq, violations)``
    where each violation is ``{"section": ..., "prev": ..., "next": ...,
    "distance_step_km": float}``.
    """
    road_to_seq: dict[str, list[str]] = {}
    violations: list[dict] = []
    tol_km = SEGMENT_LENGTH_KM / 20  # 1 m

    for section, group in df.groupby("section", sort=True):
        sub = group.sort_values("distance", kind="stable")
        seg_ids = sub["seg_id"].tolist()
        distances = sub["distance"].tolist()
        road_to_seq[section] = seg_ids
        for i in range(1, len(seg_ids)):
            d_step = distances[i] - distances[i - 1]
            if abs(d_step - SEGMENT_LENGTH_KM) > tol_km:
                violations.append({
                    "section": section,
                    "prev": seg_ids[i - 1],
                    "next": seg_ids[i],
                    "distance_step_km": float(d_step),
                    "expected_distance_step_km": SEGMENT_LENGTH_KM,
                })
    return road_to_seq, violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path,
                        help="IRAP_Vietnam dataset root.")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    images = layout.images_dir(data_dir)
    attr_meta_in = layout.attr_meta_path(data_dir)
    rows_path = layout.rows_path(data_dir)
    out = layout.metadata_dir(data_dir)

    if not images.is_dir():
        print(f"ERROR: {images} not found.", file=sys.stderr)
        return 1
    if not attr_meta_in.is_file():
        print(f"ERROR: {attr_meta_in} not found.", file=sys.stderr)
        return 1
    if not rows_path.is_file():
        print(f"ERROR: {rows_path} not found.", file=sys.stderr)
        return 1

    with open(attr_meta_in, "r", encoding="utf-8") as f:
        attr_meta = json.load(f)
    idx_to_attr = {int(v): k for k, v in attr_meta["attribute_to_idx"].items()}
    attribute_names = [idx_to_attr[i] for i in sorted(idx_to_attr)]

    df = pd.read_parquet(rows_path)
    print(f"Loaded {len(df)} rows from {rows_path}.")

    # Some attributes may have been excluded during parsing (e.g. Vietnam
    # attributes with no ground truth). Only keep those present in the parquet.
    parquet_cols = set(df.columns)
    attribute_names = [a for a in attribute_names if a in parquet_cols]

    num_rows_in = len(df)
    print(f"Matching rows to images under {images} by seg_id...")
    seg_id_to_image_paths = index_images_by_seg_id(images)
    df, seg_to_path, match_stats = match_rows_to_images(df, seg_id_to_image_paths)
    print(f"  {len(df)} rows kept; "
          f"{match_stats['num_rows_no_image']} dropped (no image found).")
    if match_stats["num_rows_prefix_mismatch"]:
        print(f"  {match_stats['num_rows_prefix_mismatch']} row(s) matched by segment ID "
              f"despite section/image-folder prefix mismatch.")
    if match_stats["num_rows_ambiguous_image_match"]:
        print(f"  {match_stats['num_rows_ambiguous_image_match']} row(s) had "
              f"multiple candidate images.")

    # Images with no matching parquet row (no labels).
    unlabeled_seg_ids = sorted(set(seg_id_to_image_paths.keys()) - set(seg_to_path.keys()), key=int)
    unlabeled_seg_to_path = {
        sid: sorted(seg_id_to_image_paths[sid])[0].as_posix()
        for sid in unlabeled_seg_ids
    }
    print(f"  {len(unlabeled_seg_ids)} image(s) have no matching row (unlabeled).")

    print("Building segment_id_to_data_paths_rel...")
    seg_to_paths = build_segment_id_to_data_paths_rel({**seg_to_path, **unlabeled_seg_to_path})
    print("Building segment_id_to_road_data...")
    seg_to_road = build_segment_id_to_road_data(df, attribute_names)
    print("Building road_id_to_segment_id_sequence and validating adjacency...")
    road_to_seq, violations = build_road_sequences_and_validate(df)
    if violations:
        print(f"WARN: {len(violations)} adjacency violation(s) "
              f"(distance step != {SEGMENT_LENGTH_KM} km).", file=sys.stderr)
        for v in violations[:10]:
            print(f"  {v}", file=sys.stderr)
        if len(violations) > 10:
            print(f"  ... and {len(violations) - 10} more.", file=sys.stderr)

    out.mkdir(parents=True, exist_ok=True)

    def _dump(name: str, obj: T.Any) -> None:
        path = out / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"Wrote {path}")

    _dump("segment_id_to_data_paths_rel.json", seg_to_paths)
    _dump("segment_id_to_road_data.json", seg_to_road)
    _dump("road_id_to_segment_id_sequence.json", road_to_seq)
    # Normalize IRAP codes to int when writing the metadata. The input JSON
    # encodes them as strings, but `required_attributes` in
    # segment_id_to_road_data.json uses ints (see line 183). BihSequence inverts
    # `attribute_value_to_irap_number` and looks up by the int code, so the two
    # sides must agree.
    attr_meta_out = out / "attribute_metadata.json"
    attr_meta_normalized = dict(attr_meta)
    attr_meta_normalized["attribute_value_to_irap_number"] = {
        attr: {value: int(code) for value, code in mapping.items()}
        for attr, mapping in attr_meta["attribute_value_to_irap_number"].items()
    }
    with open(attr_meta_out, "w", encoding="utf-8") as f:
        json.dump(attr_meta_normalized, f, indent=2, ensure_ascii=False)
    print(f"Wrote {attr_meta_out}")

    unlabeled_path = layout.unlabeled_seg_ids_path(data_dir)
    with open(unlabeled_path, "w", encoding="utf-8") as f:
        json.dump(unlabeled_seg_ids, f, indent=2, ensure_ascii=False)
    print(f"Wrote {unlabeled_path}")

    report = {
        "num_rows_in": int(num_rows_in),
        "num_rows_kept": int(len(df)),
        **{k: int(v) if isinstance(v, int) else v for k, v in match_stats.items()},
        "num_unlabeled_images": len(unlabeled_seg_ids),
        "unlabeled_examples": unlabeled_seg_ids[:20],
        "num_sections": len(road_to_seq),
        "section_lengths": {s: len(seq) for s, seq in road_to_seq.items()},
        "num_adjacency_violations": len(violations),
        "adjacency_violation_examples": violations[:20],
        "data_dir_name": data_dir.name,
    }
    report_path = layout.build_report_path(data_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
