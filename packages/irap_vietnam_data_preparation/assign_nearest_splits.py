"""Assign the sequences that splits.json leaves unassigned to the split of the nearest sequence.

Usage:
    python assign_nearest_splits.py <data_dir> [--dry-run]

The splits are geographically separated, so a sequence that build_metadata.py added, e.g. from
new coding tables or images, belongs to the split of the assigned sequences around it. The
script reads the metadata and splits.json like split_editor.py, which leaves a sequence without
seg_ids in splits.json unassigned, and writes splits.json like its Save button. The seg_ids of
splits.json that are in no sequence are therefore dropped, and ``unlabeled_unlocated`` is
written anew. Check the result in the split editor.

- A labeled sequence gets the split of the assigned labeled sequence with the segment nearest
  to one of its segments.
- An unlabeled sequence (the unlabeled segments of a recording) then gets the split of the
  labeled sequence with the segment nearest to its centroid.

A sequence stays unassigned, with a warning, if it is not clearly nearest to one split (see
`MIN_SPLIT_DISTANCE_RATIO`), so that the splits stay separated. So does an unlabeled sequence
that is nearest to a labeled sequence that stays unassigned.
"""

import argparse
import dataclasses as dc
import json
import sys
import typing as T
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import layout
import split_editor
from build_metadata import compute_distance_km
from split_editor import REAL_SPLITS, UNASSIGNED

# A sequence is clearly nearest to one split if the nearest segment of every other split is
# more than this many times as far as the nearest segment. In the checked data (2026-11-26),
# the ratio is above 3 for every unassigned sequence.
MIN_SPLIT_DISTANCE_RATIO = 2.0
# The number of points whose distances to the reference points are computed at once.
NUM_POINTS_PER_CHUNK = 256


@dc.dataclass(frozen=True)
class NearestSplit:
    """The split nearest to a sequence, and how clearly it is the nearest."""

    split: str
    distance_km: float  # to the nearest reference point, which is in `split`
    other_split: str | None  # the nearest other split, None if only `split` has points
    other_split_distance_km: float  # to the nearest point of `other_split`, inf without it

    @property
    def is_clear(self) -> bool:
        return self.other_split_distance_km > MIN_SPLIT_DISTANCE_RATIO * self.distance_km


def find_nearest_split(points: np.ndarray, split_to_points: T.Mapping[str, np.ndarray]
                       ) -> NearestSplit:
    """Finds the split with the reference point nearest to one of `points`.

    Args:
        points: The (latitude, longitude) of the points of a sequence in degrees, shape (N, 2).
        split_to_points: The reference points of each split, shape (M, 2) with M > 0, of at
            least one split.
    """
    split_to_distance_km = {
        split: min(
            float(compute_distance_km(points[i:i + NUM_POINTS_PER_CHUNK, None], ref_points).min())
            for i in range(0, len(points), NUM_POINTS_PER_CHUNK))
        for split, ref_points in split_to_points.items()}
    split, *other_splits = sorted(split_to_distance_km, key=split_to_distance_km.get)
    other_split = other_splits[0] if other_splits else None
    return NearestSplit(
        split, split_to_distance_km[split], other_split,
        split_to_distance_km[other_split] if other_split else float("inf"))


def get_split_to_points(sequence_to_points: T.Mapping[str, np.ndarray],
                        sequence_to_split: T.Mapping[str, str]) -> dict[str, np.ndarray]:
    """Returns the points, shape (M, 2), of the sequences of each split in `sequence_to_split`."""
    split_to_point_arrays = defaultdict(list)
    for seq, split in sequence_to_split.items():
        split_to_point_arrays[split].append(sequence_to_points[seq])
    return {split: np.concatenate(arrays) for split, arrays in split_to_point_arrays.items()}


def assign_nearest_splits(
    sequence_to_points: T.Mapping[str, np.ndarray],
    sequence_to_split: T.Mapping[str, str],
    split_to_reference_points: T.Mapping[str, np.ndarray],
) -> tuple[dict[str, str], dict[str, NearestSplit]]:
    """Assigns each unassigned sequence that is clearly nearest to one split to that split.

    Args:
        sequence_to_points: The (latitude, longitude) of the points of each sequence in
            degrees, shape (N, 2).
        sequence_to_split: The split of each sequence, `UNASSIGNED` for the unassigned ones.
        split_to_reference_points: The points of each split, shape (M, 2), that the
            unassigned sequences are compared with. With the points of `UNASSIGNED`
            sequences, a sequence that is nearest to them stays unassigned.

    Returns:
        The new ``sequence -> split``, and the nearest split of each sequence that was
        unassigned. Those that are not clear (`NearestSplit.is_clear`) stay unassigned.
    """
    sequence_to_nearest_split = {
        seq: find_nearest_split(sequence_to_points[seq], split_to_reference_points)
        for seq, split in sequence_to_split.items() if split == UNASSIGNED}
    new_sequence_to_split = {
        **sequence_to_split,
        **{seq: nearest.split for seq, nearest in sequence_to_nearest_split.items()
           if nearest.is_clear}}
    return new_sequence_to_split, sequence_to_nearest_split


def print_assignment_report(layer: split_editor.Layer,
                            sequence_to_nearest_split: T.Mapping[str, NearestSplit]) -> None:
    """Prints the nearest split of each sequence that was unassigned, the least clear first."""

    def format_split(split: str) -> str:
        return "unassigned labeled sequence" if split == UNASSIGNED else layer.split_key(split)

    split_to_num_sequences: Counter = Counter()
    split_to_num_segments: Counter = Counter()
    still_unassigned_sequences = []
    for seq, nearest in sorted(
            sequence_to_nearest_split.items(),
            key=lambda item: item[1].other_split_distance_km / max(item[1].distance_km, 1e-9)):
        num_segments = len(layer.data[seq]["segs"])
        other = ("nothing else" if nearest.other_split is None else
                 f"{format_split(nearest.other_split)} {nearest.other_split_distance_km:.2f} km")
        is_assigned = nearest.is_clear and nearest.split != UNASSIGNED
        print(f"  {seq} ({num_segments} segments): {format_split(nearest.split)}"
              f" {nearest.distance_km:.2f} km, {other}"
              + ("" if is_assigned else " – left unassigned"))
        if is_assigned:
            split_to_num_sequences[nearest.split] += 1
            split_to_num_segments[nearest.split] += num_segments
        else:
            still_unassigned_sequences.append(seq)
    summary = ", ".join(
        f"{layer.split_key(split)}: {split_to_num_sequences[split]} sequences"
        f" ({split_to_num_segments[split]} segments)" for split in REAL_SPLITS)
    print(f"Assigned {split_to_num_sequences.total()} of {len(sequence_to_nearest_split)}"
          f" unassigned {layer.name} sequences: {summary}.")
    if still_unassigned_sequences:
        print(f"WARNING: {len(still_unassigned_sequences)} {layer.name} sequence(s) are not"
              f" clearly nearest to one split, or nearest to an unassigned labeled sequence,"
              f" and stay unassigned: {', '.join(still_unassigned_sequences)}.",
              file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path, help="iRAP-Vietnam dataset root.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the assignments without writing splits.json.")
    args = parser.parse_args(argv)

    splits_path = layout.splits_path(args.data_dir)
    if not splits_path.is_file():
        print(f"ERROR: {splits_path} not found. Assign the splits in split_editor.py first.",
              file=sys.stderr)
        return 1
    metadata = split_editor.load_metadata(args.data_dir)
    layers = labeled, unlabeled = split_editor.make_layers(metadata)
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    assignments, warnings = split_editor.make_initial_assignments(splits, layers)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if not any(split in REAL_SPLITS for split in assignments[labeled.name].values()):
        print(f"ERROR: {splits_path} assigns no labeled sequence.", file=sys.stderr)
        return 1

    labeled_points = {seq: np.array(entry["coords"], dtype=float)
                      for seq, entry in labeled.data.items()}
    labeled_assignment, labeled_nearest = assign_nearest_splits(
        labeled_points, assignments[labeled.name],
        get_split_to_points(labeled_points, {
            seq: split for seq, split in assignments[labeled.name].items()
            if split != UNASSIGNED}))
    print_assignment_report(labeled, labeled_nearest)

    unlabeled_points = {seq: np.array([entry["centroid"]], dtype=float)
                        for seq, entry in unlabeled.data.items()}
    unlabeled_assignment, unlabeled_nearest = assign_nearest_splits(
        unlabeled_points, assignments[unlabeled.name],
        get_split_to_points(labeled_points, labeled_assignment))
    print_assignment_report(unlabeled, unlabeled_nearest)

    new_splits = split_editor.make_splits(
        layers, {labeled.name: labeled_assignment, unlabeled.name: unlabeled_assignment},
        metadata.unlocated_seg_ids)
    print("Segments per split (before -> after): " + ", ".join(
        f"{key}: {len(splits.get(key, []))} -> {len(new_splits.get(key, []))}"
        for key in dict.fromkeys([*splits, *new_splits])) + ".")
    if args.dry_run:
        print(f"Dry run: {splits_path} is not written.")
        return 0
    splits_path.write_text(json.dumps(new_splits, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"Wrote {splits_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
