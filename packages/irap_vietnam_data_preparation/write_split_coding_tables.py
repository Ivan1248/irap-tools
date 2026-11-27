"""Write the coding-table rows of the segments in the labeled splits and of those in the unlabeled splits.

Usage:
    python write_split_coding_tables.py <data_dir>

Reads the coding tables like parse_coding_tables.py, ``<data_dir>/_work/rows.parquet`` and
``<data_dir>/splits.json``. Writes to ``<data_dir>/_work/``:
    labeled_splits_coding_table.xlsx     # the rows of rows.parquet (the rows that
                                         # parse_coding_tables.py keeps) whose seg_id is in
                                         # train, val or test
    unlabeled_splits_coding_table.xlsx   # all rows whose seg_id is in an unlabeled_* split

A written table has the rows of the input tables unchanged, in the order of the files and rows,
with the columns of all input tables and a last column, ``Split``, that names the split of the
row's seg_id. The labeled table has one row per seg_id, the row that supplies its labels.

The rows in the unlabeled splits are rows that the pipeline dropped although the images of their
segments are in the dataset, e.g. for the offset of their annotation location. A seg_id can have
several of them.
"""

import argparse
import json
import sys
import typing as T
from collections import defaultdict
from pathlib import Path

import pandas as pd

import layout
from layout import LABELED_SPLIT_NAMES, is_unlabeled_split
from parse_coding_tables import (
    SEG_ID_COLUMN,
    ParseInputs,
    ParseReport,
    ParseSetupError,
    excel_row,
    parse_seg_id,
    prepare_parse_inputs,
    read_coding_tables,
)

SPLIT_COLUMN = "Split"
ROW_LOCATION_COLUMNS = ["source_file", "source_row"]


def load_coding_tables(inputs: ParseInputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenates the rows of the coding tables that parse_coding_tables.py reads.

    Returns:
        The rows, and for each row its file name, Excel row number and seg_id (None if it has
        none) in the columns ``source_file``, ``source_row`` and ``seg_id``, with the same index.
    """
    tables = []
    row_sources = []
    for path, table, cols in read_coding_tables(inputs, ParseReport()):
        if SPLIT_COLUMN in table.columns:
            raise ValueError(f"{path.name} already has a {SPLIT_COLUMN!r} column.")
        tables.append(table)
        row_sources.append(pd.DataFrame({
            "source_file": path.name,
            "source_row": table.index.map(excel_row),
            "seg_id": table[cols[SEG_ID_COLUMN]].map(
                lambda cell: None if (seg_id := parse_seg_id(cell)) is None else str(seg_id)),
        }))
    if not tables:
        raise ValueError("none of the files is a coding table.")
    return (pd.concat(tables, ignore_index=True),
            pd.concat(row_sources, ignore_index=True))


def select_split_rows(table: pd.DataFrame, row_seg_ids: pd.Series,
                      split_to_seg_ids: T.Mapping[str, T.Sequence[str]]) -> pd.DataFrame:
    """Selects the rows of `table` whose seg_id is in a split, and adds the column SPLIT_COLUMN.

    Args:
        table: The rows of coding tables.
        row_seg_ids: The seg_id of each row of `table`, or None.
        split_to_seg_ids: The seg_ids of the splits that the rows are selected for. A seg_id in
            several splits, such as an unlabeled segment of two road sequences, gets their
            names, separated by commas.
    """
    seg_id_to_splits = defaultdict(list)
    for split, seg_ids in split_to_seg_ids.items():
        for seg_id in seg_ids:
            seg_id_to_splits[seg_id].append(split)
    row_splits = row_seg_ids.map(lambda seg_id: ", ".join(seg_id_to_splits.get(seg_id, ())))
    is_selected = row_splits != ""
    return table[is_selected].assign(**{SPLIT_COLUMN: row_splits[is_selected]})


def write_split_coding_table(path: Path, table: pd.DataFrame, row_seg_ids: pd.Series,
                             split_to_seg_ids: T.Mapping[str, T.Sequence[str]]) -> None:
    """Writes the rows that :func:`select_split_rows` selects to an .xlsx file."""
    split_table = select_split_rows(table, row_seg_ids, split_to_seg_ids)
    split_table.to_excel(path, index=False, engine="openpyxl")
    print(f"Wrote {path} ({len(split_table)} rows: "
          + ", ".join(f"{split}: {num_rows}" for split, num_rows
                      in split_table[SPLIT_COLUMN].value_counts().items()) + ").")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path, help="iRAP-Vietnam dataset root.")
    args = parser.parse_args(argv)

    splits_path = layout.splits_path(args.data_dir)
    rows_path = layout.rows_path(args.data_dir)
    for path, step in [(splits_path, "Assign the splits in split_editor.py"),
                       (rows_path, "Run parse_coding_tables.py")]:
        if not path.is_file():
            print(f"ERROR: {path} not found. {step} first.", file=sys.stderr)
            return 1
    splits: dict[str, list[str]] = json.loads(splits_path.read_text(encoding="utf-8"))
    unknown_splits = [s for s in splits if s not in LABELED_SPLIT_NAMES and not is_unlabeled_split(s)]
    if unknown_splits:
        print(f"ERROR: {splits_path} has unknown splits: {', '.join(unknown_splits)}.",
              file=sys.stderr)
        return 1
    kept_rows = pd.read_parquet(rows_path)
    if not set(ROW_LOCATION_COLUMNS) <= set(kept_rows.columns):
        print(f"ERROR: {rows_path} has no {' and '.join(ROW_LOCATION_COLUMNS)} columns."
              f" Re-run parse_coding_tables.py.", file=sys.stderr)
        return 1

    try:
        inputs = prepare_parse_inputs(args.data_dir)
        table, row_sources = load_coding_tables(inputs)
    except (ParseSetupError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Read {len(table)} rows, {row_sources['seg_id'].notna().sum()} of them with a seg_id.")

    row_is_kept = pd.MultiIndex.from_frame(row_sources[ROW_LOCATION_COLUMNS]).isin(
        pd.MultiIndex.from_frame(kept_rows[ROW_LOCATION_COLUMNS]))
    location_to_seg_id = row_sources[row_is_kept].set_index(ROW_LOCATION_COLUMNS)["seg_id"]
    kept_location_to_seg_id = kept_rows.set_index(ROW_LOCATION_COLUMNS)["seg_id"]
    all_kept_rows_are_found = len(location_to_seg_id) == len(kept_rows)
    seg_ids_are_equal = location_to_seg_id.eq(
        kept_location_to_seg_id.reindex(location_to_seg_id.index)).all()
    if not (all_kept_rows_are_found and seg_ids_are_equal):
        print(f"ERROR: the rows of {rows_path} are not in the coding tables with the same seg_ids."
              f" Have the coding tables changed? Re-run parse_coding_tables.py.", file=sys.stderr)
        return 1

    kept_seg_ids = set(kept_rows["seg_id"])
    labeled_seg_ids_without_rows = [seg_id for split in LABELED_SPLIT_NAMES
                                    for seg_id in splits.get(split, [])
                                    if seg_id not in kept_seg_ids]
    if labeled_seg_ids_without_rows:
        print(f"WARNING: {len(labeled_seg_ids_without_rows)} seg_id(s) of the labeled splits"
              f" have no row in {rows_path}, e.g."
              f" {', '.join(labeled_seg_ids_without_rows[:10])}. Is {splits_path} up to date?",
              file=sys.stderr)

    layout.work_dir(args.data_dir).mkdir(parents=True, exist_ok=True)
    write_split_coding_table(
        layout.labeled_splits_coding_table_path(args.data_dir),
        table[row_is_kept], row_sources["seg_id"][row_is_kept],
        {split: seg_ids for split, seg_ids in splits.items() if split in LABELED_SPLIT_NAMES})
    write_split_coding_table(
        layout.unlabeled_splits_coding_table_path(args.data_dir),
        table, row_sources["seg_id"],
        {split: seg_ids for split, seg_ids in splits.items() if is_unlabeled_split(split)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
