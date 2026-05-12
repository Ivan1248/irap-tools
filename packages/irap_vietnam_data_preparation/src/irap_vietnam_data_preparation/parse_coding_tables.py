"""Parse IRAP-Vietnam coding tables into a normalized per-segment table.

Usage:
    python irap_vietnam_data_preparation/parse_coding_tables.py <data_dir>

Reads (under ``<data_dir>/_raw/``):
    coding-tables.zip                # OR an unzipped coding-tables/ directory
    attribute_metadata.json          # BiH-compatible: attribute_to_idx +
                                     # attribute_value_to_irap_number

Writes (into ``<data_dir>/_work/``):
    rows.parquet                     # one row per kept segment
    parse_report.json                # counts, warnings, dropped-row breakdown

Validation rules (see irap_vietnam_data_preparation/vietnam_data_preparation.md):

- Header lookup is by name; column order is not assumed.
- Required scalar columns: Section, Distance, Length, Latitude start,
  Longitude start, Image Reference FPZ, Comments.
- Required attribute columns: every key of attribute_to_idx in the supplied
  metadata. Missing any required column => fatal error.
- Files lacking the labeling columns entirely (Section + at least one
  attribute) are skipped as "non-coding files".
- Rows are dropped (and counted) when:
    * Length != 0.02 (within 1e-6),
    * any required scalar field is missing,
    * Image Reference FPZ does not contain a parseable segment id,
    * any required attribute cell is empty (for attributes that have at least
      one non-empty value in the file; attributes with zero non-empty values
      across the whole file are stored as None instead of causing a row drop),
    * any attribute IRAP code is not in attribute_value_to_irap_number.
- A file mixing multiple Section values triggers a warning but is processed.
- Duplicate segment ids across the entire dataset are resolved by keeping
  the row with the most non-empty attribute cells; warns on every collision.
"""

import argparse
import difflib
import json
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import typing as T

import pandas as pd
from tqdm import tqdm

import layout


REQUIRED_SCALAR_COLUMNS = (
    "Section",
    "Distance",
    "Length",
    "Latitude start",
    "Longitude start",
    "Image Reference FPZ",
    "Comments",
)

EXPECTED_LENGTH_KM = 0.02
LENGTH_TOLERANCE = 1e-6
ATTR_NAME_MAPPING_FILE = "attribute_name_mapping.json"

SEG_ID_RE = re.compile(r"seg\.?\s*no\.?\s*(\d+)", re.IGNORECASE)
SEG_ID_FALLBACK_RE = re.compile(r"\b(\d{4,})\b")


def resolve_coding_tables_dir(in_dir: Path) -> Path:
    """Return a directory containing the coding tables.

    If ``coding-tables/`` exists, use it. Otherwise unzip
    ``coding-tables.zip`` into a sibling ``coding-tables/`` and return that.
    """
    unzipped = in_dir / "coding-tables"
    if unzipped.is_dir():
        return unzipped
    zip_path = in_dir / "coding-tables.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(
            f"Neither '{unzipped}' nor '{zip_path}' found. Expected coding "
            f"tables in --in."
        )
    print(f"Unzipping {zip_path} -> {unzipped}...")
    unzipped.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(unzipped)
    return unzipped


def load_attribute_metadata(path: Path) -> tuple[list[str], dict[str, set[int]]]:
    """Load ``attribute_metadata.json``.

    Returns:
        attribute_names: ordered list (by ``attribute_to_idx`` value).
        valid_codes_per_attr: ``{attr_name: set(irap_codes)}``.
    """
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if "attribute_to_idx" not in meta or "attribute_value_to_irap_number" not in meta:
        raise ValueError(
            f"{path} missing 'attribute_to_idx' or "
            f"'attribute_value_to_irap_number'."
        )
    idx_to_attr = {int(v): k for k, v in meta["attribute_to_idx"].items()}
    attribute_names = [idx_to_attr[i] for i in sorted(idx_to_attr)]
    valid_codes = {
        attr: {int(c) for c in meta["attribute_value_to_irap_number"][attr].values()}
        for attr in attribute_names
    }
    return attribute_names, valid_codes


def load_attribute_name_mapping(
    script_dir: Path,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Load ``attribute_name_mapping.json`` from the script directory.

    Returns:
        attr_name_mapping: ``{metadata_attr_name: table_column_name}``
        ignored_attr_mapping: ``{metadata_attr_name: [table_column_names]}``

    If the file is absent, returns empty dicts (no mapping applied).
    """
    path = script_dir / ATTR_NAME_MAPPING_FILE
    if not path.is_file():
        return {}, {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    attr_name_mapping = data.get("metadata_to_table", {})
    ignored_attr_mapping = data.get("ignored_metadata_attributes", {})
    return attr_name_mapping, ignored_attr_mapping


def find_table_files(coding_tables_dir: Path) -> list[Path]:
    """List candidate spreadsheet files (``.xlsx``/``.xls``) under the dir."""
    files = sorted(
        p for p in coding_tables_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".xlsx", ".xls")
        and not p.name.startswith("~$")  # Excel lock files
    )
    return files


def normalize_header(name: T.Any) -> str:
    """Return a trimmed string header name (or empty string)."""
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return ""
    return str(name).strip()


def is_blank(value: T.Any) -> bool:
    """True if a cell is empty/NaN/blank-string."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def parse_seg_id(cell: T.Any) -> int | None:
    """Extract the segment id integer from an Image Reference FPZ cell."""
    if is_blank(cell):
        return None
    s = str(cell)
    m = SEG_ID_RE.search(s)
    if m:
        return int(m.group(1))
    m = SEG_ID_FALLBACK_RE.search(s)
    if m:
        return int(m.group(1))
    return None


def parse_float(cell: T.Any) -> float | None:
    """Coerce a cell to float; return None on blank/parse failure."""
    if is_blank(cell):
        return None
    try:
        return float(cell)
    except (TypeError, ValueError):
        return None


def parse_int_code(cell: T.Any) -> int | None:
    """Coerce a cell to int; return None on blank/parse failure."""
    if is_blank(cell):
        return None
    try:
        # Excel often loads numeric codes as floats (e.g. 1.0).
        f = float(cell)
    except (TypeError, ValueError):
        return None
    i = int(f)
    if float(i) != f:
        return None
    return i


def classify_road_volume(cell: T.Any) -> int | None:
    """Map a raw intersecting road volume count to an IRAP code.

    Codes match attribute_metadata.json:
        1: ≥15,000 vehicles
        2: 10,000 to <15,000 vehicles
        3: 5,000 to <10,000 vehicles
        4: 1,000 to <5,000 vehicles
        5: 100 to <1,000 vehicles
        6: 1 to <100 vehicles
        7: Not applicable (0 or blank)
    """
    v = parse_float(cell)
    if v is None:
        return None
    if v >= 15000:
        return 1
    if v >= 10000:
        return 2
    if v >= 5000:
        return 3
    if v >= 1000:
        return 4
    if v >= 100:
        return 5
    if v >= 1:
        return 6
    return 7  # 0 vehicles — not applicable


# Attributes whose table cells contain raw values (not IRAP codes) that need
# classification into codes.  The classifier is called instead of
# ``parse_int_code`` and the result is not validated against
# ``valid_codes_per_attr`` (the classifier itself produces only valid codes).
ATTR_VALUE_CLASSIFIERS: dict[str, T.Callable[[T.Any], int | None]] = {
    "Intersecting road volume": classify_road_volume,
}


def read_sheet(path: Path) -> pd.DataFrame:
    """Read the (single) sheet of an XLS/XLSX file as a DataFrame.

    Errors out if the file has more than one non-empty sheet.
    """
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    sheets = pd.read_excel(path, sheet_name=None, engine=engine, dtype=object)
    non_empty = {n: df for n, df in sheets.items() if not df.empty}
    if not non_empty:
        return pd.DataFrame()
    if len(non_empty) > 1:
        raise ValueError(
            f"{path}: expected 1 sheet, found {len(non_empty)}: {list(non_empty)}"
        )
    return next(iter(non_empty.values()))


def file_is_coding_table(
    df: pd.DataFrame,
    attribute_names: T.Sequence[str],
    *,
    attr_name_mapping: dict[str, str] | None = None,
) -> bool:
    """True if the file looks like a labeling sheet.

    Heuristic: must have a "Section" column AND at least one of the canonical
    attribute columns (case-insensitive after whitespace stripping). Files
    lacking both are silently skipped.
    """
    headers = {normalize_header(c).lower() for c in df.columns}
    attr_name_mapping = attr_name_mapping or {}
    table_attr_names = [attr_name_mapping.get(a, a) for a in attribute_names]
    return "section" in headers and any(a.lower() in headers for a in table_attr_names)


def validate_columns(
    df: pd.DataFrame,
    attribute_names: T.Sequence[str],
    *,
    file: Path,
    attr_name_mapping: dict[str, str] | None = None,
    ignored_attr_mapping: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Return ``{logical_name: actual_column_name}`` for every required column.

    Header lookup is case-insensitive (after whitespace stripping). Attribute
    column names are first translated via ``attr_name_mapping`` (metadata name
    -> table column name) before lookup.

    Raises ``ValueError`` if any required column is missing. The message lists
    missing required columns next to their closest header found in the file,
    and separately lists headers present in the file but unused by the
    schema (so renames vs. truly-extra columns are easy to tell apart).

    When all required columns are found, also prints:
    - Per ignored attribute: how many non-empty values its mapped table column
      contains (WARN; so callers can verify the skip is intentional).
    - A list of non-scalar columns in the table that have no values at all
      (INFO; useful for detecting unused/placeholder attribute columns).
    """
    attr_name_mapping = attr_name_mapping or {}
    ignored_attr_mapping = ignored_attr_mapping or {}

    actual = {normalize_header(c): c for c in df.columns}
    ci_index: dict[str, str] = {}
    for h in actual:
        ci_index.setdefault(h.lower(), h)

    def resolve_one(name: str) -> str | None:
        if name in actual:
            return name
        return ci_index.get(name.lower())

    # Scalar columns: look up by their own name.
    resolved: dict[str, str | None] = {r: resolve_one(r) for r in REQUIRED_SCALAR_COLUMNS}

    # Attribute columns: look up by their mapped table column name.
    for attr in attribute_names:
        table_col = attr_name_mapping.get(attr, attr)
        resolved[attr] = resolve_one(table_col)

    missing = [r for r, h in resolved.items() if h is None]
    if missing:
        matched = sorted(h for h in resolved.values() if h is not None)
        consumed = set(matched)
        unused_found = sorted(set(actual) - consumed)

        suggestion_lines = []
        for r in missing:
            cands = difflib.get_close_matches(r, unused_found, n=1, cutoff=0.6)
            if cands:
                ratio = difflib.SequenceMatcher(None, r, cands[0]).ratio()
                suggestion_lines.append(
                    f"  - {r!r}  ->  closest in table: {cands[0]!r} "
                    f"(sim={ratio:.2f})"
                )
            else:
                suggestion_lines.append(f"  - {r!r}  ->  no close match")
        matched_lines = [f"  - {h!r}" for h in matched]
        unused_lines = [f"  - {h!r}" for h in unused_found]

        raise ValueError(
            f"{file}: column header mismatch.\n"
            f"\nMissing required columns ({len(missing)}):\n"
            + "\n".join(suggestion_lines)
            + f"\n\nMatched required columns ({len(matched)}):\n"
            + "\n".join(matched_lines)
            + f"\n\nUnused columns in table ({len(unused_found)}):\n"
            + "\n".join(unused_lines)
        )

    result = {r: actual[h] for r, h in resolved.items() if h is not None}

    # Warn about ignored attributes: find their table column and count values.
    for meta_attr, table_cols in ignored_attr_mapping.items():
        for table_col in table_cols:
            h = resolve_one(table_col)
            if h is not None:
                actual_col = actual[h]
                num_vals = sum(1 for v in df[actual_col] if not is_blank(v))
                print(
                    f"WARN: {file.name}: ignoring '{meta_attr}' "
                    f"(table col: '{actual_col}') ({num_vals} non-empty value(s))",
                    file=sys.stderr,
                )
                break
        else:
            cols_str = ", ".join(f"'{c}'" for c in table_cols)
            print(
                f"WARN: {file.name}: ignoring '{meta_attr}' "
                f"(expected table col(s): {cols_str}) — column not found in table",
                file=sys.stderr,
            )

    return result


# -----------------------------------------------------------------------------
# Per-row processing
# -----------------------------------------------------------------------------


class RowDropReason:
    LENGTH_MISMATCH = "length_mismatch"
    MISSING_SCALAR = "missing_scalar"
    BAD_SEG_ID = "bad_seg_id"
    MISSING_ATTRIBUTE = "missing_attribute"
    UNKNOWN_CODE = "unknown_code"


def process_file(
    path: Path,
    attribute_names: T.Sequence[str],
    valid_codes_per_attr: T.Mapping[str, set[int]],
    drop_counts: Counter,
    unknown_codes: dict[str, Counter],
    *,
    attr_name_mapping: dict[str, str] | None = None,
    ignored_attr_mapping: dict[str, list[str]] | None = None,
    empty_attr_file_counts: Counter | None = None,
    missing_rows_per_attr_per_file: dict[str, dict[str, int]] | None = None,
    non_empty_rows_per_file: dict[str, int] | None = None,
    drop_counts_per_file: dict[str, Counter] | None = None,
    empty_attrs_per_file: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Parse one spreadsheet file and return a list of normalized row dicts.

    Records are dicts with keys:
        section, distance, length, lat, lon, seg_id, comments, source_file,
        and one entry per attribute (IRAP code as int).

    Mutates ``drop_counts``, ``unknown_codes``, ``empty_attr_file_counts``,
    ``missing_rows_per_attr_per_file``, ``non_empty_rows_per_file``, and
    ``drop_counts_per_file`` for the parse report.
    """
    df = read_sheet(path)
    if df.empty:
        return []
    if not file_is_coding_table(df, attribute_names, attr_name_mapping=attr_name_mapping):
        return []

    cols = validate_columns(
        df, attribute_names, file=path,
        attr_name_mapping=attr_name_mapping,
        ignored_attr_mapping=ignored_attr_mapping,
    )

    empty_attrs = [attr for attr in attribute_names
                   if df[cols[attr]].apply(is_blank).all()]
    if empty_attrs:
        if empty_attr_file_counts is not None:
            for attr in empty_attrs:
                empty_attr_file_counts[attr] += 1
        if empty_attrs_per_file is not None:
            empty_attrs_per_file[path.name] = list(empty_attrs)
        print(
            f"INFO: {path.name}: {len(empty_attrs)} attribute(s) with no values: "
            + ", ".join(f"'{a}'" for a in empty_attrs),
            file=sys.stderr,
        )
    per_file_skip = set(empty_attrs)

    # A row with no values for any attribute is treated as empty (padding /
    # non-coding row); only non-empty rows count for the missing-value check.
    attr_blank = pd.DataFrame(
        {a: df[cols[a]].apply(is_blank) for a in attribute_names}
    )
    non_empty_mask = ~attr_blank.all(axis=1)
    num_non_empty = int(non_empty_mask.sum())
    if non_empty_rows_per_file is not None:
        non_empty_rows_per_file[path.name] = num_non_empty
    missing_per_attr: dict[str, int] = {}
    for attr in attribute_names:
        if attr in per_file_skip:
            continue
        num_blank = int((attr_blank[attr] & non_empty_mask).sum())
        if num_blank > 0:
            missing_per_attr[attr] = num_blank
            print(
                f"WARN: {path.name}: attribute '{attr}' missing values in "
                f"{num_blank}/{num_non_empty} non-empty row(s)",
                file=sys.stderr,
            )
    if missing_rows_per_attr_per_file is not None and missing_per_attr:
        missing_rows_per_attr_per_file[path.name] = missing_per_attr

    rows: list[dict] = []
    sections_in_file: set[str] = set()
    file_drops: Counter = Counter()

    def drop(reason: str) -> None:
        drop_counts[reason] += 1
        file_drops[reason] += 1

    for _, raw in df.iterrows():
        # Skip wholly empty rows (XLSX often pads with blanks).
        if all(is_blank(raw[cols[k]]) for k in REQUIRED_SCALAR_COLUMNS):
            continue

        length = parse_float(raw[cols["Length"]])
        if length is None or abs(length - EXPECTED_LENGTH_KM) > LENGTH_TOLERANCE:
            drop(RowDropReason.LENGTH_MISMATCH)
            continue

        section = raw[cols["Section"]]
        distance = parse_float(raw[cols["Distance"]])
        lat = parse_float(raw[cols["Latitude start"]])
        lon = parse_float(raw[cols["Longitude start"]])
        if is_blank(section) or distance is None or lat is None or lon is None:
            drop(RowDropReason.MISSING_SCALAR)
            continue
        section = str(section).strip()
        sections_in_file.add(section)

        seg_id = parse_seg_id(raw[cols["Image Reference FPZ"]])
        if seg_id is None:
            drop(RowDropReason.BAD_SEG_ID)
            continue

        attrs: dict[str, int | None] = {}
        bad_attr = False
        for attr in attribute_names:
            if attr in per_file_skip:
                attrs[attr] = None
                continue
            classifier = ATTR_VALUE_CLASSIFIERS.get(attr)
            if classifier is not None:
                code = classifier(raw[cols[attr]])
                if code is None:
                    drop(RowDropReason.MISSING_ATTRIBUTE)
                    bad_attr = True
                    break
                attrs[attr] = code
                continue
            code = parse_int_code(raw[cols[attr]])
            if code is None:
                drop(RowDropReason.MISSING_ATTRIBUTE)
                bad_attr = True
                break
            if code not in valid_codes_per_attr[attr]:
                unknown_codes[attr][code] += 1
                drop(RowDropReason.UNKNOWN_CODE)
                bad_attr = True
                break
            attrs[attr] = code
        if bad_attr:
            continue

        comments = raw[cols["Comments"]]
        comments = "" if is_blank(comments) else str(comments)

        rec = {
            "section": section,
            "distance": distance,
            "length": length,
            "lat": lat,
            "lon": lon,
            "seg_id": str(seg_id),
            "comments": comments,
            "source_file": path.name,
        }
        rec.update(attrs)
        rows.append(rec)

    if drop_counts_per_file is not None and file_drops:
        drop_counts_per_file[path.name] = file_drops

    return rows


def num_filled_attribute_cells(rec: dict, attribute_names: T.Sequence[str]) -> int:
    """Count non-blank attribute cells in a record (used for dup resolution)."""
    return sum(1 for a in attribute_names if not is_blank(rec.get(a)))


def resolve_duplicates(
    rows: list[dict],
    attribute_names: T.Sequence[str],
) -> tuple[list[dict], list[dict]]:
    """Return (kept_rows, dropped_duplicate_rows). Keep row with most filled cells.

    Ties broken by source_file then row order (stable).
    """
    by_seg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_seg[r["seg_id"]].append(r)

    kept: list[dict] = []
    dropped: list[dict] = []
    for seg_id, group in by_seg.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        # Stable sort by (-filled, original index): keep the first.
        scored = sorted(
            enumerate(group),
            key=lambda ix_r: (-num_filled_attribute_cells(ix_r[1], attribute_names),
                              ix_r[0]),
        )
        winner = scored[0][1]
        kept.append(winner)
        for _, r in scored[1:]:
            dropped.append(r)
        files = [r["source_file"] for r in group]
        print(f"WARN: duplicate seg_id {seg_id} across {files}; "
              f"kept row from {winner['source_file']}", file=sys.stderr)
    return kept, dropped


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", type=Path,
                        help="IRAP_Vietnam dataset root.")
    args = parser.parse_args(argv)

    raw = layout.raw_dir(args.data_dir)
    out = layout.work_dir(args.data_dir)
    if not raw.is_dir():
        print(f"ERROR: {raw} not found. Expected a directory containing "
              f"attribute_metadata.json and coding-tables[.zip].",
              file=sys.stderr)
        return 2
    attr_meta_path = layout.attr_meta_path(args.data_dir)
    if not attr_meta_path.is_file():
        print(f"ERROR: {attr_meta_path} not found.", file=sys.stderr)
        return 2

    attribute_names, valid_codes = load_attribute_metadata(attr_meta_path)
    attr_name_mapping, ignored_attr_mapping = load_attribute_name_mapping(
        Path(__file__).parent
    )
    if ignored_attr_mapping:
        ignored = sorted(ignored_attr_mapping)
        attribute_names = [a for a in attribute_names if a not in ignored_attr_mapping]
        valid_codes = {k: v for k, v in valid_codes.items() if k not in ignored_attr_mapping}
        print(f"Ignoring {len(ignored)} attribute(s) per mapping file: "
              + ", ".join(repr(a) for a in ignored))
    coding_tables_dir = resolve_coding_tables_dir(raw)
    files = find_table_files(coding_tables_dir)
    if not files:
        print(f"ERROR: no .xls/.xlsx files under {coding_tables_dir}",
              file=sys.stderr)
        return 1
    print(f"Found {len(files)} spreadsheet file(s) under {coding_tables_dir}")

    drop_counts: Counter = Counter()
    unknown_codes: dict[str, Counter] = defaultdict(Counter)
    empty_attr_file_counts: Counter = Counter()
    missing_rows_per_attr_per_file: dict[str, dict[str, int]] = {}
    non_empty_rows_per_file: dict[str, int] = {}
    drop_counts_per_file: dict[str, Counter] = {}
    empty_attrs_per_file: dict[str, list[str]] = {}
    rows_kept_per_file: dict[str, int] = {}
    skipped_files: dict[str, str] = {}
    processed_files: list[str] = []
    all_rows: list[dict] = []

    for path in tqdm(files, desc="Parsing", unit="file"):
        df_head = None
        try:
            df_head = read_sheet(path)
        except Exception as e:
            print(f"ERROR reading {path}: {e}", file=sys.stderr)
            raise
        if df_head.empty:
            print(f"INFO: {path.name}: skipping — empty sheet", file=sys.stderr)
            skipped_files[path.name] = "empty sheet"
            continue
        if not file_is_coding_table(
                df_head, attribute_names, attr_name_mapping=attr_name_mapping):
            print(f"INFO: {path.name}: skipping — not a coding table "
                  f"(no 'Section' column or no recognizable attribute column)",
                  file=sys.stderr)
            skipped_files[path.name] = (
                "not a coding table (no 'Section' column or no recognizable "
                "attribute column)"
            )
            continue
        try:
            rows = process_file(
                path, attribute_names, valid_codes,
                drop_counts, unknown_codes,
                attr_name_mapping=attr_name_mapping,
                ignored_attr_mapping=ignored_attr_mapping,
                empty_attr_file_counts=empty_attr_file_counts,
                missing_rows_per_attr_per_file=missing_rows_per_attr_per_file,
                non_empty_rows_per_file=non_empty_rows_per_file,
                drop_counts_per_file=drop_counts_per_file,
                empty_attrs_per_file=empty_attrs_per_file,
            )
        except ValueError as e:
            print(f"WARN: {path.name}: skipping — column validation failed:\n{e}",
                  file=sys.stderr)
            skipped_files[path.name] = f"column validation failed: {e}"
            continue
        processed_files.append(path.name)
        rows_kept_per_file[path.name] = len(rows)
        all_rows.extend(rows)

    print(f"Collected {len(all_rows)} rows before duplicate resolution.")
    kept, dup_dropped = resolve_duplicates(all_rows, attribute_names)
    print(f"After duplicate resolution: {len(kept)} kept, "
          f"{len(dup_dropped)} duplicate row(s) dropped.")

    out.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(kept)
    parquet_path = out / layout.ROWS_PARQUET
    df_out.to_parquet(parquet_path, index=False)
    print(f"Wrote {parquet_path} ({len(df_out)} rows)")

    def file_entry(fname: str) -> dict:
        entry: dict = {
            "status": "processed",
            "num_rows_non_empty": non_empty_rows_per_file.get(fname, 0),
            "num_rows_kept": rows_kept_per_file.get(fname, 0),
        }
        file_drops = drop_counts_per_file.get(fname)
        if file_drops:
            entry["rows_dropped_by_reason"] = dict(file_drops)
        empty_attrs = empty_attrs_per_file.get(fname)
        if empty_attrs:
            entry["empty_attributes"] = empty_attrs
        missing_per_attr = missing_rows_per_attr_per_file.get(fname)
        if missing_per_attr:
            entry["num_rows_missing_per_attribute"] = dict(
                sorted(missing_per_attr.items(), key=lambda kv: (-kv[1], kv[0]))
            )
        return entry

    files_section: dict[str, dict] = {}
    for fname in processed_files:
        files_section[fname] = file_entry(fname)
    for fname, reason in sorted(skipped_files.items()):
        files_section[fname] = {"status": "skipped", "skip_reason": reason}
    files_section = dict(sorted(files_section.items()))

    report = {
        "meta": {
            "parsed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "coding_tables_dir": str(coding_tables_dir.relative_to(args.data_dir)),
            "attribute_metadata_path": str(attr_meta_path.relative_to(args.data_dir)),
            "attribute_to_table_column": {
                a: attr_name_mapping.get(a, a) for a in attribute_names
            },
            "ignored_attributes": sorted(ignored_attr_mapping),
        },
        "summary": {
            "files": {
                "total": len(files),
                "processed": len(processed_files),
                "skipped": len(skipped_files),
            },
            "rows": {
                "kept": len(kept),
                "dropped_total": sum(drop_counts.values()),
                "dropped_by_reason": dict(drop_counts),
                "dropped_duplicates": len(dup_dropped),
            },
            "num_sections": df_out["section"].nunique() if not df_out.empty else 0,
        },
        "files": files_section,
        "attributes": {
            attr: {
                **({"num_files_empty": nfe} if (nfe := int(empty_attr_file_counts[attr])) > 0 else dict()),
                **({"num_files_unknown_code": uc} if len(uc := dict(unknown_codes.get(attr, Counter()))) > 0 else dict()),
            }
            for attr in attribute_names
        },
    }
    report_path = out / layout.PARSE_REPORT
    report_text = json.dumps(report, indent=2, ensure_ascii=False)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
