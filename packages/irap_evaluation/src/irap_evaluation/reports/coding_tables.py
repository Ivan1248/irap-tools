"""Export of predictions as iRAP coding tables."""

import dataclasses as dc
import functools
import json
import math
import typing as T
from importlib import resources
from pathlib import Path

import pandas as pd

from irap_data.metadata import IRAPMetadata, SegmentLocation, get_segment_location

from ..predictions import INVALID_IRAP_CODE, Predictions, select_segments, to_irap_codes

#: The fields that `predictions_to_coding_table` fills, besides the attributes.
CODING_TABLE_FIELDS = ("coder_name", "coding_date", "segment_id", "section", "distance_km",
                       "length_km", "latitude_start", "longitude_start", "latitude_end",
                       "longitude_end")


@dc.dataclass(frozen=True)
class CodingTableTemplate:
    """A coding-table layout.

    Attributes:
        columns: The column headers, in order.
        field_to_column: Each of `CODING_TABLE_FIELDS` -> its column.
        attribute_to_column: Attribute -> the column that receives its predicted iRAP code.

    Raises:
        ValueError: If a field is not mapped, or a field or attribute is mapped to a column that
            is not in `columns` or to the same column as another.
    """

    columns: tuple[str, ...]
    field_to_column: T.Mapping[str, str]
    attribute_to_column: T.Mapping[str, str]

    def __post_init__(self):
        if missing := [f for f in CODING_TABLE_FIELDS if f not in self.field_to_column]:
            raise ValueError(f"Fields without a column: {missing}.")
        mapped = [*self.field_to_column.values(), *self.attribute_to_column.values()]
        if unknown := sorted(set(mapped) - set(self.columns)):
            raise ValueError(f"Mapped columns not in the template: {unknown}.")
        if len(set(mapped)) != len(mapped):
            raise ValueError("Several fields or attributes are mapped to the same column.")


def load_coding_table_template(path: str | Path | None = None) -> CodingTableTemplate:
    """Loads a template JSON file. None loads the packaged iRAP layout."""
    text = (Path(path).read_text(encoding="utf-8") if path is not None else
            resources.files("irap_evaluation.reports").joinpath("data/coding_table_template.json")
            .read_text(encoding="utf-8"))
    d = json.loads(text)
    return CodingTableTemplate(tuple(d["columns"]), d["field_to_column"], d["attribute_to_column"])


def get_unmapped_attributes(template: CodingTableTemplate,
                            attributes: T.Iterable[str]) -> list[str]:
    """The attributes whose predictions have no column in the template."""
    return [a for a in attributes if a not in template.attribute_to_column]


def get_unfilled_columns(template: CodingTableTemplate, attributes: T.Iterable[str]) -> list[str]:
    """The template columns that stay blank for predictions of `attributes`."""
    filled = {*template.field_to_column.values(),
              *_get_attribute_columns(template, attributes).values()}
    return [c for c in template.columns if c not in filled]


def _select_in_road_order(predictions: Predictions, metadata: IRAPMetadata) -> Predictions:
    """The predictions ordered by road sequence and position, then those of segments of no
    sequence in their order in `predictions`."""
    road_rank = {road_id: i for i, road_id in enumerate(metadata.road_id_to_segment_id_sequence)}
    sequence_index = metadata.sequence_index

    def key(segment_id: str):
        if segment_id not in sequence_index:
            return (math.inf, 0)  # sorted() is stable
        road_id, position = sequence_index[segment_id]
        return (road_rank[road_id], position)

    return select_segments(predictions, sorted(predictions.segment_ids, key=key))


def _get_next_segment_ids_with_road_data(metadata: IRAPMetadata) -> dict[str, str]:
    """Segment ID -> the next segment with road data in its road sequence, if there is one.

    Segments without road data can lie between the two.
    """
    road_data = metadata.segment_id_to_road_data
    segment_id_to_next = {}
    for sequence in metadata.road_id_to_segment_id_sequence.values():
        next_id = None
        for sid in reversed(sequence):
            if next_id is not None:
                segment_id_to_next[sid] = next_id
            if sid in road_data:
                next_id = sid
    return segment_id_to_next


#: Tolerance of the distance at which the next segment of a section starts.
ADJACENCY_EPS_M = 1.0


def _is_adjacent(location: SegmentLocation, next_location: SegmentLocation) -> bool:
    """Whether `next_location` starts on the same section where `location` ends."""
    return (next_location.section == location.section and
            abs(next_location.distance_m - location.distance_m - location.length_m)
            <= ADJACENCY_EPS_M)


def _make_blank_table(template: CodingTableTemplate, num_rows: int) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series([None] * num_rows, dtype=object)
                         for column in template.columns})


def _get_attribute_columns(template: CodingTableTemplate,
                           attributes: T.Iterable[str]) -> dict[str, str]:
    """Attribute -> column, for the attributes that have a column."""
    return {a: template.attribute_to_column[a] for a in attributes
            if a in template.attribute_to_column}


def predictions_to_coding_table(
    predictions: Predictions,
    metadata: IRAPMetadata,
    template: CodingTableTemplate,
    *,
    coder_name: str,
    coding_date: str,
) -> pd.DataFrame:
    """One coding-table row per predicted segment, with the argmax iRAP code per attribute.

    Rows are ordered by road sequence and position. The end coordinates are the start of the
    next segment with road data if it starts where the segment ends, else blank. Invalid
    predictions and the columns that no field or predicted attribute fills are blank.

    Args:
        metadata: The metadata of the release, for the segment locations.
        coder_name: Written to the coder-name column, e.g. the model name.
        coding_date: Written to the coding-date column, `YYYY-MM-DD`.
    """
    selected = _select_in_road_order(predictions, metadata)
    road_data = metadata.segment_id_to_road_data
    segment_id_to_next = _get_next_segment_ids_with_road_data(metadata)
    get_location = functools.cache(lambda sid: get_segment_location(road_data[sid]))

    def get_location_fields(segment_id: str) -> dict[str, T.Any]:
        if segment_id not in road_data:
            return {}
        location = get_location(segment_id)
        fields = dict(section=location.section, distance_km=location.distance_m / 1000,
                      length_km=location.length_m / 1000, latitude_start=location.latitude_deg,
                      longitude_start=location.longitude_deg)
        next_id = segment_id_to_next.get(segment_id)
        if next_id is not None and _is_adjacent(location, end := get_location(next_id)):
            fields.update(latitude_end=end.latitude_deg, longitude_end=end.longitude_deg)
        return fields

    field_rows = [dict(coder_name=coder_name, coding_date=coding_date, segment_id=sid,
                       **get_location_fields(sid))
                  for sid in selected.segment_ids]
    table = _make_blank_table(template, selected.num_segments)
    for field, column in template.field_to_column.items():
        table[column] = [row.get(field) for row in field_rows]
    codes = to_irap_codes(selected)
    for attr, column in _get_attribute_columns(template, predictions.attributes).items():
        table[column] = pd.arrays.IntegerArray(codes[attr], codes[attr] == INVALID_IRAP_CODE)
    return table


def predictions_to_confidence_table(predictions: Predictions,
                                    metadata: IRAPMetadata,
                                    template: CodingTableTemplate) -> pd.DataFrame:
    """The probability of each exported code, in the layout and row order of
    `predictions_to_coding_table`.

    Only the segment ID column and the attribute columns are filled.

    Raises:
        ValueError: For hard predictions, which have no probabilities.
    """
    if predictions.output_kind == "hard":
        raise ValueError("Hard predictions have no probabilities for a confidence table.")
    selected = _select_in_road_order(predictions, metadata)
    table = _make_blank_table(template, selected.num_segments)
    table[template.field_to_column["segment_id"]] = list(selected.segment_ids)
    for attr, column in _get_attribute_columns(template, predictions.attributes).items():
        table[column] = selected.probs[attr].max(1)
    return table


def write_coding_table(path: str | Path, table: pd.DataFrame,
                       confidence_table: pd.DataFrame | None = None) -> list[Path]:
    """Writes a coding table as `.xlsx` or `.csv`, depending on the suffix of `path`.

    An `.xlsx` file gets the confidence table as a second sheet. With `.csv`, it goes to a
    separate `<stem>.confidence.csv` file.

    Returns:
        The written files.

    Raises:
        ValueError: For another suffix.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".xlsx":
        with pd.ExcelWriter(path) as writer:
            table.to_excel(writer, sheet_name="Coding table", index=False)
            if confidence_table is not None:
                confidence_table.to_excel(writer, sheet_name="Confidence", index=False)
        return [path]
    if path.suffix == ".csv":
        table.to_csv(path, index=False)
        if confidence_table is None:
            return [path]
        confidence_path = path.with_suffix(".confidence.csv")
        confidence_table.to_csv(confidence_path, index=False)
        return [path, confidence_path]
    raise ValueError(f"Expected a .xlsx or .csv path, got {path}.")
