import dataclasses as dc

import numpy as np
import pandas as pd
import pytest

from irap_data.attrs import IRAP_VIETNAM_ATTRS_ALL
from irap_evaluation.reports.coding_tables import (
    get_unfilled_columns,
    get_unmapped_attributes,
    load_coding_table_template,
    predictions_to_coding_table,
    predictions_to_confidence_table,
    write_coding_table,
)
from irap_evaluation.predictions import Predictions, to_class_indices, to_irap_codes

from synthetic_vietnam import ROAD_LENGTH, SPLIT, make_predictions


@pytest.fixture
def template():
    return load_coding_table_template()


def test_packaged_template(template):
    assert len(template.columns) == 69
    assert template.columns[:4] == ("Coder name", "Coding date", "Road survey date",
                                    "Image reference")
    # Every Vietnam attribute without a column has a documented reason (see the template).
    assert get_unmapped_attributes(template, IRAP_VIETNAM_ATTRS_ALL) == [
        "Service road", "Speed limit", "Intersecting road volume"]
    unfilled = get_unfilled_columns(template, ["Grade"])
    assert "Grade" not in unfilled and "Curvature" in unfilled and "Section" not in unfilled


def test_template_rejects_unknown_columns(template):
    with pytest.raises(ValueError, match="not in the template"):
        dc.replace(template, attribute_to_column={"Grade": "Gradient"})


def test_coding_table(metadata, template, tmp_path):
    segments = list(reversed(metadata.splits[SPLIT]))
    predictions = make_predictions(metadata.vocabulary, segments, name="m", seed=0)
    table = predictions_to_coding_table(predictions, metadata, template, coder_name="m",
                                        coding_date="2026-09-28")
    assert tuple(table.columns) == template.columns
    # Rows follow the road sequences, whatever the order of the predictions.
    assert list(table["Image reference"][:ROAD_LENGTH]) == \
           metadata.road_id_to_segment_id_sequence["road0"]
    first, second = (metadata.segment_id_to_road_data[s] for s in table["Image reference"][:2])
    assert table["Latitude end"][0] == second["lat"]
    assert table["Distance"][1] == pytest.approx(second["distance_km"])
    assert pd.isna(table["Latitude end"][ROAD_LENGTH - 1])  # the last segment of a road
    codes = to_irap_codes(predictions)
    row_of = {sid: i for i, sid in enumerate(predictions.segment_ids)}
    rows = [row_of[s] for s in table["Image reference"]]
    np.testing.assert_array_equal(table["Carriageway"],
                                  codes["Carriageway label"][rows])
    assert table["Speed limit"].isna().all()

    confidence = predictions_to_confidence_table(predictions, metadata, template)
    assert list(confidence["Image reference"]) == list(table["Image reference"])
    np.testing.assert_allclose(confidence["Lane width"].astype(float),
                               predictions.probs["Lane width"][rows].max(1))
    paths = write_coding_table(tmp_path / "table.xlsx", table, confidence)
    assert pd.read_excel(paths[0], sheet_name=None).keys() == {"Coding table", "Confidence"}
    csv_paths = write_coding_table(tmp_path / "table.csv", table, confidence)
    assert [p.name for p in csv_paths] == ["table.csv", "table.confidence.csv"]


def test_confidence_table_refuses_hard_predictions(metadata, template, predictions):
    hard = Predictions.from_class_indices(predictions.header, predictions.attribute_to_irap_codes,
                                          predictions.segment_ids, to_class_indices(predictions))
    with pytest.raises(ValueError, match="no probabilities"):
        predictions_to_confidence_table(hard, metadata, template)
