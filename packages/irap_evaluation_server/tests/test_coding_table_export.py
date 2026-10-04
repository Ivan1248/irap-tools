import datetime
import io
import json

import irap_evaluation as ie
import pandas as pd
import pytest
from irap_data.metadata import MetaFiles
from irap_evaluation.reports import coding_tables
from run_helpers import OTHER_SPLIT, add_model, make_model, with_split
from synthetic_vietnam import SPLIT, make_predictions

from irap_evaluation_server.coding_table_export import (
    export_model_coding_table,
    parse_coding_date,
    write_model_coding_table,
)
from irap_evaluation_server.config import DatasetConfig
from irap_evaluation_server.datasets import load_dataset_contexts


@pytest.mark.parametrize("is_hard", [False, True])
def test_coding_table_equals_irap_evaluation(archive, dataset_contexts, tmp_path, is_hard):
    context = dataset_contexts["vietnam"]
    submission = add_model(archive, dataset_contexts,
                           make_model(dataset_contexts, is_hard=is_hard))
    path = tmp_path / "table.xlsx"
    write_model_coding_table(archive, context, [submission], path, coder_name="Bo",
                             coding_date="2026-10-03")
    predictions = ie.read_predictions(archive.get_predictions_path(submission.id))
    template = coding_tables.load_coding_table_template()
    expected = coding_tables.predictions_to_coding_table(
        [predictions], context.metadata, template, coder_name="Bo", coding_date="2026-10-03")
    segment_column = template.field_to_column["segment_id"]
    sheets = pd.read_excel(path, sheet_name=None, dtype={segment_column: str})
    assert list(sheets) == (["Coding table"] if is_hard else ["Coding table", "Confidence"])
    stored = sheets["Coding table"]
    assert list(stored.columns) == list(expected.columns)
    assert len(stored) == len(expected) == predictions.num_segments
    assert list(stored[segment_column]) == list(expected[segment_column])

    csv_path = tmp_path / "table.csv"
    write_model_coding_table(archive, context, [submission], csv_path, coder_name="Bo",
                             coding_date="2026-10-03")
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".csv"] == ["table.csv"]
    with pytest.raises(ValueError, match="xlsx or .csv"):
        write_model_coding_table(archive, context, [submission], tmp_path / "t.json",
                                 coder_name="Bo", coding_date="2026-10-03")


@pytest.fixture
def disjoint_split_contexts(metadata_dir):
    """The dataset with the roads 0 and 1 in `SPLIT` and the others in `OTHER_SPLIT`."""
    splits_path = metadata_dir / MetaFiles.SPLITS
    road_path = metadata_dir / MetaFiles.ROAD_ID_TO_SEGMENT_ID_SEQUENCE
    sequences = list(json.loads(road_path.read_text(encoding="utf-8")).values())
    splits = {SPLIT: [s for seq in sequences[:2] for s in seq],
              OTHER_SPLIT: [s for seq in sequences[2:] for s in seq]}
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    return load_dataset_contexts({"vietnam": DatasetConfig(
        dataset_dir=metadata_dir, metadata_dir=metadata_dir, analysis_splits=())})


def test_coding_table_of_several_splits(archive, disjoint_split_contexts, tmp_path):
    context = disjoint_split_contexts["vietnam"]
    metadata = context.metadata
    first, second = (
        add_model(archive, disjoint_split_contexts, with_split(make_predictions(
            metadata.vocabulary, list(metadata.splits[split]), name="m", seed=0), split))
        for split in (SPLIT, OTHER_SPLIT))
    path = tmp_path / "table.csv"
    write_model_coding_table(archive, context, [first, second], path, coder_name="Bo",
                             coding_date="2026-10-03")
    assert len(pd.read_csv(path)) == first.num_segments + second.num_segments
    other = add_model(archive, disjoint_split_contexts,
                      make_model(disjoint_split_contexts, name="other"))
    with pytest.raises(ValueError, match="one model"):
        write_model_coding_table(archive, context, [first, other], path, coder_name="Bo",
                                 coding_date="2026-10-03")


def test_export_model_coding_table(archive, disjoint_split_contexts):
    metadata = disjoint_split_contexts["vietnam"].metadata
    first, second = (
        add_model(archive, disjoint_split_contexts, with_split(make_predictions(
            metadata.vocabulary, list(metadata.splits[split]), name="m", seed=0), split))
        for split in (SPLIT, OTHER_SPLIT))

    def export(splits, file_format="csv", coding_date=""):
        return export_model_coding_table(archive, disjoint_split_contexts, first.model.id,
                                         splits, file_format, coder_name="",
                                         coding_date=coding_date)

    coding_table = export([OTHER_SPLIT, SPLIT])
    assert coding_table.file_name == f"{first.model.label}.{OTHER_SPLIT}_{SPLIT}.coding_table.csv"
    stored = pd.read_csv(io.BytesIO(coding_table.content))
    assert len(stored) == first.num_segments + second.num_segments
    coder_column = coding_tables.load_coding_table_template().field_to_column["coder_name"]
    assert set(stored[coder_column]) == {"m"}  # The method name.
    for splits, file_format, coding_date, message in [
            ([], "csv", "", "at least one split"),
            (["missing"], "csv", "", "no active files"),
            ([SPLIT], "json", "", "Unknown format"),
            ([SPLIT], "csv", "03.10.2026", "YYYY-MM-DD")]:
        with pytest.raises(ValueError, match=message):
            export(splits, file_format, coding_date)
    with pytest.raises(ValueError, match="formula"):
        export_model_coding_table(archive, disjoint_split_contexts, first.model.id, [SPLIT],
                                  "csv", coder_name="=1+1", coding_date="")
    export_model_coding_table(archive, disjoint_split_contexts, first.model.id, [SPLIT], "xlsx",
                              coder_name="=1+1", coding_date="")  # As text.


def test_parse_coding_date():
    assert parse_coding_date("") == datetime.datetime.now(datetime.UTC).date().isoformat()
    assert parse_coding_date("2026-10-03") == "2026-10-03"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_coding_date("03.10.2026")
