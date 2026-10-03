import datetime

import irap_evaluation as ie
import pandas as pd
import pytest
from irap_evaluation.reports import coding_tables
from run_helpers import add_run, make_run

from irap_evaluation_server.coding_table_export import (
    parse_coding_date,
    write_submission_coding_table,
)


@pytest.mark.parametrize("is_hard", [False, True])
def test_coding_table_equals_irap_evaluation(archive, dataset_contexts, tmp_path, is_hard):
    context = dataset_contexts["vietnam"]
    submission = add_run(archive, dataset_contexts, make_run(dataset_contexts, is_hard=is_hard))
    path = tmp_path / "table.xlsx"
    write_submission_coding_table(archive, context, submission, path, coder_name="Bo",
                                  coding_date="2026-10-03")
    predictions = ie.read_predictions(archive.get_predictions_path(submission.id))
    template = coding_tables.load_coding_table_template()
    expected = coding_tables.predictions_to_coding_table(
        predictions, context.metadata, template, coder_name="Bo", coding_date="2026-10-03")
    segment_column = template.field_to_column["segment_id"]
    sheets = pd.read_excel(path, sheet_name=None, dtype={segment_column: str})
    assert list(sheets) == (["Coding table"] if is_hard else ["Coding table", "Confidence"])
    stored = sheets["Coding table"]
    assert list(stored.columns) == list(expected.columns)
    assert len(stored) == len(expected) == predictions.num_segments
    assert list(stored[segment_column]) == list(expected[segment_column])

    csv_path = tmp_path / "table.csv"
    write_submission_coding_table(archive, context, submission, csv_path, coder_name="Bo",
                                  coding_date="2026-10-03")
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".csv"] == ["table.csv"]
    with pytest.raises(ValueError, match="xlsx or .csv"):
        write_submission_coding_table(archive, context, submission, tmp_path / "t.json",
                                      coder_name="Bo", coding_date="2026-10-03")


def test_parse_coding_date():
    assert parse_coding_date("") == datetime.datetime.now(datetime.UTC).date().isoformat()
    assert parse_coding_date("2026-10-03") == "2026-10-03"
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        parse_coding_date("03.10.2026")
