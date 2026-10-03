"""The export of a submission as an iRAP coding table (see
`irap_evaluation.reports.coding_tables`), with the packaged template."""

import datetime
from pathlib import Path

import irap_evaluation as ie
from irap_evaluation.reports import coding_tables

from .archive import Submission, SubmissionArchive
from .database import get_utc_now
from .datasets import DatasetContext

#: The file formats (suffixes without the dot).
CODING_TABLE_FORMATS = ("xlsx", "csv")


def parse_coding_date(text: str) -> str:
    """The coding date of an export, `YYYY-MM-DD`, today (in UTC) for ''.

    Raises:
        ValueError: If it is not a date in that format.
    """
    if not text:
        return get_utc_now().date().isoformat()
    try:
        return datetime.date.fromisoformat(text).isoformat()
    except ValueError:
        raise ValueError(f"The coding date must be YYYY-MM-DD, got {text!r}.") from None


def write_submission_coding_table(archive: SubmissionArchive, context: DatasetContext,
                                  submission: Submission, path: Path, *, coder_name: str,
                                  coding_date: str) -> None:
    """Writes the coding table of a submission to `path`, a `.xlsx` or `.csv` file.

    A `.xlsx` file of probabilistic predictions gets the confidence table as its second sheet
    (`coding_tables.predictions_to_confidence_table`). A `.csv` file has only the codes, since
    the confidence table would be a second file.

    Args:
        context: The dataset of the submission, whose metadata gives the segment locations.
        coding_date: `YYYY-MM-DD`.

    Raises:
        ValueError: For another suffix.
    """
    if path.suffix.lstrip(".") not in CODING_TABLE_FORMATS:
        raise ValueError(f"Expected a .xlsx or .csv path, got {path}.")
    predictions = ie.read_predictions(archive.get_predictions_path(submission.id))
    template = coding_tables.load_coding_table_template()
    table = coding_tables.predictions_to_coding_table(
        predictions, context.metadata, template, coder_name=coder_name, coding_date=coding_date)
    confidence_table = (
        coding_tables.predictions_to_confidence_table(predictions, context.metadata, template)
        if path.suffix == ".xlsx" and predictions.output_kind == "probs" else None)
    coding_tables.write_coding_table(path, table, confidence_table)
