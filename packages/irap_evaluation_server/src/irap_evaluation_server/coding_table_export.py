"""The export of the submissions of a model, e.g. one per split, as one iRAP coding table (see
`irap_evaluation.reports.coding_tables`), with the packaged template."""

import dataclasses as dc
import datetime
import tempfile
import typing as T
from pathlib import Path

import irap_evaluation as ie
from irap_evaluation.reports import coding_tables
from irap_evaluation.reports.evaluation_report import to_file_name

from .archive import ModelArchive, Submission
from .database import get_utc_now
from .datasets import DatasetContext
from .model_listing import ModelSummary

#: The file formats (suffixes without the dot).
CODING_TABLE_FORMATS = ("xlsx", "csv")


@dc.dataclass(frozen=True)
class CodingTableFile:
    file_name: str
    content: bytes


def parse_coding_date(text: str) -> str:
    """Parses the coding date of an export as `YYYY-MM-DD`, today (in UTC) for ''.

    Raises:
        ValueError: If it is not a date in that format.
    """
    if not text:
        return get_utc_now().date().isoformat()
    try:
        return datetime.date.fromisoformat(text).isoformat()
    except ValueError:
        raise ValueError(f"The coding date must be YYYY-MM-DD, got {text!r}.") from None


def write_model_coding_table(archive: ModelArchive, context: DatasetContext,
                             submissions: T.Sequence[Submission], path: Path, *,
                             coder_name: str, coding_date: str) -> None:
    """Writes the coding table of submissions of one model to `path`, a `.xlsx` or `.csv` file,
    with the segments of all of them in road order.

    A `.xlsx` file of probabilistic predictions gets the confidence table as its second sheet
    (`coding_tables.predictions_to_confidence_table`). A `.csv` file has only the codes, since
    the confidence table would be a second file.

    Args:
        context: The dataset of the submissions, whose metadata gives the segment locations.
        submissions: Of one model, at least one, e.g. its active submissions of some splits.
        coding_date: `YYYY-MM-DD`.

    Raises:
        ValueError: For another suffix, no submissions, or submissions of several models.
    """
    if path.suffix.lstrip(".") not in CODING_TABLE_FORMATS:
        raise ValueError(f"Expected a .xlsx or .csv path, got {path}.")
    if not submissions:
        raise ValueError("At least one submission is required.")
    if len({s.model.id for s in submissions}) > 1:
        raise ValueError("A coding table is of the submissions of one model.")
    predictions_seq = [ie.read_predictions(archive.get_predictions_path(s.id))
                       for s in submissions]
    template = coding_tables.load_coding_table_template()
    table = coding_tables.predictions_to_coding_table(
        predictions_seq, context.metadata, template, coder_name=coder_name,
        coding_date=coding_date)
    confidence_table = (
        coding_tables.predictions_to_confidence_table(predictions_seq, context.metadata,
                                                      template)
        if path.suffix == ".xlsx" and all(p.output_kind == "probs" for p in predictions_seq)
        else None)
    coding_tables.write_coding_table(path, table, confidence_table)


def export_model_coding_table(archive: ModelArchive,
                              dataset_contexts: T.Mapping[str, DatasetContext], model_id: int,
                              splits: T.Sequence[str], file_format: str, *, coder_name: str,
                              coding_date: str) -> CodingTableFile:
    """Makes the coding table of the active files of some splits of a model
    (`write_model_coding_table`).

    Args:
        splits: The splits of the active files, in the order of the file name.
        file_format: One of `CODING_TABLE_FORMATS`.
        coder_name: '' for the method name.
        coding_date: See `parse_coding_date`.

    Raises:
        LookupError: If there is no such model.
        ValueError: For another format, an invalid date, a dataset that is not configured, no
            splits, a split without an active file, and see `write_model_coding_table`.
    """
    if file_format not in CODING_TABLE_FORMATS:
        raise ValueError(f"Unknown format {file_format!r}.")
    coding_date = parse_coding_date(coding_date)
    model = archive.get_model(model_id)
    context = dataset_contexts.get(model.dataset)
    if context is None:
        raise ValueError(f"The dataset {model.dataset!r} is not configured.")
    split_to_active = ModelSummary.from_submissions(
        model, archive.list_model_submissions(model_id)).split_to_submission
    if not splits:
        raise ValueError("Choose at least one split.")
    if unknown := [s for s in splits if s not in split_to_active]:
        raise ValueError(f"The model has no active files of the splits {unknown}.")
    submissions = [split_to_active[s] for s in dict.fromkeys(splits)]
    file_name = to_file_name(f"{model.label}.{'+'.join(s.split for s in submissions)}"
                             f".coding_table.{file_format}")
    with tempfile.TemporaryDirectory(prefix="coding_table_") as directory:
        path = Path(directory) / file_name
        write_model_coding_table(archive, context, submissions, path,
                                 coder_name=coder_name or model.method_name,
                                 coding_date=coding_date)
        return CodingTableFile(file_name, path.read_bytes())
