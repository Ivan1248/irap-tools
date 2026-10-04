"""Display texts and HTML fragments shared by the pages."""

import html
import typing as T
from datetime import datetime

import irap_evaluation as ie

from ..archive import ActionLogEntry, Model
from .routes import get_model_path

#: The display name of each evaluation set (`datasets.EVALUATION_SET_NAMES`).
EVALUATION_SET_LABELS = {ie.REFERENCE_SET_NAME: "Reference set",
                         ie.MODEL_COMPATIBLE_SET_NAME: "Model-compatible set"}

_ACTION_DETAIL_LABELS = {"file_name": "File", "archive_member": "File in the archive",
                         "notes": "Notes", "original_seed": "Seed in the file",
                         "uploaded_file_sha256": "SHA-256 of the uploaded file",
                         "replaced_submission_id": "Replaced file",
                         "members": "Members", "intersect_segments": "Only common segments",
                         "old_description": "Old description", "description": "Description",
                         "by": "Part of"}


def format_utc_time(time: datetime) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC")


def format_splits(splits: T.Sequence[str] | None) -> str:
    """Formats training or early stopping splits (`irap_evaluation.ModelInfo`): 'unknown' for
    None, 'none' for ()."""
    if splits is None:
        return "unknown"
    return ", ".join(splits) or "none"


def format_context_offsets(offsets: T.Sequence[int] | None) -> str:
    """Formats context offsets: 'unknown' for None."""
    return "unknown" if offsets is None else ", ".join(map(str, offsets))


def make_model_link_html(model_id: int, text: str, split: str = "",
                         attributes: T.Sequence[str] = ()) -> str:
    """Makes a link to the Model page (`routes.get_model_path`) with an escaped text."""
    path = html.escape(get_model_path(model_id, split, attributes))
    return f'<a href="{path}">{html.escape(text)}</a>'


def make_code_spans_html(text: str) -> str:
    """Escapes `text` and puts each span between backticks in `<code>`, e.g. a file name in a
    note.

    Raises:
        ValueError: If a backtick is unpaired.
    """
    parts = text.split("`")
    if len(parts) % 2 == 0:
        raise ValueError(f"Unpaired backtick in {text!r}.")
    return "".join(f"<code>{html.escape(p)}</code>" if i % 2 == 1 else html.escape(p)
                   for i, p in enumerate(parts))


def make_table_html(header_cells: T.Sequence[str], rows: T.Sequence[str],
                    fits_content: bool = False) -> str:
    """Makes a `data-table` (`styles.css`) of escaped header texts and `<tr>` rows."""
    return make_table_html_from_header(
        "".join(f"<th>{html.escape(cell)}</th>" for cell in header_cells), rows, fits_content)


def make_table_html_from_header(header_html: str, rows: T.Sequence[str],
                                fits_content: bool = False) -> str:
    """Makes a `data-table` (`styles.css`) of `<th>` header cells and `<tr>` rows.

    Args:
        fits_content: Whether the table is as wide as its content instead of its container.
    """
    classes = "data-table fits-content" if fits_content else "data-table"
    return (f'<table class="{classes}"><thead><tr>{header_html}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def make_ensemble_members_html(members: T.Sequence[T.Mapping[str, T.Any]]) -> str:
    """Formats the members of an ensemble from the details of its action (`ensembles`), with
    links to their models."""
    return ", ".join(f"{make_model_link_html(m['model_id'], m['label'])}"
                     f" #{m['submission_id']} (weight {m['weight']:g})" for m in members)


def _make_action_details_html(details: T.Mapping[str, T.Any]) -> str:
    def to_html(key: str, value: T.Any) -> str:
        if key == "members":
            return make_ensemble_members_html(value)
        if key == "replaced_submission_id":
            return f"#{value}"
        if key == "by":
            return _ACTION_TEXTS[value]
        if isinstance(value, list):
            return "<br>".join(html.escape(str(v)) for v in value) or "–"
        return html.escape("–" if value in (None, "") else str(value))

    return "".join(f'<div><span class="muted">{html.escape(_ACTION_DETAIL_LABELS.get(k, k))}:'
                   f"</span> {to_html(k, v)}</div>" for k, v in details.items())


#: The text of each `archive.ActionKind` in the action log.
_ACTION_TEXTS = {"upload": "upload", "ensemble": "ensemble", "delete": "delete file",
                 "delete_model": "delete model", "restore_model": "restore model",
                 "edit_description": "edit description"}


def make_action_table_html(entries: T.Sequence[ActionLogEntry],
                           model_id_to_model: T.Mapping[int, Model]) -> str:
    """Makes the table of actions, with links to their models.

    Args:
        model_id_to_model: Has the model of each action.
    """
    def make_row(entry: ActionLogEntry) -> str:
        model = model_id_to_model[entry.model_id]
        submission = "" if entry.submission_id is None else f"#{entry.submission_id}"
        return (f"<tr><td>{format_utc_time(entry.time)}</td><td>{html.escape(entry.actor)}</td>"
                f"<td>{_ACTION_TEXTS[entry.action]}</td>"
                f"<td>{make_model_link_html(model.id, model.label)}</td>"
                f"<td>{submission}</td><td>{_make_action_details_html(entry.details)}</td></tr>")

    return make_table_html(["Time", "Who", "Action", "Model", "File", "Details"],
                           [make_row(e) for e in entries])
