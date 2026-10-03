"""Display texts and HTML fragments shared by the pages."""

import html
import typing as T
from datetime import datetime

import irap_evaluation as ie

from ..archive import ActionLogEntry
from .routes import get_submission_path

#: The display name of each evaluation set (`datasets.EVALUATION_SET_NAMES`).
EVALUATION_SET_LABELS = {ie.REFERENCE_SET_NAME: "Reference set",
                         ie.MODEL_COMPATIBLE_SET_NAME: "Model-compatible set"}

_ACTION_DETAIL_LABELS = {"file_name": "File", "archive_member": "File in the archive",
                         "notes": "Notes", "original_seed": "Seed in the file",
                         "uploaded_file_sha256": "SHA-256 of the uploaded file",
                         "members": "Members", "intersect_segments": "Only common segments"}


def format_utc_time(time: datetime) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC")


def make_submission_link_html(submission_id: int, attributes: T.Sequence[str] = ()) -> str:
    """A link to the submission page (see `routes.get_submission_path`)."""
    path = html.escape(get_submission_path(submission_id, attributes))
    return f'<a href="{path}">#{submission_id}</a>'


def make_table_html(header_cells: T.Sequence[str], rows: T.Sequence[str],
                    fits_content: bool = False) -> str:
    """A `data-table` (`styles.css`) of escaped header texts and `<tr>` rows."""
    return make_table_html_from_header(
        "".join(f"<th>{html.escape(cell)}</th>" for cell in header_cells), rows, fits_content)


def make_table_html_from_header(header_html: str, rows: T.Sequence[str],
                                fits_content: bool = False) -> str:
    """A `data-table` (`styles.css`) of `<th>` header cells and `<tr>` rows.

    Args:
        fits_content: Whether the table is as wide as its content instead of its container.
    """
    classes = "data-table fits-content" if fits_content else "data-table"
    return (f'<table class="{classes}"><thead><tr>{header_html}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def make_ensemble_members_html(members: T.Sequence[T.Mapping[str, T.Any]]) -> str:
    """The members of an ensemble in the details of its action (see `ensembles`), with links."""
    return ", ".join(f"{make_submission_link_html(m['submission_id'])}"
                     f" {html.escape(m['run_label'])} (weight {m['weight']:g})" for m in members)


def _make_action_details_html(details: T.Mapping[str, T.Any]) -> str:
    def to_html(key: str, value: T.Any) -> str:
        if key == "members":
            return make_ensemble_members_html(value)
        if isinstance(value, list):
            return "<br>".join(html.escape(str(v)) for v in value) or "–"
        return html.escape("–" if value is None else str(value))

    return "".join(f'<div><span class="muted">{html.escape(_ACTION_DETAIL_LABELS.get(k, k))}:'
                   f"</span> {to_html(k, v)}</div>" for k, v in details.items())


def make_action_table_html(entries: T.Sequence[ActionLogEntry]) -> str:
    rows = [f"<tr><td>{format_utc_time(e.time)}</td><td>{html.escape(e.actor)}</td>"
            f"<td>{e.action}</td>"
            f"<td>{'' if e.submission_id is None else make_submission_link_html(e.submission_id)}"
            f"</td><td>{_make_action_details_html(e.details)}</td></tr>"
            for e in entries]
    return make_table_html(["Time", "Who", "Action", "Submission", "Details"], rows)
