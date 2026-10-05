"""The Models page (the list of models with the upload form), and the endpoints for uploading and
downloading prediction files."""

import dataclasses as dc
import html
import typing as T
from pathlib import Path

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from irap_evaluation.reports.evaluation_report import to_valid_file_name
from nicegui import app, run, ui

from ..accounts import can_write
from ..archive import ModelArchive, Submission
from ..components.account_sessions import AccountSessions
from ..components.confirmation_dialog import confirm_changes
from ..components.formatting import (
    format_splits,
    format_utc_time,
    make_code_spans_html,
    make_model_link_html,
    make_table_html_from_header,
)
from ..components.native_controls import (
    create_file_upload_input,
    create_native_button,
    create_native_checkbox,
    create_native_input,
    create_native_select,
    set_native_input_placeholder,
    set_status,
)
from ..components.page_frame import create_page_frame, create_write_permission_panel
from ..components.routes import (
    MODELS_PATH,
    PREDICTIONS_DOWNLOAD_PATH,
    SCORES_PATH,
    UPLOAD_PATH,
    get_model_path,
    make_query_path,
)
from ..datasets import DatasetContext
from ..method_ranking import describe_split_use
from ..model_listing import MethodGroup, ModelSummary, summarize_models_by_method
from ..scoring_worker import ScoringWorker
from ..uploads import (
    MAX_UPLOAD_NUM_BYTES,
    MAX_UPLOAD_SIZE_TEXT,
    apply_upload,
    parse_optional_text,
    parse_seed,
    plan_upload,
    read_upload_headers,
)

#: Uploaded files that are not stored are removed after this time, e.g. those of a closed page
#: whose client NiceGUI has not deleted.
_ABANDONED_UPLOAD_AGE_S = 24 * 3600
_UPLOAD_TOO_LARGE_MESSAGE = f"A file can have at most {MAX_UPLOAD_SIZE_TEXT}."
_UPLOAD_FORM_TITLE = "Upload predictions"

#: The columns of the models table after the split columns.
_SUMMARY_COLUMNS = ("Output", "Segments", "Attributes", "Description", "Updated")


def get_download_file_name(submission: Submission) -> str:
    return to_valid_file_name(f"{submission.label}.{submission.split}.predictions.parquet")


def _make_submission_cell_html(summary: ModelSummary, split: str) -> str:
    """Makes the cell of a split: a link to the model page with the scores of its active
    submission, the submission details in the title, and a mark if the model used the split."""
    submission = summary.split_to_submission.get(split)
    if submission is None:
        return '<td class="mark"></td>'
    title = (f"File #{submission.id}, uploaded {format_utc_time(submission.uploaded_at)} by"
             f" {submission.submitter}, {submission.num_segments} segments,"
             f" {submission.num_attributes} attributes")
    link = make_model_link_html(summary.model.id, "✓", split)
    mark = describe_split_use(submission.model_info.get_split_use(split), split)
    mark_html = f'<div class="interval">{html.escape(mark)}</div>' if mark else ""
    return f'<td class="mark" title="{html.escape(title)}">{link}{mark_html}</td>'


def _make_summary_cells_html(summary: ModelSummary) -> str:
    attribute_range = summary.attribute_range
    attributes = ("" if attribute_range is None else str(attribute_range[0])
                  if attribute_range[0] == attribute_range[1]
                  else f"{attribute_range[0]}–{attribute_range[1]}")
    return (f"<td>{summary.output_kind or ''}</td>"
            f'<td class="number">{summary.num_segments or ""}</td>'
            f'<td class="number">{attributes}</td>'
            f"<td>{html.escape(summary.model.description)}</td>"
            f"<td>{format_utc_time(summary.updated_at)}</td>")


def _make_method_row_html(group: MethodGroup, num_columns: int) -> str:
    info = group.model_info
    splits = ""
    if info is not None:
        splits = f"trained on {format_splits(info.training_splits)}"
        if info.early_stopping_splits:
            splits += f", early stopping on {format_splits(info.early_stopping_splits)}"
        splits = f' <span class="muted">· {html.escape(splits)}</span>'
    method_name = ("" if group.shown_method_name == group.method_name
                   else f' <span class="muted">({html.escape(group.method_name)})</span>')
    return (f'<tr class="method-row"><td colspan="{num_columns}">'
            f"<b>{html.escape(group.shown_method_name)}</b>{method_name}{splits}</td></tr>")


def make_models_table_html(groups: T.Sequence[MethodGroup], split_names: T.Sequence[str]) -> str:
    """Makes the table of the models of a dataset under their methods
    (`model_listing.summarize_models_by_method`), with a column per split.

    Args:
        split_names: The splits of the dataset, the order of the split columns.
    """
    header_html = "".join([
        "<th>Model</th>", *(f'<th class="mark">{html.escape(s)}</th>' for s in split_names),
        *(f"<th>{html.escape(c)}</th>" for c in _SUMMARY_COLUMNS)])
    num_columns = 1 + len(split_names) + len(_SUMMARY_COLUMNS)
    rows = []
    for group in groups:
        rows.append(_make_method_row_html(group, num_columns))
        for summary in group.models:
            model = summary.model
            deleted = " (deleted)" if model.is_deleted else ""
            rows.append(
                f'<tr class="{"deleted" if model.is_deleted else ""}">'
                f"<td>{make_model_link_html(model.id, model.shown_label)}{deleted}</td>"
                f"{''.join(_make_submission_cell_html(summary, s) for s in split_names)}"
                f"{_make_summary_cells_html(summary)}</tr>")
    return make_table_html_from_header(header_html, rows)


async def _write_upload(request: Request, path: Path) -> None:
    """Writes the body of an upload request to `path`. Removes `path` if that fails, also if the
    client disconnects.

    Raises:
        HTTPException: 413 if the body has more than `uploads.MAX_UPLOAD_NUM_BYTES`.
    """
    num_bytes = 0
    try:
        with path.open("wb") as file:
            async for chunk in request.stream():
                num_bytes += len(chunk)
                if num_bytes > MAX_UPLOAD_NUM_BYTES:
                    raise HTTPException(status_code=413, detail=_UPLOAD_TOO_LARGE_MESSAGE)
                file.write(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


@dc.dataclass
class _UploadFormState:
    """The state of the upload form of one page.

    Attributes:
        upload_id: The latest upload of the file input (see `create_file_upload_input`).
        upload_path: The file of that upload (see `ModelArchive.get_upload_path`), from
            when it is uploaded until a check takes it.
    """

    upload_id: int | None = None
    upload_path: Path | None = None
    file_name: str = ""
    description: str = ""
    method_display_name: str = ""
    seed_text: str = ""


def _make_method_display_name_placeholder(archive: ModelArchive, upload_path: Path,
                                          file_name: str) -> str:
    """Makes the placeholder of the method display name input for an upload, with the display
    name that a blank input keeps (`ModelArchive.get_inherited_method_display_name`)."""
    headers = read_upload_headers(upload_path, file_name=file_name)
    return f"blank: {archive.get_inherited_method_display_name(headers) or 'none'}"


def _create_upload_form(archive: ModelArchive,
                        dataset_contexts: T.Mapping[str, DatasetContext],
                        worker: ScoringWorker, get_actor: T.Callable[[], str]) -> None:
    """Creates the form for uploading a prediction file or a .zip archive of the files of one
    model.

    The upload is checked and planned (`uploads.plan_upload`), and the user confirms its changes,
    if any (`archive.ModelUpdate.confirmations`). A stored upload opens the page of its model.

    Args:
        get_actor: Returns the name of the signed-in account that can write, or raises
            `ValueError`.
    """
    form = _UploadFormState()

    def remove_pending_upload() -> None:
        if form.upload_path is not None:
            form.upload_path.unlink(missing_ok=True)
            form.upload_path = None

    async def show_inherited_method_display_name(upload_path: Path, upload_id: int,
                                                 file_name: str) -> None:
        """Shows the method display name that a blank input keeps as the input's placeholder.
        It shows none if the headers cannot be read, e.g. of an invalid file, since Add then
        reports why."""
        client = ui.context.client
        try:
            placeholder = await run.io_bound(_make_method_display_name_placeholder, archive,
                                             upload_path, file_name)
        except (OSError, ValueError):  # Also irap_evaluation.PredictionFormatError.
            return
        # None if the server is stopping.
        if placeholder is not None and form.upload_id == upload_id and not client.is_deleted:
            set_native_input_placeholder(display_name_input, placeholder)

    async def on_upload_status_changed(status: dict) -> None:
        if status["status"] == "uploading":
            remove_pending_upload()
            form.upload_id = status["upload_id"]
            set_native_input_placeholder(display_name_input, "")
            set_status(status_label, f"Uploading {status['file_name']}…")
            return
        upload_path = (archive.get_upload_path(status["upload_name"])
                       if status["status"] == "uploaded" else None)
        if status["upload_id"] != form.upload_id:  # A newer upload replaced this one.
            if upload_path is not None:
                upload_path.unlink(missing_ok=True)
        elif upload_path is not None:
            form.upload_path, form.file_name = upload_path, status["file_name"]
            set_status(status_label, f"{form.file_name} is uploaded. Click Add to check and"
                                     f" store it.")
            await show_inherited_method_display_name(upload_path, form.upload_id,
                                                     form.file_name)
        else:
            set_status(status_label, f"The upload failed: {status.get('message', status)}",
                       is_error=True)

    def set_adding(is_adding: bool) -> None:
        add_button.props["disabled"] = is_adding
        add_button.update()

    async def on_add_clicked() -> None:
        if form.upload_path is None:
            set_status(status_label, "Choose a file first.", is_error=True)
            return
        if not form.upload_path.exists():
            form.upload_path = None
            set_status(status_label, f"{form.file_name} is no longer on the server, since"
                                     f" uploads are removed after"
                                     f" {_ABANDONED_UPLOAD_AGE_S / 3600:g} h. Choose it again.",
                       is_error=True)
            return
        try:
            actor = get_actor()
            seed = parse_seed(form.seed_text)
        except ValueError as e:
            set_status(status_label, str(e), is_error=True)
            return
        # The check takes the file, so that a new upload does not remove it.
        upload_path, upload_id, file_name = form.upload_path, form.upload_id, form.file_name
        form.upload_path = None

        client = ui.context.client

        def give_back_upload() -> None:
            # Kept for a retry, e.g. with another seed, unless the page is closed
            # (`remove_pending_upload`) or has a newer upload.
            if form.upload_id == upload_id and not client.is_deleted:
                form.upload_path = upload_path
            else:
                upload_path.unlink(missing_ok=True)

        set_status(status_label, f"Checking {file_name}…")
        set_adding(True)
        try:
            submissions = await _check_and_store(archive, dataset_contexts, upload_path,
                                                 file_name=file_name, seed=seed,
                                                 description=form.description,
                                                 method_display_name=form.method_display_name,
                                                 actor=actor)
        except ValueError as e:  # Also irap_evaluation.PredictionFormatError.
            give_back_upload()
            if not client.is_deleted:
                set_status(status_label, f"{file_name} is not stored: {e}", is_error=True)
            return
        except Exception as e:
            give_back_upload()
            if not client.is_deleted:
                set_status(status_label, f"Internal error while checking {file_name}: {e!r}."
                                         f" See the server log.", is_error=True)
            raise
        finally:
            if not client.is_deleted:
                set_adding(False)
        if not submissions:  # Cancelled, or the server is stopping.
            give_back_upload()
            if not client.is_deleted:
                set_status(status_label, f"{file_name} is not stored.")
            return
        await run.io_bound(worker.update_model_submissions, submissions[0].model.id)
        if not client.is_deleted:
            ui.navigate.to(get_model_path(submissions[0].model.id, submissions[0].split))

    ui.context.client.on_delete(remove_pending_upload)

    with ui.element("section").classes("panel"):
        ui.label(_UPLOAD_FORM_TITLE).classes("section-title")
        ui.html(make_code_spans_html(
            "A `.predictions.parquet` file in the `irap_evaluation` format, whose header gives"
            " the dataset, split, method, seed and training splits, or a `.zip` archive with"
            " one such file per split of one model, stored only if all are accepted. A file"
            " of a split that the model already has replaces its active file, which is deleted"
            " but can still be downloaded."),
            sanitize=False).classes("muted")
        with ui.element("div").classes("form-row"):
            create_file_upload_input("File", UPLOAD_PATH, ".parquet,.zip",
                                     on_upload_status_changed)
            display_name_input = create_native_input(
                "Method display name", "", lambda v: setattr(form, "method_display_name", v),
                size=30)
            create_native_input("Seed", "", lambda v: setattr(form, "seed_text", v),
                                placeholder="from the file", size=10)
            create_native_input("Description", "", lambda v: setattr(form, "description", v),
                                placeholder="blank: keep the model's", size=40)
            add_button = create_native_button("Add", on_add_clicked)
        status_label = ui.label().classes("muted")


async def _check_and_store(archive: ModelArchive,
                           dataset_contexts: T.Mapping[str, DatasetContext], upload_path: Path, *,
                           file_name: str, seed: int | None, description: str,
                           method_display_name: str, actor: str) -> list[Submission]:
    """Plans an upload, asks the user to confirm its changes, if any, and applies it.

    Args:
        description, method_display_name: As entered, blank to keep the stored one.
        actor: The name of the signed-in account.

    Returns:
        The new submissions, or [] if the user cancelled, closed the page, or the server is
        stopping.

    Raises:
        ValueError: See `uploads.plan_upload` and `uploads.apply_upload`.
    """
    client = ui.context.client
    planned = await run.io_bound(plan_upload, archive, dataset_contexts, upload_path,
                                 file_name=file_name, seed=seed,
                                 description=parse_optional_text(description),
                                 method_display_name=parse_optional_text(method_display_name))
    if planned is None:  # The server is stopping.
        return []
    update = planned.update
    try:
        if client.is_deleted:
            return []
        if update.confirmations and not await confirm_changes(
                f"Upload of {update.label} on {update.dataset}", update.confirmations, "Store"):
            return []
        return await run.io_bound(apply_upload, archive, planned, submitter=actor) or []
    finally:
        planned.discard()  # Also if the page is closed, or the server is stopping.


def register_models_pages(archive: ModelArchive,
                          dataset_contexts: T.Mapping[str, DatasetContext],
                          worker: ScoringWorker, sessions: AccountSessions) -> None:
    @app.post(UPLOAD_PATH)
    async def receive_upload(request: Request, file_name: str = "") -> dict:
        # Before the body is read, so that visitors cannot make the server store it.
        try:
            sessions.require_writer_name()
        except ValueError as e:
            raise HTTPException(status_code=403, detail=str(e)) from e
        if int(request.headers.get("content-length", 0)) > MAX_UPLOAD_NUM_BYTES:
            raise HTTPException(status_code=413, detail=_UPLOAD_TOO_LARGE_MESSAGE)
        await run.io_bound(archive.remove_uploads, older_than_s=_ABANDONED_UPLOAD_AGE_S)
        path = archive.make_upload_path(file_name)
        await _write_upload(request, path)
        return {"status": "uploaded", "upload_name": path.name, "file_name": file_name}

    @app.get(PREDICTIONS_DOWNLOAD_PATH)
    def download_predictions(submission_id: int) -> FileResponse:
        try:
            submission = archive.get_submission(submission_id)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        return FileResponse(archive.get_predictions_path(submission_id),
                            filename=get_download_file_name(submission),
                            media_type="application/vnd.apache.parquet")

    @ui.page("/")
    def index_page() -> RedirectResponse:
        return RedirectResponse(SCORES_PATH)

    @ui.page(MODELS_PATH, title="Models · iRAP evaluation")
    def models_page(dataset: str = "", deleted: bool = False) -> None:
        account = sessions.get_account()
        if dataset and dataset not in dataset_contexts:
            with create_page_frame(MODELS_PATH, account):
                ui.label(f"Unknown dataset {dataset!r}. Datasets:"
                         f" {', '.join(dataset_contexts)}.").classes("error")
            return
        filters = {"dataset": dataset, "deleted": deleted}

        def on_filter_changed(key: str, value: str | bool) -> None:
            filters[key] = value
            ui.navigate.history.replace(make_query_path(
                MODELS_PATH, {k: "1" if v is True else v for k, v in filters.items()}))
            show_tables.refresh()

        @ui.refreshable
        def show_tables() -> None:
            models = archive.list_models(include_deleted=filters["deleted"])
            submissions = archive.list_submissions(include_deleted_models=filters["deleted"])
            datasets = [filters["dataset"]] if filters["dataset"] else list(dataset_contexts)
            shown_datasets = [d for d in datasets if any(m.dataset == d for m in models)]
            if not shown_datasets:
                ui.label("No models.").classes("muted")
            for name in shown_datasets:
                groups = summarize_models_by_method([m for m in models if m.dataset == name],
                                                    submissions)
                ui.label(name).classes("subsection-title")
                ui.html(make_models_table_html(groups, dataset_contexts[name].split_names),
                        sanitize=False).classes("w-full overflow-x-auto")

        with create_page_frame(MODELS_PATH, account):
            if can_write(account):
                _create_upload_form(archive, dataset_contexts, worker,
                                    sessions.require_writer_name)
            # Visitors see no upload panel, which would push the models down.
            elif account is not None:
                create_write_permission_panel(account, _UPLOAD_FORM_TITLE, "upload predictions")
            with ui.element("section").classes("panel"):
                ui.label("Models").classes("section-title")
                with ui.element("div").classes("form-row"):
                    create_native_select("Dataset", {"": "All", **{n: n for n in dataset_contexts}},
                                         filters["dataset"],
                                         lambda v: on_filter_changed("dataset", v))
                    create_native_checkbox("Show deleted", filters["deleted"],
                                           lambda v: on_filter_changed("deleted", v))
                ui.label("A ✓ opens the Model page with the scores of the split's active file."
                         " Hover over it for the file details.").classes("muted")
                show_tables()
