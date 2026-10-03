"""The Submissions page (the list with the upload form), and the endpoints for uploading and
downloading prediction files."""

import dataclasses as dc
import html
import shutil
import typing as T
from pathlib import Path

from fastapi import HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from irap_evaluation.reports.evaluation_report import to_file_name
from nicegui import app, run, ui

from ..archive import Submission, SubmissionArchive, add_upload, is_archive_upload
from ..components.formatting import format_utc_time, make_submission_link_html, make_table_html
from ..components.native_controls import (
    create_file_upload_input,
    create_native_button,
    create_native_checkbox,
    create_native_input,
    create_native_select,
    set_status,
)
from ..components.page_frame import create_page_frame, get_submitter_name
from ..components.routes import (
    SCORES_PATH,
    PREDICTIONS_DOWNLOAD_PATH,
    SUBMISSIONS_PATH,
    UPLOAD_PATH,
    get_submission_path,
    make_query_path,
)
from ..datasets import DatasetContext

#: Uploaded files that are not stored are removed after this time, e.g. those of a closed page
#: whose client NiceGUI has not deleted.
_ABANDONED_UPLOAD_AGE_S = 24 * 3600


def get_download_file_name(submission: Submission) -> str:
    return to_file_name(f"{submission.run_label}.{submission.split}.predictions.parquet")


def make_submissions_table_html(submissions: T.Sequence[Submission]) -> str:
    rows = []
    for s in submissions:
        deleted = " (deleted)" if s.is_deleted else ""
        rows.append(
            f'<tr class="{"deleted" if s.is_deleted else ""}">'
            f"<td>{make_submission_link_html(s.id)}{deleted}</td>"
            f"<td>{html.escape(s.run_label)}</td><td>{s.dataset}</td><td>{s.split}</td>"
            f"<td>{s.output_kind}</td><td class='number'>{s.num_segments}</td>"
            f"<td class='number'>{s.num_attributes}</td><td>{html.escape(s.submitter)}</td>"
            f"<td>{format_utc_time(s.uploaded_at)}</td><td>{html.escape(s.description)}</td>"
            f"</tr>")
    return make_table_html(["#", "Run", "Dataset", "Split", "Output", "Segments", "Attributes",
                            "Submitter", "Uploaded", "Description"], rows)


def _parse_seed(text: str) -> int | None:
    if not text.strip():
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"The seed must be an integer, not {text.strip()!r}.") from None


def _write_upload(source: T.BinaryIO, path: Path) -> None:
    with path.open("wb") as file:
        shutil.copyfileobj(source, file)


@dc.dataclass
class _UploadFormState:
    """The state of the upload form of one page.

    Attributes:
        upload_id: The latest upload of the file input (see `create_file_upload_input`).
        upload_path: The file of that upload (see `SubmissionArchive.get_upload_path`), from
            when it is uploaded until a check takes it.
    """

    upload_id: int | None = None
    upload_path: Path | None = None
    file_name: str = ""
    description: str = ""
    seed_text: str = ""


#: Gets the ids of submissions that were added, deleted or restored, e.g. to score them. It is
#: called off the event loop, since it may read the database.
SubmissionsChangedHandler = T.Callable[[T.Sequence[int]], None]


def _create_upload_form(archive: SubmissionArchive,
                        dataset_contexts: T.Mapping[str, DatasetContext],
                        on_archive_stored: T.Callable[[], None],
                        on_submissions_changed: SubmissionsChangedHandler) -> None:
    """The form for uploading a prediction file or a .zip archive of the files of one run.

    A stored file opens its submission page. After an archive is stored, the page stays, its
    status lists the new submissions, and `on_archive_stored` is called, e.g. to refresh the
    list.
    """
    form = _UploadFormState()

    def remove_pending_upload() -> None:
        if form.upload_path is not None:
            form.upload_path.unlink(missing_ok=True)
            form.upload_path = None

    def on_upload_status_changed(status: dict) -> None:
        if status["status"] == "uploading":
            remove_pending_upload()
            form.upload_id = status["upload_id"]
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
        else:
            set_status(status_label, f"The upload failed: {status.get('message', status)}",
                       is_error=True)

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
        if not get_submitter_name():
            set_status(status_label, "Enter your name in the top bar first.", is_error=True)
            return
        try:
            seed = _parse_seed(form.seed_text)
        except ValueError as e:
            set_status(status_label, str(e), is_error=True)
            return
        # The check takes the file, so that a new upload does not remove it.
        upload_path, upload_id, file_name = form.upload_path, form.upload_id, form.file_name
        form.upload_path = None

        def give_back_upload() -> None:
            if form.upload_id == upload_id:  # Kept for a retry, e.g. with another seed.
                form.upload_path = upload_path
            else:
                upload_path.unlink(missing_ok=True)

        set_status(status_label, f"Checking {file_name}…")
        add_button.props["disabled"] = True
        add_button.update()
        try:
            results = await run.io_bound(
                add_upload, archive, dataset_contexts, upload_path, file_name=file_name,
                submitter=get_submitter_name(), description=form.description, seed=seed)
        except ValueError as e:  # Also irap_evaluation.PredictionFormatError.
            give_back_upload()
            set_status(status_label, f"{file_name} is not stored: {e}", is_error=True)
            return
        except Exception as e:
            give_back_upload()
            set_status(status_label, f"Internal error while checking {file_name}: {e!r}. See"
                                     f" the server log.", is_error=True)
            raise
        finally:
            add_button.props["disabled"] = False
            add_button.update()
        if results is None:  # The server is stopping.
            give_back_upload()
            set_status(status_label, f"{file_name} is not stored, since the server is"
                                     f" stopping.", is_error=True)
            return
        await run.io_bound(on_submissions_changed, [s.id for s, _ in results])
        if not is_archive_upload(upload_path):
            ui.navigate.to(get_submission_path(results[0][0].id))
            return
        stored = ", ".join(f"#{s.id} ({s.split})" for s, _ in results)
        set_status(status_label, f"Stored {stored} from {file_name}.")
        on_archive_stored()

    ui.context.client.on_delete(remove_pending_upload)

    with ui.element("section").classes("panel"):
        ui.label("Upload predictions").classes("section-title")
        ui.label("A .predictions.parquet file in the format of irap_evaluation, whose header gives"
                 " the dataset, split, method and seed, or a .zip archive of the files of one"
                 " run, one per split. The files of an archive are stored only if all are"
                 " accepted.").classes("muted")
        with ui.element("div").classes("form-row"):
            create_file_upload_input("File", UPLOAD_PATH, ".parquet,.zip",
                                     on_upload_status_changed)
            create_native_input("Description", "", lambda v: setattr(form, "description", v),
                                size=40)
            create_native_input("Seed", "", lambda v: setattr(form, "seed_text", v),
                                placeholder="from the file", size=10)
            add_button = create_native_button("Add", on_add_clicked)
        status_label = ui.label().classes("muted")


def register_submission_pages(archive: SubmissionArchive,
                              dataset_contexts: T.Mapping[str, DatasetContext],
                              on_submissions_changed: SubmissionsChangedHandler) -> None:
    @app.post(UPLOAD_PATH)
    async def receive_upload(file: UploadFile) -> dict:
        await run.io_bound(archive.remove_uploads, older_than_s=_ABANDONED_UPLOAD_AGE_S)
        path = archive.make_upload_path(file.filename or "")
        await run.io_bound(_write_upload, file.file, path)
        return {"status": "uploaded", "upload_name": path.name, "file_name": file.filename}

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

    @ui.page(SUBMISSIONS_PATH, title="Submissions · iRAP evaluation")
    def submissions_page(dataset: str = "", deleted: bool = False) -> None:
        filters = {"dataset": dataset if dataset in dataset_contexts else "", "deleted": deleted}

        def on_filter_changed(key: str, value: str | bool) -> None:
            filters[key] = value
            ui.navigate.history.replace(make_query_path(
                SUBMISSIONS_PATH, {k: "1" if v is True else v for k, v in filters.items()}))
            show_table.refresh()

        @ui.refreshable
        def show_table() -> None:
            submissions = [s for s in archive.list_submissions(include_deleted=filters["deleted"])
                           if filters["dataset"] in ("", s.dataset)]
            if not submissions:
                ui.label("No submissions.").classes("muted")
                return
            ui.html(make_submissions_table_html(submissions), sanitize=False).classes("w-full")

        with create_page_frame(SUBMISSIONS_PATH):
            _create_upload_form(archive, dataset_contexts, show_table.refresh,
                                on_submissions_changed)
            with ui.element("section").classes("panel"):
                ui.label("Submissions").classes("section-title")
                with ui.element("div").classes("form-row"):
                    create_native_select("Dataset", {"": "All", **{n: n for n in dataset_contexts}},
                                         filters["dataset"],
                                         lambda v: on_filter_changed("dataset", v))
                    create_native_checkbox("Show deleted", filters["deleted"],
                                           lambda v: on_filter_changed("deleted", v))
                show_table()
