"""The page of one submission: its details, scores, coding-table export and actions, and the
endpoints for downloading its scores and coding table."""

import html
import shutil
import tempfile
import typing as T
import urllib.parse
from pathlib import Path

import irap_evaluation as ie
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from irap_evaluation.reports.evaluation_report import to_file_name
from nicegui import app, run, ui
from starlette.background import BackgroundTask

from ..archive import ADDING_ACTION_KINDS, Submission, SubmissionArchive
from ..coding_table_export import (
    CODING_TABLE_FORMATS,
    parse_coding_date,
    write_submission_coding_table,
)
from ..components.analysis_view import AnalysisView
from ..components.attribute_filter import AttributeFilter, AttributeSubset
from ..components.formatting import (
    EVALUATION_SET_LABELS,
    format_utc_time,
    make_action_table_html,
    make_ensemble_members_html,
)
from ..components.native_controls import create_native_button, set_status
from ..components.page_frame import create_page_frame, get_submitter_name
from ..components.refresh_timer import RefreshTimer
from ..components.routes import (
    ATTRIBUTE_PARAMETER,
    CODING_TABLE_DOWNLOAD_PATH,
    SCORES_DOWNLOAD_PATH,
    SUBMISSION_PATH,
    SUBMISSIONS_PATH,
    get_coding_table_download_path,
    get_download_path,
    get_scores_download_path,
    get_submission_path,
)
from ..components.score_tables import make_interval_note, make_run_scores_html
from ..components.work_requests import WorkRequester
from ..datasets import DatasetContext
from ..method_ranking import collect_scored_attributes, get_unscored_reason
from ..scoring import ResultState, ScoredRun, compute_method_scores, make_interval_request
from ..scoring_worker import ScoringWorker


def _make_attachment_header(file_name: str) -> str:
    """A Content-Disposition header, with the RFC 5987 form for non-ASCII names."""
    quoted = urllib.parse.quote(file_name)
    return (f'attachment; filename="{file_name}"' if quoted == file_name
            else f"attachment; filename*=utf-8''{quoted}")


def _create_submission_details(archive: SubmissionArchive, worker: ScoringWorker,
                               submission_id: int, on_deletion_changed: T.Callable[[], None]
                               ) -> None:
    @ui.refreshable
    def show_details() -> None:
        submission = archive.get_submission(submission_id)
        actions = archive.list_actions(submission_id=submission_id)
        notes = [note for a in actions if a.action in ADDING_ACTION_KINDS
                 for note in a.details["notes"]]
        ensemble_action = next((a for a in actions if a.action == "ensemble"), None)
        offsets = submission.context_offsets
        rows = {
            "Run": submission.run_label,
            "Dataset and split": f"{submission.dataset}/{submission.split}",
            "Output": submission.output_kind,
            "Context offsets": "unknown" if offsets is None else ", ".join(map(str, offsets)),
            "Segments": str(submission.num_segments),
            "Attributes": str(submission.num_attributes),
            "Submitter": submission.submitter,
            "Uploaded": format_utc_time(submission.uploaded_at),
            "Description": submission.description or "–",
            "SHA-256": submission.file_sha256,
        }
        if submission.is_deleted:
            rows["Deleted"] = format_utc_time(submission.deleted_at)
        with ui.element("section").classes("panel"):
            ui.label(f"Submission #{submission.id}"
                     f"{' (deleted)' if submission.is_deleted else ''}").classes("section-title")
            with ui.element("div").classes("key-values"):
                for key, value in rows.items():
                    ui.label(key).classes("key")
                    ui.label(value)
                if ensemble_action is not None:
                    ui.label("Ensemble of").classes("key")
                    ui.html(make_ensemble_members_html(ensemble_action.details["members"]),
                            sanitize=False)
            for note in notes:
                ui.label(note).classes("muted")
            with ui.element("div").classes("form-row items-center"):
                ui.link("Download the prediction file", get_download_path(submission.id))
                create_native_button(
                    "Restore" if submission.is_deleted else "Delete",
                    lambda: on_deletion_clicked(not submission.is_deleted))
            status_label = ui.label().classes("muted")

        async def on_deletion_clicked(is_deleted: bool) -> None:
            try:
                # Off the event loop, since it may wait for the database lock.
                result = await run.io_bound(archive.set_submission_deleted, submission_id,
                                            is_deleted, actor=get_submitter_name())
            except ValueError as e:
                set_status(status_label, str(e), is_error=True)
                return
            except Exception as e:
                set_status(status_label, f"Internal error: {e!r}. See the server log.",
                           is_error=True)
                raise
            if result is not None:  # None if the server is stopping.
                await run.io_bound(worker.update_submissions, [submission_id])
                show_details.refresh()
                on_deletion_changed()

    show_details()


def _create_submission_actions(archive: SubmissionArchive,
                               submission_id: int) -> T.Callable[[], None]:
    """The actions on the submission. Returns their refresh."""
    @ui.refreshable
    def show_actions() -> None:
        ui.label("Actions").classes("section-title")
        ui.html(make_action_table_html(archive.list_actions(submission_id=submission_id)),
                sanitize=False)

    with ui.element("section").classes("panel"):
        show_actions()
    return show_actions.refresh


def _create_submission_scores(archive: SubmissionArchive,
                              dataset_contexts: T.Mapping[str, DatasetContext],
                              worker: ScoringWorker, submission_id: int,
                              attribute_query: T.Sequence[str]) -> T.Callable[[], None]:
    """The scores panel. Returns its refresh."""
    def on_subset_changed(subset: AttributeSubset) -> None:
        ui.navigate.history.replace(get_submission_path(submission_id, subset.to_query()))
        interval_requester.delay_requests()
        show_scores.refresh()

    interval_requester = WorkRequester(worker)

    @ui.refreshable
    def show_scores() -> None:
        submission = archive.get_submission(submission_id)
        context = dataset_contexts.get(submission.dataset)
        if context is None:
            ui.label(f"The dataset {submission.dataset!r} is not configured, so the submission"
                     f" is not scored.").classes("muted")
            return
        scoring = worker.store.get_run_scoring(submission_id)
        is_pending = worker.is_scoring_pending(submission_id)
        reason = get_unscored_reason(submission, scoring,
                                     worker.is_scoring_current(submission, scoring), is_pending)
        refresh_timer.watch([lambda: worker.is_scoring_pending(submission_id)]
                            if is_pending else [])
        if reason is not None:
            ui.label(reason).classes("muted" if is_pending or submission.is_deleted
                                     else "error")
            return
        runs = [ScoredRun(submission, s) for s in scoring.evaluation_set_scores]
        subset = attribute_filter.set_options(collect_scored_attributes(runs))
        if not runs:  # E.g. a split without labels. Otherwise the details show the notes.
            for note in scoring.notes:
                ui.label(note).classes("muted")
            return
        if not subset.selected:
            ui.label("Select at least one attribute.").classes("error")
            return
        if submission.is_deleted:
            ui.label("Deleted submissions are not in the method table or the analysis."
                     ).classes("muted")
        is_analysed = (not submission.is_deleted
                       and submission.split in context.config.analysis_splits)
        set_to_scores = {r.scores.evaluation_set: compute_method_scores(
                             submission.run_label, [r], subset.selected) for r in runs}
        set_to_state = interval_requester.update(
            {name: make_interval_request(s.runs, subset.selected, worker.settings)
             for name, s in set_to_scores.items() if s.error is None})
        for name, scores in set_to_scores.items():
            offsets = ", ".join(map(str, scores.runs[0].scores.context_offsets))
            ui.label(EVALUATION_SET_LABELS[name] if name == ie.REFERENCE_SET_NAME
                     else f"{EVALUATION_SET_LABELS[name]}, context offsets {offsets}").classes(
                "subsection-title")
            ui.label(f"Averages over {subset.label}").classes("muted")
            ui.html(make_run_scores_html(scores, set_to_state.get(name, ResultState())),
                    sanitize=False).classes("w-full")
            with ui.element("div").classes("form-row"):
                ui.link("Download the scores over all attributes (JSON)",
                        get_scores_download_path(submission_id, name))
                if is_analysed:
                    ui.link("Analyze the predictions", AnalysisView(
                        submission.dataset, submission.split, evaluation_set=name,
                        run_id=submission_id).to_path(subset.to_query()))
        ui.label(make_interval_note(worker.settings, is_mean_over_runs=False)).classes("muted")
        refresh_timer.watch(interval_requester.get_pending_checks())

    with ui.element("section").classes("panel"):
        ui.label("Scores").classes("section-title")
        attribute_filter = AttributeFilter(attribute_query, on_subset_changed)
        refresh_timer = RefreshTimer(show_scores.refresh)
        show_scores()
    return show_scores.refresh


def register_submission_page(archive: SubmissionArchive,
                             dataset_contexts: T.Mapping[str, DatasetContext],
                             worker: ScoringWorker) -> None:
    @app.get(SCORES_DOWNLOAD_PATH)
    def download_scores(submission_id: int, evaluation_set: str) -> JSONResponse:
        try:
            submission = archive.get_submission(submission_id)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        report = worker.store.get_score_report(submission_id, evaluation_set)
        if report is None:
            raise HTTPException(status_code=404, detail=f"Submission #{submission_id} has no"
                                                        f" scores on the {evaluation_set} set.")
        file_name = to_file_name(f"{submission.run_label}.{submission.split}.{evaluation_set}"
                                 f".scores.json")
        return JSONResponse(report,
                            headers={"Content-Disposition": _make_attachment_header(file_name)})

    @app.get(CODING_TABLE_DOWNLOAD_PATH)
    def download_coding_table(submission_id: int, file_format: str, coder: str = "",
                              date: str = "") -> FileResponse:
        # A sync endpoint, which FastAPI runs in a thread, since it reads the file.
        if file_format not in CODING_TABLE_FORMATS:
            raise HTTPException(status_code=404, detail=f"Unknown format {file_format!r}.")
        try:
            submission = archive.get_submission(submission_id)
            coding_date = parse_coding_date(date)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        context = dataset_contexts.get(submission.dataset)
        if context is None:
            raise HTTPException(status_code=404, detail=f"The dataset {submission.dataset!r} is"
                                                        f" not configured.")
        file_name = to_file_name(f"{submission.run_label}.{submission.split}.coding_table"
                                 f".{file_format}")
        temporary_dir = Path(tempfile.mkdtemp(prefix="coding_table_"))
        path = temporary_dir / file_name
        try:
            write_submission_coding_table(archive, context, submission, path,
                                          coder_name=coder or submission.method.name,
                                          coding_date=coding_date)
        except BaseException:
            shutil.rmtree(temporary_dir)
            raise
        return FileResponse(path, filename=file_name,
                            background=BackgroundTask(shutil.rmtree, temporary_dir))

    @ui.page(SUBMISSION_PATH, title="Submission · iRAP evaluation")
    def submission_page(submission_id: int, request: Request) -> None:
        with create_page_frame(SUBMISSIONS_PATH):
            try:
                submission = archive.get_submission(submission_id)
            except LookupError as e:
                ui.label(str(e)).classes("error")
                return
            refreshes: list[T.Callable[[], None]] = []
            _create_submission_details(archive, worker, submission_id,
                                       lambda: [refresh() for refresh in refreshes])
            refreshes.append(_create_submission_scores(
                archive, dataset_contexts, worker, submission_id,
                request.query_params.getlist(ATTRIBUTE_PARAMETER)))
            _create_coding_table_export(submission)
            refreshes.append(_create_submission_actions(archive, submission_id))


def _make_coding_table_form_html(submission: Submission) -> str:
    """A plain GET form of the coding-table endpoint, so that a download sends the values that
    the inputs have when the button is clicked."""
    buttons = "".join(
        f'<button class="native-control" type="submit"'
        f' formaction="{html.escape(get_coding_table_download_path(submission.id, f))}">'
        f"Download .{f}</button>" for f in CODING_TABLE_FORMATS)
    return (f'<form method="get" class="form-row">'
            f'<label class="field">Coder<input class="native-control" name="coder" size="24"'
            f' value="{html.escape(submission.method.name)}"></label>'
            f'<label class="field">Coding date<input class="native-control" type="date"'
            f' name="date" value="{parse_coding_date("")}"></label>{buttons}</form>')


def _create_coding_table_export(submission: Submission) -> None:
    """The form that downloads the submission as an iRAP coding table."""
    with ui.element("section").classes("panel"):
        ui.label("Coding table").classes("section-title")
        ui.label("The predicted iRAP code of each attribute and segment, in the iRAP"
                 " coding-table layout, by road and position. Invalid predictions and columns"
                 " without a predicted attribute are blank. A .xlsx file of probabilistic"
                 " predictions has the probability of each code on a second sheet.").classes(
            "muted")
        ui.html(_make_coding_table_form_html(submission), sanitize=False)
