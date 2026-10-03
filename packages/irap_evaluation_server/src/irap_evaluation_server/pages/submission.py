"""The page of one submission: its details, actions and scores, and the endpoint for downloading
its scores."""

import typing as T
import urllib.parse

import irap_evaluation as ie
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from irap_evaluation.reports.evaluation_report import to_file_name
from nicegui import app, run, ui

from ..archive import SubmissionArchive
from ..components.attribute_filter import AttributeFilter, AttributeSubset
from ..components.formatting import format_utc_time, make_action_table_html
from ..components.interval_requests import IntervalRequester
from ..components.native_controls import create_native_button, set_status
from ..components.page_frame import create_page_frame, get_submitter_name
from ..components.refresh_timer import RefreshTimer
from ..components.routes import (
    ATTRIBUTE_PARAMETER,
    SCORES_DOWNLOAD_PATH,
    SUBMISSION_PATH,
    SUBMISSIONS_PATH,
    get_download_path,
    get_scores_download_path,
    get_submission_path,
)
from ..components.score_tables import IntervalState, make_interval_note, make_run_scores_html
from ..datasets import DatasetContext
from ..method_ranking import collect_scored_attributes, get_unscored_reason
from ..scoring import ScoredRun, compute_method_scores, make_interval_request
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
        notes = [note for a in actions if a.action == "upload" for note in a.details["notes"]]
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

    interval_requester = IntervalRequester(worker)

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
            ui.label("Deleted submissions are not in the method table.").classes("muted")
        set_to_scores = {r.scores.evaluation_set: compute_method_scores(
                             submission.run_label, [r], subset.selected) for r in runs}
        set_to_state = interval_requester.update(
            {name: make_interval_request(s.runs, subset.selected, worker.settings)
             for name, s in set_to_scores.items() if s.error is None})
        for name, scores in set_to_scores.items():
            offsets = ", ".join(map(str, scores.runs[0].scores.context_offsets))
            ui.label("Reference set" if name == ie.REFERENCE_SET_NAME
                     else f"Model-compatible set, context offsets {offsets}").classes(
                "subsection-title")
            ui.label(f"Averages over {subset.label}").classes("muted")
            ui.html(make_run_scores_html(scores, set_to_state.get(name, IntervalState())),
                    sanitize=False).classes("w-full")
            ui.link("Download the scores over all attributes (JSON)",
                    get_scores_download_path(submission_id, name))
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

    @ui.page(SUBMISSION_PATH, title="Submission · iRAP evaluation")
    def submission_page(submission_id: int, request: Request) -> None:
        with create_page_frame(SUBMISSIONS_PATH):
            try:
                archive.get_submission(submission_id)
            except LookupError as e:
                ui.label(str(e)).classes("error")
                return
            refreshes: list[T.Callable[[], None]] = []
            _create_submission_details(archive, worker, submission_id,
                                       lambda: [refresh() for refresh in refreshes])
            refreshes.append(_create_submission_scores(
                archive, dataset_contexts, worker, submission_id,
                request.query_params.getlist(ATTRIBUTE_PARAMETER)))
            refreshes.append(_create_submission_actions(archive, submission_id))
