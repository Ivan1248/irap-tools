"""The Model page: a model's details, files, scores, coding-table export and actions, and the
endpoint for downloading the scores of a submission."""

import asyncio
import dataclasses as dc
import functools
import html
import typing as T
import urllib.parse

import irap_evaluation as ie
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from irap_evaluation.reports.evaluation_report import to_valid_file_name
from nicegui import app, run, ui

from ..accounts import can_write, is_admin
from ..archive import ADDING_ACTION_KINDS, ActionLogEntry, Model, ModelArchive, Submission
from ..coding_table_export import (
    CODING_TABLE_FORMATS,
    export_model_coding_table,
    parse_coding_date,
)
from ..components.account_sessions import AccountSessions
from ..components.analysis_view import AnalysisView
from ..components.attribute_filter import AttributeFilter, AttributeSubset
from ..components.changes import run_change_in_thread, run_in_thread_showing_errors
from ..components.confirmation_dialog import confirm_changes
from ..components.formatting import (
    EVALUATION_SET_LABELS,
    format_context_offsets,
    format_splits,
    format_utc_time,
    make_action_table_html,
    make_code_spans_html,
    make_ensemble_members_html,
    make_table_html,
)
from ..components.native_controls import (
    create_native_button,
    create_native_checkbox,
    create_native_input,
    create_native_select,
    set_status,
)
from ..components.page_frame import create_page_frame
from ..components.refresh_timer import RefreshTimer
from ..components.routes import (
    ATTRIBUTE_PARAMETER,
    MODEL_PATH,
    MODELS_PATH,
    SCORES_DOWNLOAD_PATH,
    get_download_path,
    get_model_path,
    get_scores_download_path,
)
from ..components.score_tables import make_interval_note, make_model_scores_html
from ..components.work_requests import WorkRequester
from ..config import DatasetConfig
from ..datasets import DatasetContext
from ..method_ranking import collect_scored_attributes, get_unscored_reason
from ..model_listing import ModelSummary
from ..scoring import (
    ResultState,
    ScoredModel,
    compute_method_scores,
    make_interval_request,
    remove_class_scores,
)
from ..scoring_worker import ScoringWorker


def _make_attachment_header(file_name: str) -> str:
    """Makes a Content-Disposition header, with the RFC 5987 form for a name that needs
    percent-encoding, e.g. a non-ASCII one."""
    quoted = urllib.parse.quote(file_name)
    return (f'attachment; filename="{file_name}"' if quoted == file_name
            else f"attachment; filename*=utf-8''{quoted}")


def _can_see_class_scores(config: DatasetConfig, split: str, is_admin: bool) -> bool:
    """Returns whether an account can see the per-class scores of a split, which are only for
    admins on a protected split (`DatasetConfig.is_protected_split`)."""
    return is_admin or not config.is_protected_split(split)


@dc.dataclass(frozen=True)
class _ModelSnapshot:
    """A model with its submissions in all states and its actions, as the panels show them."""

    model: Model
    submissions: list[Submission]
    actions: list[ActionLogEntry]

    @classmethod
    def load(cls, archive: ModelArchive, model_id: int) -> T.Self:
        """
        Raises:
            LookupError: If there is no such model.
        """
        return cls(archive.get_model(model_id), archive.list_model_submissions(model_id),
                   archive.list_actions(model_id=model_id))

    @property
    def summary(self) -> ModelSummary:
        return ModelSummary.from_submissions(self.model, self.submissions)


class _ModelPage:
    """The panels of the page and their refreshes after a change of the model. A user who cannot
    write sees them without the controls that change the model."""

    def __init__(self, archive: ModelArchive,
                 dataset_contexts: T.Mapping[str, DatasetContext], worker: ScoringWorker,
                 coding_table_lock: asyncio.Lock, get_actor: T.Callable[[], str], can_write: bool,
                 is_admin: bool, snapshot: _ModelSnapshot):
        """
        Args:
            coding_table_lock: Held while a coding table is made, shared by the pages.
            get_actor: See `run_change_in_thread`.
            can_write: Whether the account that was signed in when the page was created could
                write.
            is_admin: Whether that account was an admin.
        """
        self.archive = archive
        self.dataset_contexts = dataset_contexts
        self.worker = worker
        self.coding_table_lock = coding_table_lock
        self.get_actor = get_actor
        self.can_write = can_write
        self.is_admin = is_admin
        self.model_id = snapshot.model.id
        #: What the panels show, loaded again after each change.
        self.snapshot = snapshot
        self.client = ui.context.client
        self.refreshes: list[T.Callable[[], None]] = []

    async def run_change(self, change: T.Callable[[str], T.Any], status_label: ui.label) -> None:
        """Runs a change of the model (`run_change_in_thread`), rescores what it affects and shows
        the model as it is then, also after a refused change, e.g. of a model that another user
        has changed.

        Args:
            change: Gets the actor (`get_actor`).
        """
        if await run_change_in_thread(change, self.get_actor, status_label):
            await run.io_bound(self.worker.update_model_submissions, self.model_id)
        snapshot = await run.io_bound(_ModelSnapshot.load, self.archive, self.model_id)
        if snapshot is None or self.client.is_deleted:  # None if the server is stopping.
            return
        self.snapshot = snapshot
        for refresh in self.refreshes:
            refresh()

    # Details ######################################################################################

    def create_details(self) -> None:
        @ui.refreshable
        def show_details() -> None:
            model, summary = self.snapshot.model, self.snapshot.summary
            info = summary.model_info
            rows = {
                "Dataset": model.dataset,
                "Method": model.method_name,
                "Seed": "–" if model.seed is None else str(model.seed),
                "Training splits": "–" if info is None else format_splits(info.training_splits),
                "Early stopping splits": ("–" if info is None
                                          else format_splits(info.early_stopping_splits)),
                "Output": summary.output_kind or "–",
                "Context offsets": ("–" if not summary.submissions
                                    else format_context_offsets(summary.context_offsets)),
                "Created": format_utc_time(model.created_at),
            }
            if model.is_deleted:
                rows["Deleted"] = format_utc_time(model.deleted_at)
            if not self.can_write:  # Otherwise they are edited below.
                rows["Method display name"] = model.method_display_name or "–"
                rows["Description"] = model.description or "–"
            ui.label(f"Model {model.shown_label}{' (deleted)' if model.is_deleted else ''}"
                     ).classes("section-title")
            with ui.element("div").classes("key-values"):
                for key, value in rows.items():
                    ui.label(key).classes("key")
                    ui.label(value)
            if self.can_write:
                create_native_button("Restore model" if model.is_deleted else "Delete model",
                                     lambda: on_deletion_clicked(not model.is_deleted))

        async def on_description_saved(description: str) -> None:
            await self.run_change(lambda actor: self.archive.set_model_description(
                self.model_id, description, actor=actor), status_label)

        async def on_display_name_saved(display_name: str) -> None:
            await self.run_change(lambda actor: self.archive.set_method_display_name(
                self.model_id, display_name, actor=actor), status_label)

        async def on_deletion_clicked(is_deleted: bool) -> None:
            await self.run_change(lambda actor: self.archive.set_model_deleted(
                self.model_id, is_deleted, actor=actor), status_label)

        description = {"text": self.snapshot.model.description}
        display_name = {"text": self.snapshot.model.method_display_name or ""}
        with ui.element("section").classes("panel"):
            show_details()
            # Not refreshed with the details, so that a refresh does not undo an edit.
            if self.can_write:
                with ui.element("div").classes("form-row items-end"):
                    create_native_input("Method display name (for all models of the method)",
                                        display_name["text"],
                                        lambda text: display_name.update(text=text),
                                        placeholder="none", size=40)
                    create_native_button("Save display name",
                                         lambda: on_display_name_saved(display_name["text"]))
                with ui.element("div").classes("form-row items-end"):
                    create_native_input("Description", description["text"],
                                        lambda text: description.update(text=text), size=60)
                    create_native_button("Save description",
                                         lambda: on_description_saved(description["text"]))
            status_label = ui.label().classes("muted")
        self.refreshes.append(show_details.refresh)

    # Files ########################################################################################

    def create_files(self) -> None:
        def make_row(submission: Submission, actions: T.Sequence[ActionLogEntry]) -> str:
            notes = [note for a in actions if a.submission_id == submission.id
                     and a.action in ADDING_ACTION_KINDS for note in a.details["notes"]]
            ensemble = next((a for a in actions if a.submission_id == submission.id
                             and a.action == "ensemble"), None)
            details = "".join(f'<div class="muted">{html.escape(n)}</div>' for n in notes)
            if ensemble is not None:
                details += (f"<div>Ensemble of"
                            f" {make_ensemble_members_html(ensemble.details['members'])}</div>")
            delete_button = ("" if submission.is_deleted or not self.can_write else
                             f' <button class="native-control" type="button"'
                             f' data-delete-id="{submission.id}">Delete</button>')
            return (f'<tr class="{"deleted" if submission.is_deleted else ""}">'
                    f"<td>{html.escape(submission.split)}</td><td>#{submission.id}</td>"
                    f"<td>{'deleted' if submission.is_deleted else 'active'}</td>"
                    f'<td class="number">{submission.num_segments}</td>'
                    f'<td class="number">{submission.num_attributes}</td>'
                    f"<td>{html.escape(submission.submitter)}</td>"
                    f"<td>{format_utc_time(submission.uploaded_at)}</td>"
                    f'<td><a href="{get_download_path(submission.id)}">Download</a>'
                    f"{delete_button}</td>"
                    f"<td>{details}</td></tr>")

        @ui.refreshable
        def show_files() -> None:
            submissions, actions = self.snapshot.submissions, self.snapshot.actions
            ui.label("Files").classes("section-title")
            ui.label("Only the active file of each split is scored. A new file of a split"
                     " replaces the active one, which is deleted. Deleted files can be downloaded"
                     " and uploaded again.").classes("muted")
            ui.html(make_table_html(
                ["Split", "#", "State", "Segments", "Attributes", "Submitter",
                 "Upload time", "", "Notes"],
                [make_row(s, actions)
                 for s in sorted(submissions, key=lambda s: (s.split, -s.id))]),
                sanitize=False).classes("w-full overflow-x-auto").on(
                "click", lambda e: on_delete_clicked(int(e.args)),
                js_handler="(e) => { const button = e.target.closest('[data-delete-id]');"
                           " if (button) emit(button.dataset.deleteId); }")

        async def on_delete_clicked(submission_id: int) -> None:
            # From the shown files, so that only files of this model are deleted.
            # `ModelArchive.delete_submission` refuses a file that was deleted meanwhile.
            submission = next((s for s in self.snapshot.submissions if s.id == submission_id),
                              None)
            if submission is None:  # E.g. from a crafted event.
                set_status(status_label, f"File #{submission_id} is not a file of this model.",
                           is_error=True)
                return
            changes = [f"Deletes the {submission.split} file #{submission.id} of"
                       f" {submission.label}, so that the model has no {submission.split} scores."
                       f" It can still be downloaded, but only an upload restores it."]
            if submission.is_in_use and len(self.snapshot.summary.submissions) == 1:
                changes.append(f"Deletes the model {submission.label}, since it has no other"
                               f" active file. An upload of a file restores it.")
            if await confirm_changes(f"Delete file #{submission.id}", changes, "Delete"):
                await self.run_change(
                    lambda actor: self.archive.delete_submission(submission.id, actor=actor),
                    status_label)

        with ui.element("section").classes("panel"):
            show_files()
            status_label = ui.label().classes("muted")
        self.refreshes.append(show_files.refresh)

    # Scores #######################################################################################

    def create_scores(self, split_query: str, attribute_query: T.Sequence[str]) -> None:
        state = {"split": split_query}

        def on_split_changed(split: str) -> None:
            state["split"] = split
            update_url()
            show_scores.refresh()

        def on_subset_changed(subset: AttributeSubset) -> None:
            update_url(subset)
            interval_requester.delay_requests()
            show_scores.refresh()

        def update_url(subset: AttributeSubset | None = None) -> None:
            subset = subset or attribute_filter.subset
            ui.navigate.history.replace(get_model_path(self.model_id, state["split"],
                                                       subset.to_query()))

        interval_requester = WorkRequester(self.worker)

        @ui.refreshable
        def show_scores() -> None:
            model, active = self.snapshot.model, self.snapshot.summary.split_to_submission
            context = self.dataset_contexts.get(model.dataset)
            if context is None:
                ui.label(f"The dataset {model.dataset!r} is not configured, so the model is not"
                         f" scored.").classes("muted")
                return
            if not active:
                ui.label("The model has no active files.").classes("muted")
                return
            splits = [s for s in context.split_names if s in active]
            if state["split"] not in active:
                if state["split"]:
                    ui.label(f"The model has no active file of the split {state['split']!r}."
                             ).classes("error")
                state["split"] = splits[0]
            with ui.element("div").classes("form-row"):
                create_native_select("Split", {s: s for s in splits}, state["split"],
                                     on_split_changed)
            submission = active[state["split"]]
            self._show_submission_scores(context, submission, attribute_filter,
                                         interval_requester, refresh_timer)

        with ui.element("section").classes("panel"):
            ui.label("Scores").classes("section-title")
            attribute_filter = AttributeFilter(attribute_query, on_subset_changed)
            refresh_timer = RefreshTimer(show_scores.refresh)
            show_scores()
        self.refreshes.append(show_scores.refresh)

    def _show_submission_scores(self, context: DatasetContext, submission: Submission,
                                attribute_filter: AttributeFilter,
                                interval_requester: WorkRequester,
                                refresh_timer: RefreshTimer) -> None:
        worker = self.worker
        scoring = worker.store.get_scoring(submission.id)
        is_pending = worker.is_scoring_pending(submission.id)
        reason = get_unscored_reason(submission, scoring,
                                     worker.is_scoring_current(submission, scoring), is_pending)
        refresh_timer.watch([lambda: worker.is_scoring_pending(submission.id)]
                            if is_pending else [])
        if reason is not None:
            ui.label(reason).classes("muted" if is_pending or not submission.is_in_use
                                     else "error")
            return
        models = [ScoredModel(submission, s) for s in scoring.evaluation_set_scores]
        subset = attribute_filter.set_options(collect_scored_attributes(models))
        if not models:  # E.g. a split without labels.
            for note in scoring.notes:
                ui.label(note).classes("muted")
            return
        if not subset.selected:
            ui.label("Select at least one attribute.").classes("error")
            return
        if not submission.is_in_use:
            ui.label("Deleted models are left out of the Scores and Analysis pages."
                     ).classes("muted")
        split_use = submission.model_info.get_split_use(submission.split)
        if (explanation := ie.explain_split_use(split_use, submission.split)) is not None:
            ui.label(explanation).classes("error" if split_use == "training" else "muted")
        is_analysed = (submission.is_in_use
                       and not context.config.is_protected_split(submission.split))
        can_see_class_scores = _can_see_class_scores(context.config, submission.split,
                                                     self.is_admin)
        set_to_scores = {m.scores.evaluation_set: compute_method_scores(
                             submission.label, [m], subset.selected) for m in models}
        set_to_state = interval_requester.update(
            {name: make_interval_request(s.models, subset.selected, worker.settings)
             for name, s in set_to_scores.items() if s.error is None})
        for name, scores in set_to_scores.items():
            offsets = format_context_offsets(scores.models[0].scores.context_offsets)
            ui.label(EVALUATION_SET_LABELS[name] if name == ie.REFERENCE_SET_NAME
                     else f"{EVALUATION_SET_LABELS[name]}, context offsets {offsets}").classes(
                "subsection-title")
            ui.label(f"Averages over {subset.label}").classes("muted")
            ui.html(make_model_scores_html(scores, set_to_state.get(name, ResultState())),
                    sanitize=False).classes("w-full")
            with ui.element("div").classes("form-row"):
                ui.link("Download the scores over all attributes (JSON)",
                        get_scores_download_path(submission.id, name))
                if not can_see_class_scores:
                    ui.label(f"Without per-class scores, since the labels of the"
                             f" {submission.split} split are protected.").classes("muted")
                if is_analysed:
                    ui.link("Open in Analysis", AnalysisView(
                        submission.dataset, submission.split, evaluation_set=name,
                        model_id=submission.model.id).to_path(subset.to_query()))
        ui.label(make_interval_note(worker.settings, is_mean_over_models=False)).classes("muted")
        refresh_timer.watch(interval_requester.get_pending_checks())

    # Coding table and actions #####################################################################

    def create_coding_table_export(self) -> None:
        # The values of the inputs, which `show_form` sets to the defaults.
        form = {"split_to_is_chosen": {}, "coder": "", "date": ""}
        state = {"is_preparing": False}

        @ui.refreshable
        def show_form() -> None:
            model, active = self.snapshot.model, self.snapshot.summary.split_to_submission
            context = self.dataset_contexts.get(model.dataset)
            active_splits = ([s for s in context.split_names if s in active] if context is not None
                             else sorted(active))
            ui.label("Coding table").classes("section-title")
            ui.html(make_code_spans_html(
                "The predicted iRAP code of each attribute and segment of the chosen splits, in"
                " the iRAP coding-table layout, ordered by road and position. Invalid"
                " predictions and columns without a predicted attribute are blank. A `.xlsx`"
                " file of probabilistic predictions has the probability of each predicted code"
                " on a second sheet. Preparing a `.xlsx` file can take tens of seconds, a"
                " `.csv` file is much faster."), sanitize=False).classes("muted")
            if not active_splits:
                ui.label("The model has no active files.").classes("muted")
                return
            form.update(split_to_is_chosen=dict.fromkeys(active_splits, True),
                        coder=model.shown_method_name, date=parse_coding_date(""))
            with ui.element("div").classes("form-row items-end"):
                for split in active_splits:
                    create_native_checkbox(
                        split, True,
                        lambda is_chosen, split=split: form["split_to_is_chosen"].update(
                            {split: is_chosen}))
                create_native_input("Coder", form["coder"], lambda text: form.update(coder=text),
                                    size=24)
                create_native_input("Coding date", form["date"],
                                    lambda text: form.update(date=text), input_type="date")
                for file_format in CODING_TABLE_FORMATS:
                    create_native_button(f"Download .{file_format}",
                                         lambda f=file_format: on_download_clicked(f))

        async def on_download_clicked(file_format: str) -> None:
            if state["is_preparing"]:  # The status label shows it.
                return
            splits = [s for s, is_chosen in form["split_to_is_chosen"].items() if is_chosen]
            state["is_preparing"] = True
            try:
                if self.coding_table_lock.locked():
                    set_status(status_label, "Waiting for the coding table of another page …")
                async with self.coding_table_lock:
                    if self.client.is_deleted:
                        return
                    set_status(status_label, f"Preparing the .{file_format} file …")
                    coding_table = await run_in_thread_showing_errors(functools.partial(
                        export_model_coding_table, self.archive, self.dataset_contexts,
                        self.model_id, splits, file_format, coder_name=form["coder"],
                        coding_date=form["date"]), status_label)
            finally:
                state["is_preparing"] = False
            if coding_table is None or self.client.is_deleted:
                return
            set_status(status_label, "")
            ui.download.content(coding_table.content, coding_table.file_name)

        with ui.element("section").classes("panel"):
            show_form()
            status_label = ui.label().classes("muted")
        self.refreshes.append(show_form.refresh)

    def create_actions(self) -> None:
        @ui.refreshable
        def show_actions() -> None:
            model = self.snapshot.model
            ui.label("Action log").classes("section-title")
            ui.html(make_action_table_html(self.snapshot.actions, {model.id: model}),
                    sanitize=False)

        with ui.element("section").classes("panel"):
            show_actions()
        self.refreshes.append(show_actions.refresh)


def register_model_page(archive: ModelArchive,
                        dataset_contexts: T.Mapping[str, DatasetContext],
                        worker: ScoringWorker, sessions: AccountSessions) -> None:
    # One coding table at a time, since anyone can make one, and it can take tens of seconds in
    # the threads of `run.io_bound`, which changes and sign-ins also use.
    coding_table_lock = asyncio.Lock()

    @app.get(SCORES_DOWNLOAD_PATH)
    def download_scores(submission_id: int, evaluation_set: str) -> JSONResponse:
        try:
            submission = archive.get_submission(submission_id)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        report = worker.store.get_score_report(submission_id, evaluation_set)
        # Without the configuration of the dataset, it is not known whether the split is
        # protected. The Model page does not show such scores either.
        context = dataset_contexts.get(submission.dataset)
        if report is None or context is None:
            raise HTTPException(status_code=404, detail=f"File #{submission_id} has no"
                                                        f" scores on the {evaluation_set} set.")
        if not _can_see_class_scores(context.config, submission.split,
                                     is_admin(sessions.get_account())):
            report = remove_class_scores(report)
        file_name = to_valid_file_name(f"{submission.label}.{submission.split}.{evaluation_set}"
                                 f".scores.json")
        return JSONResponse(report,
                            headers={"Content-Disposition": _make_attachment_header(file_name)})

    @ui.page(MODEL_PATH, title="Model · iRAP evaluation")
    def model_page(model_id: int, request: Request) -> None:
        account = sessions.get_account()
        with create_page_frame(MODELS_PATH, account):
            try:
                snapshot = _ModelSnapshot.load(archive, model_id)
            except LookupError as e:
                ui.label(str(e)).classes("error")
                return
            page = _ModelPage(archive, dataset_contexts, worker, coding_table_lock,
                              sessions.require_writer_name, can_write(account),
                              is_admin(account), snapshot)
            page.create_details()
            page.create_files()
            page.create_scores(request.query_params.get("split", ""),
                               request.query_params.getlist(ATTRIBUTE_PARAMETER))
            page.create_coding_table_export()
            page.create_actions()
