"""The Model page: a model's details, files, scores, coding-table export and actions, and the
endpoints for downloading the files of a model and the scores of a submission."""

import asyncio
import dataclasses as dc
import functools
import html
import inspect
import secrets
import typing as T
import urllib.parse
from pathlib import Path

import irap_evaluation as ie
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from nicegui import app, background_tasks, run, ui
from nicegui.elements.mixins.text_element import TextElement
from starlette.background import BackgroundTask

from ..accounts import can_write, is_admin
from ..archive import (
    ADDING_ACTION_KINDS,
    ARCHIVE_SUFFIX,
    ActionLogEntry,
    Model,
    ModelArchive,
    Submission,
    make_label_file_name,
)
from ..coding_table_export import (
    CODING_TABLE_FORMATS,
    parse_coding_date,
    plan_model_coding_table,
    write_model_coding_table,
)
from ..components.account_sessions import AccountSessions
from ..components.analysis_view import AnalysisView
from ..components.attribute_filter import AttributeFilter, AttributeSubset
from ..components.changes import (
    run_change_in_thread,
    run_in_process_showing_errors,
    run_in_thread_showing_errors,
)
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
from ..components.latest_updates import LatestUpdates
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
    CODING_TABLE_DOWNLOAD_PATH,
    MODEL_PATH,
    MODEL_PREDICTIONS_DOWNLOAD_PATH,
    MODELS_PATH,
    SCORES_DOWNLOAD_PATH,
    get_download_path,
    get_model_download_path,
    get_model_path,
    get_scores_download_path,
    make_query_path,
)
from ..components.score_tables import (
    MODEL_SCORES_METRIC_NAMES,
    make_interval_note,
    make_model_scores_html,
)
from ..config import DatasetConfig
from ..datasets import DatasetContext
from ..method_ranking import collect_scored_attributes, get_unscored_reason
from ..model_listing import ModelSummary
from ..scoring import ScoredModel, SubmissionScoring, compute_method_scores, remove_class_scores
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
    """A model with its submissions and its actions, as the panels show them."""

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


@dc.dataclass(frozen=True)
class _SubmissionScoring:
    """The stored scoring of a submission and its state (`_load_submission_scoring`).

    Attributes:
        unscored_reason: Why it has no current scores (`get_unscored_reason`), or None.
        models: The submission with its scores on each evaluation set, none if `unscored_reason`
            is not None.
    """

    submission: Submission
    scoring: SubmissionScoring | None
    is_pending: bool
    unscored_reason: str | None
    models: list[ScoredModel]


def _load_submission_scoring(worker: ScoringWorker, submission: Submission) -> _SubmissionScoring:
    scoring = worker.store.get_scoring(submission.id)
    is_pending = worker.is_scoring_pending(submission.id)
    reason = get_unscored_reason(scoring, worker.is_scoring_current(submission, scoring),
                                 is_pending)
    models = ([] if reason is not None
              else [ScoredModel(submission, s) for s in scoring.evaluation_set_scores])
    return _SubmissionScoring(submission, scoring, is_pending, reason, models)


def _make_set_score_tables_html(worker: ScoringWorker, models: T.Iterable[ScoredModel],
                                subset: AttributeSubset) -> dict[str, str]:
    """Makes the score tables of a submission over the attributes of `subset`
    (`make_model_scores_html`) on each evaluation set that it is scored on.

    Args:
        models: The submission with its scores on each evaluation set.
    """
    set_to_tables_html = {}
    for model in models:
        scores = compute_method_scores(model.submission.label, [model], subset.selected)
        resamples = worker.get_method_resamples(scores, subset.selected,
                                                MODEL_SCORES_METRIC_NAMES)
        set_to_tables_html[model.scores.evaluation_set] = make_model_scores_html(
            scores, resamples, worker.settings.confidence)
    return set_to_tables_html


@dc.dataclass(frozen=True)
class _ShownSubmissionScores:
    """What the Scores panel shows of the submission of the chosen split.

    Attributes:
        split_error: Why the split of the URL is not shown, or ''.
        subset: The attribute subset, None if the submission has no current scores.
        set_to_tables_html: Evaluation set name -> its score tables
            (`_make_set_score_tables_html`), none if no attribute is selected.
    """

    split_error: str
    loaded: _SubmissionScoring
    subset: AttributeSubset | None
    set_to_tables_html: dict[str, str]


@dc.dataclass(frozen=True)
class _PreparedFile:
    path: Path
    file_name: str


class _CodingTableFiles:
    """The coding tables that the Model pages prepare, shared by the pages and sent by the
    endpoint `CODING_TABLE_DOWNLOAD_PATH`, which removes the file. A file that is not
    downloaded, e.g. after the browser left, is removed by `ModelArchive.remove_uploads`."""

    def __init__(self) -> None:
        #: Held while a coding table is made: one at a time, since anyone can make one, and a
        #: .xlsx file takes a process tens of seconds.
        self.lock = asyncio.Lock()
        #: The token of the download path -> the file.
        self._token_to_file: dict[str, _PreparedFile] = {}

    def add(self, path: Path, file_name: str) -> str:
        """Adds a prepared file. Returns its download path, which works once."""
        token = secrets.token_urlsafe(16)
        self._token_to_file[token] = _PreparedFile(path, file_name)
        return CODING_TABLE_DOWNLOAD_PATH.format(token=token)

    def pop(self, token: str) -> _PreparedFile | None:
        return self._token_to_file.pop(token, None)


class _ModelPage:
    """The panels of the page and their refreshes after a change of the model. A user who cannot
    write sees them without the controls that change the model and without the action log."""

    def __init__(self, archive: ModelArchive,
                 dataset_contexts: T.Mapping[str, DatasetContext], worker: ScoringWorker,
                 coding_tables: _CodingTableFiles, get_actor: T.Callable[[], str],
                 can_write: bool, is_admin: bool, snapshot: _ModelSnapshot):
        """
        Args:
            coding_tables: Shared by the pages.
            get_actor: See `run_change_in_thread`.
            can_write: Whether the account that was signed in when the page was created could
                write.
            is_admin: Whether that account was an admin.
        """
        self.archive = archive
        self.dataset_contexts = dataset_contexts
        self.worker = worker
        self.coding_tables = coding_tables
        self.get_actor = get_actor
        self.can_write = can_write
        self.is_admin = is_admin
        self.model_id = snapshot.model.id
        #: What the panels show, loaded again after each change.
        self.snapshot = snapshot
        self.client = ui.context.client
        #: The refreshes of the panels after a change, which may return an awaitable.
        self.refreshes: list[T.Callable[[], T.Any]] = []

    async def run_change(self, change: T.Callable[[str], T.Any], status_label: ui.label) -> None:
        """Runs a change of the model (`run_change_in_thread`), rescores what it affects and shows
        the model as it is then, also after a refused change, e.g. of a model that another user
        has changed. Goes to the Models page if the model has been deleted.

        Args:
            change: Gets the actor (`get_actor`).
        """
        model = self.snapshot.model
        if await run_change_in_thread(change, self.get_actor, status_label):
            await run.io_bound(self.worker.update_model_submissions, model)
        try:
            snapshot = await run.io_bound(_ModelSnapshot.load, self.archive, self.model_id)
        except LookupError:  # Deleted by the change or by another user.
            if not self.client.is_deleted:
                ui.navigate.to(make_query_path(MODELS_PATH, {"dataset": model.dataset}))
            return
        if snapshot is None or self.client.is_deleted:  # None if the server is stopping.
            return
        self.snapshot = snapshot
        for refresh in self.refreshes:
            if inspect.isawaitable(result := refresh()):
                await result

    # Details ######################################################################################

    def create_details(self) -> None:
        @ui.refreshable
        def show_details() -> None:
            model, summary = self.snapshot.model, self.snapshot.summary
            info = summary.model_info
            rows = {
                "Dataset": model.dataset,
                "Method": model.method_name,
                "Seed": str(model.seed),
                "Training splits": "–" if info is None else format_splits(info.training_splits),
                "Early stopping splits": ("–" if info is None
                                          else format_splits(info.early_stopping_splits)),
                "Output": summary.output_kind or "–",
                "Context offsets": ("–" if not summary.submissions
                                    else format_context_offsets(summary.context_offsets)),
                "Created": format_utc_time(model.created_at),
            }
            if not self.can_write:  # Otherwise they are edited below.
                rows["Method display name"] = model.method_display_name or "–"
                rows["Description"] = model.description or "–"
            ui.label(f"Model {model.shown_label}").classes("section-title")
            with ui.element("div").classes("key-values"):
                for key, value in rows.items():
                    ui.label(key).classes("key")
                    ui.label(value)
            if self.can_write:
                create_native_button("Delete model", on_delete_model_clicked)

        async def on_description_saved(description: str) -> None:
            await self.run_change(lambda actor: self.archive.set_model_description(
                self.model_id, description, actor=actor), status_label)

        async def on_display_name_saved(display_name: str) -> None:
            await self.run_change(lambda actor: self.archive.set_method_display_name(
                self.model_id, display_name, actor=actor), status_label)

        async def on_delete_model_clicked() -> None:
            label = self.snapshot.model.label
            changes = [f"Permanently deletes the model {label} with all its files and their"
                       f" scores. To keep a copy, first download the files (.zip) under Files,"
                       f" which can be uploaded again."]
            if await confirm_changes(f"Delete model {label}", changes, "Delete"):
                await self.run_change(lambda actor: self.archive.delete_model(
                    self.model_id, actor=actor), status_label)

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
        def make_row(submission: Submission, actions: T.Sequence[ActionLogEntry],
                     can_delete: bool) -> str:
            notes = [note for a in actions if a.submission_id == submission.id
                     and a.action in ADDING_ACTION_KINDS for note in a.details["notes"]]
            ensemble = next((a for a in actions if a.submission_id == submission.id
                             and a.action == "ensemble"), None)
            details = "".join(f'<div class="muted">{html.escape(n)}</div>' for n in notes)
            if ensemble is not None:
                details += (f"<div>Ensemble of"
                            f" {make_ensemble_members_html(ensemble.details['members'])}</div>")
            delete_button = ("" if not can_delete else
                             f' <button class="native-control" type="button"'
                             f' data-delete-id="{submission.id}">Delete</button>')
            return (f"<tr><td>{html.escape(submission.split)}</td><td>#{submission.id}</td>"
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
            # The last file is deleted with the model (`ModelArchive.delete_submission`).
            can_delete = self.can_write and len(submissions) > 1
            ui.label("Files").classes("section-title")
            ui.label("A model has one file per split. A new file of a split replaces the old one,"
                     " which is deleted permanently. Deleting a file also removes it"
                     " permanently. The last file is deleted with the model.").classes("muted")
            ui.html(f'<a href="{get_model_download_path(self.model_id)}">Download the files'
                    f" (.zip)</a>, which can be uploaded again as the files of this model.",
                    sanitize=False)
            ui.html(make_table_html(
                ["Split", "#", "Segments", "Attributes", "Submitter", "Upload time", "", "Notes"],
                [make_row(s, actions, can_delete)
                 for s in sorted(submissions, key=lambda s: s.split)],
                number_headers=["Segments", "Attributes"]),
                sanitize=False).classes("w-full overflow-x-auto").on(
                "click", lambda e: on_delete_clicked(int(e.args)),
                js_handler="(e) => { const button = e.target.closest('[data-delete-id]');"
                           " if (button) emit(button.dataset.deleteId); }")

        async def on_delete_clicked(submission_id: int) -> None:
            # From the shown files, so that only files of this model are deleted.
            submission = next((s for s in self.snapshot.submissions if s.id == submission_id),
                              None)
            if submission is None:  # E.g. from a crafted event.
                set_status(status_label, f"File #{submission_id} is not a file of this model.",
                           is_error=True)
                return
            changes = [f"Permanently deletes the {submission.split} file #{submission.id} of"
                       f" {submission.label}, so that the model has no {submission.split} scores."
                       f" To keep a copy, download it first."]
            if await confirm_changes(f"Delete file #{submission.id}", changes, "Delete"):
                await self.run_change(lambda actor: self.archive.delete_submission(
                    submission.id, actor=actor), status_label)

        with ui.element("section").classes("panel"):
            show_files()
            status_label = ui.label().classes("muted")
        self.refreshes.append(show_files.refresh)

    # Scores #######################################################################################

    def create_scores(self, split_query: str, attribute_query: T.Sequence[str]) -> None:
        state = {"split": split_query}
        latest_updates = LatestUpdates()

        async def on_split_changed(split: str) -> None:
            state["split"] = split
            update_url()
            await update_scores()

        async def on_subset_changed(subset: AttributeSubset) -> None:
            update_url(subset)
            await update_scores()

        def update_url(subset: AttributeSubset | None = None) -> None:
            subset = subset or attribute_filter.subset
            ui.navigate.history.replace(get_model_path(self.model_id, state["split"],
                                                       subset.to_query()))

        async def update_scores() -> None:
            """Loads and computes the scores of the chosen split in threads, and shows them if no
            later update has started meanwhile."""
            is_latest = latest_updates.start()
            split_to_submission = self.snapshot.summary.split_to_submission
            context = self.dataset_contexts.get(self.snapshot.model.dataset)
            if context is None or not split_to_submission:
                show_scores.refresh(None)
                return
            split_error = ""
            if state["split"] not in split_to_submission:
                if state["split"]:
                    split_error = f"The model has no file of the split {state['split']!r}."
                state["split"] = next(s for s in context.split_names if s in split_to_submission)
            submission = split_to_submission[state["split"]]
            loaded = await run.io_bound(_load_submission_scoring, self.worker, submission)
            if loaded is None or not is_latest():
                return
            subset, set_to_tables_html = None, {}
            if loaded.unscored_reason is None:
                subset = attribute_filter.set_options(collect_scored_attributes(loaded.models))
                if subset.selected:
                    set_to_tables_html = await run.io_bound(_make_set_score_tables_html,
                                                            self.worker, loaded.models, subset)
                    if set_to_tables_html is None or not is_latest():
                        return
            show_scores.refresh(_ShownSubmissionScores(split_error, loaded, subset,
                                                       set_to_tables_html))

        @ui.refreshable
        def show_scores(shown: _ShownSubmissionScores | None) -> None:
            model = self.snapshot.model
            split_to_submission = self.snapshot.summary.split_to_submission
            context = self.dataset_contexts.get(model.dataset)
            if context is None:
                ui.label(f"The dataset {model.dataset!r} is not configured, so the model is not"
                         f" scored.").classes("muted")
                return
            if not split_to_submission:
                ui.label("The model has no files.").classes("muted")
                return
            if shown is None:
                ui.label("Loading the scores…").classes("muted")
                return
            if shown.split_error:
                ui.label(shown.split_error).classes("error")
            splits = [s for s in context.split_names if s in split_to_submission]
            with ui.element("div").classes("form-row"):
                create_native_select("Split", {s: s for s in splits}, state["split"],
                                     on_split_changed)
            self._show_submission_scores(context, shown, refresh_timer)

        with ui.element("section").classes("panel"):
            ui.label("Scores").classes("section-title")
            attribute_filter = AttributeFilter(attribute_query, on_subset_changed)
            refresh_timer = RefreshTimer(update_scores)
            show_scores(None)
        background_tasks.create(update_scores(), name="update scores")
        self.refreshes.append(update_scores)

    def _show_submission_scores(self, context: DatasetContext, shown: _ShownSubmissionScores,
                                refresh_timer: RefreshTimer) -> None:
        worker = self.worker
        loaded = shown.loaded
        submission = loaded.submission
        refresh_timer.watch([lambda: worker.is_scoring_pending(submission.id)]
                            if loaded.is_pending else [])
        if loaded.unscored_reason is not None:
            ui.label(loaded.unscored_reason).classes("muted" if loaded.is_pending else "error")
            return
        if not loaded.models:  # E.g. a split without labels.
            for note in loaded.scoring.notes:
                ui.label(note).classes("muted")
            return
        if not shown.subset.selected:
            ui.label("Select at least one attribute.").classes("error")
            return
        split_use = submission.model_info.get_split_use(submission.split)
        if (explanation := ie.explain_split_use(split_use, submission.split)) is not None:
            ui.label(explanation).classes("error" if split_use == "training" else "muted")
        is_analysed = not context.config.is_protected_split(submission.split)
        can_see_class_scores = _can_see_class_scores(context.config, submission.split,
                                                     self.is_admin)
        for model in loaded.models:
            name = model.scores.evaluation_set
            offsets = format_context_offsets(model.scores.context_offsets)
            ui.label(EVALUATION_SET_LABELS[name] if name == ie.REFERENCE_SET_NAME
                     else f"{EVALUATION_SET_LABELS[name]}, context offsets {offsets}").classes(
                "subsection-title")
            ui.label(f"Averages over {shown.subset.label}").classes("muted")
            ui.html(shown.set_to_tables_html[name], sanitize=False).classes("w-full")
            with ui.element("div").classes("form-row"):
                ui.link("Download the scores over all attributes (JSON)",
                        get_scores_download_path(submission.id, name))
                if not can_see_class_scores:
                    ui.label(f"Without per-class scores, since the labels of the"
                             f" {submission.split} split are protected.").classes("muted")
                if is_analysed:
                    ui.link("Open in Analysis", AnalysisView(
                        submission.dataset, submission.split, evaluation_set=name,
                        model_id=submission.model.id).to_path(shown.subset.to_query()))
        ui.label(make_interval_note(worker.settings, is_mean_over_models=False)).classes("muted")

    # Coding table and actions #####################################################################

    def create_coding_table_export(self) -> None:
        # The values of the inputs, which `show_form` sets to the defaults.
        form = {"split_to_is_chosen": {}, "coder": "", "date": ""}
        state = {"is_preparing": False}

        @ui.refreshable
        def show_form() -> None:
            model = self.snapshot.model
            split_to_submission = self.snapshot.summary.split_to_submission
            context = self.dataset_contexts.get(model.dataset)
            splits = ([s for s in context.split_names if s in split_to_submission]
                      if context is not None else sorted(split_to_submission))
            ui.label("Coding table").classes("section-title")
            ui.html(make_code_spans_html(
                "The predicted iRAP code of each attribute and segment of the chosen splits, in"
                " the iRAP coding-table layout, ordered by road and position. Invalid"
                " predictions and columns without a predicted attribute are blank. A `.xlsx`"
                " file of probabilistic predictions has the probability of each predicted code"
                " on a second sheet. Preparing a `.xlsx` file can take tens of seconds, a"
                " `.csv` file is much faster."), sanitize=False).classes("muted")
            if not splits:
                ui.label("The model has no files.").classes("muted")
                return
            form.update(split_to_is_chosen=dict.fromkeys(splits, True),
                        coder=model.shown_method_name, date=parse_coding_date(""))
            with ui.element("div").classes("form-row items-end"):
                for split in splits:
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
                if self.coding_tables.lock.locked():
                    set_status(status_label, "Waiting for the coding table of another page …",
                               is_busy=True)
                async with self.coding_tables.lock:
                    if self.client.is_deleted:
                        return
                    set_status(status_label, f"Preparing the .{file_format} file …",
                               is_busy=True)
                    job = await run_in_thread_showing_errors(functools.partial(
                        plan_model_coding_table, self.archive, self.dataset_contexts,
                        self.model_id, splits, file_format, coder_name=form["coder"],
                        coding_date=form["date"]), status_label)
                    if job is None:
                        return
                    download_path = self.archive.make_download_path(f".{file_format}")
                    try:
                        # None if cancelled, e.g. since the server is stopping.
                        is_written = await run_in_process_showing_errors(functools.partial(
                            write_model_coding_table, job.prediction_paths, job.metadata,
                            download_path, coder_name=job.coder_name,
                            coding_date=job.coding_date), status_label) is not None
                    except BaseException:
                        download_path.unlink(missing_ok=True)
                        raise
            finally:
                state["is_preparing"] = False
            if not is_written or self.client.is_deleted:
                download_path.unlink(missing_ok=True)
                return
            set_status(status_label, "")
            # Over HTTP, so that the page's websocket does not carry a file of tens of MB as one
            # message.
            ui.download.from_url(self.coding_tables.add(download_path, job.file_name))

        with ui.element("section").classes("panel"):
            show_form()
            status_label = ui.label().classes("muted")
        self.refreshes.append(show_form.refresh)

    def create_actions(self) -> None:
        if not self.can_write:
            return

        @ui.refreshable
        def show_actions() -> None:
            model = self.snapshot.model
            ui.html(make_action_table_html(self.snapshot.actions, {model.id: model}),
                    sanitize=False)

        # Collapsed at first. Not refreshed with the table, so that a refresh does not close it.
        with ui.element("section").classes("panel"), \
                ui.element("details").classes("collapsible-section"):
            TextElement(tag="summary", text="Action log").classes("section-title")
            show_actions()
        self.refreshes.append(show_actions.refresh)


def register_model_page(archive: ModelArchive,
                        dataset_contexts: T.Mapping[str, DatasetContext],
                        worker: ScoringWorker, sessions: AccountSessions) -> None:
    coding_tables = _CodingTableFiles()

    @app.get(CODING_TABLE_DOWNLOAD_PATH)
    def download_coding_table(token: str) -> FileResponse:
        prepared = coding_tables.pop(token)
        if prepared is None:
            raise HTTPException(status_code=404, detail="The coding table was already downloaded"
                                                        " or does not exist. Prepare it again.")
        return FileResponse(prepared.path, filename=prepared.file_name,
                            background=BackgroundTask(prepared.path.unlink))

    @app.get(MODEL_PREDICTIONS_DOWNLOAD_PATH)
    def download_model_predictions(model_id: int) -> FileResponse:
        try:
            model = archive.get_model(model_id)
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        # Removed after it is sent, or by `remove_uploads` if the download is cut off.
        path = archive.make_download_path(ARCHIVE_SUFFIX)
        try:
            with path.open("wb") as file:
                archive.write_model_predictions_zip(model_id, file)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return FileResponse(path, filename=make_label_file_name(model.label, ".predictions.zip"),
                            media_type="application/zip", background=BackgroundTask(path.unlink))

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
        file_name = make_label_file_name(
            submission.label, f".{submission.split}.{evaluation_set}.scores.json")
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
            page = _ModelPage(archive, dataset_contexts, worker, coding_tables,
                              sessions.require_writer_name, can_write(account),
                              is_admin(account), snapshot)
            page.create_details()
            page.create_files()
            page.create_scores(request.query_params.get("split", ""),
                               request.query_params.getlist(ATTRIBUTE_PARAMETER))
            page.create_coding_table_export()
            page.create_actions()
