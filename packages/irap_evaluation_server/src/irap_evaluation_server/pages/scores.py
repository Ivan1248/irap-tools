"""The Scores page: the methods of a dataset split, with the means over their runs of the
attribute averages, or of one per-attribute metric."""

import dataclasses as dc
import html
import typing as T

import irap_evaluation as ie
from fastapi import Request
from nicegui import ui

from ..archive import Submission, SubmissionArchive
from ..components.attribute_filter import AttributeFilter, AttributeSubset
from ..components.formatting import make_submission_link_html
from ..components.interval_requests import IntervalRequester
from ..components.native_controls import create_native_select
from ..components.page_frame import create_page_frame
from ..components.refresh_timer import RefreshTimer
from ..components.routes import ATTRIBUTE_PARAMETER, SCORES_PATH, make_query_path
from ..components.score_tables import (
    make_interval_note,
    make_method_table_html,
    make_per_attribute_table_html,
)
from ..datasets import DatasetContext
from ..method_ranking import (
    DEFAULT_SORT_METRIC,
    RANKING_METRIC_NAMES,
    PER_ATTRIBUTE_METRIC_NAMES,
    collect_scored_attributes,
    rank_methods,
    get_unscored_reason,
)
from ..scoring import RunScoring, make_interval_request, select_scored_runs
from ..scoring_worker import ScoringWorker

_EVALUATION_SET_LABELS = {ie.REFERENCE_SET_NAME: "Reference set",
                          ie.MODEL_COMPATIBLE_SET_NAME: "Model-compatible sets"}


@dc.dataclass(frozen=True)
class _ScoresView:
    """What the page shows, as in its URL.

    Attributes:
        metric_name: The per-attribute metric of the per-attribute table, or '' for the
            attribute averages.
    """

    dataset: str
    split: str
    evaluation_set: str
    sort_metric: str
    metric_name: str

    def to_path(self, subset: AttributeSubset | None) -> str:
        return make_query_path(SCORES_PATH, {
            "dataset": self.dataset, "split": self.split,
            "set": "" if self.evaluation_set == ie.REFERENCE_SET_NAME else self.evaluation_set,
            "sort": "" if self.sort_metric == DEFAULT_SORT_METRIC else self.sort_metric,
            "metric": self.metric_name,
            ATTRIBUTE_PARAMETER: () if subset is None else subset.to_query()})


def _parse_view(query: T.Mapping[str, str],
                dataset_contexts: T.Mapping[str, DatasetContext]) -> _ScoresView:
    """
    Raises:
        ValueError: For an unknown dataset, split, evaluation set or metric.
    """
    dataset = query.get("dataset") or next(iter(dataset_contexts))
    if dataset not in dataset_contexts:
        raise ValueError(f"Unknown dataset {dataset!r}. Datasets: {', '.join(dataset_contexts)}.")
    context = dataset_contexts[dataset]
    split = query.get("split") or (context.config.analysis_splits or context.split_names)[0]
    if split not in context.split_names:
        raise ValueError(f"The dataset {dataset!r} has no split {split!r}. Its splits:"
                         f" {', '.join(context.split_names)}.")
    evaluation_set = query.get("set") or ie.REFERENCE_SET_NAME
    if evaluation_set not in _EVALUATION_SET_LABELS:
        raise ValueError(f"Unknown evaluation set {evaluation_set!r}. Evaluation sets:"
                         f" {', '.join(_EVALUATION_SET_LABELS)}.")
    sort_metric = query.get("sort") or DEFAULT_SORT_METRIC
    if sort_metric not in RANKING_METRIC_NAMES:
        raise ValueError(f"Unknown sort metric {sort_metric!r}. Metrics:"
                         f" {', '.join(RANKING_METRIC_NAMES)}.")
    metric_name = query.get("metric", "")
    if metric_name and metric_name not in PER_ATTRIBUTE_METRIC_NAMES:
        raise ValueError(f"Unknown per-attribute metric {metric_name!r}. Metrics:"
                         f" {', '.join(PER_ATTRIBUTE_METRIC_NAMES)}.")
    return _ScoresView(dataset=dataset, split=split, evaluation_set=evaluation_set,
                            sort_metric=sort_metric, metric_name=metric_name)


def register_scores_page(archive: SubmissionArchive,
                              dataset_contexts: T.Mapping[str, DatasetContext],
                              worker: ScoringWorker) -> None:
    @ui.page(SCORES_PATH, title="Scores · iRAP evaluation")
    def scores_page(request: Request) -> None:
        with create_page_frame(SCORES_PATH):
            try:
                view = _parse_view(request.query_params, dataset_contexts)
            except ValueError as e:
                ui.label(str(e)).classes("error")
                ui.link("Open the default Scores page", SCORES_PATH)
                return
            _create_scores(archive, dataset_contexts, worker, view,
                                request.query_params.getlist(ATTRIBUTE_PARAMETER))


def _create_scores(archive: SubmissionArchive,
                        dataset_contexts: T.Mapping[str, DatasetContext], worker: ScoringWorker,
                        view: _ScoresView, attribute_query: T.Sequence[str]) -> None:
    def navigate_to(**changes: str) -> None:
        # Another dataset has other attributes.
        subset = None if "dataset" in changes else attribute_filter.subset
        ui.navigate.to(dc.replace(view, **changes).to_path(subset))

    def on_view_changed(**changes: str) -> None:
        nonlocal view
        view = dc.replace(view, **changes)
        ui.navigate.history.replace(view.to_path(attribute_filter.subset))
        show_table.refresh()

    def on_subset_changed(subset: AttributeSubset) -> None:
        ui.navigate.history.replace(view.to_path(subset))
        interval_requester.delay_requests()
        show_table.refresh()

    context = dataset_contexts[view.dataset]
    interval_requester = IntervalRequester(worker)

    @ui.refreshable
    def show_table() -> None:
        submissions, submission_id_to_scoring, current = _load_view_scorings(archive, worker,
                                                                             view)
        runs = select_scored_runs(submissions, current, view.evaluation_set)
        subset = attribute_filter.set_options(collect_scored_attributes(runs))
        pending_ids = [s.id for s in submissions if worker.is_scoring_pending(s.id)]

        title = f"{view.dataset}/{view.split}, {_EVALUATION_SET_LABELS[view.evaluation_set]}"
        if runs:
            title += (f": {view.metric_name} of {subset.label}" if view.metric_name
                      else f": averages over {subset.label}")
        ui.label(title).classes("section-title")
        if not runs:
            ui.label("No runs are scored on this evaluation set.").classes("muted")
            notes = {n for s in current.values() if s.get_scores(view.evaluation_set) is None
                     for n in s.notes}
            for note in sorted(notes):
                ui.label(note).classes("muted")
        elif not subset.selected:
            ui.label("Select at least one attribute.").classes("error")
        else:
            rows = rank_methods(runs, subset.selected, view.sort_metric)
            method_to_state = interval_requester.update(
                {r.method_name: make_interval_request(r.runs, subset.selected, worker.settings)
                 for r in rows if r.error is None})
            if view.metric_name:
                table_html = make_per_attribute_table_html(rows, method_to_state,
                                                           view.metric_name, subset.selected)
            else:
                table_html = make_method_table_html(
                    rows, method_to_state, view.sort_metric, subset.to_query(),
                    view.evaluation_set == ie.MODEL_COMPATIBLE_SET_NAME)
            ui.html(table_html, sanitize=False).classes("w-full overflow-x-auto").on(
                "click", lambda e: on_view_changed(sort_metric=e.args),
                js_handler="(e) => { const th = e.target.closest('th[data-sort]');"
                           " if (th) emit(th.dataset.sort); }")
            ui.label(make_interval_note(worker.settings, is_mean_over_runs=True)).classes("muted")
            if view.evaluation_set == ie.MODEL_COMPATIBLE_SET_NAME:
                ui.label("Each method is scored on the segments that its context offsets"
                         " select, so the rows cover different segments.").classes("muted")

        unlisted = [(s, reason) for s in submissions
                    if (reason := get_unscored_reason(s, submission_id_to_scoring.get(s.id),
                                                      s.id in current, s.id in pending_ids))]
        if unlisted:
            ui.label("Not in the method table").classes("section-title")
            ui.html("".join(f"<div>{make_submission_link_html(s.id)}"
                            f" {html.escape(s.run_label)}: {html.escape(reason)}</div>"
                            for s, reason in unlisted), sanitize=False)
        refresh_timer.watch([*(lambda i=i: worker.is_scoring_pending(i) for i in pending_ids),
                             *interval_requester.get_pending_checks()])

    with ui.element("section").classes("panel"):
        with ui.element("div").classes("form-row"):
            create_native_select("Dataset", {n: n for n in dataset_contexts}, view.dataset,
                                 lambda v: navigate_to(dataset=v, split=""))
            create_native_select("Split", {n: n for n in context.split_names}, view.split,
                                 lambda v: navigate_to(split=v))
            if view.evaluation_set != ie.REFERENCE_SET_NAME or _has_model_compatible_scores(
                    archive, worker, view):
                create_native_select("Evaluation set", _EVALUATION_SET_LABELS,
                                     view.evaluation_set, lambda v: navigate_to(evaluation_set=v))
            create_native_select(
                "View", {"": "Attribute averages",
                         **{m: f"Per attribute: {m}" for m in PER_ATTRIBUTE_METRIC_NAMES}},
                view.metric_name, lambda v: on_view_changed(metric_name=v))
        attribute_filter = AttributeFilter(attribute_query, on_subset_changed)
    with ui.element("section").classes("panel"):
        refresh_timer = RefreshTimer(show_table.refresh)
        show_table()


def _load_view_scorings(archive: SubmissionArchive, worker: ScoringWorker, view: _ScoresView
                        ) -> tuple[list[Submission], dict[int, RunScoring], dict[int, RunScoring]]:
    """The submissions of the dataset split of `view` that are not deleted, the stored scorings
    (`ScoreStore.get_run_scorings`) and the current ones."""
    submissions = [s for s in archive.list_submissions()
                   if (s.dataset, s.split) == (view.dataset, view.split)]
    submission_id_to_scoring = worker.store.get_run_scorings(s.id for s in submissions)
    return (submissions, submission_id_to_scoring,
            worker.select_current_scorings(submissions, submission_id_to_scoring))


def _has_model_compatible_scores(archive: SubmissionArchive, worker: ScoringWorker,
                                 view: _ScoresView) -> bool:
    submissions, _, current = _load_view_scorings(archive, worker, view)
    return bool(select_scored_runs(submissions, current, ie.MODEL_COMPATIBLE_SET_NAME))
