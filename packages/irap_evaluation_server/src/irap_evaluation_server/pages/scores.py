"""The Scores page: the methods of a dataset split, with the means over their runs of the
attribute averages, or of one per-attribute metric."""

import dataclasses as dc
import html
import typing as T

import irap_evaluation as ie
from fastapi import Request
from nicegui import ui

from ..components.attribute_filter import AttributeFilter, AttributeSubset
from ..components.comparison_view import ComparisonView
from ..components.formatting import EVALUATION_SET_LABELS, make_submission_link_html
from ..components.native_controls import create_native_select
from ..components.page_frame import create_page_frame, show_view_error
from ..components.refresh_timer import RefreshTimer
from ..components.routes import ATTRIBUTE_PARAMETER, SCORES_PATH, make_query_path
from ..components.score_tables import (
    make_interval_note,
    make_method_table_html,
    make_per_attribute_table_html,
)
from ..components.view_queries import (
    VIEW_OPTIONS,
    parse_dataset_and_split,
    parse_evaluation_set,
    parse_per_attribute_metric,
)
from ..components.work_requests import WorkRequester
from ..datasets import DatasetContext
from ..method_comparison import get_default_partner, list_comparable_methods
from ..method_ranking import (
    DEFAULT_SORT_METRIC,
    RANKING_METRIC_NAMES,
    collect_scored_attributes,
    get_unscored_reason,
    rank_methods,
)
from ..scoring import MethodScores, make_interval_request, select_scored_runs
from ..scoring_worker import ScoringWorker

#: Plural for the model-compatible sets, since each method is scored on its own.
_EVALUATION_SET_LABELS = {**EVALUATION_SET_LABELS,
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

    def to_path(self, attributes: T.Sequence[str] = ()) -> str:
        """
        Args:
            attributes: The attribute subset (`AttributeSubset.to_query`).
        """
        return make_query_path(SCORES_PATH, {
            "dataset": self.dataset, "split": self.split,
            "set": "" if self.evaluation_set == ie.REFERENCE_SET_NAME else self.evaluation_set,
            "sort": "" if self.sort_metric == DEFAULT_SORT_METRIC else self.sort_metric,
            "metric": self.metric_name, ATTRIBUTE_PARAMETER: attributes})


def _parse_view(query: T.Mapping[str, str],
                dataset_contexts: T.Mapping[str, DatasetContext]) -> _ScoresView:
    """
    Raises:
        ValueError: For an unknown dataset, split, evaluation set or metric.
    """
    dataset, split = parse_dataset_and_split(query, dataset_contexts)
    sort_metric = query.get("sort") or DEFAULT_SORT_METRIC
    if sort_metric not in RANKING_METRIC_NAMES:
        raise ValueError(f"Unknown sort metric {sort_metric!r}. Metrics:"
                         f" {', '.join(RANKING_METRIC_NAMES)}.")
    return _ScoresView(dataset=dataset, split=split, evaluation_set=parse_evaluation_set(query),
                       sort_metric=sort_metric, metric_name=parse_per_attribute_metric(query))


def register_scores_page(dataset_contexts: T.Mapping[str, DatasetContext],
                         worker: ScoringWorker) -> None:
    @ui.page(SCORES_PATH, title="Scores · iRAP evaluation")
    def scores_page(request: Request) -> None:
        with create_page_frame(SCORES_PATH):
            try:
                view = _parse_view(request.query_params, dataset_contexts)
            except ValueError as e:
                show_view_error(str(e), SCORES_PATH)
                return
            _create_scores(dataset_contexts, worker, view,
                           request.query_params.getlist(ATTRIBUTE_PARAMETER))


def _create_scores(dataset_contexts: T.Mapping[str, DatasetContext], worker: ScoringWorker,
                   view: _ScoresView, attribute_query: T.Sequence[str]) -> None:
    def navigate_to(**changes: str) -> None:
        # Another dataset has other attributes.
        attributes = () if "dataset" in changes else attribute_filter.subset.to_query()
        ui.navigate.to(dc.replace(view, **changes).to_path(attributes))

    def on_view_changed(**changes: str) -> None:
        nonlocal view
        view = dc.replace(view, **changes)
        ui.navigate.history.replace(view.to_path(attribute_filter.subset.to_query()))
        show_table.refresh()

    def on_subset_changed(subset: AttributeSubset) -> None:
        ui.navigate.history.replace(view.to_path(subset.to_query()))
        interval_requester.delay_requests()
        show_table.refresh()

    context = dataset_contexts[view.dataset]
    interval_requester = WorkRequester(worker)

    @ui.refreshable
    def show_table() -> None:
        submissions, submission_id_to_scoring, current = worker.load_split_scorings(
            view.dataset, view.split)
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
                    view.evaluation_set == ie.MODEL_COMPATIBLE_SET_NAME,
                    _make_comparison_paths(view, rows, subset))
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
                    worker, view):
                create_native_select("Evaluation set", _EVALUATION_SET_LABELS,
                                     view.evaluation_set, lambda v: navigate_to(evaluation_set=v))
            create_native_select("View", VIEW_OPTIONS, view.metric_name,
                                 lambda v: on_view_changed(metric_name=v))
        attribute_filter = AttributeFilter(attribute_query, on_subset_changed)
    with ui.element("section").classes("panel"):
        refresh_timer = RefreshTimer(show_table.refresh)
        show_table()


def _make_comparison_paths(view: _ScoresView, rows: T.Sequence[MethodScores],
                           subset: AttributeSubset) -> dict[str, str]:
    """Method name -> the Comparison page of it and its default partner
    (`method_comparison.get_default_partner`), for the methods that can be compared.

    Args:
        rows: The methods of the table, best first.
    """
    names = [r.method_name for r in list_comparable_methods(rows)]
    return {name: ComparisonView(view.dataset, view.split, name, partner).to_path(
                subset.to_query())
            for name in names if (partner := get_default_partner(names, name)) is not None}


def _has_model_compatible_scores(worker: ScoringWorker, view: _ScoresView) -> bool:
    submissions, _, current = worker.load_split_scorings(view.dataset, view.split)
    return bool(select_scored_runs(submissions, current, ie.MODEL_COMPATIBLE_SET_NAME))
