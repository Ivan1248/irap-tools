"""The Comparison page: two methods of a dataset split on its reference set, with the differences of
their metrics and the intervals of the differences (see `method_comparison`)."""

import dataclasses as dc
import html
import typing as T

import irap_evaluation as ie
from fastapi import Request
from nicegui import ui

from ..components.attribute_filter import AttributeFilter, AttributeSubset
from ..components.comparison_view import ComparisonView
from ..components.native_controls import create_native_select
from ..components.page_frame import create_page_frame, show_view_error
from ..components.refresh_timer import RefreshTimer
from ..components.routes import ATTRIBUTE_PARAMETER, COMPARISON_PATH
from ..components.score_tables import make_comparison_note, make_comparison_table_html
from ..components.view_queries import VIEW_OPTIONS
from ..components.work_requests import WorkRequester
from ..datasets import DatasetContext
from ..method_comparison import (
    ComparisonRequest,
    list_comparable_methods,
    make_comparison_request,
    select_method_pair,
)
from ..method_ranking import DEFAULT_SORT_METRIC, collect_scored_attributes, rank_methods
from ..scoring import MethodScores, ResultState, ScoringSettings, select_scored_runs
from ..scoring_worker import ScoringWorker


def register_comparison_page(dataset_contexts: T.Mapping[str, DatasetContext],
                             worker: ScoringWorker) -> None:
    @ui.page(COMPARISON_PATH, title="Comparison · iRAP evaluation")
    def comparison_page(request: Request) -> None:
        with create_page_frame(COMPARISON_PATH):
            try:
                view = ComparisonView.from_query(request.query_params, dataset_contexts)
            except ValueError as e:
                show_view_error(str(e), COMPARISON_PATH)
                return
            _create_comparison(dataset_contexts, worker, view,
                               request.query_params.getlist(ATTRIBUTE_PARAMETER))


def _describe_runs(row: MethodScores) -> str:
    return f"{row.method_name} ({row.num_runs} run{'' if row.num_runs == 1 else 's'})"


def _create_comparison(dataset_contexts: T.Mapping[str, DatasetContext], worker: ScoringWorker,
                       view: ComparisonView, attribute_query: T.Sequence[str]) -> None:
    def navigate_to(**changes: str) -> None:
        # Another dataset has other attributes and methods.
        attributes = () if "dataset" in changes else attribute_filter.subset.to_query()
        ui.navigate.to(dc.replace(view, **changes).to_path(attributes))

    def on_view_changed(**changes: str) -> None:
        nonlocal view
        view = dc.replace(view, **changes)
        ui.navigate.history.replace(view.to_path(attribute_filter.subset.to_query()))
        show_comparison.refresh()

    def on_subset_changed(subset: AttributeSubset) -> None:
        ui.navigate.history.replace(view.to_path(subset.to_query()))
        requester.delay_requests()
        show_comparison.refresh()

    context = dataset_contexts[view.dataset]
    requester = WorkRequester(worker)

    @ui.refreshable
    def show_comparison() -> None:
        submissions, _, current = worker.load_split_scorings(view.dataset, view.split)
        runs = select_scored_runs(submissions, current, ie.REFERENCE_SET_NAME)
        subset = attribute_filter.set_options(collect_scored_attributes(runs))
        pending_ids = [s.id for s in submissions if worker.is_scoring_pending(s.id)]
        if pending_ids:
            ui.label(f"Scoring {len(pending_ids)} submissions…").classes("muted")
        all_rows = (rank_methods(runs, subset.selected, DEFAULT_SORT_METRIC) if subset.selected
                    else [])
        rows = list_comparable_methods(all_rows)
        pair = None
        if not runs:
            ui.label("No runs are scored on the reference set.").classes("muted")
        elif not subset.selected:
            ui.label("Select at least one attribute.").classes("error")
        elif len(rows) < 2:
            ui.label("Two methods whose runs can be combined are needed.").classes("muted")
        else:
            try:
                pair = _make_method_pair(rows, view, subset, worker.settings)
            except ValueError as e:
                ui.label(str(e)).classes("error")
            with ui.element("div").classes("form-row"):
                options = {r.method_name: _describe_runs(r) for r in rows}
                create_native_select(
                    "Method A", options,
                    view.method_a if pair is None else pair.scores_a.method_name,
                    lambda v: on_view_changed(method_a=v))
                create_native_select(
                    "Method B", options,
                    view.method_b if pair is None else pair.scores_b.method_name,
                    lambda v: on_view_changed(method_b=v))
        for row in all_rows:
            if row.error is not None:
                ui.label(f"{row.method_name} cannot be compared: {row.error}").classes("muted")
        # Without a comparison, this withdraws an earlier request of the page.
        key_to_state = requester.update({} if pair is None else {"comparison": pair.request})
        if pair is not None:
            _show_comparison_table(worker.settings, view, pair, key_to_state["comparison"],
                                   subset)
        refresh_timer.watch([*(lambda i=i: worker.is_scoring_pending(i) for i in pending_ids),
                             *requester.get_pending_checks()])

    with ui.element("section").classes("panel"):
        with ui.element("div").classes("form-row"):
            create_native_select("Dataset", {n: n for n in dataset_contexts}, view.dataset,
                                 lambda v: navigate_to(dataset=v, split="", method_a="",
                                                       method_b=""))
            create_native_select("Split", {n: n for n in context.split_names}, view.split,
                                 lambda v: navigate_to(split=v))
            create_native_select("View", VIEW_OPTIONS, view.metric_name,
                                 lambda v: on_view_changed(metric_name=v))
        attribute_filter = AttributeFilter(attribute_query, on_subset_changed)
    with ui.element("section").classes("panel"):
        refresh_timer = RefreshTimer(show_comparison.refresh)
        show_comparison()


@dc.dataclass(frozen=True)
class _MethodPair:
    """Two methods with their means over runs (without an error), and their comparison."""

    scores_a: MethodScores
    scores_b: MethodScores
    request: ComparisonRequest


def _make_method_pair(rows: T.Sequence[MethodScores], view: ComparisonView,
                      subset: AttributeSubset, settings: ScoringSettings) -> _MethodPair:
    """
    Raises:
        ValueError: See `select_method_pair` and `make_comparison_request`.
    """
    scores_a, scores_b = select_method_pair(rows, view.method_a, view.method_b)
    return _MethodPair(scores_a, scores_b, make_comparison_request(
        scores_a.runs, scores_b.runs, subset.selected, settings))


def _show_comparison_table(settings: ScoringSettings, view: ComparisonView, pair: _MethodPair,
                           state: ResultState, subset: AttributeSubset) -> None:
    row_a, row_b = pair.scores_a, pair.scores_b
    what = view.metric_name or "attribute averages"
    ui.label(f"{view.dataset}/{view.split}, reference set: {what} of {subset.label}").classes(
        "section-title")
    ui.html(make_comparison_table_html(row_a, row_b, state, view.metric_name, subset.selected),
            sanitize=False).classes("w-full overflow-x-auto")
    if state.cached is not None and state.cached.value is None:
        ui.label(f"The comparison failed: {state.cached.message}").classes("error")
    notes = [make_comparison_note(settings, has_single_run=1 in (row_a.num_runs,
                                                                 row_b.num_runs)),
             f"A: {_describe_runs(row_a)}, B: {_describe_runs(row_b)}."]
    if view.metric_name:
        notes.append("Differences of single attributes are exploratory: with many attributes,"
                     " some intervals exclude 0 by chance. Base conclusions on the main metric,"
                     f" {DEFAULT_SORT_METRIC}.")
    for row in (row_a, row_b):
        if row.missing_attributes:
            notes.append(f"{row.method_name} does not predict"
                         f" {', '.join(row.missing_attributes)}, which count as invalid cells.")
    ui.html("".join(f'<div class="muted">{html.escape(n)}</div>' for n in notes),
            sanitize=False)
