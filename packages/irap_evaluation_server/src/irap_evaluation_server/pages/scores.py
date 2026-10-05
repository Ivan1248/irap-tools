"""The Scores page: the methods of a dataset split, with the means over their models of the
attribute averages, or of one per-attribute metric, and optionally their differences from a
reference method (see `method_comparison`). The methods trained on the split are left out unless
the user shows them."""

import dataclasses as dc
import html
import typing as T

import irap_evaluation as ie
from fastapi import Request
from nicegui import ui

from ..components.account_sessions import AccountSessions
from ..components.attribute_filter import AttributeFilter, AttributeSubset
from ..components.formatting import EVALUATION_SET_LABELS, make_model_link_html
from ..components.native_controls import (
    create_native_checkbox,
    create_native_input,
    create_native_select,
)
from ..components.page_frame import create_page_frame, show_view_error
from ..components.refresh_timer import RefreshTimer
from ..components.routes import ATTRIBUTE_PARAMETER, SCORES_PATH, make_query_path
from ..components.score_tables import (
    ReferenceComparisons,
    make_comparison_note,
    make_interval_note,
    make_method_table_html,
    make_per_attribute_table_html,
)
from ..components.view_queries import (
    VIEW_OPTIONS,
    parse_dataset_and_split,
    parse_evaluation_set,
    parse_per_attribute_metric,
    select_matching_names,
)
from ..components.work_requests import WorkRequester
from ..datasets import DatasetContext
from ..method_comparison import list_comparable_methods, make_comparison_request
from ..method_ranking import (
    DEFAULT_SORT_METRIC,
    RANKING_METRIC_NAMES,
    collect_scored_attributes,
    get_method_split_use,
    get_unscored_reason,
    rank_methods,
    select_shown_methods,
)
from ..scoring import (
    MethodScores,
    ResultState,
    ScoredModel,
    ScoringSettings,
    WorkRequest,
    make_interval_request,
    select_scored_models,
)
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
        method_pattern: The regular expression of the shown methods (`select_matching_names`),
            '' for all.
        reference_method: The method that the others are compared with, on the reference set
            only, or '' for none.
        shows_trained_methods: Whether the methods trained on the split are shown.
    """

    dataset: str
    split: str
    evaluation_set: str
    sort_metric: str
    metric_name: str
    method_pattern: str = ""
    reference_method: str = ""
    shows_trained_methods: bool = False

    def to_path(self, attributes: T.Sequence[str] = ()) -> str:
        """
        Args:
            attributes: The attribute subset (`AttributeSubset.to_query`).
        """
        return make_query_path(SCORES_PATH, {
            "dataset": self.dataset, "split": self.split,
            "set": "" if self.evaluation_set == ie.REFERENCE_SET_NAME else self.evaluation_set,
            "sort": "" if self.sort_metric == DEFAULT_SORT_METRIC else self.sort_metric,
            "metric": self.metric_name, "methods": self.method_pattern,
            "ref": self.reference_method, "trained": "1" if self.shows_trained_methods else "",
            ATTRIBUTE_PARAMETER: attributes})


def _parse_view(query: T.Mapping[str, str],
                dataset_contexts: T.Mapping[str, DatasetContext]) -> _ScoresView:
    """Parses the view from the query parameters of `_ScoresView.to_path`. The page checks the
    method pattern and the reference method against the scored models.

    Raises:
        ValueError: For an unknown dataset, split, evaluation set or metric, or a reference
            method on the model-compatible sets.
    """
    dataset, split = parse_dataset_and_split(query, dataset_contexts)
    evaluation_set = parse_evaluation_set(query)
    sort_metric = query.get("sort") or DEFAULT_SORT_METRIC
    if sort_metric not in RANKING_METRIC_NAMES:
        raise ValueError(f"Unknown sort metric {sort_metric!r}. Metrics:"
                         f" {', '.join(RANKING_METRIC_NAMES)}.")
    reference_method = query.get("ref", "")
    if reference_method and evaluation_set != ie.REFERENCE_SET_NAME:
        raise ValueError("Methods are compared only on the reference set.")
    return _ScoresView(dataset=dataset, split=split, evaluation_set=evaluation_set,
                       sort_metric=sort_metric, metric_name=parse_per_attribute_metric(query),
                       method_pattern=query.get("methods", ""),
                       reference_method=reference_method,
                       shows_trained_methods=query.get("trained") == "1")


def register_scores_page(dataset_contexts: T.Mapping[str, DatasetContext],
                         worker: ScoringWorker, sessions: AccountSessions) -> None:
    @ui.page(SCORES_PATH, title="Scores · iRAP evaluation")
    def scores_page(request: Request) -> None:
        with create_page_frame(SCORES_PATH, sessions.get_account()):
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
        requester.delay_requests()
        show_table.refresh()

    def on_method_pattern_changed(method_pattern: str) -> None:
        requester.delay_requests()
        on_view_changed(method_pattern=method_pattern)

    context = dataset_contexts[view.dataset]
    requester = WorkRequester(worker)

    def show_methods(models: T.Sequence[ScoredModel], subset: AttributeSubset,
                     matching_method_names: T.AbstractSet[str]) -> None:
        rows = rank_methods(models, subset.selected, view.sort_metric)
        name_to_comparable = {r.method_name: r for r in list_comparable_methods(rows)}
        with ui.element("div").classes("form-row"):
            if view.evaluation_set == ie.REFERENCE_SET_NAME:
                create_native_select("Compare with",
                                     {"": "–", **{n: r.shown_method_name
                                                  for n, r in name_to_comparable.items()}},
                                     view.reference_method,
                                     lambda v: on_view_changed(reference_method=v))
            if view.shows_trained_methods or any(
                    get_method_split_use(r, view.split) == "training" for r in rows):
                create_native_checkbox(f"Show methods trained on {view.split}",
                                       view.shows_trained_methods,
                                       lambda v: on_view_changed(shows_trained_methods=v))
        reference = name_to_comparable.get(view.reference_method)
        if view.reference_method and reference is None:
            ui.label(f"{view.reference_method} cannot be compared, since it has no scores on"
                     f" this split or its models cannot be combined.").classes("error")
        elif reference is not None and reference.method_name not in matching_method_names:
            ui.label(f"{reference.shown_method_name} is shown as the reference, although the filter"
                     f" excludes it.").classes("muted")
        reference_method = "" if reference is None else reference.method_name
        shown, num_hidden = select_shown_methods(
            rows, matching_method_names, reference_method,
            hidden_training_split=None if view.shows_trained_methods else view.split)
        if num_hidden:
            ui.label(f"Number of hidden methods trained on {view.split}: {num_hidden}.").classes(
                "muted")
        method_to_intervals, comparisons, comparison_errors = _request_results(
            requester, shown, reference, subset.selected, worker.settings)
        if not shown:
            if not num_hidden:  # Otherwise, the label above says that the matches are hidden.
                ui.label("The filter matches no method.").classes("muted")
            return
        if view.metric_name:
            table_html = make_per_attribute_table_html(shown, method_to_intervals,
                                                       view.metric_name, subset.selected,
                                                       view.split, comparisons)
        else:
            table_html = make_method_table_html(
                shown, method_to_intervals, view.sort_metric, view.split, subset.to_query(),
                view.evaluation_set == ie.MODEL_COMPATIBLE_SET_NAME, comparisons)
        ui.html(table_html, sanitize=False).classes("w-full overflow-x-auto").on(
            "click", lambda e: on_view_changed(sort_metric=e.args),
            js_handler="(e) => { const th = e.target.closest('th[data-sort]');"
                       " if (th) emit(th.dataset.sort); }")
        ui.label(make_interval_note(worker.settings, is_mean_over_models=True)).classes(
            "muted")
        if reference is not None:
            ui.label(make_comparison_note(worker.settings, reference.shown_method_name, shown)
                     ).classes("muted")
        for error in comparison_errors:
            ui.label(error).classes("error")
        if view.evaluation_set == ie.MODEL_COMPATIBLE_SET_NAME:
            ui.label("Each method is scored on the segments that its context offsets"
                     " select, so the rows cover different segments.").classes("muted")

    @ui.refreshable
    def show_table() -> None:
        submissions, submission_id_to_scoring, current = worker.load_split_scorings(
            view.dataset, view.split)
        models = select_scored_models(submissions, current, view.evaluation_set)
        subset = attribute_filter.set_options(collect_scored_attributes(models))
        pending_ids = [s.id for s in submissions if worker.is_scoring_pending(s.id)]

        title = f"{view.dataset}/{view.split}, {_EVALUATION_SET_LABELS[view.evaluation_set]}"
        if models:
            title += (f": {view.metric_name} of {subset.label}" if view.metric_name
                      else f": averages over {subset.label}")
        ui.label(title).classes("section-title")
        matching_method_names = None
        if not models:
            ui.label("No models are scored on this evaluation set.").classes("muted")
            notes = {n for s in current.values() if s.get_scores(view.evaluation_set) is None
                     for n in s.notes}
            for note in sorted(notes):
                ui.label(note).classes("muted")
        elif not subset.selected:
            ui.label("Select at least one attribute.").classes("error")
        else:
            # The filter matches the shown method names.
            name_to_shown = {m.submission.model.method_name: m.submission.model.shown_method_name
                             for m in models}
            try:
                matching_shown_names = select_matching_names(view.method_pattern,
                                                             set(name_to_shown.values()))
                matching_method_names = {n for n, s in name_to_shown.items()
                                         if s in matching_shown_names}
            except ValueError as e:
                ui.label(str(e)).classes("error")
        if matching_method_names is None:
            requester.update({})  # Withdraws the requests of an earlier render.
        else:
            show_methods(models, subset, matching_method_names)

        unlisted = [(s, reason) for s in submissions
                    if (reason := get_unscored_reason(submission_id_to_scoring.get(s.id),
                                                      s.id in current, s.id in pending_ids))]
        if unlisted:
            ui.label("Unscored files").classes("section-title")
            ui.html("".join(f"<div>{make_model_link_html(s.model.id, s.shown_label, s.split)}"
                            f": {html.escape(reason)}</div>"
                            for s, reason in unlisted), sanitize=False)
        refresh_timer.watch([*(lambda i=i: worker.is_scoring_pending(i) for i in pending_ids),
                             *requester.get_pending_checks()])

    with ui.element("section").classes("panel"):
        with ui.element("div").classes("form-row"):
            # Another dataset has other methods, and the other evaluation set no comparisons.
            create_native_select("Dataset", {n: n for n in dataset_contexts}, view.dataset,
                                 lambda v: navigate_to(dataset=v, split="", reference_method=""))
            create_native_select("Split", {n: n for n in context.split_names}, view.split,
                                 lambda v: navigate_to(split=v))
            if view.evaluation_set != ie.REFERENCE_SET_NAME or _has_model_compatible_scores(
                    worker, view):
                create_native_select(
                    "Evaluation set", _EVALUATION_SET_LABELS, view.evaluation_set,
                    lambda v: navigate_to(evaluation_set=v, reference_method=""))
            create_native_select("View", VIEW_OPTIONS, view.metric_name,
                                 lambda v: on_view_changed(metric_name=v))
            create_native_input("Method filter (regex)", view.method_pattern,
                                on_method_pattern_changed, placeholder="all", size=24)
        attribute_filter = AttributeFilter(attribute_query, on_subset_changed)
    with ui.element("section").classes("panel"):
        refresh_timer = RefreshTimer(show_table.refresh)
        show_table()


def _request_results(requester: WorkRequester, rows: T.Sequence[MethodScores],
                     reference: MethodScores | None, attributes: T.Sequence[str],
                     settings: ScoringSettings
                     ) -> tuple[dict[str, ResultState], ReferenceComparisons | None, list[str]]:
    """Requests the intervals of the rows, and their comparisons with the reference method, row
    by row, so that the worker computes them in the order of the rows.

    Args:
        reference: The reference method, without an error, or None.

    Returns:
        Method name -> the state of its intervals, the comparisons (None without a reference),
        and the errors of the rows that cannot be compared with the reference.
    """
    key_to_request: dict[tuple[str, str], WorkRequest] = {}
    comparison_errors = []
    for row in rows:
        if row.error is not None:
            continue
        key_to_request["intervals", row.method_name] = make_interval_request(
            row.models, attributes, settings)
        if reference is None or row.method_name == reference.method_name:
            continue
        try:
            key_to_request["comparison", row.method_name] = make_comparison_request(
                row.models, reference.models, attributes, settings)
        except ValueError as e:
            comparison_errors.append(f"{row.shown_method_name} cannot be compared with"
                                     f" {reference.shown_method_name}: {e}")
    key_to_state = requester.update(key_to_request)

    def select_states(kind: str) -> dict[str, ResultState]:
        return {name: state for (k, name), state in key_to_state.items() if k == kind}

    comparisons = (None if reference is None
                   else ReferenceComparisons(reference, select_states("comparison")))
    return select_states("intervals"), comparisons, comparison_errors


def _has_model_compatible_scores(worker: ScoringWorker, view: _ScoresView) -> bool:
    submissions, _, current = worker.load_split_scorings(view.dataset, view.split)
    return bool(select_scored_models(submissions, current, ie.MODEL_COMPATIBLE_SET_NAME))
