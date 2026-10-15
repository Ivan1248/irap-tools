"""The Scores page: the methods of a dataset split, with the means over their models of the
attribute averages, or of one per-attribute metric, and optionally their differences from a
reference method (see `method_comparison`). The methods trained on the split are left out unless
the user shows them."""

import dataclasses as dc
import typing as T

import irap_evaluation as ie
from fastapi import Request
from nicegui import run, ui

from ..components.account_sessions import AccountSessions
from ..components.attribute_filter import AttributeFilter, AttributeSubset
from ..components.formatting import EVALUATION_SET_LABELS, make_unscored_files_html
from ..components.latest_updates import LatestUpdates
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
    ScoresScope,
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
from ..datasets import DatasetContext
from ..method_comparison import compare_with_reference, list_comparable_methods
from ..method_ranking import (
    DEFAULT_SORT_METRIC,
    RANKING_METRIC_NAMES,
    collect_scored_attributes,
    get_method_split_use,
    rank_methods,
    select_shown_methods,
)
from ..scoring import MethodScores, ScoredModel
from ..scoring_worker import ScoringWorker, SubmissionScorings

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
    async def scores_page(request: Request) -> None:
        with create_page_frame(SCORES_PATH, sessions.get_account()):
            try:
                view = _parse_view(request.query_params, dataset_contexts)
            except ValueError as e:
                show_view_error(str(e), SCORES_PATH)
                return
            await _create_scores(dataset_contexts, worker, view,
                                 request.query_params.getlist(ATTRIBUTE_PARAMETER))


@dc.dataclass(frozen=True)
class _SplitScorings:
    """The scorings of the files of the split of a view (`_load_split_scorings`).

    Attributes:
        models: The models scored on the evaluation set of the view.
    """

    scorings: SubmissionScorings
    models: list[ScoredModel]


def _load_split_scorings(worker: ScoringWorker, view: _ScoresView) -> _SplitScorings:
    scorings = worker.load_split_scorings(view.dataset, view.split)
    return _SplitScorings(scorings, scorings.select_scored_models(view.evaluation_set))


@dc.dataclass(frozen=True)
class _MethodTable:
    """The methods of a view and their table (`_compute_method_table`).

    Attributes:
        rows: All methods, ranked (`rank_methods`).
        name_to_comparable: Method name -> its row, for the methods that can be compared
            (`list_comparable_methods`).
        reference: The reference method of the view, None if there is none or it cannot be
            compared.
        is_reference_excluded: Whether the method filter excludes the reference method, which is
            shown anyway.
        shown: The rows of the table (`select_shown_methods`).
        num_hidden: The number of matching methods that are hidden, since they were trained on
            the split.
        table_html: '' if no row is shown.
        comparison_errors: Why some rows cannot be compared with the reference method.
    """

    rows: list[MethodScores]
    name_to_comparable: dict[str, MethodScores]
    reference: MethodScores | None
    is_reference_excluded: bool
    shown: list[MethodScores]
    num_hidden: int
    table_html: str
    comparison_errors: list[str]


def _compute_method_table(worker: ScoringWorker, view: _ScoresView,
                          models: T.Sequence[ScoredModel], subset: AttributeSubset,
                          matching_method_names: T.AbstractSet[str]) -> _MethodTable:
    """Ranks the methods of a view over the attributes of `subset` and makes their table, with
    the comparisons with the reference method, if any.

    Args:
        matching_method_names: The methods that the method filter matches.
    """
    rows = rank_methods(models, subset.selected, view.sort_metric)
    name_to_comparable = {r.method_name: r for r in list_comparable_methods(rows)}
    reference = name_to_comparable.get(view.reference_method)
    shown, num_hidden = select_shown_methods(
        rows, matching_method_names, "" if reference is None else reference.method_name,
        hidden_training_split=None if view.shows_trained_methods else view.split)
    table_html, comparison_errors = "", []
    if shown:
        confidence = worker.settings.confidence
        metric_names = (view.metric_name,) if view.metric_name else RANKING_METRIC_NAMES
        method_to_resamples = {
            r.method_name: worker.get_method_resamples(r, subset.selected, metric_names)
            for r in shown}
        comparisons = None
        if reference is not None:
            method_to_differences, comparison_errors = compare_with_reference(
                shown, method_to_resamples, reference, confidence)
            comparisons = ReferenceComparisons(reference, method_to_differences)
        scope = ScoresScope(view.dataset, view.split, subset.to_query())
        if view.metric_name:
            table_html = make_per_attribute_table_html(
                shown, method_to_resamples, confidence, view.metric_name, subset.selected, scope,
                comparisons)
        else:
            table_html = make_method_table_html(
                shown, method_to_resamples, confidence, view.sort_metric, scope, comparisons)
    return _MethodTable(
        rows, name_to_comparable, reference,
        reference is not None and reference.method_name not in matching_method_names, shown,
        num_hidden, table_html, comparison_errors)


@dc.dataclass(frozen=True)
class _ScoresContent:
    """What the table panel of the page shows (`_compute_scores_content`).

    Attributes:
        filter_error: Why the method filter is invalid, or None.
        table: None if no model is scored on the evaluation set, no attribute is selected, or
            the method filter is invalid.
    """

    view: _ScoresView
    split_scorings: _SplitScorings
    subset: AttributeSubset
    filter_error: str | None
    table: _MethodTable | None


def _compute_scores_content(worker: ScoringWorker, view: _ScoresView,
                            split_scorings: _SplitScorings,
                            subset: AttributeSubset) -> _ScoresContent:
    filter_error, table = None, None
    if split_scorings.models and subset.selected:
        # The filter matches the shown method names.
        name_to_shown = {m.submission.model.method_name: m.submission.model.shown_method_name
                         for m in split_scorings.models}
        try:
            matching_shown_names = select_matching_names(view.method_pattern,
                                                         set(name_to_shown.values()))
        except ValueError as e:
            filter_error = str(e)
        else:
            matching_method_names = {n for n, s in name_to_shown.items()
                                     if s in matching_shown_names}
            table = _compute_method_table(worker, view, split_scorings.models, subset,
                                          matching_method_names)
    return _ScoresContent(view, split_scorings, subset, filter_error, table)


async def _create_scores(dataset_contexts: T.Mapping[str, DatasetContext], worker: ScoringWorker,
                         view: _ScoresView, attribute_query: T.Sequence[str]) -> None:
    """Creates the page. The scores are loaded and computed in threads (`update_table`), so that
    a large table does not hold up the event loop, e.g. for other users."""
    def navigate_to(**changes: str) -> None:
        # Another dataset has other attributes.
        attributes = () if "dataset" in changes else attribute_filter.subset.to_query()
        ui.navigate.to(dc.replace(view, **changes).to_path(attributes))

    async def on_view_changed(**changes: str) -> None:
        nonlocal view
        view = dc.replace(view, **changes)
        ui.navigate.history.replace(view.to_path(attribute_filter.subset.to_query()))
        await update_table()

    async def on_subset_changed(subset: AttributeSubset) -> None:
        ui.navigate.history.replace(view.to_path(subset.to_query()))
        await update_table()

    async def update_table(split_scorings: _SplitScorings | None = None) -> None:
        """Loads and computes the table of the view in threads, and shows it if no later update
        has started meanwhile.

        Args:
            split_scorings: The scorings of the split if they are loaded, e.g. at the start.
        """
        is_latest = latest_updates.start()
        shown_view = view
        if split_scorings is None:
            split_scorings = await run.io_bound(_load_split_scorings, worker, shown_view)
            if split_scorings is None or not is_latest():
                return
        subset = attribute_filter.set_options(collect_scored_attributes(split_scorings.models))
        content = await run.io_bound(_compute_scores_content, worker, shown_view, split_scorings,
                                     subset)
        if content is not None and is_latest():
            show_table.refresh(content)

    context = dataset_contexts[view.dataset]

    def show_methods(shown_view: _ScoresView, table: _MethodTable) -> None:
        with ui.element("div").classes("form-row"):
            if shown_view.evaluation_set == ie.REFERENCE_SET_NAME:
                create_native_select("Compare with",
                                     {"": "–", **{n: r.shown_method_name
                                                  for n, r in table.name_to_comparable.items()}},
                                     shown_view.reference_method,
                                     lambda v: on_view_changed(reference_method=v))
            if shown_view.shows_trained_methods or any(
                    get_method_split_use(r, shown_view.split) == "training" for r in table.rows):
                create_native_checkbox(f"Show methods trained on {shown_view.split}",
                                       shown_view.shows_trained_methods,
                                       lambda v: on_view_changed(shows_trained_methods=v))
        reference = table.reference
        if shown_view.reference_method and reference is None:
            ui.label(f"{shown_view.reference_method} cannot be compared, since it has no scores"
                     f" on this split or its models cannot be combined.").classes("error")
        elif table.is_reference_excluded:
            ui.label(f"{reference.shown_method_name} is shown as the reference, although the filter"
                     f" excludes it.").classes("muted")
        if table.num_hidden:
            ui.label(f"Number of hidden methods trained on {shown_view.split}:"
                     f" {table.num_hidden}.").classes("muted")
        if not table.shown:
            if not table.num_hidden:  # Otherwise, the label above says that the matches are hidden.
                ui.label("The filter matches no method.").classes("muted")
            return
        ui.html(table.table_html, sanitize=False).classes("w-full overflow-x-auto").on(
            "click", lambda e: on_view_changed(sort_metric=e.args),
            js_handler="(e) => { const th = e.target.closest('th[data-sort]');"
                       " if (th) emit(th.dataset.sort); }")
        for error in table.comparison_errors:
            ui.label(error).classes("error")

    @ui.refreshable
    def show_table(content: _ScoresContent | None) -> None:
        if content is None:
            ui.label("Loading the scores…").classes("muted")
            return
        shown_view, split_scorings, subset = content.view, content.split_scorings, content.subset
        title = (f"{shown_view.dataset}/{shown_view.split},"
                 f" {_EVALUATION_SET_LABELS[shown_view.evaluation_set]}")
        if split_scorings.models:
            title += (f": {shown_view.metric_name} of {subset.label}" if shown_view.metric_name
                      else f": averages over {subset.label}")
        ui.label(title).classes("section-title")
        if not split_scorings.models:
            ui.label("No models are scored on this evaluation set.").classes("muted")
            notes = {n for s in split_scorings.scorings.current.values()
                     if s.get_scores(shown_view.evaluation_set) is None for n in s.notes}
            for note in sorted(notes):
                ui.label(note).classes("muted")
        elif not subset.selected:
            ui.label("Select at least one attribute.").classes("error")
        elif content.filter_error is not None:
            ui.label(content.filter_error).classes("error")
        if content.table is not None:
            show_methods(shown_view, content.table)

        if unlisted := split_scorings.scorings.list_unscored():
            ui.label("Unscored files").classes("section-title")
            ui.html(make_unscored_files_html(unlisted, lambda s: s.shown_label), sanitize=False)
        refresh_timer.watch([lambda i=i: worker.is_scoring_pending(i)
                             for i in split_scorings.scorings.pending_ids])

    latest_updates = LatestUpdates()
    initial_scorings = await run.io_bound(_load_split_scorings, worker, view)
    if initial_scorings is None:  # The server is stopping.
        return
    with ui.element("section").classes("panel"):
        with ui.element("div").classes("form-row"):
            # Another dataset has other methods, and the other evaluation set no comparisons.
            create_native_select("Dataset", {n: n for n in dataset_contexts}, view.dataset,
                                 lambda v: navigate_to(dataset=v, split="", reference_method=""))
            create_native_select("Split", {n: n for n in context.split_names}, view.split,
                                 lambda v: navigate_to(split=v))
            if (view.evaluation_set != ie.REFERENCE_SET_NAME
                    or initial_scorings.scorings.select_scored_models(
                        ie.MODEL_COMPATIBLE_SET_NAME)):
                create_native_select(
                    "Evaluation set", _EVALUATION_SET_LABELS, view.evaluation_set,
                    lambda v: navigate_to(evaluation_set=v, reference_method=""))
            create_native_select("View", VIEW_OPTIONS, view.metric_name,
                                 lambda v: on_view_changed(metric_name=v))
            create_native_input("Method filter (regex)", view.method_pattern,
                                lambda v: on_view_changed(method_pattern=v), placeholder="all",
                                size=24)
        attribute_filter = AttributeFilter(attribute_query, on_subset_changed)
    with ui.element("section").classes("panel"):
        refresh_timer = RefreshTimer(update_table)
        show_table(None)
    await update_table(initial_scorings)
