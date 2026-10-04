"""The Analysis page: the outcomes of a model, or of two compared models, for one attribute on an
evaluation set of an analysis split (`config.DatasetConfig.analysis_splits`), with the confusion
matrix, the map and the details of a segment (see `prediction_analysis`), and the endpoint of the
segment images."""

import asyncio
import dataclasses as dc
import functools
import html
import typing as T

import irap_evaluation as ie
import numpy as np
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from nicegui import app, run, ui

from ..components.analysis_html import (
    make_class_table_html,
    make_confusion_matrix_html,
    make_segment_detail_html,
    make_segment_list_html,
)
from ..components.analysis_view import (
    AnalysisView,
    list_analysis_datasets,
    list_compared_models,
    parse_cell,
    resolve_view,
    select_attribute_options,
)
from ..components.attribute_filter import AttributeFilter, AttributeSubset
from ..components.deck_map import DeckMap, make_map_legend_html
from ..components.formatting import EVALUATION_SET_LABELS, make_model_link_html
from ..components.native_controls import create_native_grouped_select, create_native_select
from ..components.page_frame import create_page_frame, show_view_error
from ..components.routes import (
    ANALYSIS_PATH,
    ATTRIBUTE_PARAMETER,
    SEGMENT_IMAGE_PATH,
    get_segment_image_path,
)
from ..components.score_tables import format_optional_metric_value
from ..datasets import DatasetContext
from ..method_ranking import (
    DEFAULT_SORT_METRIC,
    collect_scored_attributes,
    describe_split_use,
    select_model_metrics,
    sort_scored_models,
)
from ..prediction_analysis import (
    AlignedPredictions,
    AlignedPredictionsCache,
    AttributeOutcomes,
    MapPoints,
    compute_attribute_outcomes,
    compute_map_points,
    get_segment_detail,
    select_cell_segments,
)
from ..scoring import (
    ScoredModel,
    ScoreStore,
    get_model_evaluation_set,
    select_scored_models,
)
from ..scoring_worker import ScoringWorker

#: The metric that orders the models, and that the summary shows.
_RANKING_METRIC = DEFAULT_SORT_METRIC
_MAX_LISTED_SEGMENTS = 500
#: Null policy -> how the class table counts invalid predictions.
_CLASS_TABLE_NOTES = {
    "first_class": "Precision, recall and F1 as scored: an invalid prediction counts as the first"
                   " class.",
    "exclude": "Precision, recall and F1 as scored: invalid predictions are left out."}
#: The initial width of the side panel next to the map.
_SIDE_PANEL_WIDTH_PX = 440


def _create_side_panel_without_map() -> ui.element:
    """Creates the side panel of a page without the map, e.g. of an error, as wide as next to
    the map."""
    return ui.element("div").classes("analysis-side").style(f"width: {_SIDE_PANEL_WIDTH_PX}px")


@dc.dataclass(frozen=True)
class _LoadedModel:
    """A model of the page with its predictions.

    Attributes:
        class_reports: Attribute -> its class scores, the `classes` of the stored
            `evaluation_report.to_json_dict` document.
    """

    scored: ScoredModel
    aligned: AlignedPredictions
    class_reports: T.Mapping[str, T.Mapping[str, T.Any]]

    @property
    def model_id(self) -> int:
        return self.scored.submission.model.id


def _load_model(predictions_cache: AlignedPredictionsCache, store: ScoreStore,
                scored_model: ScoredModel, evaluation_set: ie.EvaluationSet) -> _LoadedModel:
    """Reads the predictions of a model, e.g. off the event loop.

    Raises:
        OSError, ValueError: If the file cannot be read (see `AlignedPredictionsCache.get`).
    """
    submission = scored_model.submission
    report = store.get_score_report(submission.id, scored_model.scores.evaluation_set)
    return _LoadedModel(scored_model, predictions_cache.get(submission, evaluation_set),
                        {} if report is None else report["classes"])


def register_analysis_page(dataset_contexts: T.Mapping[str, DatasetContext],
                           worker: ScoringWorker, predictions_cache: AlignedPredictionsCache,
                           map_error: str | None) -> None:
    """
    Args:
        map_error: Why the map libraries are missing (`map_libraries.ensure_map_libraries`), or
            None if they are served.
    """
    @app.get(SEGMENT_IMAGE_PATH)
    def get_segment_image(dataset: str, segment_id: str) -> FileResponse:
        context = dataset_contexts.get(dataset)
        path = None if context is None else context.find_segment_image(segment_id)
        if path is None:
            raise HTTPException(status_code=404, detail=f"No image of segment {segment_id!r}.")
        return FileResponse(path)

    @ui.page(ANALYSIS_PATH, title="Analysis · iRAP evaluation")
    async def analysis_page(request: Request) -> None:
        with create_page_frame(ANALYSIS_PATH, fills_window=True):
            try:
                view = AnalysisView.from_query(request.query_params, dataset_contexts)
            except ValueError as e:
                with _create_side_panel_without_map():
                    show_view_error(str(e), ANALYSIS_PATH)
                return
            await _create_analysis(dataset_contexts, worker, predictions_cache, map_error, view,
                                   request.query_params.getlist(ATTRIBUTE_PARAMETER))


def _open_changed_view(context: DatasetContext, view: AnalysisView,
                       attribute_query: T.Sequence[str], **changes: str) -> None:
    """Opens the page of another dataset, split or evaluation set, with the default models, cell
    and segment.

    Args:
        context: The context of the dataset of `view`.
        attribute_query: The attribute subset of the page, which is kept for the same dataset.
        changes: Of the dataset, the split or the evaluation set.
    """
    changed = dc.replace(view, model_id=None, compare_id=None, cell=None, segment_id="",
                         **changes)
    if changed.dataset != view.dataset:
        ui.navigate.to(AnalysisView(dataset=changed.dataset, split="",
                                    evaluation_set=changed.evaluation_set).to_path())
        return
    # The attribute is kept if the reference set of the split has labels of it. A model-compatible
    # set depends on the model, which is not known before the page is loaded.
    if (changed.evaluation_set != ie.REFERENCE_SET_NAME
            or context.get_reference_set(changed.split).attr_to_num_labeled.get(
                view.attribute_name, 0) == 0):
        changed = dc.replace(changed, attribute_name="")
    ui.navigate.to(changed.to_path(attribute_query))


def _create_view_controls(context: DatasetContext, analysis_datasets: T.Sequence[str],
                          view: AnalysisView, open_view: T.Callable[..., None]) -> None:
    """Creates the selects of the dataset, the split and the evaluation set.

    Args:
        analysis_datasets: The options of the Dataset select.
        open_view: Opens the page of the changed view, given the changes as keyword arguments
            (see `_open_changed_view`).
    """
    with ui.element("div").classes("analysis-controls"):
        create_native_select("Dataset", {n: n for n in analysis_datasets}, view.dataset,
                             lambda v: open_view(dataset=v))
        create_native_select("Split", {s: s for s in context.config.analysis_splits}, view.split,
                             lambda v: open_view(split=v))
        create_native_select("Evaluation set", EVALUATION_SET_LABELS, view.evaluation_set,
                             lambda v: open_view(evaluation_set=v))


def _get_ranking_average(scored_model: ScoredModel,
                         attributes: T.Sequence[str]) -> float | None:
    """Returns the average of `_RANKING_METRIC` over `attributes`, None if the model lacks it."""
    metrics = select_model_metrics(scored_model, attributes)
    return None if metrics is None else metrics.averages.get(_RANKING_METRIC)


def _make_model_label(scored_model: ScoredModel, attributes: T.Sequence[str]) -> str:
    """Makes the option text of a model: its label, its score and a mark if it used the split
    (`method_ranking.describe_split_use`)."""
    submission = scored_model.submission
    average = _get_ranking_average(scored_model, attributes)
    score = ("" if average is None
             else f" · {_RANKING_METRIC} {format_optional_metric_value(average, 3)}")
    mark = describe_split_use(submission.model_info.get_split_use(submission.split),
                              submission.split)
    return f"{submission.label}{score}{f' · {mark}' if mark else ''}"


def _group_model_options(models: T.Sequence[ScoredModel],
                         attributes: T.Sequence[str]) -> dict[str, dict[str, str]]:
    """Groups the options of a grouped select by method: method name -> model id -> option text.
    The groups are in the order of their first model, so for models sorted best first, the
    method of the best model is first."""
    groups: dict[str, dict[str, str]] = {}
    for model in models:
        groups.setdefault(model.submission.model.method_name, {})[
            str(model.submission.model.id)] = _make_model_label(model, attributes)
    return groups


def _describe_attribute_scores(scored_model: ScoredModel, attribute: str) -> str:
    """Describes the scores of a model for one attribute, e.g. 'mF1 .512 · 3 invalid', with
    '(not predicted)' for a missing attribute."""
    scores = scored_model.scores
    f1 = format_optional_metric_value(scores.metrics.per_attribute["mF1"].get(attribute), 3)
    missing = " (not predicted)" if attribute in scores.missing_attributes else ""
    return f"mF1 {f1} · {scores.num_invalid[attribute]} invalid{missing}"


async def _create_analysis(dataset_contexts: T.Mapping[str, DatasetContext],
                           worker: ScoringWorker, predictions_cache: AlignedPredictionsCache,
                           map_error: str | None, view: AnalysisView,
                           attribute_query: T.Sequence[str]) -> None:
    """Loads the models of the view, resolves the view (`resolve_view`), and creates the page."""
    context = dataset_contexts[view.dataset]
    client = ui.context.client
    loading_label = ui.label("Loading predictions…").classes("muted p-3")

    def show_error(message: str, offers_default_view: bool = True) -> None:
        """Shows the error with the view controls, so that another view can be chosen.

        Args:
            offers_default_view: Whether to link the default view of the split if the URL has
                other models, attributes, cell or segment, which the error can depend on.
        """
        loading_label.delete()
        default_path = AnalysisView(dataset=view.dataset, split=view.split,
                                    evaluation_set=view.evaluation_set).to_path()
        with _create_side_panel_without_map():
            _create_view_controls(
                context, list_analysis_datasets(dataset_contexts), view,
                functools.partial(_open_changed_view, context, view, attribute_query))
            ui.label(message).classes("error")
            if offers_default_view and view.to_path(attribute_query) != default_path:
                ui.link(f"Open the default view of {view.dataset}/{view.split}", default_path)

    # The page is sent first, since reading the files takes tenths of a second (see the README).
    await client.connected()
    submissions, _, current = worker.load_split_scorings(view.dataset, view.split)
    models = select_scored_models(submissions, current, view.evaluation_set)
    if not models:
        show_error(f"No models of {view.dataset}/{view.split} are scored on the"
                   f" {EVALUATION_SET_LABELS[view.evaluation_set].lower()}.",
                   offers_default_view=False)
        return
    subset = AttributeSubset.from_query(attribute_query, collect_scored_attributes(models))
    models = sort_scored_models(models, _RANKING_METRIC, subset.selected)
    try:
        resolved = resolve_view(view, models, subset,
                                functools.partial(get_model_evaluation_set, context))
    except ValueError as e:
        show_error(str(e))
        return
    load = functools.partial(_load_model, predictions_cache, worker.store,
                             evaluation_set=resolved.evaluation_set)
    try:
        # The first map of a dataset also locates its segments (`segment_coordinates`).
        points, model_a, model_b = await asyncio.gather(
            run.io_bound(compute_map_points, context.metadata, resolved.evaluation_set),
            run.io_bound(load, resolved.model_a),
            run.io_bound(load, resolved.model_b) if resolved.model_b is not None
            else asyncio.sleep(0, result=None))
    except (OSError, ValueError) as e:  # Also irap_evaluation.PredictionFormatError.
        if not client.is_deleted:
            show_error(f"The predictions cannot be read: {e}")
        return
    if client.is_deleted:  # The user has left the page while the files were read.
        return
    loading_label.delete()
    _AnalysisPage(context, worker, predictions_cache, resolved.view, models, points, model_a,
                  model_b).create(list_analysis_datasets(dataset_contexts), map_error,
                                  attribute_query)


class _AnalysisPage:
    """The controls, matrix, map and details of the page, and their updates."""

    def __init__(self, context: DatasetContext, worker: ScoringWorker,
                 predictions_cache: AlignedPredictionsCache, view: AnalysisView,
                 models: T.Sequence[ScoredModel], points: MapPoints, model_a: _LoadedModel,
                 model_b: _LoadedModel | None):
        """
        Args:
            view: A resolved view (`resolve_view`).
            models: The scored models of the split on the evaluation set.
            points: The map points of the evaluation set (`compute_map_points`).
            model_a: The model of the confusion matrix.
            model_b: The model that A is compared with, or None.
        """
        self.context = context
        self.worker = worker
        self.predictions_cache = predictions_cache
        self.client = ui.context.client
        #: Without the models, which are `model_a` and `model_b` (`_update_url` adds them).
        self.view = dc.replace(view, model_id=None, compare_id=None)
        self.models = models
        self.model_a, self.model_b = model_a, model_b
        self.evaluation_set = model_a.aligned.evaluation_set
        self.segment_id_to_index = {s: i for i, s in enumerate(self.evaluation_set.segment_ids)}
        self.points = points
        #: Counts the changes of each model, so that a slow file read is discarded when another
        #: change of the same model started after it.
        self.num_model_changes = {"A": 0, "B": 0}
        self.outcomes: AttributeOutcomes | None = None
        #: The segments of the selected cell (`select_cell_segments`), None without a cell.
        self.cell_segments: np.ndarray | None = None

    def create(self, analysis_datasets: T.Sequence[str], map_error: str | None,
               attribute_query: T.Sequence[str]) -> None:
        """Creates the page content and shows the view.

        Args:
            analysis_datasets: The options of the Dataset select.
            map_error: See `register_analysis_page`.
            attribute_query: The attribute subset of the URL.
        """
        with ui.splitter(value=_SIDE_PANEL_WIDTH_PX, limits=(320, 900)).props("unit=px").classes(
                "w-full h-full") as outer:
            with outer.before, ui.element("div").classes("analysis-side"):
                self._create_side_panel(analysis_datasets, attribute_query)
            with outer.after:
                self._create_map_panel(map_error)
        self._update_outcomes()
        self._update_map_selected()

    def _create_side_panel(self, analysis_datasets: T.Sequence[str],
                           attribute_query: T.Sequence[str]) -> None:
        """Creates the controls, the confusion matrix and the segments of a cell."""
        _create_view_controls(
            self.context, analysis_datasets, self.view,
            lambda **changes: _open_changed_view(self.context, self.view,
                                                 self.attribute_filter.subset.to_query(),
                                                 **changes))
        self.attribute_filter = AttributeFilter(attribute_query, self._on_subset_changed)
        self.attribute_filter.set_options(collect_scored_attributes(self.models))
        with ui.element("div").classes("analysis-controls"):
            self.show_model_selects = ui.refreshable(self._show_model_selects)
            self.show_model_selects()
        self.show_attribute_picker = ui.refreshable(self._show_attribute_picker)
        self.show_attribute_picker()
        self.summary_html = ui.html("", sanitize=False)
        ui.label("Confusion matrix").classes("section-title")
        ui.label("Rows: labels. Columns: predictions of A, then invalid ones.").classes("muted")
        self.matrix_html = ui.html("", sanitize=False).on(
            "click", lambda e: self._on_cell_clicked(e.args),
            js_handler="(e) => { const cell = e.target.closest('[data-cell]');"
                       " if (cell) emit(cell.dataset.cell); }")
        self.class_table_html = ui.html("", sanitize=False)
        ui.label(_CLASS_TABLE_NOTES[self.worker.settings.null_policy]).classes("muted")
        ui.label("Segments").classes("section-title")
        self.segment_status = ui.label().classes("muted")
        self.segment_list_html = ui.html("", sanitize=False).classes("segment-list").on(
            "click", lambda e: self._select_segment(int(e.args)),
            js_handler="(e) => { const button = e.target.closest('[data-segment]');"
                       " if (button) emit(button.dataset.segment); }")

    def _create_map_panel(self, map_error: str | None) -> None:
        """Creates the map and its legend, and the details of the selected segment below them."""
        with ui.splitter(horizontal=True, reverse=True, value=280, limits=(120, 700)).props(
                "unit=px").classes("w-full h-full") as splitter:
            with splitter.before, ui.element("div").classes("map-container"):
                self.deck_map = None
                if map_error is not None:
                    ui.label(f"The map cannot be shown: {map_error}").classes("error p-3")
                else:
                    segment_ids = self.evaluation_set.segment_ids
                    self.deck_map = DeckMap([segment_ids[i] for i in self.points.segment_indices],
                                            self.points.coordinates, self._on_point_picked)
                self.legend_html = ui.html("", sanitize=False)
            with splitter.after:
                self.detail_html = ui.html("", sanitize=False).classes("w-full h-full")

    # Controls #####################################################################################

    def _show_model_selects(self) -> None:
        subset = self.attribute_filter.subset
        attributes = subset.selected
        models = sort_scored_models(self.models, _RANKING_METRIC, attributes)
        with ui.element("div").classes("wide"):
            create_native_grouped_select(
                "Model A", _group_model_options(models, attributes), str(self.model_a.model_id),
                self._on_model_changed)
        with ui.element("div").classes("wide"):
            create_native_grouped_select(
                "Compare with (B)",
                _group_model_options(list_compared_models(models, self.model_a.scored),
                                     attributes),
                "" if self.model_b is None else str(self.model_b.model_id),
                self._on_compared_model_changed, ungrouped={"": "–"})
        links = ", ".join(
            f"{role}: {make_model_link_html(m.model_id, s.label, s.split)}"
            for role, m in (("A", self.model_a), ("B", self.model_b)) if m is not None
            for s in [m.scored.submission])
        ui.html(f'<span class="muted">{links}. Grouped by method, best {_RANKING_METRIC} over'
                f" {html.escape(subset.label)} first.</span>", sanitize=False).classes("wide")

    def _show_attribute_picker(self) -> None:
        attributes = select_attribute_options(self.attribute_filter.subset, self.model_a.scored)
        if not attributes:
            ui.label("Select at least one attribute that model A is scored on.").classes("error")
            return
        create_native_select(
            "Attribute",
            {a: f"{a} · {_describe_attribute_scores(self.model_a.scored, a)}"
             for a in attributes},
            self.view.attribute_name, self._on_attribute_changed)
        unlabeled = [a for a, n in self.evaluation_set.attr_to_num_labeled.items() if n == 0]
        if unlabeled:
            names = html.escape(", ".join(unlabeled))
            ui.html(f'<span class="muted" title="{names}">Number of attributes without labels'
                    f" in the set (not scored): {len(unlabeled)} (hover to list).</span>",
                    sanitize=False)

    # Updates ######################################################################################

    def _update_url(self) -> None:
        view = dc.replace(self.view, model_id=self.model_a.model_id,
                          compare_id=None if self.model_b is None else self.model_b.model_id)
        ui.navigate.history.replace(view.to_path(self.attribute_filter.subset.to_query()))

    def _update_outcomes(self, is_model_a_changed: bool = True) -> None:
        """Shows the outcomes of the models for the attribute: everything but the controls and
        the selected segment on the map.

        Args:
            is_model_a_changed: Whether the attribute or model A has changed, and not only model
                B, which the matrix, the class table and the cell do not depend on.
        """
        attribute = self.view.attribute_name
        self.outcomes = compute_attribute_outcomes(
            self.model_a.aligned, attribute,
            None if self.model_b is None else self.model_b.aligned)
        self._update_summary()
        point_outcomes = self.outcomes.segment_outcomes[self.points.segment_indices]
        counts = np.bincount(point_outcomes, minlength=len(self.outcomes.outcome_names))
        num_unlocated = self.points.num_unlocated
        self.legend_html.set_content(make_map_legend_html(
            self.outcomes.outcome_names, counts.tolist(),
            note=f"Number of segments without a location (not shown): {num_unlocated}."
            if num_unlocated else ""))
        if self.deck_map is not None:
            self.deck_map.set_outcomes(point_outcomes, self.outcomes.outcome_names)
        if is_model_a_changed:
            class_report = self.model_a.class_reports.get(attribute)
            self.class_table_html.set_content(
                '<div class="error">The class scores are not stored.</div>'
                if class_report is None
                else make_class_table_html(class_report, self.outcomes.confusion_matrix))
            self._update_cell()
        self._update_detail()

    def _update_summary(self) -> None:
        """Shows the averages of the models over the subset, and their scores of the attribute."""
        subset = self.attribute_filter.subset
        attribute = html.escape(self.view.attribute_name)
        parts = []
        for name, loaded_model in (("A", self.model_a), ("B", self.model_b)):
            if loaded_model is None:
                continue
            average = _get_ranking_average(loaded_model.scored, subset.selected)
            parts.append(
                f"{name}: {_RANKING_METRIC} {format_optional_metric_value(average, 3)} over"
                f" {html.escape(subset.label)} – {attribute}:"
                f" {_describe_attribute_scores(loaded_model.scored, self.view.attribute_name)}")
        self.summary_html.set_content(f'<div class="muted">{"<br>".join(parts)}</div>')

    def _get_selected_segment(self) -> int | None:
        return self.segment_id_to_index.get(self.view.segment_id)

    def _update_cell(self) -> None:
        """Shows the selected cell: in the matrix, its segments, and their points on the map."""
        attribute = self.view.attribute_name
        vocabulary = self.evaluation_set.vocabulary
        self.matrix_html.set_content(make_confusion_matrix_html(
            self.outcomes.confusion_matrix, vocabulary.get_values(attribute),
            vocabulary.get_irap_codes(attribute), self.view.cell))
        segment_to_point = self.points.segment_to_point
        if self.view.cell is None:
            self.cell_segments = None
            self.segment_status.set_text("Click a cell of the confusion matrix to list its"
                                         " segments.")
        else:
            self.cell_segments = select_cell_segments(self.model_a.aligned, attribute,
                                                      *self.view.cell)
            num_unlocated = int((segment_to_point[self.cell_segments] < 0).sum())
            text = f"Number of segments: {len(self.cell_segments)}"
            if num_unlocated:
                text += f" ({num_unlocated} not on the map)"
            order = []
            if len(self.cell_segments) > _MAX_LISTED_SEGMENTS:
                order.append(f"the first {_MAX_LISTED_SEGMENTS} listed")
            if self.model_a.aligned.predictions.output_kind == "probs":
                order.append("most confident first")
            if order:
                text += f" – {', '.join(order)}"
            self.segment_status.set_text(text + ".")
        self._update_segment_list()
        if self.deck_map is not None:
            points = (None if self.cell_segments is None
                      else segment_to_point[self.cell_segments])
            self.deck_map.set_highlighted(None if points is None else points[points >= 0])
        self._update_url()

    def _update_segment(self) -> None:
        """Shows the selected segment: in the segment list, on the map, and its details."""
        self._update_segment_list()
        self._update_map_selected()
        self._update_detail()
        self._update_url()

    def _update_map_selected(self) -> None:
        if self.deck_map is not None:
            selected = self._get_selected_segment()
            point = -1 if selected is None else int(self.points.segment_to_point[selected])
            self.deck_map.set_selected(None if point < 0 else point)

    def _update_segment_list(self) -> None:
        self.segment_list_html.set_content(
            "" if self.cell_segments is None else make_segment_list_html(
                self.cell_segments[:_MAX_LISTED_SEGMENTS].tolist(),
                self.evaluation_set.segment_ids, self._get_selected_segment()))

    def _update_detail(self) -> None:
        selected = self._get_selected_segment()
        if selected is None:
            self.detail_html.set_content(
                '<div class="muted p-2">Select a segment on the map or in the list.</div>')
            return
        aligned_seq = [m.aligned for m in (self.model_a, self.model_b) if m is not None]
        detail = get_segment_detail(self.context.metadata, aligned_seq, selected,
                                    self.view.attribute_name)
        dataset = self.context.name
        self.detail_html.set_content(make_segment_detail_html(
            detail, None if self.context.config.images_dir is None
            else lambda s: get_segment_image_path(dataset, s)))

    # Event handlers ###############################################################################

    def _on_cell_clicked(self, text: str) -> None:
        cell = parse_cell(text)
        self.view = dc.replace(self.view, cell=None if cell == self.view.cell else cell)
        self._update_cell()

    def _select_segment(self, segment_index: int) -> None:
        self.view = dc.replace(self.view,
                               segment_id=self.evaluation_set.segment_ids[segment_index])
        self._update_segment()

    def _on_point_picked(self, point: int) -> None:
        self._select_segment(int(self.points.segment_indices[point]))

    def _on_attribute_changed(self, attribute: str) -> None:
        self.view = dc.replace(self.view, attribute_name=attribute, cell=None)
        self._update_outcomes()

    def _on_subset_changed(self, subset: AttributeSubset) -> None:
        self.show_model_selects.refresh()
        self.show_attribute_picker.refresh()
        attributes = select_attribute_options(subset, self.model_a.scored)
        if attributes and self.view.attribute_name not in attributes:
            self.view = dc.replace(self.view, attribute_name=attributes[0], cell=None)
            self._update_outcomes()
        else:  # Only the averages over the subset change.
            self._update_summary()
            self._update_url()

    def _find_model(self, model_id_text: str) -> ScoredModel:
        return next(m for m in self.models if str(m.submission.model.id) == model_id_text)

    async def _load_latest_model(self, role: T.Literal["A", "B"],
                                 scored_model: ScoredModel) -> _LoadedModel | None:
        """Reads the predictions of a model off the event loop. Returns None if they cannot be
        read (the user is notified), the user has left the page, or another change of the same
        role started meanwhile."""
        self.num_model_changes[role] += 1
        change_number = self.num_model_changes[role]
        try:
            loaded_model = await run.io_bound(_load_model, self.predictions_cache,
                                              self.worker.store, scored_model,
                                              self.evaluation_set)
        except (OSError, ValueError) as e:  # Also irap_evaluation.PredictionFormatError.
            if not self.client.is_deleted:
                ui.notify(f"The predictions of {scored_model.submission.label} cannot be"
                          f" read: {e}", type="negative")
            return None
        if self.client.is_deleted or change_number != self.num_model_changes[role]:
            return None
        return loaded_model

    async def _on_model_changed(self, value: str) -> None:
        scored_model = self._find_model(value)
        if (scored_model.scores.evaluation_set_fingerprint
                != self.model_a.scored.scores.evaluation_set_fingerprint):
            # Another model-compatible set has other segments, so the page is created again.
            view = dc.replace(self.view, model_id=scored_model.submission.model.id, cell=None,
                              segment_id="")
            ui.navigate.to(view.to_path(self.attribute_filter.subset.to_query()))
            return
        if (model_a := await self._load_latest_model("A", scored_model)) is None:
            return
        self.model_a = model_a
        if self.model_b is not None and self.model_b.model_id == model_a.model_id:
            self.model_b = None
        self.view = dc.replace(self.view, cell=None)
        self.show_model_selects.refresh()
        self.show_attribute_picker.refresh()
        self._update_outcomes()

    async def _on_compared_model_changed(self, value: str) -> None:
        model_b = None
        if value:
            scored_model = self._find_model(value)
            if (model_b := await self._load_latest_model("B", scored_model)) is None:
                return
            if model_b.model_id not in {m.submission.model.id for m in list_compared_models(
                    self.models, self.model_a.scored)}:
                # Model A has changed to B while B was read.
                self.show_model_selects.refresh()
                return
        else:
            self.num_model_changes["B"] += 1  # Discards a pending read of B.
        self.model_b = model_b
        self.show_model_selects.refresh()
        self._update_outcomes(is_model_a_changed=False)
        self._update_url()
