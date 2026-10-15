"""HTML of the score tables of the Scores, Method and Model pages.

A value cell shows the value and, below it, its bootstrap interval, computed from the resampled
values of its row (`scoring_worker.ScoringWorker.get_method_resamples`). Its background is shaded by
how good the value is among the values that it is compared with, and it is bold if it is not
significantly worse than the best (`method_ranking.compute_highlights`). With a reference method,
the cells of the other methods also show their difference from it (`ReferenceComparisons`). The
notes below the tables are short, with links to the Methodology page (`methodology`).
"""

import dataclasses as dc
import html
import math
import typing as T

import irap_evaluation as ie
import numpy as np
from irap_evaluation.metrics import IRAP_ATTRIBUTE_METRIC_NAMES

from ..method_comparison import get_better_method
from ..method_ranking import (
    RANKING_METRIC_NAMES,
    CellHighlight,
    compute_highlights,
    describe_split_use,
    get_method_split_use,
)
from ..methodology import COMPARISONS_SECTION, EVALUATION_SETS_SECTION, INTERVALS_SECTION
from ..scoring import MethodScores, are_scored_on_same_set
from .formatting import (
    format_context_offsets,
    make_link_html,
    make_method_link_html,
    make_model_link_html,
    make_table_html,
    make_table_html_from_header,
    make_title_attribute,
    make_truncated_name_html,
)
from .routes import get_methodology_path

V = T.TypeVar("V")

#: Gets the value of a cell from the values of a row, e.g. its resampled values, None if they
#: lack it.
ValueSelector = T.Callable[[ie.MetricValues[V]], V | None]

#: The resampled values of a row (`scoring_worker.ScoringWorker.get_method_resamples`), or None.
RowResamples = ie.MetricValues[np.ndarray] | None

#: The mark of a method with a single model (`_is_single_model`), which `_SINGLE_MODEL_NOTE`
#: explains.
_SINGLE_MODEL_MARK = "single model"
_SINGLE_MODEL_NOTE = (f'"{_SINGLE_MODEL_MARK}": its intervals do not cover the variation between'
                      f" seeds.")

#: The metrics of `make_model_scores_html` and `make_method_scores_html`, e.g. for
#: `ScoringWorker.get_method_resamples`.
MODEL_SCORES_METRIC_NAMES = (*RANKING_METRIC_NAMES, *IRAP_ATTRIBUTE_METRIC_NAMES)


@dc.dataclass(frozen=True)
class ScoresScope:
    """What the scores of a table are of: a dataset split and an attribute subset, which the
    links of the table keep.

    Attributes:
        attributes_query: The attribute subset (`AttributeSubset.to_query`).
    """

    dataset: str
    split: str
    attributes_query: tuple[str, ...]

    def make_method_link_html(self, method_name: str, text: str) -> str:
        return make_method_link_html(self.dataset, method_name, text, self.split,
                                     self.attributes_query)

    def make_model_link_html(self, model_id: int, text: str) -> str:
        return make_model_link_html(model_id, text, self.split, self.attributes_query)


def select_average_value(metric_name: str) -> ValueSelector:
    return lambda values: values.averages.get(metric_name)


def select_attribute_value(metric_name: str, attribute: str) -> ValueSelector:
    return lambda values: values.per_attribute.get(metric_name, {}).get(attribute)


def _select_row_value(values: ie.MetricValues[V] | None, select_value: ValueSelector) -> V | None:
    return None if values is None else select_value(values)


def _compute_cell_interval(resamples: np.ndarray | None,
                           confidence: float) -> ie.BootstrapInterval | None:
    return None if resamples is None else ie.compute_percentile_interval(resamples, confidence)


def _compute_row_interval(resamples: RowResamples, select_value: ValueSelector,
                          confidence: float) -> ie.BootstrapInterval | None:
    return _compute_cell_interval(_select_row_value(resamples, select_value), confidence)


def _make_methodology_link_html(section_id: str, text: str) -> str:
    """Makes a link to a section of the Methodology page with an escaped text."""
    return make_link_html(get_methodology_path(section_id), text)


def _make_note_lines_html(*lines_html: str) -> str:
    return "".join(f'<div class="muted">{line}</div>' for line in lines_html)


def _make_shading_note(axis: T.Literal["column", "row"]) -> str:
    """Explains the shading (`CellHighlight.strength`).

    Args:
        axis: Along which the values are compared with each other.
    """
    return f"Darker: better in its {axis}."


def _make_interval_link_html(confidence: float) -> str:
    return _make_methodology_link_html(INTERVALS_SECTION, f"{confidence:.0%} confidence interval")


def make_model_scores_note_html(confidence: float) -> str:
    """Explains the intervals and the shading of `make_model_scores_html`."""
    return _make_note_lines_html(
        f"Below each value: its {_make_interval_link_html(confidence)}, which does not cover the"
        f" variation between seeds. {_make_shading_note('column')}")


def make_method_scores_note_html(confidence: float, num_models: int) -> str:
    """Explains the intervals and the shading of `make_method_scores_html`.

    Args:
        num_models: The number of models of the mean.
    """
    if num_models == 1:
        return make_model_scores_note_html(confidence)
    return _make_note_lines_html(
        f"Below each value: its {_make_interval_link_html(confidence)}. In the rows of the seeds,"
        f" it does not cover the variation between seeds. {_make_shading_note('column')}")


def _is_single_model(row: MethodScores) -> bool:
    """Returns whether the intervals of the row cover only one model (`_SINGLE_MODEL_NOTE`)."""
    return row.num_models == 1 and row.error is None


def format_metric_value(value: float, num_decimals: int) -> str:
    """Formats a value without a leading zero, e.g. '.123' or '-.123', or as 'NaN', '∞' or
    '-∞'."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    text = f"{value:.{num_decimals}f}"
    sign, unsigned = ("-", text[1:]) if text.startswith("-") else ("", text)
    return sign + unsigned[1:] if unsigned.startswith("0.") else text


def format_optional_metric_value(value: float | None, num_decimals: int) -> str:
    """Formats a value like `format_metric_value`, or None as '–', e.g. a metric that a model
    lacks."""
    return "–" if value is None else format_metric_value(value, num_decimals)


def format_signed_metric_value(value: float, num_decimals: int) -> str:
    """Formats a value like `format_metric_value`, with '+' before a positive value, e.g. a
    difference."""
    return ("+" if value > 0 else "") + format_metric_value(value, num_decimals)


def _get_interval_text(interval: ie.BootstrapInterval | None,
                       is_signed: bool = False) -> tuple[str, str]:
    """Returns the text of an interval, '–' for None, and a title that explains it ('' for
    none).

    Args:
        is_signed: Whether the bounds have a sign, e.g. those of a difference.
    """
    if interval is None:
        return "–", ""
    format_bound = format_signed_metric_value if is_signed else format_metric_value
    text = (f"{format_bound(interval.low, num_decimals=3)}–"
            f"{format_bound(interval.high, num_decimals=3)}")
    if not interval.is_partially_undefined:
        return text, ""
    return text + "*", (f"Number of resamples left out, since the value is undefined:"
                        f" {interval.num_undefined}.")


def _make_interval_html(interval: ie.BootstrapInterval | None) -> str:
    text, title = _get_interval_text(interval)
    return f'<div class="interval"{make_title_attribute(title)}>{text}</div>'


def make_value_cell_html(value: float | None, interval: ie.BootstrapInterval | None,
                         note: str = "", highlight: CellHighlight | None = None,
                         difference_html: str = "") -> str:
    """Makes a cell with the value and its interval, or '–' if the value is None.

    Args:
        note: Shown instead of the interval, e.g. 'not predicted'.
        difference_html: Shown below the interval, e.g. the difference from a reference method
            (`_make_difference_html`).
    """
    if value is None:
        return '<td class="number">–</td>'
    below = (f'<div class="interval">{html.escape(note)}</div>' if note
             else _make_interval_html(interval)) + difference_html
    text = format_metric_value(value, num_decimals=4)
    if highlight is not None and highlight.is_not_significantly_worse:
        text = f"<b>{text}</b>"
    # The shade is `--strength` in `styles.css`.
    style = (f' style="--strength: {highlight.strength:.2f}"'
             if highlight is not None and highlight.strength else "")
    return f'<td class="number"{style}>{text}{below}</td>'


# Comparisons with a reference method ##############################################################

@dc.dataclass(frozen=True)
class ReferenceComparisons:
    """The comparisons of the methods of a table with a reference method
    (`method_comparison.compare_with_reference`).

    Attributes:
        method_to_differences: Method name -> its differences from the reference, for the
            compared methods.
    """

    reference: MethodScores
    method_to_differences: T.Mapping[str, ie.MetricValues[ie.MetricDifference]]


def _make_comparison_note(comparisons: ReferenceComparisons, confidence: float) -> str:
    """Explains the differences from the reference method."""
    return (f"{_make_methodology_link_html(COMPARISONS_SECTION, 'Δ')}: the difference from"
            f" {html.escape(comparisons.reference.shown_method_name)}, with its {confidence:.0%}"
            f" confidence interval. better/worse: the interval excludes 0.")


def _make_difference_html(difference: ie.MetricDifference, metric_name: str) -> str:
    """Makes the difference of a value from that of the reference method, with its interval, and
    whether the value is clearly better or worse (`method_comparison.get_better_method`)."""
    interval_text, title = _get_interval_text(difference.interval, is_signed=True)
    better = get_better_method(difference.interval, metric_name)
    verdict = {"A": " <b>better</b>", "B": " <b>worse</b>", None: ""}[better]
    return (f'<div class="interval"{make_title_attribute(title)}>'
            f"Δ {format_signed_metric_value(difference.difference, 4)} ({interval_text})"
            f"{verdict}</div>")


def _make_method_label_html(row: MethodScores, comparisons: ReferenceComparisons | None,
                            scope: ScoresScope) -> str:
    """Makes the shown method name as a link to the Method page, cut off with a tooltip if it is
    long (`truncated-name` in `styles.css`), and below it, whether it is the reference method,
    used the split of the scores (`method_ranking.describe_split_use`) or has a single model."""
    marks = []
    if comparisons is not None and row.method_name == comparisons.reference.method_name:
        marks.append("reference")
    if split_use := describe_split_use(get_method_split_use(row, scope.split), scope.split):
        marks.append(split_use)
    if _is_single_model(row):
        marks.append(_SINGLE_MODEL_MARK)
    marks_html = (f'<div class="muted">({html.escape(", ".join(marks))})</div>' if marks
                  else "")
    link_html = scope.make_method_link_html(row.method_name, row.shown_method_name)
    return make_truncated_name_html(row.shown_method_name, link_html) + marks_html


# Tables ###########################################################################################

def _make_compared_cells_html(rows: T.Sequence[MethodScores],
                              row_resamples: T.Sequence[RowResamples],
                              select_value: ValueSelector, metric_name: str, confidence: float,
                              is_tested: bool, comparisons: ReferenceComparisons | None,
                              notes: T.Sequence[str] | None = None) -> list[str]:
    """Makes the cells of the values of rows that are compared with each other, with their
    intervals and differences from the reference method.

    Args:
        row_resamples: The resampled values of each row.
        select_value: Gets the value of the cells from the metrics, resampled values and
            differences of a row.
        metric_name: The metric of the value.
        is_tested: Whether the values are tested against the best (`compute_highlights`), which
            needs them on the same resamples, e.g. of methods on the same evaluation set.
        notes: The note of each cell (`make_value_cell_html`).
    """
    values = [_select_row_value(r.metrics, select_value) for r in rows]
    resamples = [_select_row_value(r, select_value) for r in row_resamples]
    highlights = compute_highlights(values, resamples if is_tested else None,
                                    ie.is_lower_better(metric_name))

    def get_difference_html(row: MethodScores) -> str:
        if comparisons is None:
            return ""
        difference = _select_row_value(comparisons.method_to_differences.get(row.method_name),
                                       select_value)
        return "" if difference is None else _make_difference_html(difference, metric_name)

    return [make_value_cell_html(v, _compute_cell_interval(r, confidence), note=n, highlight=h,
                                 difference_html=get_difference_html(row))
            for row, v, r, n, h in zip(rows, values, resamples, notes or [""] * len(rows),
                                       highlights, strict=True)]


def _make_seed_links_html(row: MethodScores, scope: ScoresScope) -> str:
    """Makes a link to each model of a row, labeled by its seed."""
    return ", ".join(scope.make_model_link_html(m.submission.model.id,
                                                m.submission.model.seed_label)
                     for m in row.models)


def _make_model_links_html(row: MethodScores, scope: ScoresScope) -> str:
    """Makes the number of models, with a link to each, labeled by seed."""
    return f'{row.num_models} <span class="muted">({_make_seed_links_html(row, scope)})</span>'


def _make_row_cells_html(row: MethodScores, metric_cells: T.Iterable[str],
                         num_metrics: int) -> str:
    """Makes the metric cells of a row, or one cell with its error across them."""
    if row.metrics is None:
        return f'<td colspan="{num_metrics}" class="error">{html.escape(row.error or "")}</td>'
    return "".join(metric_cells)


def _are_tested(rows: T.Iterable[MethodScores]) -> bool:
    """Returns whether the values of the rows are tested against the best
    (`compute_highlights`), which needs the rows with values on the same evaluation set
    (`scoring.are_scored_on_same_set`)."""
    return are_scored_on_same_set(m for r in rows if r.metrics is not None for m in r.models)


def _make_method_notes_html(rows: T.Sequence[MethodScores], confidence: float,
                            axis: T.Literal["column", "row"], is_tested: bool,
                            comparisons: ReferenceComparisons | None) -> str:
    """Explains a table of methods: their values and intervals, the highlights, which are tested
    against the best if `is_tested` (`_are_tested`), the single models, and the differences from
    the reference method.

    Args:
        axis: Along which the values are compared with each other.
    """
    if is_tested:
        bold = (f"{_make_methodology_link_html(COMPARISONS_SECTION, 'Bold')}: not significantly"
                f" worse than the best.")
    else:
        segments = _make_methodology_link_html(EVALUATION_SETS_SECTION, "different segments")
        bold = f"No bold, since the methods are scored on {segments}."
    lines = [(f"Values: means over the seeds of each method. Below each: its"
              f" {_make_interval_link_html(confidence)}. {_make_shading_note(axis)} {bold}")]
    if any(_is_single_model(r) for r in rows):
        lines.append(_SINGLE_MODEL_NOTE)
    if comparisons is not None:
        lines.append(_make_comparison_note(comparisons, confidence))
    return _make_note_lines_html(*lines)


def _make_notes_html(row: MethodScores) -> str:
    notes = []
    if row.missing_attributes:
        title_attribute = make_title_attribute(", ".join(row.missing_attributes))
        notes.append(f"<span{title_attribute}>{len(row.missing_attributes)} not predicted</span>")
    if row.models[0].scores.evaluation_set == ie.MODEL_COMPATIBLE_SET_NAME:
        notes.append(
            f"context offsets {format_context_offsets(row.models[0].scores.context_offsets)}")
    return "<br>".join(notes)


def _make_sortable_metric_header_html(metric_name: str, is_sorted: bool) -> str:
    """Makes the header of a metric column with a `data-sort` attribute (`make_method_table_html`)
    and the description of the metric as its tooltip."""
    title = f"{ie.describe_metric(metric_name)}\nClick to sort by it, best first."
    return (f'<th class="number sortable{" sorted" if is_sorted else ""}"'
            f' data-sort="{metric_name}"{make_title_attribute(title)}>'
            f'{metric_name}{" ▾" if is_sorted else ""}</th>')


def make_method_table_html(rows: T.Sequence[MethodScores],
                           method_to_resamples: T.Mapping[str, RowResamples], confidence: float,
                           sort_metric: str, scope: ScoresScope,
                           comparisons: ReferenceComparisons | None = None) -> str:
    """Makes a table of method rows × the attribute averages of `RANKING_METRIC_NAMES`.

    The metric headers have a `data-sort` attribute with the metric name, for a click handler.

    Args:
        method_to_resamples: Method name -> its resampled values of `RANKING_METRIC_NAMES`, or
            None.
        confidence: The confidence of the intervals.
        scope: For the links to the methods and models, and the marks of the methods that used
            the split.
        comparisons: The comparisons with a reference method, or None.
    """
    metric_headers = "".join(_make_sortable_metric_header_html(n, n == sort_metric)
                             for n in RANKING_METRIC_NAMES)
    header = f"<th>Method</th><th>Models</th>{metric_headers}<th>Notes</th>"
    row_resamples = [method_to_resamples.get(r.method_name) for r in rows]
    is_tested = _are_tested(rows)
    metric_columns = [
        _make_compared_cells_html(rows, row_resamples, select_average_value(n), n, confidence,
                                  is_tested=is_tested, comparisons=comparisons)
        for n in RANKING_METRIC_NAMES]
    html_rows = []
    for row, metric_cells in zip(rows, zip(*metric_columns)):
        cells = _make_row_cells_html(row, metric_cells, len(RANKING_METRIC_NAMES))
        html_rows.append(f"<tr><td>{_make_method_label_html(row, comparisons, scope)}</td>"
                         f"<td>{_make_model_links_html(row, scope)}</td>{cells}"
                         f"<td>{_make_notes_html(row)}</td></tr>")
    return (make_table_html_from_header(header, html_rows)
            + _make_method_notes_html(rows, confidence, "column", is_tested, comparisons))


def make_per_attribute_table_html(rows: T.Sequence[MethodScores],
                                  method_to_resamples: T.Mapping[str, RowResamples],
                                  confidence: float, metric_name: str,
                                  attributes: T.Sequence[str], scope: ScoresScope,
                                  comparisons: ReferenceComparisons | None = None) -> str:
    """Makes a table of attribute rows × method columns of one per-attribute metric, for the rows
    without an error, in their order.

    Args:
        method_to_resamples: Method name -> its resampled values of `metric_name`, or None.
        scope: For the links to the methods, and the marks of the methods that used the split.
        comparisons: The comparisons with a reference method, or None.
    """
    rows = [r for r in rows if r.metrics is not None]
    row_resamples = [method_to_resamples.get(r.method_name) for r in rows]
    is_tested = _are_tested(rows)
    html_rows = []
    for attribute in attributes:
        cells = _make_compared_cells_html(
            rows, row_resamples, select_attribute_value(metric_name, attribute), metric_name,
            confidence, is_tested=is_tested, comparisons=comparisons,
            notes=["not predicted" if attribute in r.missing_attributes else "" for r in rows])
        html_rows.append(f"<tr><td>{html.escape(attribute)}</td>{''.join(cells)}</tr>")
    # Vertical, so that the method columns are as narrow as their values.
    method_headers = "".join(
        f'<th class="number vertical-header"><div>{_make_method_label_html(r, comparisons, scope)}'
        f"</div></th>" for r in rows)
    metric_note = f'<div class="muted">{html.escape(ie.describe_metric(metric_name))}</div>'
    return (make_table_html_from_header(f"<th>Attribute</th>{method_headers}", html_rows)
            + metric_note
            + _make_method_notes_html(rows, confidence, "row", is_tested, comparisons))


def _make_row_value_cell_html(row: MethodScores, resamples: RowResamples, average_name: str,
                              confidence: float) -> str:
    """Makes the cell of an attribute average of a row, with its interval, without shading."""
    select_value = select_average_value(average_name)
    return make_value_cell_html(_select_row_value(row.metrics, select_value),
                                _compute_row_interval(resamples, select_value, confidence))


def _format_num_invalid(scores: MethodScores, attribute: str) -> str:
    """Formats the number of invalid predictions of an attribute, the mean over the models if
    there are several."""
    counts = [m.scores.num_invalid[attribute] for m in scores.models]
    return str(counts[0]) if len(counts) == 1 else f"{sum(counts) / len(counts):.1f}"


def _make_attribute_scores_table_html(scores: MethodScores, resamples: RowResamples,
                                     confidence: float) -> str:
    """Makes a table of the iRAP metrics and invalid predictions of each attribute of a subset, of
    one model or the mean over the models of a method. The attributes are compared with each other
    in each metric column, without tests against the best (bold), which are meant for comparing
    methods.

    Args:
        scores: `scoring.compute_method_scores`, without an error.
        resamples: Its resampled values of `IRAP_ATTRIBUTE_METRIC_NAMES`, or None.
        confidence: The confidence of the intervals.

    Raises:
        ValueError: If `scores` has an error.
    """
    if (metrics := scores.metrics) is None:
        raise ValueError(f"The scores of {scores.method_name} have an error: {scores.error}")
    attributes = list(metrics.per_attribute[IRAP_ATTRIBUTE_METRIC_NAMES[0]])
    metric_columns = []
    for n in IRAP_ATTRIBUTE_METRIC_NAMES:
        values = [metrics.per_attribute[n].get(a) for a in attributes]
        highlights = compute_highlights(values, None, ie.is_lower_better(n))
        intervals = [_compute_row_interval(resamples, select_attribute_value(n, a), confidence)
                     for a in attributes]
        metric_columns.append([make_value_cell_html(v, i, highlight=h)
                               for v, i, h in zip(values, intervals, highlights)])
    attribute_rows = []
    for attribute, metric_cells in zip(attributes, zip(*metric_columns)):
        is_missing = attribute in scores.missing_attributes
        num_invalid = (f"{_format_num_invalid(scores, attribute)}"
                       f"{' (not predicted)' if is_missing else ''}")
        attribute_rows.append(f"<tr><td>{html.escape(attribute)}</td>{''.join(metric_cells)}"
                              f'<td class="number">{num_invalid}</td></tr>')
    invalid_header = ("Invalid predictions" if scores.num_models == 1
                      else "Invalid predictions (mean)")
    number_headers = [*IRAP_ATTRIBUTE_METRIC_NAMES, invalid_header]
    return make_table_html(
        ["Attribute", *number_headers], attribute_rows, number_headers,
        header_titles={n: ie.describe_metric(n) for n in IRAP_ATTRIBUTE_METRIC_NAMES})


def make_model_scores_html(scores: MethodScores, resamples: RowResamples,
                           confidence: float) -> str:
    """Makes the tables of the scores of one model over a subset of the attributes: the attribute
    averages, and the iRAP metrics and invalid predictions of each attribute
    (`_make_attribute_scores_table_html`). `make_model_scores_note_html` explains them.

    Args:
        scores: `scoring.compute_method_scores` of the model alone.
        resamples: Its resampled values of `MODEL_SCORES_METRIC_NAMES`, or None.
        confidence: The confidence of the intervals.
    """
    if scores.metrics is None:
        return f'<div class="error">{html.escape(scores.error or "")}</div>'
    average_rows = [f"<tr><td{make_title_attribute(ie.describe_metric(n))}>{n}</td>"
                    f"{_make_row_value_cell_html(scores, resamples, n, confidence)}</tr>"
                    for n in scores.metrics.averages]
    average_table = make_table_html(["Average", "Value"], average_rows, number_headers=["Value"])
    attribute_table = _make_attribute_scores_table_html(scores, resamples, confidence)
    return f'<div class="score-tables">{average_table}{attribute_table}</div>'


def make_method_scores_html(method_scores: MethodScores, method_resamples: RowResamples,
                            seed_scores: T.Sequence[MethodScores],
                            seed_resamples: T.Sequence[RowResamples], confidence: float,
                            scope: ScoresScope) -> str:
    """Makes the tables of the scores of a method over a subset of the attributes: the attribute
    averages of the mean over its models and of each model alone (`_make_seed_table_html`), and
    the iRAP metrics and invalid predictions of each attribute of the mean
    (`_make_attribute_scores_table_html`), unless the mean has an error.
    `make_method_scores_note_html` explains them.

    Args:
        method_scores: `scoring.compute_method_scores` of the models of the method.
        method_resamples: Its resampled values of `MODEL_SCORES_METRIC_NAMES`, or None.
        seed_scores: `scoring.compute_method_scores` of each model alone, in the order of the
            rows.
        seed_resamples: The resampled values of each of `seed_scores` of `RANKING_METRIC_NAMES`,
            or None.
        confidence: The confidence of the intervals.
        scope: For the links to the models.
    """
    seed_table = _make_seed_table_html(method_scores, method_resamples, seed_scores,
                                       seed_resamples, confidence, scope)
    tables_html = f"<div>{seed_table}</div>"
    if method_scores.metrics is not None:
        attribute_table = _make_attribute_scores_table_html(method_scores, method_resamples,
                                                            confidence)
        tables_html += f'<div><div class="muted">Mean over the seeds</div>{attribute_table}</div>'
    return f'<div class="score-tables">{tables_html}</div>'


def _make_seed_table_html(method_scores: MethodScores, method_resamples: RowResamples,
                          seed_scores: T.Sequence[MethodScores],
                          seed_resamples: T.Sequence[RowResamples], confidence: float,
                          scope: ScoresScope) -> str:
    """Makes a table of the mean over the models of a method and of each model alone (labeled by
    seed, with a link) × the attribute averages of `RANKING_METRIC_NAMES`. The models are
    compared with each other in each column, without tests against the best (bold), which are
    meant for comparing methods. The mean is not shaded.

    Args: See `make_method_scores_html`.
    """
    mean_cells = [_make_row_value_cell_html(method_scores, method_resamples, n, confidence)
                  for n in RANKING_METRIC_NAMES]
    seed_columns = [
        _make_compared_cells_html(seed_scores, seed_resamples, select_average_value(n), n,
                                  confidence, is_tested=False, comparisons=None)
        for n in RANKING_METRIC_NAMES]
    num_metrics = len(RANKING_METRIC_NAMES)
    html_rows = [f"<tr><td>Mean</td>"
                 f"{_make_row_cells_html(method_scores, mean_cells, num_metrics)}</tr>"]
    html_rows += [f"<tr><td>{_make_seed_links_html(row, scope)}</td>"
                  f"{_make_row_cells_html(row, metric_cells, num_metrics)}</tr>"
                  for row, metric_cells in zip(seed_scores, zip(*seed_columns), strict=True)]
    return make_table_html(["Seed", *RANKING_METRIC_NAMES], html_rows, RANKING_METRIC_NAMES,
                           header_titles={n: ie.describe_metric(n) for n in RANKING_METRIC_NAMES})
