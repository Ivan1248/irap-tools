"""HTML of the score tables of the Scores page and the Model page.

A value cell shows the value and, below it, its bootstrap interval, computed from the resampled
values of its row (`scoring_worker.ScoringWorker.get_method_resamples`). Its background is shaded by
how good the value is among the values that it is compared with, and it is bold if it is not
significantly worse than the best (`method_ranking.compute_highlights`). With a reference method,
the cells of the other methods also show their difference from it (`ReferenceComparisons`).
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
    NOT_WORSE_SIGNIFICANCE_LEVEL,
    RANKING_METRIC_NAMES,
    CellHighlight,
    compute_highlights,
    describe_split_use,
    get_method_split_use,
)
from ..scoring import MethodScores, ScoringSettings, are_scored_on_same_set
from .formatting import (
    format_context_offsets,
    make_model_link_html,
    make_table_html,
    make_table_html_from_header,
    make_title_attribute,
    make_truncated_name_html,
)

V = T.TypeVar("V")

#: Gets the value of a cell from the values of a row, e.g. its resampled values, None if they
#: lack it.
ValueSelector = T.Callable[[ie.MetricValues[V]], V | None]

#: The resampled values of a row (`scoring_worker.ScoringWorker.get_method_resamples`), or None.
RowResamples = ie.MetricValues[np.ndarray] | None

SINGLE_MODEL_NOTE = "With a single seed, the interval cannot account for the choice of seeds."

#: The metrics of `make_model_scores_html`, e.g. for `ScoringWorker.get_method_resamples`.
MODEL_SCORES_METRIC_NAMES = (*RANKING_METRIC_NAMES, *IRAP_ATTRIBUTE_METRIC_NAMES)


def select_average_value(metric_name: str) -> ValueSelector:
    return lambda values: values.averages.get(metric_name)


def select_attribute_value(metric_name: str, attribute: str) -> ValueSelector:
    return lambda values: values.per_attribute.get(metric_name, {}).get(attribute)


def _select_row_value(values: ie.MetricValues[V] | None, select_value: ValueSelector) -> V | None:
    return None if values is None else select_value(values)


def _compute_cell_interval(resamples: np.ndarray | None,
                           confidence: float) -> ie.BootstrapInterval | None:
    return None if resamples is None else ie.compute_percentile_interval(resamples, confidence)


def make_interval_note(settings: ScoringSettings, is_mean_over_models: bool) -> str:
    """Explains the intervals below the values.

    Args:
        is_mean_over_models: Whether the values are means over the models of a method, or of a
            single model.
    """
    roads = (f"the choice of roads ({settings.num_resamples} bootstrap resamples of road"
             f" sequences)")
    if is_mean_over_models:
        return (f"Values: means over the seeds of each method. Below each: its"
                f" {settings.confidence:.0%} confidence interval, which accounts for both {roads}"
                f" and the choice of seeds.")
    return (f"Below each value: its {settings.confidence:.0%} confidence interval, which accounts"
            f" for {roads}. {SINGLE_MODEL_NOTE}")


def _is_single_model(row: MethodScores) -> bool:
    """Returns whether the intervals of the row cover only one model (`SINGLE_MODEL_NOTE`)."""
    return row.num_models == 1 and row.error is None


def _make_highlight_note_html(axis: T.Literal["column", "row"], is_tested: bool) -> str:
    """Explains the highlights.

    Args:
        axis: Along which the values are compared with each other.
        is_tested: Whether the values are tested against the best (bold).
    """
    text = f"Darker is better (in a {axis})."
    if is_tested:
        text += (f" Bold: the best, and what is not significantly worse than it (one-sided paired"
                 f" bootstrap test at {NOT_WORSE_SIGNIFICANCE_LEVEL:.0%}).")
    return f'<div class="muted">{text}</div>'


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

    reference_method_name: str
    method_to_differences: T.Mapping[str, ie.MetricValues[ie.MetricDifference]]


def make_comparison_note(settings: ScoringSettings, reference_method: str,
                         rows: T.Iterable[MethodScores]) -> str:
    """Explains the differences from the reference method (`ReferenceComparisons`).

    Args:
        rows: The methods of the table, for the note about single models.
    """
    text = (f"Δ: the difference from {reference_method}, with its {settings.confidence:.0%}"
            f" confidence interval, which accounts for the choice of roads (the same"
            f" {settings.num_resamples} resamples for both methods) and of seeds. better/worse:"
            f" the interval excludes 0. With many methods or attributes, some intervals exclude 0"
            f" by chance.")
    return text + (f" {SINGLE_MODEL_NOTE}" if any(_is_single_model(r) for r in rows) else "")


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
                            split: str) -> str:
    """Makes the shown method name, cut off with a tooltip if it is long (`truncated-name` in
    `styles.css`), and below it, whether it is the reference method or used the split of the
    scores (`method_ranking.describe_split_use`)."""
    marks = []
    if comparisons is not None and row.method_name == comparisons.reference_method_name:
        marks.append("reference")
    if split_use := describe_split_use(get_method_split_use(row, split), split):
        marks.append(split_use)
    marks_html = (f'<div class="muted">({html.escape(", ".join(marks))})</div>' if marks
                  else "")
    return make_truncated_name_html(row.shown_method_name) + marks_html


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


def _make_model_links_html(row: MethodScores, split: str,
                           attributes_query: T.Sequence[str]) -> str:
    """Makes the number of models, with a link to each, labeled by seed."""
    links = ", ".join(make_model_link_html(model.id, f"seed{model.seed}", split, attributes_query)
                      for model in (m.submission.model for m in row.models))
    return f'{row.num_models} <span class="muted">({links})</span>'


def _are_tested(rows: T.Iterable[MethodScores]) -> bool:
    """Returns whether the values of the rows are tested against the best
    (`compute_highlights`), which needs the rows with values on the same evaluation set
    (`scoring.are_scored_on_same_set`)."""
    return are_scored_on_same_set(m for r in rows if r.metrics is not None for m in r.models)


def _make_method_highlight_note_html(axis: T.Literal["column", "row"], is_tested: bool) -> str:
    """Explains the highlights of methods (`_make_highlight_note_html`), and why they are not
    tested if they are not (`_are_tested`)."""
    untested_note = ("" if is_tested else
                     '<div class="muted">The methods are scored on the segments that their'
                     " context offsets select, which differ, so they are not tested against each"
                     " other.</div>")
    return _make_highlight_note_html(axis, is_tested) + untested_note


def _make_notes_html(row: MethodScores) -> str:
    notes = []
    if row.missing_attributes:
        title_attribute = make_title_attribute(", ".join(row.missing_attributes))
        notes.append(f"<span{title_attribute}>{len(row.missing_attributes)} not predicted</span>")
    if row.models[0].scores.evaluation_set == ie.MODEL_COMPATIBLE_SET_NAME:
        notes.append(
            f"context offsets {format_context_offsets(row.models[0].scores.context_offsets)}")
    if _is_single_model(row):
        notes.append("single model*")
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
                           sort_metric: str, split: str, attributes_query: T.Sequence[str],
                           comparisons: ReferenceComparisons | None = None) -> str:
    """Makes a table of method rows × the attribute averages of `RANKING_METRIC_NAMES`.

    The metric headers have a `data-sort` attribute with the metric name, for a click handler.

    Args:
        method_to_resamples: Method name -> its resampled values of `RANKING_METRIC_NAMES`, or
            None.
        confidence: The confidence of the intervals.
        split: The split of the scores, for the links to the models and the marks of the methods
            that used it.
        attributes_query: The attribute subset for the links to the models
            (`AttributeSubset.to_query`).
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
        if row.metrics is None:
            cells = (f'<td colspan="{len(RANKING_METRIC_NAMES)}" class="error">'
                     f"{html.escape(row.error or '')}</td>")
        else:
            cells = "".join(metric_cells)
        method_label_html = _make_method_label_html(row, comparisons, split)
        html_rows.append(f"<tr><td>{method_label_html}</td>"
                         f"<td>{_make_model_links_html(row, split, attributes_query)}</td>{cells}"
                         f"<td>{_make_notes_html(row)}</td></tr>")
    footnote = (f'<div class="muted">* {SINGLE_MODEL_NOTE}</div>'
                if any(_is_single_model(r) for r in rows) else "")
    return (make_table_html_from_header(header, html_rows) + footnote
            + _make_method_highlight_note_html("column", is_tested))


def make_per_attribute_table_html(rows: T.Sequence[MethodScores],
                                  method_to_resamples: T.Mapping[str, RowResamples],
                                  confidence: float, metric_name: str,
                                  attributes: T.Sequence[str], split: str,
                                  comparisons: ReferenceComparisons | None = None) -> str:
    """Makes a table of attribute rows × method columns of one per-attribute metric, for the rows
    without an error, in their order.

    Args:
        method_to_resamples: Method name -> its resampled values of `metric_name`, or None.
        split: The split of the scores, for the marks of the methods that used it.
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
    method_headers = "".join(
        f'<th class="number">{_make_method_label_html(r, comparisons, split)}</th>' for r in rows)
    metric_note = f'<div class="muted">{html.escape(ie.describe_metric(metric_name))}</div>'
    return (make_table_html_from_header(f"<th>Attribute</th>{method_headers}", html_rows)
            + metric_note + _make_method_highlight_note_html("row", is_tested))


def make_model_scores_html(scores: MethodScores, resamples: RowResamples,
                           confidence: float) -> str:
    """Makes the tables of the scores of one model over a subset of the attributes: the attribute
    averages, and the iRAP metrics and invalid cells of each attribute.

    Args:
        scores: `scoring.compute_method_scores` of the model alone.
        resamples: Its resampled values of `MODEL_SCORES_METRIC_NAMES`, or None.
        confidence: The confidence of the intervals.
    """
    [model] = scores.models
    if scores.metrics is None:
        return f'<div class="error">{html.escape(scores.error or "")}</div>'

    def compute_interval(select_value: ValueSelector) -> ie.BootstrapInterval | None:
        return _compute_cell_interval(_select_row_value(resamples, select_value), confidence)

    average_rows = [f"<tr><td{make_title_attribute(ie.describe_metric(n))}>{n}</td>"
                    f"{make_value_cell_html(v, compute_interval(select_average_value(n)))}</tr>"
                    for n, v in scores.metrics.averages.items()]
    attributes = list(scores.metrics.per_attribute[IRAP_ATTRIBUTE_METRIC_NAMES[0]])
    # The attributes are compared within each metric column, without tests against the best
    # (bold), which are meant for comparing methods.
    metric_columns = []
    for n in IRAP_ATTRIBUTE_METRIC_NAMES:
        values = [scores.metrics.per_attribute[n].get(a) for a in attributes]
        highlights = compute_highlights(values, None, ie.is_lower_better(n))
        metric_columns.append([
            make_value_cell_html(v, compute_interval(select_attribute_value(n, a)), highlight=h)
            for a, v, h in zip(attributes, values, highlights)])
    attribute_rows = []
    for attribute, metric_cells in zip(attributes, zip(*metric_columns)):
        cells = "".join(metric_cells)
        is_missing = attribute in scores.missing_attributes
        num_invalid = (f"{model.scores.num_invalid[attribute]}"
                       f"{' (not predicted)' if is_missing else ''}")
        attribute_rows.append(f"<tr><td>{html.escape(attribute)}</td>{cells}"
                              f'<td class="number">{num_invalid}</td></tr>')
    average_table = make_table_html(["Average", "Value"], average_rows, number_headers=["Value"])
    number_headers = [*IRAP_ATTRIBUTE_METRIC_NAMES, "Invalid cells"]
    attribute_table = make_table_html(
        ["Attribute", *number_headers], attribute_rows, number_headers,
        header_titles={n: ie.describe_metric(n) for n in IRAP_ATTRIBUTE_METRIC_NAMES})
    return (f'<div class="score-tables">{average_table}{attribute_table}</div>'
            f"{_make_highlight_note_html('the other attributes', is_tested=False)}")
