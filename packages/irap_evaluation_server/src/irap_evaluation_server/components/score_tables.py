"""HTML of the score tables of the Scores page and the Model page.

A value cell shows the value and, below it, its bootstrap interval, or the state of its
computation. Its background is shaded by how good the value is among the values that it is compared
with (`method_ranking.compute_highlights`). With a reference method, the cells of the other methods
also show their difference from it (`ReferenceComparisons`).
"""

import dataclasses as dc
import html
import math
import typing as T

import irap_evaluation as ie
from irap_evaluation.metrics import IRAP_ATTRIBUTE_METRIC_NAMES

from ..method_comparison import get_better_method
from ..method_ranking import (
    RANKING_METRIC_NAMES,
    CellHighlight,
    compute_highlights,
    describe_split_use,
    get_method_split_use,
)
from ..scoring import MethodScores, ResultState, ScoringSettings
from .formatting import (
    format_context_offsets,
    make_model_link_html,
    make_table_html,
    make_table_html_from_header,
)

V = T.TypeVar("V")

#: Gets the value of a cell from the values of a row, e.g. its interval, None if they lack it.
ValueSelector = T.Callable[[ie.MetricValues[V]], V | None]
IntervalSelector = ValueSelector[ie.BootstrapInterval]

SINGLE_MODEL_NOTE = "With a single seed, the interval cannot account for the choice of seeds."


def select_average_value(metric_name: str) -> ValueSelector:
    return lambda values: values.averages.get(metric_name)


def select_attribute_value(metric_name: str, attribute: str) -> ValueSelector:
    return lambda values: values.per_attribute.get(metric_name, {}).get(attribute)


def _select_cell_interval(state: ResultState,
                          select_interval: IntervalSelector) -> ie.BootstrapInterval | None:
    """Selects the computed interval of a cell, or None."""
    if state.cached is None or state.cached.value is None:
        return None
    return select_interval(state.cached.value)


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


def _make_highlight_note_html(compared_with: str, has_best_or_tied: bool) -> str:
    """Explains the highlights.

    Args:
        compared_with: What each value is compared with, e.g. 'the other methods in its column'.
    """
    text = f"Shading: each value relative to {compared_with}, darkest is best."
    if has_best_or_tied:
        text += " Bold: the best value, and the values whose interval overlaps its interval."
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


def _make_title_attribute(title: str) -> str:
    return f' title="{html.escape(title)}"' if title else ""


def _get_interval_text(state: ResultState, select_interval: IntervalSelector,
                       is_signed: bool = False) -> tuple[str, str]:
    """Returns the text of the interval of a cell, or of the state of its computation, and a
    title that explains it ('' for none).

    Args:
        is_signed: Whether the bounds have a sign, e.g. those of a difference.
    """
    if state.cached is not None and state.cached.value is None:
        return "failed", state.cached.message
    if state.cached is None:
        return "computing…" if state.is_pending else "–", ""
    interval = select_interval(state.cached.value)
    if interval is None:
        return "–", ""
    format_bound = format_signed_metric_value if is_signed else format_metric_value
    text = (f"{format_bound(interval.low, num_decimals=3)}–"
            f"{format_bound(interval.high, num_decimals=3)}")
    if not interval.is_partially_undefined:
        return text, ""
    return text + "*", (f"Number of resamples left out, since the value is undefined:"
                        f" {interval.num_undefined}.")


def _make_interval_html(state: ResultState, select_interval: IntervalSelector) -> str:
    text, title = _get_interval_text(state, select_interval)
    return f'<div class="interval"{_make_title_attribute(title)}>{text}</div>'


def make_value_cell_html(value: float | None, state: ResultState,
                         select_interval: IntervalSelector, note: str = "",
                         highlight: CellHighlight | None = None,
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
             else _make_interval_html(state, select_interval)) + difference_html
    text = format_metric_value(value, num_decimals=4)
    if highlight is not None and highlight.is_best_or_tied:
        text = f"<b>{text}</b>"
    # The shade is `--strength` in `styles.css`.
    style = (f' style="--strength: {highlight.strength:.2f}"'
             if highlight is not None and highlight.strength else "")
    return f'<td class="number"{style}>{text}{below}</td>'


def _make_compared_cells_html(values: T.Sequence[float | None], states: T.Sequence[ResultState],
                              select_interval: IntervalSelector, is_lower_better: bool,
                              difference_htmls: T.Sequence[str],
                              notes: T.Sequence[str] | None = None) -> list[str]:
    """Makes the cells of values that are compared with each other, with their intervals.

    Args:
        states: The intervals of the row of each value.
        difference_htmls: The difference of each cell (`make_value_cell_html`).
        notes: The note of each cell (`make_value_cell_html`).
    """
    highlights = compute_highlights(
        values, [_select_cell_interval(s, select_interval) for s in states], is_lower_better)
    return [make_value_cell_html(v, s, select_interval, note=n, highlight=h, difference_html=d)
            for v, s, n, h, d in zip(values, states, notes or [""] * len(values), highlights,
                                     difference_htmls, strict=True)]


# Comparisons with a reference method ##############################################################

@dc.dataclass(frozen=True)
class ReferenceComparisons:
    """The comparisons of the methods of a table with a reference method
    (`method_comparison.ComparisonRequest`, with the method as A and the reference as B).

    Attributes:
        reference: The means over the models of the reference method, without an error.
        method_to_state: Method name -> the state of its comparison, for the compared methods.
    """

    reference: MethodScores
    method_to_state: T.Mapping[str, ResultState]


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


def _select_difference_interval(
        select_difference: ValueSelector[ie.MetricDifference]) -> IntervalSelector:
    """Makes a selector of the interval of the difference that `select_difference` gets from the
    result of a comparison."""
    def select(differences: ie.MetricValues[ie.MetricDifference]) -> ie.BootstrapInterval | None:
        difference = select_difference(differences)
        return None if difference is None else difference.interval
    return select


def _make_difference_html(difference: float, state: ResultState,
                          select_difference: ValueSelector[ie.MetricDifference],
                          metric_name: str) -> str:
    """Makes the difference of a value from that of the reference method, with its interval, and
    whether the value is clearly better or worse (`method_comparison.get_better_method`).

    Args:
        state: The comparison of the method with the reference.
    """
    select_interval = _select_difference_interval(select_difference)
    interval_text, title = _get_interval_text(state, select_interval, is_signed=True)
    better = get_better_method(_select_cell_interval(state, select_interval), metric_name)
    verdict = {"A": " <b>better</b>", "B": " <b>worse</b>", None: ""}[better]
    return (f'<div class="interval"{_make_title_attribute(title)}>'
            f"Δ {format_signed_metric_value(difference, 4)} ({interval_text}){verdict}</div>")


def _make_difference_htmls(rows: T.Sequence[MethodScores],
                           comparisons: ReferenceComparisons | None,
                           select_value: ValueSelector, metric_name: str) -> list[str]:
    """Makes the difference of the value of each row from that of the reference method
    (`_make_difference_html`), '' for the rows that are not compared or lack the value.

    Args:
        select_value: Gets the value from the metrics of a row, and the difference from a
            comparison.
        metric_name: The metric of the value.
    """
    if comparisons is None:
        return [""] * len(rows)
    reference_value = select_value(comparisons.reference.metrics)

    def make_html(row: MethodScores) -> str:
        state = comparisons.method_to_state.get(row.method_name)
        value = None if row.metrics is None else select_value(row.metrics)
        if state is None or value is None or reference_value is None:
            return ""
        return _make_difference_html(value - reference_value, state, select_value, metric_name)

    return [make_html(r) for r in rows]


def _make_method_label(row: MethodScores, comparisons: ReferenceComparisons | None,
                       split: str) -> str:
    """Makes the shown method name, marked if it is the reference method or used the split of the
    scores (`method_ranking.describe_split_use`)."""
    marks = []
    if comparisons is not None and row.method_name == comparisons.reference.method_name:
        marks.append("reference")
    if split_use := describe_split_use(get_method_split_use(row, split), split):
        marks.append(split_use)
    name = row.shown_method_name
    return f"{name} ({', '.join(marks)})" if marks else name


# Tables ###########################################################################################

def _make_model_links_html(row: MethodScores, split: str,
                           attributes_query: T.Sequence[str]) -> str:
    """Makes the number of models, with a link to each, labeled by seed."""
    links = ", ".join(make_model_link_html(model.id, "model" if model.seed is None
                                           else f"seed{model.seed}", split, attributes_query)
                      for model in (m.submission.model for m in row.models))
    return f'{row.num_models} <span class="muted">({links})</span>'


def _make_notes_html(row: MethodScores, is_model_compatible_view: bool) -> str:
    notes = []
    if row.missing_attributes:
        title_attribute = _make_title_attribute(", ".join(row.missing_attributes))
        notes.append(f"<span{title_attribute}>{len(row.missing_attributes)} not predicted</span>")
    if is_model_compatible_view:
        notes.append(
            f"context offsets {format_context_offsets(row.models[0].scores.context_offsets)}")
    if _is_single_model(row):
        notes.append("single model*")
    return "<br>".join(notes)


def make_method_table_html(rows: T.Sequence[MethodScores],
                           method_to_intervals: T.Mapping[str, ResultState],
                           sort_metric: str, split: str, attributes_query: T.Sequence[str],
                           is_model_compatible_view: bool,
                           comparisons: ReferenceComparisons | None = None) -> str:
    """Makes a table of method rows × the attribute averages of `RANKING_METRIC_NAMES`.

    The metric headers have a `data-sort` attribute with the metric name, for a click handler.

    Args:
        method_to_intervals: Method name -> the state of its intervals.
        split: The split of the scores, for the links to the models and the marks of the methods
            that used it.
        attributes_query: The attribute subset for the links to the models
            (`AttributeSubset.to_query`).
        is_model_compatible_view: Whether the rows are scored on model-compatible sets, whose
            context offsets are shown.
        comparisons: The comparisons with a reference method, or None.
    """
    metric_headers = "".join(
        f'<th class="sortable{" sorted" if n == sort_metric else ""}" data-sort="{n}"'
        f' title="Sort by {n}, best first">{n}{" ▾" if n == sort_metric else ""}</th>'
        for n in RANKING_METRIC_NAMES)
    header = f"<th>Method</th><th>Models</th>{metric_headers}<th>Notes</th>"
    states = [method_to_intervals.get(r.method_name, ResultState()) for r in rows]
    metric_columns = [
        _make_compared_cells_html(
            [None if r.metrics is None else r.metrics.averages.get(n) for r in rows], states,
            select_average_value(n), ie.is_lower_better(n),
            _make_difference_htmls(rows, comparisons, select_average_value(n), n))
        for n in RANKING_METRIC_NAMES]
    html_rows = []
    for row, metric_cells in zip(rows, zip(*metric_columns)):
        if row.metrics is None:
            cells = (f'<td colspan="{len(RANKING_METRIC_NAMES)}" class="error">'
                     f"{html.escape(row.error or '')}</td>")
        else:
            cells = "".join(metric_cells)
        method_label = _make_method_label(row, comparisons, split)
        html_rows.append(f"<tr><td>{html.escape(method_label)}</td>"
                         f"<td>{_make_model_links_html(row, split, attributes_query)}</td>{cells}"
                         f"<td>{_make_notes_html(row, is_model_compatible_view)}</td></tr>")
    footnote = (f'<div class="muted">* {SINGLE_MODEL_NOTE}</div>'
                if any(_is_single_model(r) for r in rows) else "")
    return (make_table_html_from_header(header, html_rows) + footnote
            + _make_highlight_note_html("the other methods in its column", has_best_or_tied=True))


def make_per_attribute_table_html(rows: T.Sequence[MethodScores],
                                  method_to_intervals: T.Mapping[str, ResultState],
                                  metric_name: str, attributes: T.Sequence[str], split: str,
                                  comparisons: ReferenceComparisons | None = None) -> str:
    """Makes a table of attribute rows × method columns of one per-attribute metric, for the rows
    without an error, in their order.

    Args:
        split: The split of the scores, for the marks of the methods that used it.
        comparisons: The comparisons with a reference method, or None.
    """
    rows = [r for r in rows if r.metrics is not None]
    states = [method_to_intervals.get(r.method_name, ResultState()) for r in rows]
    is_lower_better = ie.is_lower_better(metric_name)
    html_rows = []
    for attribute in attributes:
        select_value = select_attribute_value(metric_name, attribute)
        cells = _make_compared_cells_html(
            [select_value(r.metrics) for r in rows], states, select_value, is_lower_better,
            _make_difference_htmls(rows, comparisons, select_value, metric_name),
            notes=["not predicted" if attribute in r.missing_attributes else "" for r in rows])
        html_rows.append(f"<tr><td>{html.escape(attribute)}</td>{''.join(cells)}</tr>")
    method_labels = [_make_method_label(r, comparisons, split) for r in rows]
    return (make_table_html(["Attribute", *method_labels], html_rows)
            + _make_highlight_note_html("the other methods in its row", has_best_or_tied=True))


def make_model_scores_html(scores: MethodScores, state: ResultState) -> str:
    """Makes the tables of the scores of one model over a subset of the attributes: the attribute
    averages, and the iRAP metrics and invalid cells of each attribute.

    Args:
        scores: `scoring.compute_method_scores` of the model alone.
    """
    [model] = scores.models
    if scores.metrics is None:
        return f'<div class="error">{html.escape(scores.error or "")}</div>'
    average_rows = [f"<tr><td>{n}</td>"
                    f"{make_value_cell_html(v, state, select_average_value(n))}</tr>"
                    for n, v in scores.metrics.averages.items()]
    attributes = list(scores.metrics.per_attribute[IRAP_ATTRIBUTE_METRIC_NAMES[0]])
    # The attributes are compared within each metric column, without ties of intervals (bold),
    # which are meant for comparing methods.
    metric_columns = []
    for n in IRAP_ATTRIBUTE_METRIC_NAMES:
        values = [scores.metrics.per_attribute[n].get(a) for a in attributes]
        highlights = compute_highlights(values, None, ie.is_lower_better(n))
        metric_columns.append([make_value_cell_html(v, state, select_attribute_value(n, a),
                                                    highlight=h)
                               for a, v, h in zip(attributes, values, highlights)])
    attribute_rows = []
    for attribute, metric_cells in zip(attributes, zip(*metric_columns)):
        cells = "".join(metric_cells)
        is_missing = attribute in scores.missing_attributes
        num_invalid = (f"{model.scores.num_invalid[attribute]}"
                       f"{' (not predicted)' if is_missing else ''}")
        attribute_rows.append(f"<tr><td>{html.escape(attribute)}</td>{cells}"
                              f'<td class="number">{num_invalid}</td></tr>')
    attribute_header = ["Attribute", *IRAP_ATTRIBUTE_METRIC_NAMES, "Invalid cells"]
    return (f'<div class="score-tables">'
            f'{make_table_html(["Average", "Value"], average_rows)}'
            f"{make_table_html(attribute_header, attribute_rows)}</div>"
            f"{_make_highlight_note_html('the other attributes', has_best_or_tied=False)}")
