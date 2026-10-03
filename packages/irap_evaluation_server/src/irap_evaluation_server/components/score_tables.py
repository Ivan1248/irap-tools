"""HTML of the score tables of the Scores page and the submission page.

A value cell shows the value and, below it, its bootstrap interval, or the state of its
computation. Its background is shaded by how good the value is among the values that it is compared
with (`method_ranking.compute_highlights`).
"""

import html
import math
import typing as T

import irap_evaluation as ie
from irap_evaluation.metrics import IRAP_ATTRIBUTE_METRIC_NAMES

from ..method_comparison import get_better_method
from ..method_ranking import RANKING_METRIC_NAMES, CellHighlight, compute_highlights
from ..scoring import MethodScores, ResultState, ScoringSettings
from .formatting import make_submission_link_html, make_table_html, make_table_html_from_header

V = T.TypeVar("V")

#: Gets the value of a cell from the values of a row, e.g. its interval, None if they lack it.
ValueSelector = T.Callable[[ie.MetricValues[V]], V | None]
IntervalSelector = ValueSelector[ie.BootstrapInterval]

SINGLE_RUN_NOTE = ("The intervals of a single run cover only the sampling of the roads, not the"
                   " variation between runs.")


def select_average_value(metric_name: str) -> ValueSelector:
    return lambda values: values.averages.get(metric_name)


def select_attribute_value(metric_name: str, attribute: str) -> ValueSelector:
    return lambda values: values.per_attribute.get(metric_name, {}).get(attribute)


def _select_cell_interval(state: ResultState,
                          select_interval: IntervalSelector) -> ie.BootstrapInterval | None:
    """The computed interval of a cell, or None."""
    if state.cached is None or state.cached.value is None:
        return None
    return select_interval(state.cached.value)


def make_interval_note(settings: ScoringSettings, is_mean_over_runs: bool) -> str:
    """Explains the intervals below the values.

    Args:
        is_mean_over_runs: Whether the values are means over the runs of a method, or of a
            single run.
    """
    text = (f"Below each value: its {settings.confidence:.0%} bootstrap interval over the road"
            f" sequences ({settings.num_resamples} resamples)")
    return text + (", the means over the runs of a method." if is_mean_over_runs
                   else f". {SINGLE_RUN_NOTE}")


def _is_single_run(row: MethodScores) -> bool:
    """Whether the intervals of the row cover only one run (`SINGLE_RUN_NOTE`)."""
    return row.num_runs == 1 and row.error is None


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
    """The value without a leading zero, e.g. '.123' or '-.123', or 'NaN' or '∞'."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "∞" if value > 0 else "−∞"
    text = f"{value:.{num_decimals}f}"
    sign, unsigned = ("-", text[1:]) if text.startswith("-") else ("", text)
    return sign + unsigned[1:] if unsigned.startswith("0.") else text


def format_optional_metric_value(value: float | None, num_decimals: int) -> str:
    """`format_metric_value`, or '–' for None, e.g. a metric that a run lacks."""
    return "–" if value is None else format_metric_value(value, num_decimals)


def _make_interval_html(state: ResultState, select_interval: IntervalSelector) -> str:
    title = ""
    if state.cached is not None and state.cached.value is None:
        text, title = "failed", state.cached.message
    elif state.cached is not None:
        interval = select_interval(state.cached.value)
        if interval is None:
            text = "–"
        else:
            text = (f"{format_metric_value(interval.low, num_decimals=3)}–"
                    f"{format_metric_value(interval.high, num_decimals=3)}")
            if interval.is_partially_undefined:
                text += "*"
                title = (f"Leaves out {interval.num_undefined} resamples where the value is"
                         f" undefined.")
    else:
        text = "computing…" if state.is_pending else "–"
    title_attribute = f' title="{html.escape(title)}"' if title else ""
    return f'<div class="interval"{title_attribute}>{text}</div>'


def make_value_cell_html(value: float | None, state: ResultState,
                         select_interval: IntervalSelector, note: str = "",
                         highlight: CellHighlight | None = None) -> str:
    """A cell with the value and its interval, or '–' if the value is None.

    Args:
        note: Shown instead of the interval, e.g. 'not predicted'.
    """
    if value is None:
        return '<td class="number">–</td>'
    below = (f'<div class="interval">{html.escape(note)}</div>' if note
             else _make_interval_html(state, select_interval))
    text = format_metric_value(value, num_decimals=4)
    if highlight is not None and highlight.is_best_or_tied:
        text = f"<b>{text}</b>"
    # The shade is `--strength` in `styles.css`.
    style = (f' style="--strength: {highlight.strength:.2f}"'
             if highlight is not None and highlight.strength else "")
    return f'<td class="number"{style}>{text}{below}</td>'


def _make_compared_cells_html(values: T.Sequence[float | None], states: T.Sequence[ResultState],
                              select_interval: IntervalSelector, is_lower_better: bool,
                              notes: T.Sequence[str] | None = None) -> list[str]:
    """The cells of values that are compared with each other, with their intervals.

    Args:
        states: The intervals of the row of each value.
        notes: The note of each cell (`make_value_cell_html`).
    """
    highlights = compute_highlights(
        values, [_select_cell_interval(s, select_interval) for s in states], is_lower_better)
    return [make_value_cell_html(v, s, select_interval, note=n, highlight=h)
            for v, s, n, h in zip(values, states, notes or [""] * len(values), highlights,
                                  strict=True)]


def _make_runs_html(row: MethodScores, attributes_query: T.Sequence[str]) -> str:
    links = ", ".join(make_submission_link_html(r.submission.id, attributes_query)
                      for r in row.runs)
    return f'{row.num_runs} <span class="muted">({links})</span>'


def _make_notes_html(row: MethodScores, is_model_compatible_view: bool) -> str:
    notes = []
    if row.missing_attributes:
        names = html.escape(", ".join(row.missing_attributes))
        notes.append(f'<span title="{names}">{len(row.missing_attributes)} not predicted</span>')
    if is_model_compatible_view:
        offsets = ", ".join(map(str, row.runs[0].scores.context_offsets))
        notes.append(f"context offsets {offsets}")
    if _is_single_run(row):
        notes.append("single run*")
    return "<br>".join(notes)


def make_method_table_html(rows: T.Sequence[MethodScores],
                           method_to_intervals: T.Mapping[str, ResultState],
                           sort_metric: str, attributes_query: T.Sequence[str],
                           is_model_compatible_view: bool,
                           method_to_comparison_path: T.Mapping[str, str]) -> str:
    """Method rows × the attribute averages of `RANKING_METRIC_NAMES`.

    The metric headers have a `data-sort` attribute with the metric name, for a click handler.

    Args:
        method_to_intervals: Method name -> the state of its intervals.
        attributes_query: The attribute subset for the links to the runs
            (`AttributeSubset.to_query`).
        is_model_compatible_view: Whether the rows are scored on model-compatible sets, whose
            context offsets are shown.
        method_to_comparison_path: Method name -> the Comparison page of it and another
            method, for a link next to its name.
    """
    metric_headers = "".join(
        f'<th class="sortable{" sorted" if n == sort_metric else ""}" data-sort="{n}"'
        f' title="Sort by {n}, best first">{n}{" ▾" if n == sort_metric else ""}</th>'
        for n in RANKING_METRIC_NAMES)
    header = f"<th>Method</th><th>Runs</th>{metric_headers}<th>Notes</th>"
    states = [method_to_intervals.get(r.method_name, ResultState()) for r in rows]
    metric_columns = [
        _make_compared_cells_html(
            [None if r.metrics is None else r.metrics.averages.get(n) for r in rows], states,
            select_average_value(n), ie.is_lower_better(n))
        for n in RANKING_METRIC_NAMES]
    html_rows = []
    for row, metric_cells in zip(rows, zip(*metric_columns)):
        if row.metrics is None:
            cells = (f'<td colspan="{len(RANKING_METRIC_NAMES)}" class="error">'
                     f"{html.escape(row.error or '')}</td>")
        else:
            cells = "".join(metric_cells)
        comparison_path = method_to_comparison_path.get(row.method_name)
        compare_link = ("" if comparison_path is None else
                        f' <a class="muted" href="{html.escape(comparison_path)}">compare</a>')
        html_rows.append(f"<tr><td>{html.escape(row.method_name)}{compare_link}</td>"
                         f"<td>{_make_runs_html(row, attributes_query)}</td>{cells}"
                         f"<td>{_make_notes_html(row, is_model_compatible_view)}</td></tr>")
    footnote = (f'<div class="muted">* {SINGLE_RUN_NOTE}</div>'
                if any(_is_single_run(r) for r in rows) else "")
    return (make_table_html_from_header(header, html_rows, fits_content=True) + footnote
            + _make_highlight_note_html("the other methods in its column", has_best_or_tied=True))


def make_per_attribute_table_html(rows: T.Sequence[MethodScores],
                                  method_to_intervals: T.Mapping[str, ResultState],
                                  metric_name: str, attributes: T.Sequence[str]) -> str:
    """Attribute rows × method columns of one per-attribute metric, for the rows without an
    error, in their order."""
    rows = [r for r in rows if r.metrics is not None]
    states = [method_to_intervals.get(r.method_name, ResultState()) for r in rows]
    is_lower_better = ie.is_lower_better(metric_name)
    html_rows = []
    for attribute in attributes:
        cells = _make_compared_cells_html(
            [r.metrics.per_attribute.get(metric_name, {}).get(attribute) for r in rows], states,
            select_attribute_value(metric_name, attribute), is_lower_better,
            notes=["not predicted" if attribute in r.missing_attributes else "" for r in rows])
        html_rows.append(f"<tr><td>{html.escape(attribute)}</td>{''.join(cells)}</tr>")
    return (make_table_html(["Attribute", *(r.method_name for r in rows)], html_rows,
                            fits_content=True)
            + _make_highlight_note_html("the other methods in its row", has_best_or_tied=True))


def make_run_scores_html(scores: MethodScores, state: ResultState) -> str:
    """The scores of one run over a subset of the attributes: the attribute averages, and the
    iRAP metrics and invalid cells of each attribute.

    Args:
        scores: `scoring.compute_method_scores` of the run alone.
    """
    [run] = scores.runs
    if scores.metrics is None:
        return f'<div class="error">{html.escape(scores.error or "")}</div>'
    average_rows = [f"<tr><td>{n}</td>"
                    f"{make_value_cell_html(v, state, select_average_value(n))}</tr>"
                    for n, v in scores.metrics.averages.items()]
    attributes = list(scores.metrics.per_attribute[IRAP_ATTRIBUTE_METRIC_NAMES[0]])
    # The attributes are compared within each metric column, without intervals, since a tie of
    # intervals is about the best method.
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
        num_invalid = (f"{run.scores.num_invalid[attribute]}"
                       f"{' (not predicted)' if is_missing else ''}")
        attribute_rows.append(f"<tr><td>{html.escape(attribute)}</td>{cells}"
                              f'<td class="number">{num_invalid}</td></tr>')
    attribute_header = ["Attribute", *IRAP_ATTRIBUTE_METRIC_NAMES, "Invalid cells"]
    return (f'<div class="score-tables">'
            f'{make_table_html(["Average", "Value"], average_rows, fits_content=True)}'
            f"{make_table_html(attribute_header, attribute_rows, fits_content=True)}</div>"
            f"{_make_highlight_note_html('the other attributes', has_best_or_tied=False)}")


# Comparisons ######################################################################################

def make_comparison_note(settings: ScoringSettings, has_single_run: bool) -> str:
    """Explains the intervals of the differences (`method_comparison.ComparisonRequest`)."""
    text = (f"Below each difference: its {settings.confidence:.0%} bootstrap interval over the"
            f" same resampled road sequences ({settings.num_resamples} resamples), with the"
            f" variation between the runs of each method. A method is clearly better if the"
            f" interval excludes 0.")
    return text + (f" {SINGLE_RUN_NOTE}" if has_single_run else "")


def _select_difference_interval(
        select_difference: ValueSelector[ie.MetricDifference]) -> IntervalSelector:
    """The interval of the difference that `select_difference` gets from the result of a
    comparison."""
    def select(differences: ie.MetricValues[ie.MetricDifference]) -> ie.BootstrapInterval | None:
        difference = select_difference(differences)
        return None if difference is None else difference.interval
    return select


def make_comparison_table_html(scores_a: MethodScores, scores_b: MethodScores,
                               state: ResultState, metric_name: str = "",
                               attributes: T.Sequence[str] = ()) -> str:
    """The metrics of methods A and B, and their difference with its interval, in rows of the
    attribute averages that both have, or of the attributes for one per-attribute metric.

    Args:
        scores_a: The means over the runs of A, without an error.
        scores_b: The same of B.
        state: The result of their comparison (`method_comparison.ComparisonRequest`).
        metric_name: The per-attribute metric, or '' for the attribute averages.
        attributes: The attributes of the rows of a per-attribute metric.
    """
    metrics_a, metrics_b = scores_a.metrics, scores_b.metrics
    if metric_name:
        values_a = metrics_a.per_attribute.get(metric_name, {})
        values_b = metrics_b.per_attribute.get(metric_name, {})
        keys = list(attributes)
    else:
        values_a, values_b = metrics_a.averages, metrics_b.averages
        keys = [n for n in RANKING_METRIC_NAMES if n in values_a and n in values_b]
    rows = []
    for key in keys:
        value_a, value_b = values_a.get(key), values_b.get(key)
        difference = None if value_a is None or value_b is None else value_a - value_b
        selector = _select_difference_interval(
            select_attribute_value(metric_name, key) if metric_name
            else select_average_value(key))
        better = get_better_method(_select_cell_interval(state, selector), metric_name or key)
        verdict = "" if better is None else f"{better} better"
        rows.append(f"<tr><td>{html.escape(key)}</td>"
                    f'<td class="number">{format_optional_metric_value(value_a, 4)}</td>'
                    f'<td class="number">{format_optional_metric_value(value_b, 4)}</td>'
                    f"{make_value_cell_html(difference, state, selector)}"
                    f"<td><b>{verdict}</b></td></tr>")
    header = ["Attribute" if metric_name else "Metric", f"A: {scores_a.method_name}",
              f"B: {scores_b.method_name}", "A − B", "Clearly better"]
    return make_table_html(header, rows, fits_content=True)
