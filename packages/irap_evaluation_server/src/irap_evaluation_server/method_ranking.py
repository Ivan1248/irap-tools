"""What the Scores page shows: the methods of a dataset split, ranked by a metric averaged over a
subset of the attributes (see `scoring` for the scores), and the highlights of compared values."""

import dataclasses as dc
import math
import typing as T

import irap_evaluation as ie
from irap_evaluation.metrics import IRAP_MAIN_METRIC, parse_metric_name

from .archive import Submission
from .scoring import (
    MethodScores,
    RunScoring,
    ScoredRun,
    compute_method_scores,
    group_scored_runs_by_method,
)

#: The columns of the method table: the attribute averages of the iRAP protocol, amF1 first.
#: Hard predictions lack the probabilistic ones.
RANKING_METRIC_NAMES = tuple(n for n in ie.get_irap_metric_names(output_kind="probs")
                                 if parse_metric_name(n).is_averaged)
DEFAULT_SORT_METRIC = f"a{IRAP_MAIN_METRIC}"
#: The per-attribute metrics of the columns, e.g. mF1 of amF1, for the per-attribute view.
PER_ATTRIBUTE_METRIC_NAMES = tuple(parse_metric_name(n).name for n in RANKING_METRIC_NAMES)


@dc.dataclass(frozen=True)
class CellHighlight:
    """How a value stands out among the values that it is compared with.

    Attributes:
        strength: 0 for the worst value, 1 for the best (min–max over the compared values).
        is_best_or_tied: Whether the value is the best, or its interval overlaps that of a best
            value.
    """

    strength: float
    is_best_or_tied: bool


def _has_finite_bounds(interval: ie.BootstrapInterval | None) -> bool:
    return interval is not None and math.isfinite(interval.low) and math.isfinite(interval.high)


def compute_highlights(values: T.Sequence[float | None],
                       intervals: T.Sequence[ie.BootstrapInterval | None] | None,
                       is_lower_better: bool) -> list[CellHighlight | None]:
    """The highlights of values that are compared with each other, e.g. of the methods in a
    method table column.

    Only finite values are compared. The others, or all values if fewer than 2 are finite, have
    None.

    Args:
        intervals: The interval of each value (None if it lacks one), or None for no
            `CellHighlight.is_best_or_tied`. A value is tied with the best if both intervals have
            finite bounds and overlap, a rough criterion that is not a statistical test.
    """
    is_tie_shown = intervals is not None
    pairs = list(zip(values, intervals if is_tie_shown else [None] * len(values), strict=True))
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if len(finite) < 2:
        return [None] * len(values)
    best, worst = (min(finite), max(finite)) if is_lower_better else (max(finite), min(finite))
    best_intervals = [i for v, i in pairs if v == best and _has_finite_bounds(i)]

    def highlight(value: float | None, interval: ie.BootstrapInterval | None
                  ) -> CellHighlight | None:
        if value is None or not math.isfinite(value):
            return None
        strength = 1.0 if best == worst else (value - worst) / (best - worst)
        is_tied = _has_finite_bounds(interval) and any(
            interval.low <= b.high and interval.high >= b.low for b in best_intervals)
        return CellHighlight(strength, is_best_or_tied=is_tie_shown and (value == best or is_tied))

    return [highlight(v, i) for v, i in pairs]


def collect_scored_attributes(runs: T.Iterable[ScoredRun]) -> tuple[str, ...]:
    """The attributes that the runs are scored on, in first-seen order.

    The runs of a view normally have the same attributes, since missing attributes are scored as
    invalid cells. Model-compatible sets can lack labels of some attributes.
    """
    return tuple(dict.fromkeys(a for r in runs for a in r.scores.attributes))


def _get_sort_key(metrics: ie.MetricValues[float] | None, metric: str,
                  name: str) -> tuple[bool, float, str]:
    """The key of `sort_method_scores` and `sort_scored_runs`."""
    value = math.nan if metrics is None else metrics.averages.get(metric, math.nan)
    is_missing = math.isnan(value)
    sign = 1 if ie.is_lower_better(metric) else -1
    return is_missing, 0.0 if is_missing else sign * value, name


def sort_method_scores(rows: T.Iterable[MethodScores], metric: str) -> list[MethodScores]:
    """Best first by the attribute average `metric` (see `irap_evaluation.is_lower_better`).

    Rows without a value (an error, a metric that the method lacks, or NaN) are last. Ties are
    ordered by method name.
    """
    return sorted(rows, key=lambda r: _get_sort_key(r.metrics, metric, r.method_name))


def select_run_metrics(run: ScoredRun,
                       attributes: T.Sequence[str]) -> ie.MetricValues[float] | None:
    """The metrics of a run over `attributes` (`irap_evaluation.select_metric_attributes`), None
    if it is not scored on all of them, or `attributes` is empty."""
    if not attributes or any(a not in run.scores.attributes for a in attributes):
        return None
    return ie.select_metric_attributes(run.scores.metrics, attributes)


def sort_scored_runs(runs: T.Iterable[ScoredRun], metric: str,
                     attributes: T.Sequence[str]) -> list[ScoredRun]:
    """Best first by the attribute average `metric` over `attributes` (`select_run_metrics`), as
    `sort_method_scores` sorts methods. Ties are ordered by run label."""
    return sorted(runs, key=lambda r: _get_sort_key(select_run_metrics(r, attributes), metric,
                                                    r.submission.run_label))


def rank_methods(runs: T.Iterable[ScoredRun], attributes: T.Sequence[str],
                        sort_metric: str) -> list[MethodScores]:
    """The scores of each method over `attributes` (`scoring.compute_method_scores`), sorted by
    `sort_metric` (`sort_method_scores`).

    Raises:
        ValueError: If `attributes` is empty.
    """
    return sort_method_scores(
        [compute_method_scores(name, method_runs, attributes)
         for name, method_runs in group_scored_runs_by_method(runs).items()],
        sort_metric)


def get_unscored_reason(submission: Submission, scoring: RunScoring | None, is_current: bool,
                        is_pending: bool) -> str | None:
    """Why a submission has no current scores, or None if it has them.

    Args:
        scoring: Its stored scoring, or None.
        is_current: Whether `scoring` is current (`scoring.is_run_scoring_current`).
        is_pending: Whether the worker is scoring it.
    """
    if is_pending:
        return "Scoring…"
    if scoring is not None and scoring.status == "error":
        return f"{scoring.message} It is scored again when the server starts."
    if scoring is None or not is_current:
        return ("Deleted submissions are not scored." if submission.is_deleted
                else "Not scored with the current settings and metadata.")
    if scoring.status == "failed":
        return scoring.message
    return None
