"""What the Scores page shows: the methods of a dataset split, ranked by a metric averaged over a
subset of the attributes (see `scoring` for the scores), and the highlights of compared values."""

import dataclasses as dc
import math
import re
import typing as T

import irap_evaluation as ie
from irap_evaluation.metrics import IRAP_MAIN_METRIC, parse_metric_name

from .archive import Submission
from .scoring import (
    MethodScores,
    ScoredModel,
    SubmissionScoring,
    compute_method_scores,
    group_scored_models_by_method,
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
    """Computes the highlights of values that are compared with each other, e.g. of the methods
    in a method table column.

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


def collect_scored_attributes(models: T.Iterable[ScoredModel]) -> tuple[str, ...]:
    """Collects the attributes that the models are scored on, in first-seen order.

    The models of a view normally have the same attributes, since missing attributes are scored
    as invalid cells. Model-compatible sets can lack labels of some attributes.
    """
    return tuple(dict.fromkeys(a for m in models for a in m.scores.attributes))


def _get_sort_key(metrics: ie.MetricValues[float] | None, metric: str,
                  name: str) -> tuple[bool, float, str]:
    """Returns the sort key of `sort_method_scores` and `sort_scored_models`."""
    value = math.nan if metrics is None else metrics.averages.get(metric, math.nan)
    is_missing = math.isnan(value)
    sign = 1 if ie.is_lower_better(metric) else -1
    return is_missing, 0.0 if is_missing else sign * value, name


def sort_method_scores(rows: T.Iterable[MethodScores], metric: str) -> list[MethodScores]:
    """Sorts methods best first by the attribute average `metric` (see
    `irap_evaluation.is_lower_better`).

    Rows without a value (an error, a metric that the method lacks, or NaN) are last. Ties are
    ordered by method name.
    """
    return sorted(rows, key=lambda r: _get_sort_key(r.metrics, metric, r.method_name))


def select_model_metrics(model: ScoredModel,
                         attributes: T.Sequence[str]) -> ie.MetricValues[float] | None:
    """Selects the metrics of a model over `attributes`
    (`irap_evaluation.select_metric_attributes`), None if it is not scored on all of them, or
    `attributes` is empty."""
    if not attributes or any(a not in model.scores.attributes for a in attributes):
        return None
    return ie.select_metric_attributes(model.scores.metrics, attributes)


def sort_scored_models(models: T.Iterable[ScoredModel], metric: str,
                       attributes: T.Sequence[str]) -> list[ScoredModel]:
    """Sorts models best first by the attribute average `metric` over `attributes`
    (`select_model_metrics`), as `sort_method_scores` sorts methods. Ties are ordered by model
    label."""
    return sorted(models, key=lambda m: _get_sort_key(select_model_metrics(m, attributes),
                                                      metric, m.submission.label))


def rank_methods(models: T.Iterable[ScoredModel], attributes: T.Sequence[str],
                 sort_metric: str) -> list[MethodScores]:
    """Computes the scores of each method over `attributes` (`scoring.compute_method_scores`),
    sorted by `sort_metric` (`sort_method_scores`).

    Raises:
        ValueError: If `attributes` is empty.
    """
    return sort_method_scores(
        [compute_method_scores(name, method_models, attributes)
         for name, method_models in group_scored_models_by_method(models).items()],
        sort_metric)


def get_method_split_use(row: MethodScores, split: str) -> ie.SplitUse:
    """Returns what the models of a method used `split` for
    (`irap_evaluation.ModelInfo.get_split_use`), which is the same for all of them
    (`irap_evaluation.check_method_models`)."""
    return row.models[0].submission.model_info.get_split_use(split)


def describe_split_use(split_use: ie.SplitUse, split: str) -> str:
    """Returns a short mark of what a model used a split for, '' if the split is held out.
    `irap_evaluation.explain_split_use` gives the explanation."""
    return {"training": f"trained on {split}", "early_stopping": f"early stopping on {split}",
            "unknown": "training splits unknown", "held_out": ""}[split_use]


def select_shown_methods(rows: T.Iterable[MethodScores], pattern: re.Pattern[str],
                         reference_method: str = "",
                         hidden_training_split: str | None = None
                         ) -> tuple[list[MethodScores], int]:
    """Selects the rows whose method name matches `pattern` (`re.Pattern.search`), and the row
    of `reference_method`, in their order.

    Args:
        reference_method: The method that the others are compared with, shown even if it does
            not match `pattern` or was trained on `hidden_training_split`.
        hidden_training_split: If not None, the methods trained on this split
            (`get_method_split_use`) are hidden.

    Returns:
        The selected rows, and the number of matching rows that are hidden.
    """
    shown, num_hidden = [], 0
    for row in rows:
        if row.method_name == reference_method:
            shown.append(row)
        elif not pattern.search(row.method_name):
            continue
        elif (hidden_training_split is not None
              and get_method_split_use(row, hidden_training_split) == "training"):
            num_hidden += 1
        else:
            shown.append(row)
    return shown, num_hidden


def get_unscored_reason(submission: Submission, scoring: SubmissionScoring | None,
                        is_current: bool, is_pending: bool) -> str | None:
    """Explains why a submission has no current scores. Returns None if it has them.

    Args:
        scoring: Its stored scoring, or None.
        is_current: Whether `scoring` is current (`scoring.is_scoring_current`).
        is_pending: Whether its scoring is queued or running.
    """
    if is_pending:
        return "Scoring…"
    if submission.is_in_use and scoring is not None and scoring.status == "error":
        return f"{scoring.message} It is scored again when the server restarts."
    if scoring is None or not is_current:  # An 'error' scoring is not current.
        return ("Deleted files and the files of deleted models are not scored."
                if not submission.is_in_use
                else "No scores for the current settings and metadata.")
    if scoring.status == "failed":
        return scoring.message
    return None
