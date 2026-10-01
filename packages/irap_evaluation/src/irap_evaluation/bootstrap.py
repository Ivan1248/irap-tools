"""Confidence intervals of methods and of their differences.

Each resample draws road sequences with replacement, recomputes the metrics of each run from the
per-sequence statistics (`EvaluationResult.statistics`), and adds the uncertainty of the mean
over the runs with a Student-t draw. `docs/evaluation.md#confidence-intervals` defines them.
"""

import dataclasses as dc
import typing as T

import numpy as np

from .evaluation import EvaluationResult, check_runs_distinct
from .metrics import MetricValues, compute_grouped_metrics, map_metric_values


@dc.dataclass(frozen=True)
class BootstrapInterval:
    """A bootstrap confidence interval.

    Attributes:
        num_undefined: The number of resamples whose value is NaN, which the bounds leave out,
            e.g. resamples with an undefined MCC, or with infinite values of both compared
            methods. The bounds are NaN if every resample is undefined.
    """

    low: float
    high: float
    num_undefined: int


@dc.dataclass(frozen=True)
class MetricDifference:
    """The difference `a - b` of a metric of two methods on the same evaluation set.

    Attributes:
        difference: The difference of the means over the runs of each method.
        interval: Its bootstrap confidence interval.
    """

    difference: float
    interval: BootstrapInterval


def draw_group_weights(num_groups: int, num_resamples: int,
                       rng: np.random.Generator) -> np.ndarray:
    """(num_resamples, num_groups) int64: how often each resample draws each group.

    Raises:
        ValueError: If there are no groups, e.g. for an empty evaluation set, or no resamples.
    """
    if num_groups < 1:
        raise ValueError("The evaluation set has no road sequences to resample.")
    if num_resamples < 1:
        raise ValueError(f"At least one resample is required, got {num_resamples}.")
    return rng.multinomial(num_groups, np.full(num_groups, 1 / num_groups), size=num_resamples)


def _draw_run_t_values(num_runs: int, num_resamples: int,
                       rng: np.random.Generator) -> np.ndarray | None:
    """(num_resamples,) Student-t(num_runs - 1) values, or None for a single run."""
    return None if num_runs == 1 else rng.standard_t(num_runs - 1, size=num_resamples)


def _compute_quantile(sorted_values: np.ndarray, quantile: float) -> float:
    """The quantile of non-empty sorted values with linear interpolation, like `np.quantile`.

    Unlike `np.quantile`, which computes inf - inf = NaN, it interpolates between equal infinite
    values to that value, and between an infinite and a finite value to the infinite one.
    """
    position = (len(sorted_values) - 1) * quantile
    index = int(np.floor(position))
    fraction = position - index
    low = float(sorted_values[index])
    high = float(sorted_values[min(index + 1, len(sorted_values) - 1)])
    if fraction == 0 or low == high:
        return low
    return (1 - fraction) * low + fraction * high


def _compute_interval(values: np.ndarray, confidence: float) -> BootstrapInterval:
    values = np.asarray(values, dtype=np.float64)
    is_undefined = np.isnan(values)
    num_undefined = int(is_undefined.sum())
    values = np.sort(values[~is_undefined])
    if values.size == 0:
        return BootstrapInterval(float("nan"), float("nan"), num_undefined)
    tail = (1 - confidence) / 2
    return BootstrapInterval(_compute_quantile(values, tail), _compute_quantile(values, 1 - tail),
                    num_undefined)


def _check_confidence(confidence: float) -> None:
    if not 0 <= confidence <= 1:
        raise ValueError(f"The confidence must be in [0, 1], got {confidence}.")


def _check_comparable(result_a: EvaluationResult, result_b: EvaluationResult) -> None:
    set_a, set_b = result_a.evaluation_set, result_b.evaluation_set
    if (set_a.segment_ids, set_a.segment_sequence_ids) != \
            (set_b.segment_ids, set_b.segment_sequence_ids):
        raise ValueError(f"{result_a.run_label!r} and {result_b.run_label!r} are on different"
                         f" segments. Compare them on the reference set, which is the same for"
                         f" every method.")
    if result_a.attributes != result_b.attributes:
        raise ValueError(f"{result_a.run_label!r} and {result_b.run_label!r} score different"
                         f" attributes:"
                         f" {sorted(set(result_a.attributes) ^ set(result_b.attributes))}.")
    if result_a.null_policy != result_b.null_policy:
        raise ValueError(f"{result_a.run_label!r} and {result_b.run_label!r} use different null"
                         f" policies: {result_a.null_policy!r} and {result_b.null_policy!r}.")
    if result_a.null_policy == "exclude" and any(
            n > 0 for r in (result_a, result_b) for n in r.num_invalid.values()):
        raise ValueError("With null_policy='exclude', the invalid cells of each run are left"
                         " out, so the runs are scored on different segments. Use"
                         " null_policy='first_class'.")


def _check_runs(runs: T.Sequence[EvaluationResult]) -> None:
    """Checks that `runs` are distinct runs of one method, comparable with each other and with
    the same metrics."""
    if not runs:
        raise ValueError("A method needs at least one run.")
    if len(names := {r.method.name for r in runs}) > 1:
        raise ValueError(f"The runs are of different methods: {sorted(names)}.")
    check_runs_distinct(r.method for r in runs)
    for run in runs[1:]:
        _check_comparable(runs[0], run)
        if run.metrics.names != runs[0].metrics.names:
            raise ValueError(f"The runs {runs[0].run_label!r} and {run.run_label!r} have"
                             f" different metrics.")


def _compute_mean(*values: float) -> float:
    """The mean, or the value itself if all values are equal, which keeps ints (e.g. `n`) and
    avoids rounding."""
    if all(v == values[0] for v in values):
        return values[0]
    return float(np.mean(values))


def compute_method_metrics(runs: T.Sequence[EvaluationResult]) -> MetricValues[float]:
    """The metrics of a method: the mean of each metric over its runs.

    Raises:
        ValueError: If `runs` is empty, the runs are not distinct runs of one method with the
            same metrics, or two runs are not comparable: they differ in segments, sequences,
            attributes or null policy, or use `null_policy='exclude'` with invalid cells, which
            scores them on different segments.
    """
    _check_runs(runs)
    return map_metric_values(_compute_mean, *(r.metrics for r in runs))


def _resample_method_metrics(
    runs: T.Sequence[EvaluationResult],
    metric_names: T.Sequence[str],
    group_weights: np.ndarray,
    run_t_values: np.ndarray | None,
) -> MetricValues[np.ndarray]:
    """The resampled values of a method: (B,) arrays for the (B, G) `group_weights`.

    Each value is the mean over the runs plus their standard deviation over sqrt(R) times
    `run_t_values`, or just the value of the run for a single run. A value whose mean is
    infinite is that mean.
    """
    per_run = [compute_grouped_metrics(r.statistics, metric_names, group_weights) for r in runs]

    def combine(*run_values) -> np.ndarray:
        values = np.stack([np.asarray(v, dtype=np.float64) for v in run_values])  # (R, B)
        mean = values.mean(0)
        if run_t_values is None:
            return mean
        with np.errstate(invalid="ignore"):  # the deviation of infinite values is NaN
            uncertainty = values.std(0, ddof=1) / np.sqrt(len(values)) * run_t_values
        return np.where(np.isinf(mean), mean, mean + uncertainty)

    return map_metric_values(combine, *per_run)


def compute_bootstrap_intervals(
    runs: T.Sequence[EvaluationResult],
    *,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> MetricValues[BootstrapInterval]:
    """Confidence intervals of the metrics of a method (`compute_method_metrics`).

    Args:
        runs: The runs of the method. With a single run, the intervals cover only the sampling
            of road sequences, and can be attached to it with
            `dataclasses.replace(result, intervals=...)`.

    Raises:
        ValueError: If the evaluation set is empty, `num_resamples` < 1, `confidence` is not in
            [0, 1], or `runs` are not valid for `compute_method_metrics`.
    """
    _check_confidence(confidence)
    _check_runs(runs)
    rng = np.random.default_rng(seed)
    weights = draw_group_weights(len(runs[0].evaluation_set.sequence_ids), num_resamples, rng)
    resampled = _resample_method_metrics(runs, runs[0].metrics.names, weights,
                                         _draw_run_t_values(len(runs), num_resamples, rng))
    return map_metric_values(lambda v: _compute_interval(v, confidence), resampled)


def compare_methods(
    runs_a: T.Sequence[EvaluationResult],
    runs_b: T.Sequence[EvaluationResult],
    metric_names: T.Sequence[str] | None = None,
    *,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> MetricValues[MetricDifference]:
    """Compares two methods on the same resampled road sequences.

    Args:
        metric_names: Metrics of both methods, attribute averages or per-attribute metrics.
            None means the attribute averages that both have.

    Raises:
        ValueError: If the runs of a method are not valid for `compute_method_metrics`, the
            runs of a are not comparable with those of b in the same sense, a method lacks a
            metric, `num_resamples` < 1 or `confidence` is not in [0, 1].
    """
    _check_confidence(confidence)
    _check_runs(runs_a)
    _check_runs(runs_b)
    _check_comparable(runs_a[0], runs_b[0])
    metrics_a, metrics_b = runs_a[0].metrics, runs_b[0].metrics
    if metric_names is None:
        metric_names = [n for n in metrics_a.averages if n in metrics_b.averages]
    common_names = set(metrics_a.names) & set(metrics_b.names)
    if missing := [n for n in metric_names if n not in common_names]:
        raise ValueError(f"Metrics not in both methods: {missing}. Method a has"
                         f" {list(metrics_a.names)}.")
    rng = np.random.default_rng(seed)
    weights = draw_group_weights(len(runs_a[0].evaluation_set.sequence_ids), num_resamples, rng)
    resampled_a, resampled_b = (
        _resample_method_metrics(runs, metric_names, weights,
                                 _draw_run_t_values(len(runs), num_resamples, rng))
        for runs in (runs_a, runs_b))

    def get_difference(resampled_value_a, resampled_value_b, point_a, point_b
                       ) -> MetricDifference:
        return MetricDifference(float(point_a - point_b),
                                _compute_interval(resampled_value_a - resampled_value_b,
                                                  confidence))

    return map_metric_values(get_difference, resampled_a, resampled_b,
                             compute_method_metrics(runs_a), compute_method_metrics(runs_b))
