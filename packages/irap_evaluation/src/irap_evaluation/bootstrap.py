"""Confidence intervals of methods and of their differences.

Each resample draws road sequences with replacement, recomputes the metrics of each model from the
per-sequence statistics (`EvaluationResult.statistics`), and adds the uncertainty of the mean
over the models with a Student-t draw. `docs/evaluation.md#confidence-intervals` defines them.
"""

import dataclasses as dc
import math
import typing as T

import numpy as np

from .evaluation import EvaluationResult, check_method_models
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

    @property
    def is_partially_undefined(self) -> bool:
        """Whether some but not all resamples are undefined. The bounds then hold only for the
        resamples where the value is defined, unlike NaN bounds, which show that every resample
        is undefined."""
        return self.num_undefined > 0 and not math.isnan(self.low)


@dc.dataclass(frozen=True)
class MetricDifference:
    """The difference `a - b` of a metric of two methods on the same evaluation set.

    Attributes:
        difference: The difference of the means over the models of each method.
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


def _draw_model_t_values(num_models: int, num_resamples: int,
                         rng: np.random.Generator) -> np.ndarray | None:
    """(num_resamples,) Student-t(num_models - 1) values, or None for a single model."""
    return None if num_models == 1 else rng.standard_t(num_models - 1, size=num_resamples)


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
    label_a, label_b = result_a.model.label, result_b.model.label
    if (set_a.segment_ids, set_a.segment_sequence_ids) != \
            (set_b.segment_ids, set_b.segment_sequence_ids):
        raise ValueError(f"{label_a!r} and {label_b!r} are on different segments. Compare them on"
                         f" the reference set, which is the same for every method.")
    if result_a.attributes != result_b.attributes:
        raise ValueError(f"{label_a!r} and {label_b!r} score different attributes:"
                         f" {sorted(set(result_a.attributes) ^ set(result_b.attributes))}.")
    if result_a.null_policy != result_b.null_policy:
        raise ValueError(f"{label_a!r} and {label_b!r} use different null policies:"
                         f" {result_a.null_policy!r} and {result_b.null_policy!r}.")
    if result_a.null_policy == "exclude" and any(
            n > 0 for r in (result_a, result_b) for n in r.num_invalid.values()):
        raise ValueError("With null_policy='exclude', the invalid cells of each model are left"
                         " out, so the models are scored on different segments. Use"
                         " null_policy='first_class'.")


def _have_same_metrics(values_a: MetricValues, values_b: MetricValues) -> bool:
    """Whether the values are of the same metrics and, per metric, of the same attributes."""
    return values_a.names == values_b.names and all(
        values_a.per_attribute[n].keys() == values_b.per_attribute[n].keys()
        for n in values_a.per_attribute)


def _check_models(models: T.Sequence[EvaluationResult]) -> None:
    """Checks that `models` are results of the models of one method (see `check_method_models`),
    comparable with each other and with the same metrics."""
    if not models:
        raise ValueError("A method needs at least one model.")
    if len(names := {r.model.method_name for r in models}) > 1:
        raise ValueError(f"The models are of different methods: {sorted(names)}.")
    check_method_models(r.model for r in models)
    for model in models[1:]:
        _check_comparable(models[0], model)
        if not _have_same_metrics(model.metrics, models[0].metrics):
            raise ValueError(f"The models {models[0].model.label!r} and {model.model.label!r}"
                             f" have different metrics.")


def _compute_mean(*values: float) -> float:
    """The mean, or the value itself if all values are equal, which keeps ints (e.g. `n`) and
    avoids rounding."""
    if all(v == values[0] for v in values):
        return values[0]
    return float(np.mean(values))


def compute_mean_metrics(model_metrics: T.Sequence[MetricValues[float]]) -> MetricValues[float]:
    """Computes the mean of each metric over models, e.g. of their metrics over a subset of the
    attributes (`metrics.select_metric_attributes`). `compute_method_metrics` also checks the
    models.

    Raises:
        ValueError: If `model_metrics` is empty, or the models differ in metrics or attributes.
    """
    if not model_metrics:
        raise ValueError("At least one model is required.")
    if not all(_have_same_metrics(v, model_metrics[0]) for v in model_metrics[1:]):
        raise ValueError("The models differ in metrics or attributes.")
    return map_metric_values(_compute_mean, *model_metrics)


def compute_method_metrics(models: T.Sequence[EvaluationResult]) -> MetricValues[float]:
    """Computes the metrics of a method: the mean of each metric over its models.

    Args:
        models: The results of the models of the method.

    Raises:
        ValueError: If `models` is empty, the results are not of the models of one method that
            `check_method_models` accepts, with the same metrics, or two models are not
            comparable: they differ in segments, sequences, attributes or null policy, or use
            `null_policy='exclude'` with invalid cells, which scores them on different segments.
    """
    _check_models(models)
    return compute_mean_metrics([r.metrics for r in models])


def _resample_method_metrics(
    models: T.Sequence[EvaluationResult],
    metric_names: T.Sequence[str],
    group_weights: np.ndarray,
    model_t_values: np.ndarray | None,
) -> MetricValues[np.ndarray]:
    """Computes the resampled values of a method: (B,) arrays for the (B, G) `group_weights`.

    Each value is the mean over the models plus their standard deviation over sqrt(M) times
    `model_t_values`, or just the value of the model for a single model. A value whose mean is
    infinite is that mean.
    """
    per_model = [compute_grouped_metrics(r.statistics, metric_names, group_weights)
                 for r in models]

    def combine(*model_values) -> np.ndarray:
        values = np.stack([np.asarray(v, dtype=np.float64) for v in model_values])  # (M, B)
        mean = values.mean(0)
        if model_t_values is None:
            return mean
        with np.errstate(invalid="ignore"):  # the deviation of infinite values is NaN
            uncertainty = values.std(0, ddof=1) / np.sqrt(len(values)) * model_t_values
        return np.where(np.isinf(mean), mean, mean + uncertainty)

    return map_metric_values(combine, *per_model)


def compute_bootstrap_intervals(
    models: T.Sequence[EvaluationResult],
    *,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> MetricValues[BootstrapInterval]:
    """Computes confidence intervals of the metrics of a method (`compute_method_metrics`).

    Args:
        models: The results of the models of the method. With a single model, the intervals
            cover only the sampling of road sequences, and can be attached to it with
            `dataclasses.replace(result, intervals=...)`.

    Raises:
        ValueError: If the evaluation set is empty, `num_resamples` < 1, `confidence` is not in
            [0, 1], or `models` are not valid for `compute_method_metrics`.
    """
    _check_confidence(confidence)
    _check_models(models)
    rng = np.random.default_rng(seed)
    weights = draw_group_weights(len(models[0].evaluation_set.sequence_ids), num_resamples, rng)
    resampled = _resample_method_metrics(models, models[0].metrics.names, weights,
                                         _draw_model_t_values(len(models), num_resamples, rng))
    return map_metric_values(lambda v: _compute_interval(v, confidence), resampled)


def compare_methods(
    models_a: T.Sequence[EvaluationResult],
    models_b: T.Sequence[EvaluationResult],
    metric_names: T.Sequence[str] | None = None,
    *,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> MetricValues[MetricDifference]:
    """Compares two methods on the same resampled road sequences.

    Args:
        models_a: The results of the models of method a.
        models_b: The results of the models of method b.
        metric_names: Metrics of both methods, attribute averages or per-attribute metrics.
            None means the attribute averages that both have.

    Raises:
        ValueError: If the models of a method are not valid for `compute_method_metrics`, the
            models of a are not comparable with those of b in the same sense, a method lacks a
            metric, `num_resamples` < 1 or `confidence` is not in [0, 1].
    """
    _check_confidence(confidence)
    _check_models(models_a)
    _check_models(models_b)
    _check_comparable(models_a[0], models_b[0])
    metrics_a, metrics_b = models_a[0].metrics, models_b[0].metrics
    if metric_names is None:
        metric_names = [n for n in metrics_a.averages if n in metrics_b.averages]
    common_names = set(metrics_a.names) & set(metrics_b.names)
    if missing := [n for n in metric_names if n not in common_names]:
        raise ValueError(f"Metrics not in both methods: {missing}. Method a has"
                         f" {list(metrics_a.names)}.")
    rng = np.random.default_rng(seed)
    weights = draw_group_weights(len(models_a[0].evaluation_set.sequence_ids), num_resamples, rng)
    resampled_a, resampled_b = (
        _resample_method_metrics(models, metric_names, weights,
                                 _draw_model_t_values(len(models), num_resamples, rng))
        for models in (models_a, models_b))

    def get_difference(resampled_value_a, resampled_value_b, point_a, point_b
                       ) -> MetricDifference:
        return MetricDifference(float(point_a - point_b),
                                _compute_interval(resampled_value_a - resampled_value_b,
                                                  confidence))

    return map_metric_values(get_difference, resampled_a, resampled_b,
                             compute_method_metrics(models_a), compute_method_metrics(models_b))
