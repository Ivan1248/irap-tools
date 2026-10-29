"""Classification metrics of the iRAP evaluation protocol, computed from sufficient statistics.

The definitions match `vidlu_irap_gaim.metrics.get_irap_metrics` (`vidlu.metrics` with
`ignore_missing_classes=True`), up to float32 rounding, since vidlu computes some metrics in
float32 and this module computes all of them in float64.

Statistics and metrics can carry leading batch dimensions, e.g. one entry per road sequence or
per bootstrap resample, so that many evaluations are computed in one vectorized call.

`metric_names` defines the `[a]<base>[_suppN]` form of metric names.
"""

import dataclasses as dc
import typing as T

import numpy as np

from irap_data.metadata import IGNORE_LABEL_INDEX

from .metric_names import (
    IRAP_IGNORE_MISSING_CLASSES,
    MACRO_METRIC_TO_PER_CLASS,
    PER_CLASS_METRICS,
    PROBABILISTIC_METRICS,
    SUPPORT_ONLY_METRICS,
    ParsedMetricName,
    parse_metric_name,
)

V = T.TypeVar("V")
U = T.TypeVar("U")


# Metric values ####################################################################################

@dc.dataclass(frozen=True)
class MetricValues(T.Generic[V]):
    """Values of metrics of several attributes, e.g. scores, intervals or differences.

    Attributes:
        averages: Attribute-average name (`a` prefix) -> value.
        per_attribute: Per-attribute name -> attribute -> value.
    """

    averages: T.Mapping[str, V]
    per_attribute: T.Mapping[str, T.Mapping[str, V]]

    @property
    def names(self) -> tuple[str, ...]:
        return (*self.averages, *self.per_attribute)

    def iterate_values(self) -> T.Iterator[V]:
        """Iterates over the values of the averages, then those of each per-attribute metric and
        attribute, in their order."""
        yield from self.averages.values()
        for attr_to_value in self.per_attribute.values():
            yield from attr_to_value.values()


def map_metric_values(f: T.Callable[..., U], *values: MetricValues) -> MetricValues[U]:
    """Applies `f` to the corresponding values of each argument, for the names and attributes of
    the first one."""
    first = values[0]
    return MetricValues(
        averages={name: f(*(v.averages[name] for v in values)) for name in first.averages},
        per_attribute={name: {attr: f(*(v.per_attribute[name][attr] for v in values))
                              for attr in attr_to_value}
                       for name, attr_to_value in first.per_attribute.items()})


# Sufficient statistics ############################################################################

@dc.dataclass(frozen=True)
class ClassificationStatistics:
    """Sufficient statistics of the classification results of one attribute.

    All arrays share the leading batch dimensions `...`.

    Attributes:
        confusion_matrix: (..., K, K) int64 example counts, rows = ground truth, columns = the
            predicted class.
        nll_sums: (..., K) float64 sums of the negative log-likelihood `-log p[target]` per
            ground-truth class, or None for hard predictions.
        brier_sums: (..., K) float64 sums of the Brier score `sum_k (p_k - 1[k == target])^2` per
            ground-truth class, or None for hard predictions.
    """

    confusion_matrix: np.ndarray
    nll_sums: np.ndarray | None = None
    brier_sums: np.ndarray | None = None

    def __post_init__(self):
        if (self.nll_sums is None) != (self.brier_sums is None):
            raise ValueError("nll_sums and brier_sums must both be given or both be None.")

    @property
    def has_probabilities(self) -> bool:
        return self.nll_sums is not None

    @property
    def support(self) -> np.ndarray:
        """(..., K) number of ground-truth examples per class."""
        return self.confusion_matrix.sum(-1)


def _check_class_indices(indices: np.ndarray, name: str, num_classes: int) -> None:
    if len(indices) and (indices.min() < 0 or indices.max() >= num_classes):
        raise ValueError(f"{name} of labeled examples must be class indices in"
                         f" [0, {num_classes}).")


def compute_classification_statistics(
    targets: np.ndarray,
    predicted_indices: np.ndarray,
    num_classes: int,
    *,
    probs: np.ndarray | None = None,
    group_indices: np.ndarray | None = None,
    num_groups: int | None = None,
) -> ClassificationStatistics:
    """Computes the statistics of one attribute, optionally per group of examples.

    Args:
        targets: (N,) class indices, with `IGNORE_LABEL_INDEX` for examples to leave out.
        predicted_indices: (N,) predicted class indices, from which the confusion matrix is
            counted. With probabilities, usually their argmax (`predictions.to_class_indices`).
        num_classes: K, the number of classes.
        probs: (N, K) predicted distributions, from which the statistics of the probabilistic
            metrics are summed, or None for hard predictions. They need not agree with
            `predicted_indices`, e.g. for invalid predictions (see `evaluation.NullPolicy`).
        group_indices: (N,) group index in [0, num_groups) of each example, or None.
        num_groups: The number of groups, required with `group_indices`.

    Returns:
        Statistics without batch dimensions, or with a leading (num_groups,) dimension if
        `group_indices` is given.
    """
    if probs is not None and probs.shape != (len(targets), num_classes):
        raise ValueError(f"probs must have the shape {(len(targets), num_classes)}, got"
                         f" {probs.shape}.")
    is_labeled = targets != IGNORE_LABEL_INDEX
    targets, predicted_indices = targets[is_labeled], predicted_indices[is_labeled]
    if group_indices is None:
        group_indices, num_groups, batch_shape = np.zeros(len(targets), np.int64), 1, ()
    else:
        if num_groups is None:
            raise ValueError("num_groups is required with group_indices.")
        group_indices, batch_shape = group_indices[is_labeled], (num_groups,)
    _check_class_indices(targets, "Targets", num_classes)
    _check_class_indices(predicted_indices, "Predicted indices", num_classes)

    group_class = group_indices * num_classes + targets
    confusion_matrix = np.bincount(group_class * num_classes + predicted_indices,
                                   minlength=num_groups * num_classes ** 2)
    confusion_matrix = confusion_matrix.reshape(*batch_shape, num_classes, num_classes)
    if probs is None:
        return ClassificationStatistics(confusion_matrix)

    probs = probs[is_labeled].astype(np.float64)
    target_probs = probs[np.arange(len(targets)), targets]
    with np.errstate(divide="ignore"):
        nll = -np.log(target_probs)
    # sum_k (p_k - 1[k == target])^2 without the (N, K) one-hot matrix.
    brier = np.einsum("nk,nk->n", probs, probs) - 2 * target_probs + 1

    def sum_per_group_class(values):
        sums = np.bincount(group_class, weights=values, minlength=num_groups * num_classes)
        return sums.reshape(*batch_shape, num_classes)

    return ClassificationStatistics(confusion_matrix, sum_per_group_class(nll),
                                    sum_per_group_class(brier))


def sum_statistics(statistics: ClassificationStatistics,
                   group_weights: np.ndarray | None = None) -> ClassificationStatistics:
    """Sums grouped statistics over their first batch dimension (the groups).

    Args:
        statistics: Statistics with a leading (G,) group dimension.
        group_weights: (..., G) multiplicity of each group, e.g. how often a bootstrap resample
            draws it, or None to sum each group once.

    Returns:
        Statistics with the batch dimensions `...` of `group_weights`, or without batch
        dimensions if it is None. A group with weight 0 adds nothing, also if its sum is
        infinite, e.g. an NLL sum with a label of probability 0.
    """
    # In float64, which numpy multiplies with BLAS, unlike int64. Counts below 2^53 are exact.
    weights = None if group_weights is None else np.asarray(group_weights, dtype=np.float64)

    def weighted_sum(x):
        if x is None:
            return None
        if weights is None:
            return x.sum(0)
        x_float = x.astype(np.float64, copy=False)
        # Only NLL sums can be infinite, and only +inf, since -log p >= 0 for p <= 1.
        is_infinite = x_float == np.inf
        if not is_infinite.any():
            weighted = np.tensordot(weights, x_float, axes=(-1, 0))
            return weighted.astype(x.dtype) if np.issubdtype(x.dtype, np.integer) else weighted
        # 0 * inf is NaN, so the infinite sums are added only where their group is drawn.
        weighted = np.tensordot(weights, np.where(is_infinite, 0.0, x_float), axes=(-1, 0))
        num_drawn_infinite = np.tensordot((weights > 0).astype(np.float64),
                                          is_infinite.astype(np.float64), axes=(-1, 0))
        return weighted + np.where(num_drawn_infinite > 0, np.inf, 0.0)

    return ClassificationStatistics(weighted_sum(statistics.confusion_matrix),
                                    weighted_sum(statistics.nll_sums),
                                    weighted_sum(statistics.brier_sums))


# Metrics ##########################################################################################

def _divide(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.true_divide(numerator, denominator)


def _compute_masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over the last axis of the entries where `mask` is True, NaN if there are none."""
    return _divide(np.where(mask, values, 0).sum(-1), mask.sum(-1))


def _compute_mean_over_defined(values: np.ndarray) -> np.ndarray:
    """Mean over the last axis of the entries that are not NaN, NaN if there are none."""
    return _compute_masked_mean(values, ~np.isnan(values))


def _compute_attribute_averages(
        average_names: T.Iterable[str],
        per_attribute: T.Mapping[str, T.Mapping[str, T.Any]]) -> dict[str, np.ndarray]:
    """Attribute averages, e.g. 'amF1', from `per_attribute`: metric name -> attribute -> value,
    arrays of the same shape. An average is the mean over the attributes where its metric is
    defined, including infinite values."""
    def compute_average(name: str) -> np.ndarray:
        values = per_attribute[parse_metric_name(name).name].values()
        return _compute_mean_over_defined(
            np.stack([np.asarray(v, dtype=np.float64) for v in values], -1))

    return {name: compute_average(name) for name in average_names}


def _compute_confusion_matrix_metrics(confusion_matrix: np.ndarray,
                                      ignore_missing_classes: bool) -> dict[str, np.ndarray]:
    true_positives = np.diagonal(confusion_matrix, axis1=-2, axis2=-1)
    num_actual = confusion_matrix.sum(-1)
    num_predicted = confusion_matrix.sum(-2)
    false_positives = num_predicted - true_positives
    precision = np.nan_to_num(_divide(true_positives, num_predicted))
    recall = np.nan_to_num(_divide(true_positives, num_actual))
    per_class = dict(
        P=precision, R=recall,
        F1=np.nan_to_num(_divide(2 * precision * recall, precision + recall)),
        IoU=np.nan_to_num(_divide(true_positives, num_actual + false_positives)))
    # A class that is neither in the ground truth nor predicted is left out of the means.
    mask = (num_actual + false_positives > 0) if ignore_missing_classes \
        else np.ones_like(num_actual, dtype=bool)
    metrics = {**per_class,
               **{"m" + k: _compute_masked_mean(v, mask) for k, v in per_class.items()}}

    n = num_predicted.sum(-1)
    num_correct = true_positives.sum(-1)
    metrics["n"] = n
    metrics["A"] = _divide(num_correct, n)
    # Chance-corrected agreement. MCC is NaN (0/0) when every ground-truth or every predicted
    # label is one class, kappa when every ground-truth and predicted label is the same class.
    n, num_correct = n.astype(np.float64), num_correct.astype(np.float64)
    num_predicted, num_actual = num_predicted.astype(np.float64), num_actual.astype(np.float64)
    agreement_by_chance = (num_predicted * num_actual).sum(-1)
    numerator = num_correct * n - agreement_by_chance
    metrics["kappa"] = _divide(numerator, n ** 2 - agreement_by_chance)
    metrics["MCC"] = _divide(numerator, np.sqrt((n ** 2 - (num_predicted ** 2).sum(-1))
                                                * (n ** 2 - (num_actual ** 2).sum(-1))))
    return metrics


def _compute_probabilistic_metrics(statistics: ClassificationStatistics) -> dict[str, np.ndarray]:
    support = statistics.support
    n = support.sum(-1)
    class_nll = _divide(statistics.nll_sums, support)  # NaN for a class without examples
    class_brier = _divide(statistics.brier_sums, support)
    return dict(NLL=_divide(statistics.nll_sums.sum(-1), n),
                Brier=_divide(statistics.brier_sums.sum(-1), n),
                cNLL=class_nll, cBrier=class_brier,
                mNLL=_compute_mean_over_defined(class_nll),
                mBrier=_compute_mean_over_defined(class_brier))


def compute_classification_metrics(
    statistics: ClassificationStatistics,
    metric_names: T.Sequence[str],
    ignore_missing_classes: bool = IRAP_IGNORE_MISSING_CLASSES,
) -> dict[str, np.ndarray]:
    """Computes per-attribute metrics (names without the `a` prefix) from the statistics.

    Args:
        statistics: Statistics with batch dimensions `...`.
        metric_names: The names to compute.
        ignore_missing_classes: Whether means over classes leave out the classes that are
            neither in the ground truth nor predicted (the iRAP protocol), rather than counting
            them as 0.

    Returns:
        Name -> an array of shape `...`, or `(..., K)` for a per-class metric.

    Raises:
        ValueError: For an averaged name, or a probabilistic metric of hard predictions.
    """
    parsed = [parse_metric_name(name) for name in metric_names]
    if averaged := [p.name for p in parsed if p.is_averaged]:
        raise ValueError(f"Attribute averages {averaged} need several attributes. See"
                         f" compute_multi_attribute_metrics.")
    probabilistic_names = [p.name for p in parsed if p.base in PROBABILISTIC_METRICS]
    if probabilistic_names and not statistics.has_probabilities:
        raise ValueError(f"Metrics {probabilistic_names} need predicted distributions, but the"
                         f" predictions are hard.")
    base_metrics = _compute_confusion_matrix_metrics(statistics.confusion_matrix,
                                                     ignore_missing_classes)
    if probabilistic_names:
        base_metrics.update(_compute_probabilistic_metrics(statistics))
    support = statistics.support

    def select(p: ParsedMetricName) -> np.ndarray:
        if p.min_support is None:
            return base_metrics[p.base]
        is_supported = support >= p.min_support
        if p.base in SUPPORT_ONLY_METRICS:
            return is_supported.sum(-1)
        return _compute_masked_mean(base_metrics[MACRO_METRIC_TO_PER_CLASS[p.base]], is_supported)

    return {p.name: select(p) for p in parsed}


def compute_multi_attribute_metrics(
    attribute_to_statistics: T.Mapping[str, ClassificationStatistics],
    metric_names: T.Sequence[str],
) -> MetricValues[np.ndarray]:
    """Computes metrics of several attributes, including averages over attributes.

    Means over classes are those of the iRAP protocol (`IRAP_IGNORE_MISSING_CLASSES`).

    Args:
        attribute_to_statistics: Statistics of each attribute, all with the same batch
            dimensions `...`.
        metric_names: `[a]<base>[_suppN]` names.

    Returns:
        Arrays of shape `...`. An attribute average is the mean over the attributes where the
        metric is defined, NaN if it is defined for none.

    Raises:
        ValueError: If there is no attribute, or for the average of a per-class metric.
    """
    if not attribute_to_statistics:
        raise ValueError("At least one attribute is required.")
    parsed = {name: parse_metric_name(name) for name in metric_names}
    if per_class := [n for n, p in parsed.items() if p.is_averaged and p.base in PER_CLASS_METRICS]:
        raise ValueError(f"Cannot average per-class metrics {per_class} over attributes.")
    per_attribute_names = list(dict.fromkeys(p.name for p in parsed.values()))
    attribute_to_metrics = {
        attr: compute_classification_metrics(stats, per_attribute_names)
        for attr, stats in attribute_to_statistics.items()}

    per_attribute = {name: {attr: metrics[name] for attr, metrics in attribute_to_metrics.items()}
                     for name in per_attribute_names}
    return MetricValues(
        averages=_compute_attribute_averages(
            [name for name, p in parsed.items() if p.is_averaged], per_attribute),
        per_attribute={name: per_attribute[name] for name, p in parsed.items()
                       if not p.is_averaged})


def compute_grouped_metrics(
    statistics: T.Mapping[str, ClassificationStatistics],
    metric_names: T.Sequence[str],
    group_weights: np.ndarray | None = None,
) -> MetricValues[np.ndarray]:
    """Computes metrics from per-group statistics.

    Args:
        statistics: Attribute -> statistics with a leading (G,) group axis.
        metric_names: See `compute_multi_attribute_metrics`.
        group_weights: (..., G) multiplicity of each group, or None to count each group once.

    Returns:
        As `compute_multi_attribute_metrics`, with batch dimensions `...`.
    """
    return compute_multi_attribute_metrics(
        {a: sum_statistics(s, group_weights) for a, s in statistics.items()}, metric_names)


def select_metric_names(values: MetricValues[V], names: T.Container[str]) -> MetricValues[V]:
    """Selects the values of the metrics in `names`."""
    return MetricValues(
        averages={n: v for n, v in values.averages.items() if n in names},
        per_attribute={n: v for n, v in values.per_attribute.items() if n in names})


def select_metric_attributes(values: MetricValues[V],
                             attributes: T.Collection[str]) -> MetricValues[V]:
    """The metric values of a subset of the attributes, with the attribute averages computed
    again over the subset, as `compute_multi_attribute_metrics` computes them.

    The values are floats, or arrays of the same shape, e.g. resampled values
    (`bootstrap.resample_model_metrics`), whose averages are computed elementwise. They must
    have the per-attribute value of each average, e.g. 'mF1' for 'amF1' (see
    `metric_names.add_per_attribute_metric_names`). The attributes keep the order of `values`, so
    that the averages equal those of scoring only the subset.

    Raises:
        ValueError: If `attributes` is empty or has attributes without values, or an average
            lacks its per-attribute values.
    """
    if not attributes:
        raise ValueError("At least one attribute is required.")
    if missing := [n for n in values.averages
                   if parse_metric_name(n).name not in values.per_attribute]:
        raise ValueError(f"No per-attribute values of the averages {missing}. Add them with"
                         f" add_per_attribute_metric_names.")
    if unknown := sorted(a for a in attributes
                         if any(a not in v for v in values.per_attribute.values())):
        raise ValueError(f"Attributes without values of every metric: {unknown}.")
    per_attribute = {name: {a: v for a, v in attr_to_value.items() if a in attributes}
                     for name, attr_to_value in values.per_attribute.items()}
    return MetricValues(
        averages={name: average.item() if average.ndim == 0 else average for name, average
                  in _compute_attribute_averages(values.averages, per_attribute).items()},
        per_attribute=per_attribute)
