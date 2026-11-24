"""Metric names of the iRAP evaluation protocol: their form, parsing and descriptions.

Metric names have the form `[a]<base>[_suppN][_{<condition>}]`:

- `<base>` is a per-attribute metric (see `CONFUSION_MATRIX_METRICS`, `PROBABILISTIC_METRICS`
  and `MANUAL_REVIEW_METRICS`). An aggregation of a per-class metric over classes has a prefix
  that names the aggregation (`ClassAggregation`), e.g. `mF1` (mean) and `minR` (minimum). `nc`
  is the number of classes, and the per-class `supp` the number of ground-truth examples of a
  class (its support).
- The `a` prefix averages it over attributes, leaving out attributes where it is undefined (NaN).
- The suffixes restrict the classes of an aggregation over classes or of `nc`
  (`CLASS_SELECTION_METRICS`): `_suppN` to the classes with at least N ≥ 1 ground-truth
  examples, and `_{<condition>}` to the classes that meet a condition on a per-class metric
  (`ClassCondition`), e.g. `nc_{R>0}` and `mF1_supp1_{supp<10}`.
"""

import re
import typing as T

from .predictions import OutputKind

# TODO: Consider dropping `_suppN`, which equals the class condition `_{supp>=N}`, if a name can
# have several class conditions.
SUPPORT_SUFFIX = "_supp"

#: Aggregations of a per-class metric over classes, by the prefix of their names: 'm' the mean,
#: 'min' the minimum and 'max' the maximum, e.g. 'mF1', 'minR' and 'maxNLL'. All aggregations of
#: a per-class metric are over the same classes (see `describe_metric`).
ClassAggregation = T.Literal["m", "min", "max"]
#: The name of a per-class metric in the names of its aggregations -> the per-class metric.
_CONFUSION_MATRIX_CLASS_METRICS = {"P": "P", "R": "R", "F1": "F1", "IoU": "IoU"}
_PROBABILISTIC_CLASS_METRICS = {"NLL": "cNLL", "Brier": "cBrier"}


def _make_class_aggregate_metrics(
        name_to_per_class: T.Mapping[str, str]) -> dict[str, tuple[ClassAggregation, str]]:
    return {aggregation + name: (aggregation, per_class)
            for aggregation in T.get_args(ClassAggregation)
            for name, per_class in name_to_per_class.items()}


_CONFUSION_MATRIX_AGGREGATE_METRICS = _make_class_aggregate_metrics(
    _CONFUSION_MATRIX_CLASS_METRICS)
_PROBABILISTIC_AGGREGATE_METRICS = _make_class_aggregate_metrics(_PROBABILISTIC_CLASS_METRICS)
#: Metric aggregated over classes -> its aggregation and per-class metric, e.g.
#: 'minR' -> ('min', 'R') and 'mNLL' -> ('m', 'cNLL').
CLASS_AGGREGATE_METRICS = {**_CONFUSION_MATRIX_AGGREGATE_METRICS,
                           **_PROBABILISTIC_AGGREGATE_METRICS}
#: The per-class metrics, which are (..., K) arrays, in the default order of
#: `evaluation.compute_class_metrics`. `supp` is the number of ground-truth examples of a class
#: (its support), which has no aggregations over classes.
PER_CLASS_METRICS = ("supp", *_CONFUSION_MATRIX_CLASS_METRICS.values(),
                     *_PROBABILISTIC_CLASS_METRICS.values())
#: Metrics computed from the confusion matrix: the per-class `supp`, `P`, `R`, `F1` and `IoU`,
#: the aggregations of the last four over classes, e.g. `mF1`, and scalars.
CONFUSION_MATRIX_METRICS = frozenset(
    {"A", "n", "supp", "MCC", "kappa", *_CONFUSION_MATRIX_CLASS_METRICS.values(),
     *_CONFUSION_MATRIX_AGGREGATE_METRICS})
#: The number of equal-width confidence bins of the expected calibration error `ECE`
#: (`metrics.ClassificationStatistics.calibration_gap_sums`), as in Guo et al., "On Calibration of
#: Modern Neural Networks", 2017.
NUM_CALIBRATION_BINS = 15
#: Metrics computed from predicted distributions. `NLL` and `Brier` are means over examples,
#: `cNLL` and `cBrier` per-class means, e.g. `mNLL` and `maxBrier` their aggregations over
#: classes, and `ECE` the expected calibration error of the confidences.
PROBABILISTIC_METRICS = frozenset(
    {"NLL", "Brier", "ECE", *_PROBABILISTIC_CLASS_METRICS.values(),
     *_PROBABILISTIC_AGGREGATE_METRICS})
#: The metric of a simulated review (`metrics.compute_review_curve`): the share of the labeled
#: examples that a reviewer corrects, least confident first, until mF1 reaches
#: `MANUAL_REVIEW_TARGET_MF1`.
MANUAL_REVIEW_TARGET_MF1 = 0.9
MANUAL_REVIEW_PROPORTION_METRIC = f"MRP{round(MANUAL_REVIEW_TARGET_MF1 * 100)}"
#: Metrics computed from the confidences of predicted distributions.
MANUAL_REVIEW_METRICS = frozenset({MANUAL_REVIEW_PROPORTION_METRIC})
#: Metrics that need predicted distributions, so hard predictions lack them.
DISTRIBUTION_METRICS = PROBABILISTIC_METRICS | MANUAL_REVIEW_METRICS
#: Metrics whose lower values are better, which are those of predicted distributions.
LOWER_IS_BETTER_METRICS = DISTRIBUTION_METRICS
#: Metrics that count classes.
CLASS_COUNT_METRICS = frozenset({"nc"})
#: Metrics over a selection of classes, which a support threshold and a class condition restrict.
CLASS_SELECTION_METRICS = frozenset(CLASS_AGGREGATE_METRICS) | CLASS_COUNT_METRICS
KNOWN_BASE_METRICS = CONFUSION_MATRIX_METRICS | DISTRIBUTION_METRICS | CLASS_COUNT_METRICS

# The iRAP protocol.
IRAP_ATTRIBUTE_METRIC_NAMES = ("mF1", "mP", "mR", "MCC")
IRAP_MAIN_METRIC = "mF1"
IRAP_IGNORE_MISSING_CLASSES = True
IRAP_CLASS_SUPPORT_THRESHOLDS = (5, 10)


def get_irap_metric_names(min_class_supports: T.Sequence[int] = IRAP_CLASS_SUPPORT_THRESHOLDS,
                          output_kind: OutputKind = "probs") -> tuple[str, ...]:
    """The metric names of the iRAP protocol, attribute averages first.

    Args:
        min_class_supports: Support thresholds of the restricted versions of the main metric.
        output_kind: The metrics of predicted distributions are included for 'probs'.
    """
    is_probabilistic = output_kind == "probs"
    averaged = [*IRAP_ATTRIBUTE_METRIC_NAMES, "A"]  # The per-attribute names of the averages.
    per_attribute = ["mF1", "A", "n", "MCC"]
    if is_probabilistic:
        averaged += ["NLL", "Brier", "mNLL"]
        per_attribute += ["NLL", "Brier", "mNLL"]
    for n in min_class_supports:
        main_name = f"{IRAP_MAIN_METRIC}{SUPPORT_SUFFIX}{n}"
        averaged.append(main_name)
        per_attribute += [main_name, f"nc{SUPPORT_SUFFIX}{n}"]
        if is_probabilistic:
            averaged.append(f"mNLL{SUPPORT_SUFFIX}{n}")
    return (*map(get_attribute_average_name, averaged), *per_attribute)


# Parsing ##########################################################################################

ComparisonOperator = T.Literal["<", "<=", "=", ">=", ">"]
# Longer operators first, so that '<=' does not match as '<'.
_COMPARISON_OPERATOR_PATTERN = "|".join(
    sorted(map(re.escape, T.get_args(ComparisonOperator)), key=len, reverse=True))
_CLASS_CONDITION_PATTERN = re.compile(
    rf"([A-Za-z][A-Za-z0-9]*)({_COMPARISON_OPERATOR_PATTERN})(-?\d+(?:\.\d+)?)")


class ClassCondition(T.NamedTuple):
    """A condition `<per_class_metric><operator><threshold>` on a per-class metric
    (`PER_CLASS_METRICS`), e.g. 'R>0'."""

    per_class_metric: str
    operator: ComparisonOperator
    threshold: float


class ParsedMetricName(T.NamedTuple):
    is_averaged: bool  # has the 'a' prefix
    name: str  # the per-attribute name, without the 'a' prefix
    base: str  # without the prefix and the suffixes, e.g. 'mF1'
    min_support: int | None  # the threshold of a '_suppN' suffix
    class_condition: ClassCondition | None  # the condition of a '_{<condition>}' suffix

    @property
    def required_base_metrics(self) -> tuple[str, ...]:
        """The base metrics that the metric is computed from: `base` and the per-class metric of
        `class_condition`."""
        condition = self.class_condition
        return (self.base,) if condition is None else (self.base, condition.per_class_metric)

    @property
    def min_class_support(self) -> int | None:
        """The minimum number of ground-truth examples of the classes that the metric selects:
        `min_support`, but at least 1 if it is computed from a per-class metric of predicted
        distributions, which is NaN for a class without examples. None for no minimum."""
        if not PROBABILISTIC_METRICS.isdisjoint(self.required_base_metrics):
            return self.min_support or 1
        return self.min_support


def _parse_class_condition(name: str, condition: str) -> ClassCondition:
    if not (match := _CLASS_CONDITION_PATTERN.fullmatch(condition)):
        raise ValueError(f"Metric {name!r} has an invalid class condition {condition!r}. A class"
                         f" condition has the form <per-class metric><operator><number>, e.g."
                         f" R>0, with an operator in {T.get_args(ComparisonOperator)}.")
    per_class_metric, operator, threshold = match.groups()
    if per_class_metric not in PER_CLASS_METRICS:
        raise ValueError(f"Metric {name!r} has a condition on {per_class_metric!r}, which is not"
                         f" a per-class metric {PER_CLASS_METRICS}.")
    return ClassCondition(per_class_metric, operator, float(threshold))


def parse_metric_name(name: str) -> ParsedMetricName:
    """Parses an `[a]<base>[_suppN][_{<condition>}]` metric name.

    Raises:
        ValueError: For an unknown base name, a support threshold or class condition on a metric
            that is not in `CLASS_SELECTION_METRICS`, a support threshold that is not a positive
            integer, or an invalid class condition.
    """
    head, condition = name, None
    if name.endswith("}"):
        head, condition_separator, condition = name[:-1].partition("_{")
        if not condition_separator:
            raise ValueError(f"Metric {name!r} ends with '}}' without a class condition '_{{'.")
    base, separator, threshold = head.partition(SUPPORT_SUFFIX)
    if separator and not (threshold.isdigit() and int(threshold) > 0):
        raise ValueError(f"Metric {name!r} has a support threshold {threshold!r} that is not a"
                         f" positive integer.")
    min_support = int(threshold) if separator else None
    is_averaged = False
    if base not in KNOWN_BASE_METRICS and base.startswith("a"):
        is_averaged, base = True, base[1:]
    if base not in KNOWN_BASE_METRICS:
        raise ValueError(f"Unknown metric {name!r}. Known base names:"
                         f" {sorted(KNOWN_BASE_METRICS)}.")
    if (separator or condition is not None) and base not in CLASS_SELECTION_METRICS:
        raise ValueError(f"Metric {name!r} restricts the classes of {base!r}, which is neither an"
                         f" aggregation over classes nor a class count.")
    return ParsedMetricName(
        is_averaged=is_averaged, name=name[1:] if is_averaged else name, base=base,
        min_support=min_support,
        class_condition=None if condition is None else _parse_class_condition(name, condition))


def get_attribute_average_name(name: str) -> str:
    """Returns the name of the attribute average of a per-attribute metric, e.g. 'amF1' for
    'mF1'.

    Raises:
        ValueError: For an invalid name (`parse_metric_name`) or an attribute average.
    """
    if parse_metric_name(name).is_averaged:
        raise ValueError(f"Metric {name!r} is already an attribute average.")
    return "a" + name


def is_per_class_metric(name: str) -> bool:
    """Whether the metric is a per-class array, e.g. 'F1'."""
    return parse_metric_name(name).base in PER_CLASS_METRICS


def is_lower_better(name: str) -> bool:
    """Whether lower values of the metric are better (`LOWER_IS_BETTER_METRICS`), e.g. for
    sorting methods."""
    return parse_metric_name(name).base in LOWER_IS_BETTER_METRICS


def add_per_attribute_metric_names(metric_names: T.Sequence[str]) -> tuple[str, ...]:
    """`metric_names` followed by the per-attribute name of each attribute average that they
    lack, e.g. 'NLL' for 'aNLL', so that `metrics.select_metric_attributes` can average the
    results over a subset of the attributes."""
    names = dict.fromkeys(metric_names)
    for name in metric_names:
        if (parsed := parse_metric_name(name)).is_averaged:
            names.setdefault(parsed.name)
    return tuple(names)


# Descriptions #####################################################################################

#: Per-class metric -> its name in the descriptions of its aggregations over classes, and the
#: title of its mean over classes.
_PER_CLASS_METRIC_WORDS = {
    "P": ("precision", "Macro precision"),
    "R": ("recall", "Macro recall"),
    "F1": ("F1", "Macro F1"),
    "IoU": ("IoU", "Mean intersection over union"),
    "cNLL": ("class NLL", "Class-balanced negative log-likelihood"),
    "cBrier": ("class Brier score", "Class-balanced Brier score"),
}


def _describe_class_aggregation(aggregation: ClassAggregation, per_class: str) -> str:
    """Describes an aggregation of a per-class metric over classes, as a value of
    `_BASE_METRIC_DESCRIPTIONS`."""
    words, mean_title = _PER_CLASS_METRIC_WORDS[per_class]
    if aggregation == "m":
        return f"{mean_title} ({{name}}): the mean of {words} over {{classes}}."
    kind, extreme = ("Minimum", "lowest") if aggregation == "min" else ("Maximum", "highest")
    return f"{kind} {words} over classes ({{name}}): the {extreme} {words} among {{classes}}."


#: Base metric name -> its description (`describe_metric`), with the placeholders `{name}` for the
#: metric name, e.g. 'mF1_supp10', and `{classes}` for the classes of an aggregation over classes.
_BASE_METRIC_DESCRIPTIONS = {
    **{name: _describe_class_aggregation(*parts)
       for name, parts in CLASS_AGGREGATE_METRICS.items()},
    "A": "Accuracy ({name}): the fraction of labeled examples whose predicted class matches the"
         " label.",
    "P": "Precision ({name}): among examples predicted as this class, the fraction whose label is"
         " this class.",
    "R": "Recall ({name}): among examples whose label is this class, the fraction predicted as"
         " this class.",
    "F1": "{name}: the harmonic mean of this class's precision and recall.",
    "IoU": "Intersection over union ({name}): for this class, the number of examples both labeled"
           " and predicted as the class, divided by the number labeled or predicted as the class.",
    "n": "{name}: the number of labeled examples.",
    "supp": "Support ({name}): the number of examples whose label is this class.",
    "nc": "{name}: the number of {classes}.",
    "MCC": "Matthews correlation coefficient ({name}): the correlation between labels and predicted"
           " classes; 1 indicates perfect agreement and 0 indicates no correlation.",
    "kappa": "Cohen's kappa ({name}): the agreement of predicted classes with labels beyond"
             " chance, 1 for perfect predictions and 0 for agreement by chance.",
    "NLL": "Negative log-likelihood ({name}): the mean over examples of −ln of the predicted"
           " probability of the label. Infinite if a label gets probability 0.",
    "Brier": "Brier score ({name}): the mean, over examples, of the sum of squared differences"
             " between the predicted class probabilities and the one-hot label (from 0 to 2).",
    "cNLL": "Class NLL ({name}): the mean negative log-likelihood of examples whose label is this"
            " class.",
    "cBrier": "Class Brier score ({name}): the mean Brier score of examples whose label is this"
              " class.",
    "ECE": "Expected calibration error ({name}): the mean absolute difference between the"
           " confidence (the predicted probability of the predicted class) and the accuracy, in"
           f" {NUM_CALIBRATION_BINS} equal-width confidence bins, weighted by the fraction of"
           " labeled examples in each bin.",
    MANUAL_REVIEW_PROPORTION_METRIC: (
        "Manual review proportion ({name}): the fraction of labeled examples that a reviewer has"
        " to correct, starting with the least confident predictions, until mF1 reaches"
        f" {MANUAL_REVIEW_TARGET_MF1}. 0 if mF1 already reaches it."),
}
#: Metrics that count something, so that neither higher nor lower values are better.
_COUNT_METRICS = frozenset({"n", "supp", *CLASS_COUNT_METRICS})
#: Per-class metric -> words for it in a class condition.
_CLASS_CONDITION_WORDS = {
    "supp": "the number of labeled examples",
    **{name: words for name, (words, _) in _PER_CLASS_METRIC_WORDS.items()}}
_COMPARISON_SYMBOLS: dict[ComparisonOperator, str] = {
    "<": "<", "<=": "≤", "=": "=", ">=": "≥", ">": ">"}


def describe_metric(name: str,
                    ignore_missing_classes: bool = IRAP_IGNORE_MISSING_CLASSES) -> str:
    """Describes a metric in words, e.g. 'Macro F1 (mF1): the mean of F1 over …', for a tooltip. An
    attribute average gets a sentence about the average, then a line break and the description of
    its per-attribute metric.

    Args:
        ignore_missing_classes: The setting of `metrics.compute_classification_metrics` that the
            metric was computed with.

    Raises:
        ValueError: For an invalid name (`parse_metric_name`).
    """
    parsed = parse_metric_name(name)
    if (min_support := parsed.min_class_support) == 1:
        classes = "labeled classes"
    elif min_support is not None:
        classes = f"classes with at least {min_support} labeled examples"
    elif ignore_missing_classes:
        classes = "classes that are labeled or predicted"
    else:
        classes = "all classes"
    if (condition := parsed.class_condition) is not None:
        classes += (f", where {_CLASS_CONDITION_WORDS[condition.per_class_metric]}"
                    f" {_COMPARISON_SYMBOLS[condition.operator]} {condition.threshold:g}")
    description = _BASE_METRIC_DESCRIPTIONS[parsed.base].format(name=parsed.name, classes=classes)
    if parsed.base not in _COUNT_METRICS:
        description += " Lower is better." if is_lower_better(name) else " Higher is better."
    if not parsed.is_averaged:
        return description
    return f"{describe_attribute_average(parsed.name)}\n{description}"


def describe_attribute_average(name: str) -> str:
    """Describes the attribute average of a metric in a sentence, e.g. 'The mean over attributes of
    mF1, …' for `name='mF1'`.

    Args:
        name: The name of the per-attribute metric, or words for it, e.g. 'the metric'.
    """
    return f"The mean over attributes of {name}, leaving out those where it is undefined."
