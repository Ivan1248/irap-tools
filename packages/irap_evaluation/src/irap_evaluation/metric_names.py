"""Metric names of the iRAP evaluation protocol: their form, parsing and descriptions.

Metric names have the form `[a|tot]<base>[_suppN][_{<condition>}]`, with two levels of prefixes,
at most one of each:

1. Over attributes (`AttributeAggregation`): `a` averages a per-attribute metric, leaving out
   attributes where it is undefined (NaN), and `tot` totals a count of `ATTRIBUTE_TOTAL_METRICS`
   (`n`, `nc`), e.g. `amF1` and `totnc_{R>0}`. Without it, the metric is per attribute.
2. Over classes inside an attribute, as part of `<base>`: `m`, `min`, `max` or `sum` aggregates a
   per-class metric (`ClassAggregation`), e.g. `mF1` (mean), `minR` (minimum) and `sumR` (sum),
   and `nc` counts classes.

So `amF1`, `aminR`, `asumR`, `anc_{R>0}`, `totnc_{R>0}` and `totn` are valid names, while
`atotnc` and `totamF1` (two aggregations over attributes) and `totmF1` (a total of a metric that
is not a count) are not.

- `<base>` is a per-attribute metric (see `CONFUSION_MATRIX_METRICS`, `PROBABILISTIC_METRICS`
  and `MANUAL_REVIEW_METRICS`), e.g. an aggregation over classes, `nc`, the number of classes,
  or the per-class `supp`, the number of ground-truth examples of a class (its support). No base
  name starts with an attribute-level prefix, so names are parsed unambiguously.
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
#: 'min' the minimum, 'max' the maximum and 'sum' the sum, e.g. 'mF1', 'minR', 'maxNLL' and
#: 'sumR'. All aggregations of a per-class metric are over the same classes (see
#: `describe_metric`). If no class is selected, the sum is 0, and the others are NaN. The classes
#: of a per-class metric of predicted distributions, which is NaN for a class without examples,
#: have examples (`ParsedMetricName.min_class_support`), so a sum has no NaN terms.
ClassAggregation = T.Literal["m", "min", "max", "sum"]
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
#: Metrics that count something, so that neither higher nor lower values are better.
COUNT_METRICS = frozenset({"n", "supp", *CLASS_COUNT_METRICS})
#: Aggregations of a per-attribute metric over attributes, by the prefix of their names: 'a' the
#: mean over the attributes where the metric is defined, and 'tot' the total (sum), e.g. 'amF1'
#: and 'totnc_{R>0}'.
AttributeAggregation = T.Literal["a", "tot"]
#: The metrics that have totals over attributes: the counts of examples and classes. A total of
#: other metrics, e.g. of F1 values, has no meaning.
ATTRIBUTE_TOTAL_METRICS = frozenset({"n", *CLASS_COUNT_METRICS})

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
    attribute_aggregation: AttributeAggregation | None  # the 'a' or 'tot' prefix, if any
    name: str  # the per-attribute name, without the prefix
    base: str  # without the prefix and the suffixes, e.g. 'mF1'
    min_support: int | None  # the threshold of a '_suppN' suffix
    class_condition: ClassCondition | None  # the condition of a '_{<condition>}' suffix

    @property
    def is_attribute_aggregate(self) -> bool:
        """Whether the metric is aggregated over attributes, e.g. 'amF1' and 'totnc_{R>0}'."""
        return self.attribute_aggregation is not None

    @property
    def required_base_metrics(self) -> tuple[str, ...]:
        """The base metrics that the metric is computed from: `base` and the per-class metric of
        `class_condition`."""
        condition = self.class_condition
        return (self.base,) if condition is None else (self.base, condition.per_class_metric)

    @property
    def needs_distributions(self) -> bool:
        """Whether the metric is computed from predicted distributions
        (`DISTRIBUTION_METRICS`), so that hard predictions lack it."""
        return not DISTRIBUTION_METRICS.isdisjoint(self.required_base_metrics)

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


def _find_attribute_aggregation(head: str) -> AttributeAggregation | None:
    """Returns the aggregation over attributes that a name starts with, or None."""
    return next((p for p in T.get_args(AttributeAggregation) if head.startswith(p)), None)


def parse_metric_name(name: str) -> ParsedMetricName:
    """Parses an `[a|tot]<base>[_suppN][_{<condition>}]` metric name (see the module
    docstring).

    Raises:
        ValueError: For an unknown base name, two aggregations over attributes, a total over
            attributes of a metric that is not in `ATTRIBUTE_TOTAL_METRICS`, a support threshold
            or class condition on a metric that is not in `CLASS_SELECTION_METRICS`, a support
            threshold that is not a positive integer, or an invalid class condition.
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
    # No known base name starts with an attribute aggregation (see the module docstring).
    attribute_aggregation = _find_attribute_aggregation(base)
    base = base.removeprefix(attribute_aggregation or "")
    if base not in KNOWN_BASE_METRICS:
        if (attribute_aggregation is not None
                and (second := _find_attribute_aggregation(base)) is not None
                and base.removeprefix(second) in KNOWN_BASE_METRICS):
            raise ValueError(f"Metric {name!r} has two aggregations over attributes. A name has"
                             f" only one aggregation over attributes (a or tot).")
        raise ValueError(f"Unknown metric {name!r}. Known base names:"
                         f" {sorted(KNOWN_BASE_METRICS)}, with an optional aggregation over"
                         f" attributes, {' or '.join(T.get_args(AttributeAggregation))}.")
    if attribute_aggregation == "tot" and base not in ATTRIBUTE_TOTAL_METRICS:
        raise ValueError(f"Metric {name!r} is a total over attributes of {base!r}, which is not a"
                         f" count. tot applies only to counts"
                         f" ({', '.join(sorted(ATTRIBUTE_TOTAL_METRICS))}, with their suffixes),"
                         f" e.g. totnc_{{R>0}}.")
    if (separator or condition is not None) and base not in CLASS_SELECTION_METRICS:
        raise ValueError(f"Metric {name!r} restricts the classes of {base!r}, which is neither an"
                         f" aggregation over classes nor a class count.")
    return ParsedMetricName(
        attribute_aggregation=attribute_aggregation,
        name=name.removeprefix(attribute_aggregation or ""), base=base,
        min_support=min_support,
        class_condition=None if condition is None else _parse_class_condition(name, condition))


def get_attribute_aggregate_name(name: str, aggregation: AttributeAggregation) -> str:
    """Returns the name of an aggregate over attributes of a per-attribute metric, e.g. 'amF1'
    for ('mF1', 'a') and 'totnc_{R>0}' for ('nc_{R>0}', 'tot').

    Raises:
        ValueError: For an invalid name (`parse_metric_name`), an attribute aggregate, or a total
            of a metric that is not in `ATTRIBUTE_TOTAL_METRICS`.
    """
    if parse_metric_name(name).is_attribute_aggregate:
        raise ValueError(f"Metric {name!r} is already an attribute aggregate.")
    if aggregation not in T.get_args(AttributeAggregation):
        raise ValueError(f"Unknown attribute aggregation {aggregation!r}. Aggregations:"
                         f" {T.get_args(AttributeAggregation)}.")
    parse_metric_name(aggregate_name := aggregation + name)  # Checks that a total is of a count.
    return aggregate_name


def get_attribute_average_name(name: str) -> str:
    """Returns the name of the attribute average of a per-attribute metric, e.g. 'amF1' for
    'mF1' (see `get_attribute_aggregate_name`)."""
    return get_attribute_aggregate_name(name, "a")


def is_count_metric(name: str) -> bool:
    """Whether the metric is a count (`COUNT_METRICS`) or a total or average of counts over
    attributes, e.g. 'nc_supp10' and 'totnc_{R>0}'."""
    return parse_metric_name(name).base in COUNT_METRICS


def is_per_class_metric(name: str) -> bool:
    """Whether the metric is a per-class array, e.g. 'F1'."""
    return parse_metric_name(name).base in PER_CLASS_METRICS


def is_lower_better(name: str) -> bool:
    """Whether lower values of the metric are better (`LOWER_IS_BETTER_METRICS`), e.g. for
    sorting methods."""
    return parse_metric_name(name).base in LOWER_IS_BETTER_METRICS


def can_compute_metric(name: str, output_kind: OutputKind) -> bool:
    """Whether predictions of an output kind have the metric: hard predictions lack those of
    predicted distributions (`ParsedMetricName.needs_distributions`)."""
    return output_kind == "probs" or not parse_metric_name(name).needs_distributions


def add_per_attribute_metric_names(metric_names: T.Sequence[str]) -> tuple[str, ...]:
    """`metric_names` followed by the per-attribute name of each attribute aggregate that they
    lack, e.g. 'NLL' for 'aNLL' and 'nc_{R>0}' for 'totnc_{R>0}', so that
    `metrics.select_metric_attributes` can aggregate the results over a subset of the
    attributes."""
    names = dict.fromkeys(metric_names)
    for name in metric_names:
        if (parsed := parse_metric_name(name)).is_attribute_aggregate:
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
    if aggregation == "sum":
        return (f"Sum of {words} over classes ({{name}}): the sum of {words} over {{classes}}, 0"
                f" if there are none.")
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
#: Per-class metric -> words for it in a class condition.
_CLASS_CONDITION_WORDS = {
    "supp": "the number of labeled examples",
    **{name: words for name, (words, _) in _PER_CLASS_METRIC_WORDS.items()}}
_COMPARISON_SYMBOLS: dict[ComparisonOperator, str] = {
    "<": "<", "<=": "≤", "=": "=", ">=": "≥", ">": ">"}


def describe_metric(name: str,
                    ignore_missing_classes: bool = IRAP_IGNORE_MISSING_CLASSES) -> str:
    """Describes a metric in words, e.g. 'Macro F1 (mF1): the mean of F1 over …', for a tooltip. An
    attribute aggregate gets a sentence about the aggregation (`describe_attribute_aggregate`),
    then a line break and the description of its per-attribute metric.

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
    if not is_count_metric(name):
        description += " Lower is better." if is_lower_better(name) else " Higher is better."
    if (aggregation := parsed.attribute_aggregation) is None:
        return description
    return f"{describe_attribute_aggregate(parsed.name, aggregation)}\n{description}"


def describe_attribute_aggregate(name: str, aggregation: AttributeAggregation) -> str:
    """Describes an aggregate of a metric over attributes in a sentence, e.g. 'The mean over
    attributes of mF1, …' for `name='mF1'` and `aggregation='a'`.

    Args:
        name: The name of the per-attribute metric, or words for it, e.g. 'the metric'.
    """
    return {"a": f"The mean over attributes of {name}, leaving out those where it is undefined.",
            "tot": f"The total over attributes of {name}: the sum of its values of the"
                   f" attributes."}[aggregation]
