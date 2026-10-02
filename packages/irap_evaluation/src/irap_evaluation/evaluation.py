"""Scoring `Predictions` on an `EvaluationSet`."""

import dataclasses as dc
import typing as T

import numpy as np

from irap_data.attrs import filter_labeled_attrs, get_attrs_to_include
from irap_data.metadata import IGNORE_LABEL_INDEX, ClassVocabulary

from .evaluation_sets import EvaluationSet
from .metrics import (
    MACRO_METRIC_TO_PER_CLASS,
    PROBABILISTIC_METRICS,
    ClassificationStatistics,
    MetricValues,
    compute_classification_metrics,
    compute_classification_statistics,
    compute_grouped_metrics,
    get_irap_metric_names,
    is_per_class_metric,
    map_metric_values,
    sum_statistics,
)
from .predictions import (
    MethodInfo,
    OutputKind,
    PredictionFormatError,
    Predictions,
    add_invalid_attributes,
    align_classes,
    select_segments,
    to_class_indices,
)

#: How an invalid cell (`Predictions.is_valid` False) is scored:
#: - 'first_class': as class 0 in the metrics of the predicted classes, as in vidlu, and as the
#:   uniform distribution in NLL and Brier.
#: - 'exclude': not scored, like a missing label. Results with invalid cells are then scored on
#:   different segments, so the means over runs and the comparisons of `bootstrap` refuse them.
#: `docs/evaluation.md#invalid-cells` gives the reasons.
NullPolicy = T.Literal["first_class", "exclude"]
NULL_POLICIES: tuple[str, ...] = T.get_args(NullPolicy)

#: How an evaluated attribute that the predictions lack is scored:
#: - 'error': not at all, `evaluate_predictions` raises an error.
#: - 'invalid': as if every cell of it were invalid, so that runs that lack attributes are scored
#:   on the same attributes as the others and are not rewarded for leaving out attributes.
#: `docs/evaluation.md#missing-attributes` gives the reasons.
MissingAttributePolicy = T.Literal["error", "invalid"]
MISSING_ATTRIBUTE_POLICIES: tuple[str, ...] = T.get_args(MissingAttributePolicy)

#: 'canonical': the iRAP attribute subset (`irap_data.attrs.get_attrs_to_include`), as in
#: `vidlu_irap_gaim.metrics`. 'all': every attribute of the dataset.
AttributeSelection = T.Literal["canonical", "all"]
ATTRIBUTE_SELECTIONS: tuple[str, ...] = T.get_args(AttributeSelection)


def _check_attributes_in_vocabulary(attributes: T.Iterable[str],
                                    vocabulary: ClassVocabulary) -> None:
    if unknown := [a for a in attributes if a not in vocabulary.attribute_names]:
        raise ValueError(f"Attributes not in the dataset: {unknown}.")


def select_evaluated_attributes(
    evaluation_set: EvaluationSet,
    attributes: AttributeSelection | T.Sequence[str] = "canonical",
) -> tuple[str, ...]:
    """The attributes to score: the candidates that have a label in `evaluation_set`.

    Args:
        attributes: The candidates, an `AttributeSelection` or attribute names. The 'canonical'
            selection can contain attributes that the dataset lacks, which are left out.

    Raises:
        ValueError: For an unknown selection, or attribute names that are not in the dataset.
    """
    if isinstance(attributes, str):
        if attributes not in ATTRIBUTE_SELECTIONS:
            raise ValueError(f"attributes must be one of {ATTRIBUTE_SELECTIONS} or attribute"
                             f" names, got {attributes!r}.")
        attributes = (get_attrs_to_include() if attributes == "canonical"
                      else evaluation_set.vocabulary.attribute_names)
    else:
        _check_attributes_in_vocabulary(attributes, evaluation_set.vocabulary)
    return filter_labeled_attrs(attributes, evaluation_set.attr_to_num_labeled)


@dc.dataclass(frozen=True)
class EvaluationResult:
    """Scores of one set of predictions on one evaluation set.

    Attributes:
        method: The method, and run, of the evaluated predictions.
        null_policy: See `NullPolicy`.
        statistics: Attribute -> statistics with a leading (G,) axis, one entry per road
            sequence of `evaluation_set.sequence_ids`, from which any metric can be recomputed,
            e.g. for a bootstrap resample. Its keys are the evaluated attributes (`attributes`).
        num_invalid: Attribute -> number of labeled segments with an invalid prediction.
        metrics: The scores. The values of `n` and `nc_suppN` are ints, the others floats. For
            per-class metrics, see `compute_class_metrics`.
        intervals: `bootstrap.BootstrapInterval`s of the metrics of this run alone, which cover the
            sampling of road sequences but not of runs (see
            `bootstrap.compute_bootstrap_intervals`), or None.
        missing_attributes: The evaluated attributes that the predictions lack, scored as
            invalid cells (`MissingAttributePolicy` 'invalid'). `num_invalid` counts their cells
            too.
    """

    method: MethodInfo
    output_kind: OutputKind
    evaluation_set: EvaluationSet
    null_policy: NullPolicy
    statistics: T.Mapping[str, ClassificationStatistics]
    num_invalid: T.Mapping[str, int]
    metrics: MetricValues[float]
    intervals: MetricValues | None = None
    missing_attributes: tuple[str, ...] = ()

    @property
    def run_label(self) -> str:
        return self.method.run_label

    @property
    def attributes(self) -> tuple[str, ...]:
        return tuple(self.statistics)


def check_runs_distinct(methods: T.Iterable[MethodInfo]) -> None:
    """Checks that each method has distinct runs: different seeds, or a single run without a
    seed.

    Raises:
        ValueError: If a method has a seed twice, or several runs of which one lacks a seed.
    """
    name_to_seeds: dict[str, list[int | None]] = {}
    for method in methods:
        name_to_seeds.setdefault(method.name, []).append(method.seed)
    for name, seeds in name_to_seeds.items():
        if len(seeds) > 1 and None in seeds:
            raise ValueError(f"Method {name!r} has {len(seeds)} runs, so each needs a seed.")
        if len(set(seeds)) < len(seeds):
            raise ValueError(f"Method {name!r} has several runs with the same seed.")


def group_runs_by_method(
        results: T.Iterable[EvaluationResult]) -> dict[str, tuple[EvaluationResult, ...]]:
    """Method name -> its results (runs), in the order of `results`.

    Raises:
        ValueError: If the runs of a method are not distinct (see `check_runs_distinct`).
    """
    results = list(results)
    check_runs_distinct(r.method for r in results)
    name_to_runs: dict[str, list[EvaluationResult]] = {}
    for result in results:
        name_to_runs.setdefault(result.method.name, []).append(result)
    return {name: tuple(runs) for name, runs in name_to_runs.items()}


def evaluate_predictions(
    predictions: Predictions,
    evaluation_set: EvaluationSet,
    *,
    attributes: T.Sequence[str] | None = None,
    null_policy: NullPolicy = "first_class",
    missing_attribute_policy: MissingAttributePolicy = "error",
    metric_names: T.Sequence[str] | None = None,
) -> EvaluationResult:
    """Scores predictions on every segment of an evaluation set.

    Predictions of other segments, e.g. outside the reference set, are not used.

    Args:
        predictions: Predictions of the same release and split as `evaluation_set`.
        attributes: The attributes to score. None means `select_evaluated_attributes` with its
            defaults.
        null_policy: See `NullPolicy`.
        missing_attribute_policy: See `MissingAttributePolicy`.
        metric_names: None means the iRAP protocol (`metrics.get_irap_metric_names`) for the
            output kind of the predictions.

    Raises:
        ValueError: For probabilistic metrics of hard predictions, per-class metric names,
            attributes that are not in the dataset, if there is no attribute to score, or if
            `null_policy='exclude'` would leave an attribute without scored segments, since
            every labeled cell of it is invalid or, with `missing_attribute_policy='invalid'`,
            it is not predicted.
        PredictionFormatError: If the release or split differ, an attribute is not predicted
            and `missing_attribute_policy='error'`, the predicted iRAP codes of an attribute
            differ from the dataset's, or a segment of the evaluation set has no prediction.
    """
    header = predictions.header
    if (header.dataset, header.split) != (evaluation_set.dataset, evaluation_set.split):
        raise PredictionFormatError(
            f"Predictions of {header.dataset}/{header.split} cannot be scored on"
            f" {evaluation_set.dataset}/{evaluation_set.split}.")
    if null_policy not in NULL_POLICIES:
        raise ValueError(f"null_policy must be one of {NULL_POLICIES}, got {null_policy!r}.")
    if missing_attribute_policy not in MISSING_ATTRIBUTE_POLICIES:
        raise ValueError(f"missing_attribute_policy must be one of {MISSING_ATTRIBUTE_POLICIES},"
                         f" got {missing_attribute_policy!r}.")
    if metric_names is None:
        metric_names = get_irap_metric_names(output_kind=predictions.output_kind)
    if per_class := [n for n in metric_names if is_per_class_metric(n)]:
        raise ValueError(f"Per-class metrics {per_class} are not kept in results. Use"
                         f" compute_class_metrics(result).")
    attributes = tuple(select_evaluated_attributes(evaluation_set) if attributes is None
                       else attributes)
    if not attributes:
        raise ValueError("There is no attribute to score.")
    _check_attributes_in_vocabulary(attributes, evaluation_set.vocabulary)
    attribute_to_irap_codes = evaluation_set.vocabulary.restrict_to_attributes(
        attributes).attribute_to_irap_codes
    missing_attributes = ()
    if missing_attribute_policy == "invalid":
        missing_attributes = tuple(a for a in attributes if a not in predictions.attributes)
        predictions = add_invalid_attributes(predictions, attribute_to_irap_codes)
    aligned = select_segments(align_classes(predictions, attribute_to_irap_codes),
                              evaluation_set.segment_ids)
    num_invalid = {attr: int((~aligned.is_valid[attr]
                              & (evaluation_set.class_indices[attr] != IGNORE_LABEL_INDEX)).sum())
                   for attr in attributes}
    if null_policy == "exclude":
        attr_to_num_labeled = evaluation_set.attr_to_num_labeled
        if all_invalid := [a for a in attributes
                           if 0 < num_invalid[a] == attr_to_num_labeled[a]]:
            raise ValueError(f"Every labeled cell of {all_invalid} is invalid or not predicted,"
                             f" so null_policy='exclude' would leave them without scored"
                             f" segments. Use null_policy='first_class'.")
    has_probabilities = predictions.output_kind == "probs"
    attr_to_predicted_indices = to_class_indices(aligned)

    def get_statistics(attr: str) -> ClassificationStatistics:
        is_valid = aligned.is_valid[attr]
        targets = evaluation_set.class_indices[attr]
        if null_policy == "exclude":
            targets = np.where(is_valid, targets, IGNORE_LABEL_INDEX)
        num_classes = len(aligned.attribute_to_irap_codes[attr])
        # See NullPolicy.
        predicted_indices = np.where(is_valid, attr_to_predicted_indices[attr], 0)
        probs = (np.where(is_valid[:, None], aligned.probs[attr], np.float32(1 / num_classes))
                 if has_probabilities else None)
        return compute_classification_statistics(
            targets, predicted_indices, num_classes, probs=probs,
            group_indices=evaluation_set.segment_sequence_indices,
            num_groups=len(evaluation_set.sequence_ids))

    statistics = {attr: get_statistics(attr) for attr in attributes}
    metrics = compute_grouped_metrics(statistics, metric_names)
    return EvaluationResult(
        method=header.method, output_kind=predictions.output_kind,
        evaluation_set=evaluation_set, null_policy=null_policy,
        statistics=statistics, num_invalid=num_invalid,
        metrics=map_metric_values(lambda v: np.asarray(v).item(), metrics),
        missing_attributes=missing_attributes)


@dc.dataclass(frozen=True)
class ClassMetrics:
    """Per-class metrics of one attribute on an evaluation set, in vocabulary class order.

    Attributes:
        irap_codes: (K,) the iRAP code of each class.
        values: (K,) the class values, e.g. 'Wide (>= 3.25m)'.
        support: (K,) int64 number of labeled segments per class.
        confusion_matrix: (K, K) int64 counts, rows = ground truth, columns = prediction.
        metrics: Name -> (K,) float64 values, e.g. 'P', 'R', 'F1' (see
            `metrics.PER_CLASS_METRICS`).
    """

    irap_codes: tuple[int, ...]
    values: tuple[str, ...]
    support: np.ndarray
    confusion_matrix: np.ndarray
    metrics: T.Mapping[str, np.ndarray]


def compute_class_metrics(result: EvaluationResult,
                          metric_names: T.Sequence[str] | None = None) -> dict[str, ClassMetrics]:
    """Attribute -> the per-class metrics of `result`, from its statistics.

    Args:
        metric_names: Per-class metric names. None means 'P', 'R', 'F1' and 'IoU', and also
            'cNLL' and 'cBrier' for 'probs' predictions.

    Raises:
        ValueError: For a name that is not a per-class metric, or a probabilistic metric of hard
            predictions.
    """
    if metric_names is None:
        metric_names = [n for n in MACRO_METRIC_TO_PER_CLASS.values()
                        if result.output_kind == "probs" or n not in PROBABILISTIC_METRICS]
    if not_per_class := [n for n in metric_names if not is_per_class_metric(n)]:
        raise ValueError(f"Not per-class metrics: {not_per_class}.")
    vocabulary = result.evaluation_set.vocabulary

    def compute(attr: str) -> ClassMetrics:
        statistics = sum_statistics(result.statistics[attr])
        return ClassMetrics(
            irap_codes=vocabulary.get_irap_codes(attr), values=vocabulary.get_values(attr),
            support=statistics.support, confusion_matrix=statistics.confusion_matrix,
            metrics=compute_classification_metrics(statistics, metric_names))

    return {attr: compute(attr) for attr in result.attributes}
