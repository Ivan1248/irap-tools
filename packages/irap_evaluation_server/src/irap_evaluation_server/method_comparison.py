"""The comparison of two methods on the reference set of a dataset split
(`irap_evaluation.compare_methods`): the differences of their means over models, with bootstrap
intervals over the same resampled road sequences, which include the variation between models (see
`irap_evaluation/docs/evaluation.md`, "Comparing methods").

Like intervals, comparisons need the per-sequence statistics, so the scoring worker computes them
from the files on request and caches them (`ComparisonRequest`, a `scoring.WorkRequest`).
"""

import dataclasses as dc
import functools
import typing as T

import irap_evaluation as ie

from .archive import ModelArchive
from .datasets import DatasetContext
from .method_ranking import PER_ATTRIBUTE_METRIC_NAMES, RANKING_METRIC_NAMES
from .scoring import (
    IntervalRequest,
    MethodScores,
    ScoredModel,
    ScoringSettings,
    check_attributes,
    compute_json_sha256,
    make_interval_request,
    metric_values_from_json,
    metric_values_to_json,
)

#: The compared metrics: the attribute averages of the method table and their per-attribute
#: values. Hard predictions lack the probabilistic ones.
COMPARED_METRIC_NAMES = (*RANKING_METRIC_NAMES, *PER_ATTRIBUTE_METRIC_NAMES)


@dc.dataclass(frozen=True)
class ComparisonRequest:
    """The differences of the metrics of method A minus method B on the reference set, over a
    subset of the attributes, with their intervals.

    Its fields are the fingerprints of what the comparison depends on, so `key` identifies it in
    the cache.

    Attributes:
        intervals_a: The request of the intervals of A, whose fields the comparison depends on.
        intervals_b: The same of B, on the same evaluation set and attributes.
    """

    intervals_a: IntervalRequest
    intervals_b: IntervalRequest

    @functools.cached_property
    def key(self) -> str:
        return compute_json_sha256({"kind": "comparison", **dc.asdict(self)})

    @property
    def submission_ids(self) -> tuple[int, ...]:
        return (*self.intervals_a.submission_ids, *self.intervals_b.submission_ids)

    def compute(self, archive: ModelArchive,
                dataset_contexts: T.Mapping[str, DatasetContext],
                settings: ScoringSettings) -> ie.MetricValues[ie.MetricDifference]:
        """See `scoring.WorkRequest.compute`. The models are scored again
        (`scoring.IntervalRequest.evaluate_models`), and the metrics of `COMPARED_METRIC_NAMES`
        that both methods have are compared.

        Raises:
            ValueError: See `scoring.IntervalRequest.evaluate_models`, and if
                `irap_evaluation.compare_methods` refuses the models.
        """
        results_a, results_b = (r.evaluate_models(archive, dataset_contexts, settings)
                                for r in (self.intervals_a, self.intervals_b))
        common_names = set(results_a[0].metrics.names) & set(results_b[0].metrics.names)
        return ie.compare_methods(
            results_a, results_b, [n for n in COMPARED_METRIC_NAMES if n in common_names],
            num_resamples=settings.num_resamples, confidence=settings.confidence,
            seed=settings.bootstrap_seed)

    def value_to_json(self, value: ie.MetricValues[ie.MetricDifference]) -> str:
        return metric_values_to_json(value)

    def value_from_json(self, text: str) -> ie.MetricValues[ie.MetricDifference]:
        return metric_values_from_json(text, lambda d: ie.MetricDifference(
            d["difference"], ie.BootstrapInterval(**d["interval"])))


def list_comparable_methods(rows: T.Iterable[MethodScores]) -> list[MethodScores]:
    """Lists the methods that can be compared, those without `MethodScores.error`, in their
    order."""
    return [r for r in rows if r.error is None]


def get_better_method(interval: ie.BootstrapInterval | None,
                      metric_name: str) -> T.Literal["A", "B"] | None:
    """Determines which method is clearly better by the interval of a difference A − B: the one
    that it favours if it excludes 0, otherwise None (also for an undefined interval). There are
    no p-values (`irap_evaluation/docs/evaluation.md`, "Comparing methods")."""
    if interval is None or not interval.low <= interval.high:  # Also NaN bounds.
        return None
    if interval.low <= 0 <= interval.high:
        return None
    is_a_higher = interval.low > 0
    return "A" if is_a_higher != ie.is_lower_better(metric_name) else "B"


def make_comparison_request(models_a: T.Sequence[ScoredModel], models_b: T.Sequence[ScoredModel],
                            attributes: T.Collection[str],
                            settings: ScoringSettings) -> ComparisonRequest:
    """
    Args:
        models_a: The models of method A, scored on the reference set.
        models_b: The models of method B, the same.

    Raises:
        ValueError: If `attributes` is empty, the models of a method cannot be combined
            (`scoring.make_interval_request`), a model is not scored on the reference set, or A
            and B are scored on different sets.
    """
    check_attributes(attributes)
    interval_requests = []
    for name, models in (("A", models_a), ("B", models_b)):
        try:
            interval_requests.append(make_interval_request(models, attributes, settings))
        except ValueError as e:
            raise ValueError(f"Method {name}: {e}") from None
    if any(m.scores.evaluation_set != ie.REFERENCE_SET_NAME for m in (*models_a, *models_b)):
        raise ValueError("Methods are compared on the reference set.")
    intervals_a, intervals_b = interval_requests
    if intervals_a.evaluation_set_fingerprint != intervals_b.evaluation_set_fingerprint:
        raise ValueError("The methods are scored on different reference sets.")
    return ComparisonRequest(intervals_a, intervals_b)
