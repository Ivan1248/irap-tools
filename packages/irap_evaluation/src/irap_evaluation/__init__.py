"""Saved predictions of iRAP road-attribute methods: file format, evaluation and ensembling.

The tables and JSON documents of results and the coding-table export are in
`irap_evaluation.reports`, and the `irap-eval` command is in `irap_evaluation.tools`. This package
does not import them.

Typical use::

    from irap_data import load_irap_metadata
    import irap_evaluation as ie

    predictions = ie.read_predictions("resnet.predictions.parquet")
    metadata = load_irap_metadata("/data/IRAP_Vietnam")
    reference_set = ie.get_reference_set(metadata, "vietnam", "val")
    result = ie.evaluate_predictions(predictions, reference_set)
    print(result.metrics.averages["amF1"])
"""

from .bootstrap import (
    BootstrapInterval,
    MetricDifference,
    compare_methods,
    compute_bootstrap_intervals,
    compute_method_metrics,
)
from .ensembling import ensemble_predictions
from .evaluation import (
    AttributeSelection,
    ClassMetrics,
    EvaluationResult,
    NullPolicy,
    check_runs_distinct,
    compute_class_metrics,
    evaluate_predictions,
    group_runs_by_method,
    select_evaluated_attributes,
)
from .evaluation_sets import (
    EvaluationSet,
    get_evaluation_set,
    get_reference_set,
    select_evaluation_sets,
    select_evaluation_subset,
)
from .metrics import (
    ClassificationStatistics,
    MetricValues,
    compute_classification_metrics,
    compute_classification_statistics,
    compute_grouped_metrics,
    compute_multi_attribute_metrics,
    get_irap_metric_names,
    map_metric_values,
    sum_statistics,
)
from .prediction_io import read_predictions, write_predictions
from .predictions import (
    INVALID_IRAP_CODE,
    MethodInfo,
    OutputKind,
    PredictionFormatError,
    PredictionHeader,
    Predictions,
    align_classes,
    get_common_dataset_and_split,
    select_segments,
    to_class_indices,
    to_irap_codes,
)
