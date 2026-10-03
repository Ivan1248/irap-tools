"""The view settings that several pages share, mostly query parameters."""

import re
import typing as T

import irap_evaluation as ie

from ..datasets import EVALUATION_SET_NAMES, DatasetContext
from ..method_ranking import PER_ATTRIBUTE_METRIC_NAMES

#: The options of the View select of a page with the attribute averages or one per-attribute
#: metric (the `metric` parameter, `parse_per_attribute_metric`).
VIEW_OPTIONS = {"": "Attribute averages",
                **{m: f"Per attribute: {m}" for m in PER_ATTRIBUTE_METRIC_NAMES}}


def parse_dataset(query: T.Mapping[str, str], dataset_contexts: T.Mapping[str, DatasetContext],
                  default: str | None = None) -> str:
    """The `dataset` parameter.

    Args:
        default: The dataset without the parameter, by default the first one.

    Raises:
        ValueError: For an unknown dataset.
    """
    dataset = query.get("dataset") or default or next(iter(dataset_contexts))
    if dataset not in dataset_contexts:
        raise ValueError(f"Unknown dataset {dataset!r}. Datasets: {', '.join(dataset_contexts)}.")
    return dataset


def parse_dataset_and_split(query: T.Mapping[str, str],
                            dataset_contexts: T.Mapping[str, DatasetContext]) -> tuple[str, str]:
    """The `dataset` and `split` parameters of a page of any split. They default to the first
    dataset and its first analysis split, or its first split.

    Raises:
        ValueError: For an unknown dataset or split.
    """
    dataset = parse_dataset(query, dataset_contexts)
    context = dataset_contexts[dataset]
    split = query.get("split") or (context.config.analysis_splits or context.split_names)[0]
    if split not in context.split_names:
        raise ValueError(f"The dataset {dataset!r} has no split {split!r}. Its splits:"
                         f" {', '.join(context.split_names)}.")
    return dataset, split


def parse_evaluation_set(query: T.Mapping[str, str]) -> str:
    """The `set` parameter, by default the reference set.

    Raises:
        ValueError: For an unknown evaluation set.
    """
    evaluation_set = query.get("set") or ie.REFERENCE_SET_NAME
    if evaluation_set not in EVALUATION_SET_NAMES:
        raise ValueError(f"Unknown evaluation set {evaluation_set!r}. Evaluation sets:"
                         f" {', '.join(EVALUATION_SET_NAMES)}.")
    return evaluation_set


def parse_per_attribute_metric(query: T.Mapping[str, str]) -> str:
    """The `metric` parameter: a per-attribute metric, or '' for the attribute averages.

    Raises:
        ValueError: For an unknown metric.
    """
    metric_name = query.get("metric", "")
    if metric_name and metric_name not in PER_ATTRIBUTE_METRIC_NAMES:
        raise ValueError(f"Unknown per-attribute metric {metric_name!r}. Metrics:"
                         f" {', '.join(PER_ATTRIBUTE_METRIC_NAMES)}.")
    return metric_name


def compile_name_pattern(text: str) -> re.Pattern[str]:
    """The case-insensitive regular expression of a name filter, e.g. of the methods. Use it with
    `re.Pattern.search`, so that a part of a name matches, and '' matches all names.

    Raises:
        ValueError: If `text` is not a valid regular expression.
    """
    try:
        return re.compile(text, re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"Invalid regular expression {text!r}: {e}.") from None
