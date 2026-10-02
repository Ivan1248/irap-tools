"""Tables and JSON documents of evaluation results, and the files of `irap-eval evaluate`."""

import collections
import dataclasses as dc
import json
import math
import re
import typing as T
from pathlib import Path

import pandas as pd

from ..bootstrap import BootstrapInterval, MetricDifference, compute_method_metrics
from ..evaluation import (
    ClassMetrics,
    EvaluationResult,
    compute_class_metrics,
    group_runs_by_method,
)
from ..metrics import MetricValues, map_metric_values, parse_metric_name

V = T.TypeVar("V")

#: The `attribute` of an attribute average in the long table.
AVERAGE_ATTRIBUTE = "average"
LONG_TABLE_COLUMNS = ("run", "method", "seed", "dataset", "split", "evaluation_set",
                      "context_offsets", "null_policy", "attribute", "metric", "value", "ci_low",
                      "ci_high", "ci_num_undefined")
#: Column-name suffix -> long-table column, of the interval bounds in `to_average_table`.
_INTERVAL_SUFFIX_TO_COLUMN = {" low": "ci_low", " high": "ci_high"}


def _to_rows(result: EvaluationResult) -> T.Iterator[dict[str, T.Any]]:
    evaluation_set = result.evaluation_set
    common = dict(run=result.run_label, method=result.method.name, seed=result.method.seed,
                  dataset=evaluation_set.dataset, split=evaluation_set.split,
                  evaluation_set=evaluation_set.name,
                  context_offsets=",".join(map(str, evaluation_set.context_offsets)),
                  null_policy=result.null_policy)
    intervals = result.intervals or map_metric_values(lambda _: None, result.metrics)

    def make_row(attribute, metric, value, interval):
        return dict(common, attribute=attribute, metric=metric, value=value,
                    ci_low=math.nan if interval is None else interval.low,
                    ci_high=math.nan if interval is None else interval.high,
                    ci_num_undefined=pd.NA if interval is None else interval.num_undefined)

    for name, value in result.metrics.averages.items():
        yield make_row(AVERAGE_ATTRIBUTE, parse_metric_name(name).name, value,
                       intervals.averages[name])
    for name, attr_to_value in result.metrics.per_attribute.items():
        for attr, value in attr_to_value.items():
            yield make_row(attr, name, value, intervals.per_attribute[name][attr])


def to_long_table(results: T.Iterable[EvaluationResult]) -> pd.DataFrame:
    """One row per (result, attribute or average, scalar metric).

    An attribute average `a<X>` gives the row (`attribute='average'`, `metric=X`), so that it
    lines up with the per-attribute rows of `X`. Without intervals, the confidence bounds are NaN
    and `ci_num_undefined` (`BootstrapInterval.num_undefined`) is missing.
    """
    table = pd.DataFrame([row for r in results for row in _to_rows(r)],
                         columns=list(LONG_TABLE_COLUMNS))
    return table.astype({"ci_num_undefined": "Int64"})


def to_per_attribute_table(long_table: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Attribute rows × run columns of one metric, with the attribute average last.

    Args:
        long_table: From `to_long_table`, with one row per (run, attribute) for `metric`,
            e.g. restricted to one evaluation set.
        metric: A per-attribute metric name, e.g. 'mF1'.

    Raises:
        ValueError: If `metric` has no per-attribute rows, or a (run, attribute) pair has
            several rows.
    """
    rows = long_table[long_table["metric"] == metric]
    if rows.empty:
        per_attribute = long_table.loc[long_table["attribute"] != AVERAGE_ATTRIBUTE, "metric"]
        raise ValueError(f"The table has no per-attribute values of {metric!r}. It has"
                         f" {list(dict.fromkeys(per_attribute))}. The attribute average aX is the"
                         f" last row of the table of X.")
    if rows.duplicated(["run", "attribute"]).any():
        raise ValueError(f"Several rows per (run, attribute) for {metric!r}. Restrict the"
                         f" table to one evaluation set, dataset and split.")
    table = rows.pivot(index="attribute", columns="run", values="value")
    attributes = [a for a in dict.fromkeys(rows["attribute"]) if a != AVERAGE_ATTRIBUTE]
    order = attributes + ([AVERAGE_ATTRIBUTE] if AVERAGE_ATTRIBUTE in table.index else [])
    return table.loc[order, list(dict.fromkeys(rows["run"]))]


def to_average_table(long_table: pd.DataFrame, with_intervals: bool = False) -> pd.DataFrame:
    """Run rows × attribute-average columns, in the metric order of `long_table`.

    Args:
        long_table: From `to_long_table`, restricted to one evaluation set, dataset and split.
        with_intervals: Whether to add `<metric> low` and `<metric> high` columns.

    Raises:
        ValueError: If a (run, metric) pair has several rows.
    """
    rows = long_table[long_table["attribute"] == AVERAGE_ATTRIBUTE]
    if rows.duplicated(["run", "metric"]).any():
        raise ValueError("Several rows per (run, metric). Restrict the table to one evaluation"
                         " set, dataset and split.")
    runs, metrics = list(dict.fromkeys(rows["run"])), list(dict.fromkeys(rows["metric"]))
    suffix_to_column = {"": "value", **(_INTERVAL_SUFFIX_TO_COLUMN if with_intervals else {})}
    pivots = {suffix: rows.pivot(index="run", columns="metric", values=column).loc[runs]
              for suffix, column in suffix_to_column.items()}
    return pd.DataFrame({f"{metric}{suffix}": pivots[suffix][metric]
                         for metric in metrics for suffix in suffix_to_column}).rename_axis("run")


def to_class_table(class_metrics: ClassMetrics) -> pd.DataFrame:
    """Class rows (indexed by iRAP code) with the class value, support and each metric."""
    return pd.DataFrame(
        {"value": class_metrics.values, "support": class_metrics.support,
         **class_metrics.metrics},
        index=pd.Index(class_metrics.irap_codes, name="irap_code"))


def to_method_average_table(
    method_to_runs: T.Mapping[str, T.Sequence[EvaluationResult]],
    method_to_intervals: T.Mapping[str, MetricValues[BootstrapInterval]] | None = None,
) -> pd.DataFrame:
    """Method rows × attribute-average columns: the number of runs, their null policy, and the
    mean of each metric over the runs (`bootstrap.compute_method_metrics`).

    Args:
        method_to_runs: Method name -> its runs on one evaluation set.
        method_to_intervals: Method name -> `bootstrap.compute_bootstrap_intervals` of its runs,
            which add `<metric> low` and `<metric> high` columns, or None.
    """
    def make_row(name: str, runs: T.Sequence[EvaluationResult]) -> dict[str, T.Any]:
        # `compute_method_metrics` checks that the runs share the null policy.
        row: dict[str, T.Any] = {"runs": len(runs), "null_policy": runs[0].null_policy}
        for metric, value in compute_method_metrics(runs).averages.items():
            column = parse_metric_name(metric).name
            row[column] = value
            if method_to_intervals is not None:
                interval = method_to_intervals[name].averages[metric]
                row[f"{column} low"], row[f"{column} high"] = interval.low, interval.high
        return row

    return pd.DataFrame.from_dict(
        {name: make_row(name, runs) for name, runs in method_to_runs.items()},
        orient="index").rename_axis("method")


def to_comparison_table(
    runs_a: T.Sequence[EvaluationResult],
    runs_b: T.Sequence[EvaluationResult],
    differences: MetricValues[MetricDifference],
) -> pd.DataFrame:
    """One row per compared metric (see `to_labeled_metric_values`) with the values of both
    methods (means over their runs) and their difference from `bootstrap.compare_methods`.

    An `undefined` column with `BootstrapInterval.num_undefined` is added if an interval leaves out
    resamples.
    """
    def make_row(d: MetricDifference, a: float, b: float) -> dict[str, float]:
        return {"a": a, "b": b, "a - b": d.difference, "low": d.interval.low,
                "high": d.interval.high, "undefined": d.interval.num_undefined}

    rows = map_metric_values(make_row, differences, compute_method_metrics(runs_a),
                             compute_method_metrics(runs_b))
    table = pd.DataFrame.from_dict(to_labeled_metric_values(rows),
                                   orient="index").rename_axis("metric")
    return table if table["undefined"].any() else table.drop(columns="undefined")


def to_labeled_metric_values(values: MetricValues[V]) -> dict[str, V]:
    """Label -> value: the name of an attribute average, `<name> (<attribute>)` for a
    per-attribute metric."""
    return {**values.averages,
            **{f"{name} ({attr})": value for name, attr_to_value in values.per_attribute.items()
               for attr, value in attr_to_value.items()}}


def get_partially_undefined_intervals(
        intervals: MetricValues[BootstrapInterval]) -> dict[str, int]:
    """Label (see `to_labeled_metric_values`) -> `BootstrapInterval.num_undefined`, for the
    intervals that leave out some but not all resamples. Such bounds hold only for the resamples where the
    value is defined, unlike NaN bounds, which show that every resample is undefined."""
    return {label: interval.num_undefined
            for label, interval in to_labeled_metric_values(intervals).items()
            if interval.num_undefined > 0 and not math.isnan(interval.low)}


def to_markdown_table(table: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    """A GitHub-flavored markdown table of `table`, including its index."""
    def format_value(value) -> str:
        if isinstance(value, float):
            return "" if math.isnan(value) else float_format.format(value)
        return str(value)

    header = [str(table.index.name or ""), *map(str, table.columns)]
    body = [[str(index), *map(format_value, row)] for index, row in zip(table.index, table.values)]
    lines = [header, ["---"] * len(header), *body]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in lines)


def _to_json_value(value: T.Any) -> T.Any:
    """`value` with NaN and infinite floats, also nested, replaced by None, since standard JSON
    has no such numbers."""
    if isinstance(value, T.Mapping):
        return {k: _to_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def to_json_dict(result: EvaluationResult) -> dict[str, T.Any]:
    """A JSON-compatible document of a result, including per-class scores and confusion
    matrices (rows = ground truth) keyed by iRAP code.

    Undefined values (NaN, e.g. the MCC of an attribute with one class) are None."""
    evaluation_set = result.evaluation_set
    attr_to_class_metrics = compute_class_metrics(result, ["P", "R", "F1"])

    def get_class_report(attr: str) -> dict[str, T.Any]:
        class_metrics = attr_to_class_metrics[attr]
        return {"irap_codes": list(class_metrics.irap_codes),
                "values": list(class_metrics.values),
                "support": class_metrics.support.tolist(),
                "precision": class_metrics.metrics["P"].tolist(),
                "recall": class_metrics.metrics["R"].tolist(),
                "f1": class_metrics.metrics["F1"].tolist(),
                "confusion_matrix": class_metrics.confusion_matrix.tolist()}

    return _to_json_value({
        "method": dc.asdict(result.method),
        "output_kind": result.output_kind,
        "dataset": evaluation_set.dataset,
        "split": evaluation_set.split,
        "evaluation_set": evaluation_set.name,
        "context_offsets": list(evaluation_set.context_offsets),
        "null_policy": result.null_policy,
        "num_segments": evaluation_set.num_segments,
        "num_sequences": len(evaluation_set.sequence_ids),
        "attributes": list(result.attributes),
        "missing_attributes": list(result.missing_attributes),
        "num_invalid": dict(result.num_invalid),
        "metrics": dc.asdict(result.metrics),
        "intervals": None if result.intervals is None else dc.asdict(result.intervals),
        "classes": {attr: get_class_report(attr) for attr in result.attributes},
    })


# Files of `irap-eval evaluate` ####################################################################

#: The long-table columns that identify an evaluation set.
EVALUATION_SET_COLUMNS = ("dataset", "split", "evaluation_set")


def to_evaluation_set_tables(
    long_table: pd.DataFrame,
    table_metrics: T.Sequence[str],
    with_intervals: bool = False,
) -> dict[tuple[str, str, str], dict[str, pd.DataFrame]]:
    """The tables of each evaluation set of `long_table`.

    Returns:
        (dataset, split, evaluation set) -> table name -> table: 'averages' from
        `to_average_table`, and 'per_attribute_<metric>' from `to_per_attribute_table` for each
        metric of `table_metrics`.
    """
    return {key: {"averages": to_average_table(table, with_intervals),
                  **{f"per_attribute_{metric}": to_per_attribute_table(table, metric)
                     for metric in table_metrics}}
            for key, table in long_table.groupby(list(EVALUATION_SET_COLUMNS), sort=False)}


def to_file_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name)


def check_run_dir_names(run_labels: T.Iterable[str]) -> None:
    """Checks that the runs get different directories in `write_evaluation_reports`.

    Raises:
        ValueError: If several run labels (`MethodInfo.run_label`) map to the same directory
            name.
    """
    num_runs_per_dir = collections.Counter(map(to_file_name, set(run_labels)))
    if clashes := sorted(d for d, n in num_runs_per_dir.items() if n > 1):
        raise ValueError(f"Several runs map to the output directories {clashes}. Rename the"
                         f" methods.")


def group_multi_run_methods(
    results: T.Iterable[EvaluationResult],
) -> dict[tuple[str, str, str], dict[str, tuple[EvaluationResult, ...]]]:
    """(dataset, split, evaluation set name) -> method name -> its runs, for the methods with
    several runs on that set.

    Raises:
        ValueError: If the runs of a method are not distinct (see
            `evaluation.check_runs_distinct`).
    """
    key_to_results: dict[tuple[str, str, str], list[EvaluationResult]] = {}
    for r in results:
        s = r.evaluation_set
        key_to_results.setdefault((s.dataset, s.split, s.name), []).append(r)
    key_to_methods = {key: {name: runs for name, runs in group_runs_by_method(rs).items()
                            if len(runs) > 1}
                      for key, rs in key_to_results.items()}
    return {key: methods for key, methods in key_to_methods.items() if methods}


def write_evaluation_reports(
    results: T.Sequence[EvaluationResult],
    out_dir: str | Path,
    *,
    table_metrics: T.Sequence[str],
    method_average_tables: T.Mapping[tuple[str, str, str], pd.DataFrame] | None = None,
) -> list[Path]:
    """Writes the files of `irap-eval evaluate` (see the package README) to `out_dir`.

    Nothing is written if a check fails.

    Args:
        results: At most one result per run and evaluation set.
        table_metrics: The metrics of the per-attribute tables, e.g. 'mF1'.
        method_average_tables: (dataset, split, evaluation set name) -> a
            `to_method_average_table`, written as `method_averages_<key>.csv`.

    Returns:
        The written files.

    Raises:
        ValueError: If several runs map to one directory (see `check_run_dir_names`), two files
            map to one path, e.g. for evaluation sets named 'a b' and 'a_b', or a table cannot be
            made (see `to_per_attribute_table` and `to_average_table`).
    """
    check_run_dir_names(r.run_label for r in results)
    out_dir = Path(out_dir)
    with_intervals = any(r.intervals is not None for r in results)
    long_table = to_long_table(results)
    table_files = [(out_dir / "summary_long.csv", long_table.set_index("run"))]
    for key, name_to_table in to_evaluation_set_tables(long_table, table_metrics,
                                                       with_intervals).items():
        table_files += [(out_dir / to_file_name(f"{table_name}_{'_'.join(key)}.csv"), table)
                        for table_name, table in name_to_table.items()]
    table_files += [(out_dir / to_file_name(f"method_averages_{'_'.join(key)}.csv"), table)
                    for key, table in (method_average_tables or {}).items()]
    json_files = [
        (out_dir / to_file_name(r.run_label)
         / to_file_name(f"metrics_{r.evaluation_set.name}.json"),
         json.dumps(to_json_dict(r), indent=2, ensure_ascii=False, allow_nan=False))
        for r in results]
    num_files_per_path = collections.Counter(path for path, _ in [*table_files, *json_files])
    if clashes := sorted(str(p) for p, n in num_files_per_path.items() if n > 1):
        raise ValueError(f"Several outputs map to the files {clashes}. Pass at most one result per"
                         f" run and evaluation set, with distinct evaluation-set names.")
    path_to_table, path_to_json = dict(table_files), dict(json_files)

    for path, text in path_to_json.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, table in path_to_table.items():
        table.to_csv(path)
    return [*path_to_json, *path_to_table]
