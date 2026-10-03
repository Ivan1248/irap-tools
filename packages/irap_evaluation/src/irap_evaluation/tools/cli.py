"""The `irap-eval` command-line interface.

Run `irap-eval <command> --help` for the options of each command.
"""

import argparse
import dataclasses as dc
import functools
import sys
import typing as T
from datetime import date
from pathlib import Path

import pandas as pd

from irap_data.metadata import load_irap_metadata

from ..bootstrap import BootstrapInterval, compare_methods, compute_bootstrap_intervals
from ..ensembling import ensemble_predictions
from ..evaluation import (
    ATTRIBUTE_SELECTIONS,
    MISSING_ATTRIBUTE_POLICIES,
    NULL_POLICIES,
    AttributeSelection,
    EvaluationResult,
    MissingAttributePolicy,
    NullPolicy,
    check_runs_distinct,
    evaluate_predictions,
    select_evaluated_attributes,
)
from ..evaluation_sets import (
    EvaluationSet,
    get_model_compatible_set,
    get_reference_set,
    select_evaluation_sets,
)
from ..metrics import (
    IRAP_MAIN_METRIC,
    MetricValues,
    get_irap_metric_names,
    map_metric_values,
    parse_metric_name,
)
from ..prediction_io import read_predictions, write_predictions
from ..predictions import MethodInfo, Predictions, get_common_dataset_and_split
from ..reports import coding_tables, evaluation_report

#: The metrics that `--table-metrics` accepts: the per-attribute metrics of the iRAP protocol.
TABLE_METRICS = tuple(n for n in get_irap_metric_names() if not parse_metric_name(n).is_averaged)


# Helpers ##########################################################################################

def _parse_context_offsets(text: str) -> tuple[int, ...]:
    return tuple(int(o) for o in text.split(","))


def _parse_method_context_offsets(items: T.Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Parses `NAME=0,-1,-4` items."""
    def parse(item: str) -> tuple[str, tuple[int, ...]]:
        name, separator, offsets = item.rpartition("=")
        if not separator or not name:
            raise ValueError(f"Expected NAME=OFFSETS, e.g. resnet=0,-1,-4, got {item!r}.")
        return name, _parse_context_offsets(offsets)

    return dict(map(parse, items))


def _read_prediction_files(paths: T.Sequence[Path]) -> list[Predictions]:
    """Reads prediction files of distinct runs (see `evaluation.check_runs_distinct`)."""
    predictions_seq = [read_predictions(p) for p in paths]
    check_runs_distinct(p.header.method for p in predictions_seq)
    return predictions_seq


def _print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}\n\n{evaluation_report.to_markdown_table(table)}")


def _print_partially_undefined_intervals(subject: str, intervals: MetricValues[BootstrapInterval],
                                         num_resamples: int) -> None:
    """Prints the intervals that leave out some resamples, if there are any."""
    if label_to_count := evaluation_report.get_partially_undefined_intervals(intervals):
        counts = ", ".join(f"{label} {n}" for label, n in label_to_count.items())
        print(f"{subject}: undefined resamples (NaN values, e.g. inf - inf), of {num_resamples},"
              f" left out of the intervals: {counts}.")


# Commands #########################################################################################

def run_validate(args: argparse.Namespace) -> None:
    for path in args.files:
        header = (predictions := read_predictions(path)).header
        num_invalid = sum(int((~v).sum()) for v in predictions.is_valid.values())
        print(f"{path}: valid. Run {header.method.run_label!r}, {header.dataset}/{header.split},"
              f" {predictions.output_kind}, {predictions.num_segments} segments,"
              f" {len(predictions.attributes)} attributes, {num_invalid} invalid cells.")


def _score(predictions: Predictions, evaluation_set: EvaluationSet, *, null_policy: NullPolicy,
           attribute_selection: AttributeSelection,
           missing_attribute_policy: MissingAttributePolicy, num_resamples: int,
           confidence: float, seed: int) -> EvaluationResult:
    """Scores predictions, with bootstrap intervals of the run if `num_resamples` > 0."""
    result = evaluate_predictions(
        predictions, evaluation_set, null_policy=null_policy,
        missing_attribute_policy=missing_attribute_policy,
        attributes=select_evaluated_attributes(evaluation_set, attribute_selection))
    if result.missing_attributes:
        print(f"Run {result.run_label!r}, {evaluation_set.name} set: scored as invalid, since"
              f" not predicted: {', '.join(result.missing_attributes)}.")
    if not num_resamples:
        return result
    return dc.replace(result, intervals=compute_bootstrap_intervals(
        [result], num_resamples=num_resamples, confidence=confidence, seed=seed))


def _select_run_evaluation_sets(
    predictions: Predictions,
    context_offsets: tuple[int, ...] | None,
    reference_set_f: T.Callable[[str, str], EvaluationSet],
    model_compatible_set_f: T.Callable[[str, str, tuple[int, ...]], EvaluationSet],
) -> list[EvaluationSet]:
    """`select_evaluation_sets` for a run, with its notes printed."""
    header = predictions.header
    name = header.method.run_label
    reference_set = reference_set_f(header.dataset, header.split)
    model_compatible_set = None if context_offsets is None else model_compatible_set_f(
        header.dataset, header.split, tuple(context_offsets))
    try:
        evaluation_sets, notes = select_evaluation_sets(predictions.segment_ids, reference_set,
                                                        model_compatible_set)
    except ValueError as e:
        raise ValueError(f"Run {name!r}: {e}") from e
    for note in notes:
        print(f"Run {name!r}: {note}")
    return evaluation_sets


def _compute_method_average_tables(
    results: T.Sequence[EvaluationResult], *, num_resamples: int, confidence: float, seed: int,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    """`evaluation_report.to_method_average_table` of each evaluation set with methods of
    several runs, with intervals if `num_resamples` > 0, whose undefined resamples are
    printed."""
    def to_table(key: tuple[str, str, str],
                 method_to_runs: T.Mapping[str, T.Sequence[EvaluationResult]]) -> pd.DataFrame:
        intervals = None if not num_resamples else {
            name: compute_bootstrap_intervals(runs, num_resamples=num_resamples,
                                              confidence=confidence, seed=seed)
            for name, runs in method_to_runs.items()}
        dataset, split, evaluation_set = key
        for name, method_intervals in (intervals or {}).items():
            _print_partially_undefined_intervals(
                f"Method {name!r}, {dataset}/{split}, {evaluation_set} set", method_intervals,
                num_resamples)
        return evaluation_report.to_method_average_table(method_to_runs, intervals)

    return {key: to_table(key, method_to_runs)
            for key, method_to_runs in evaluation_report.group_multi_run_methods(results).items()}


def run_evaluate(args: argparse.Namespace) -> None:
    predictions_seq = _read_prediction_files(args.files)
    evaluation_report.check_run_dir_names(p.header.method.run_label for p in predictions_seq)
    method_context_offsets = _parse_method_context_offsets(args.context_offsets)
    if unknown := set(method_context_offsets) - {p.header.method.name for p in predictions_seq}:
        raise ValueError(f"--context-offsets names unknown methods: {sorted(unknown)}.")
    metadata = load_irap_metadata(args.metadata_dir)
    # Cached, since the runs of a release and split share their reference set.
    reference_set_f = functools.cache(functools.partial(get_reference_set, metadata))
    model_compatible_set_f = functools.cache(functools.partial(get_model_compatible_set, metadata))
    score = functools.partial(
        _score, null_policy=args.null_policy, attribute_selection=args.attributes,
        missing_attribute_policy=args.missing_attribute_policy, num_resamples=args.bootstrap,
        confidence=args.confidence, seed=args.seed)
    results = [
        score(predictions, evaluation_set)
        for predictions in predictions_seq
        for evaluation_set in _select_run_evaluation_sets(
            predictions,
            method_context_offsets.get(predictions.header.method.name,
                                       predictions.header.context_offsets),
            reference_set_f, model_compatible_set_f)]
    if not results:
        raise ValueError("No run has an evaluation set with labels (see the notes above), so"
                         " nothing is scored.")
    for result in results:
        if result.intervals is not None:
            _print_partially_undefined_intervals(
                f"Run {result.run_label!r}, {result.evaluation_set.name} set", result.intervals,
                args.bootstrap)
    # Means over runs need 'first_class' (NullPolicy).
    if args.null_policy == "first_class":
        method_average_tables = _compute_method_average_tables(
            results, num_resamples=args.bootstrap, confidence=args.confidence, seed=args.seed)
    else:
        method_average_tables = {}
        if evaluation_report.group_multi_run_methods(results):
            print(f"No means over the runs of a method: the null policy {args.null_policy} scores"
                  f" each run on its own segments. Use --null-policy first_class for them.")

    evaluation_report.write_evaluation_reports(results, args.out, table_metrics=args.table_metrics,
                                               method_average_tables=method_average_tables)
    metric = args.table_metrics[0]
    evaluation_set_tables = evaluation_report.to_evaluation_set_tables(
        evaluation_report.to_long_table(results), [metric])
    for key, name_to_table in evaluation_set_tables.items():
        dataset, split, evaluation_set = key
        _print_table(f"Attribute averages per run, {dataset}/{split}, {evaluation_set} set",
                     name_to_table["averages"])
        if key in method_average_tables:
            _print_table(f"Attribute averages of the methods with several runs (means over runs),"
                         f" {dataset}/{split}, {evaluation_set} set",
                         method_average_tables[key])
        _print_table(f"{metric} per attribute, {dataset}/{split}, {evaluation_set} set",
                     name_to_table[f"per_attribute_{metric}"])
    print(f"\nWrote the results to {args.out}.")


def run_compare(args: argparse.Namespace) -> None:
    predictions_seq = _read_prediction_files(args.files)
    method_names = {p.header.method.name for p in predictions_seq}
    if missing := [n for n in (args.a, args.b) if n not in method_names]:
        raise ValueError(f"No files of the methods {missing}. The files are of"
                         f" {sorted(method_names)}.")
    if other := sorted(method_names - {args.a, args.b}):
        raise ValueError(f"Files of other methods than a and b: {other}.")
    dataset, split = get_common_dataset_and_split(p.header for p in predictions_seq)
    reference_set = get_reference_set(load_irap_metadata(args.metadata_dir), dataset, split)
    attributes = select_evaluated_attributes(reference_set, args.attributes)

    def evaluate_runs(method_name: str) -> list[EvaluationResult]:
        return [evaluate_predictions(p, reference_set, attributes=attributes,
                                     null_policy="first_class")
                for p in predictions_seq if p.header.method.name == method_name]

    runs_a, runs_b = evaluate_runs(args.a), evaluate_runs(args.b)
    differences = compare_methods(runs_a, runs_b, args.metrics, num_resamples=args.bootstrap,
                                  seed=args.seed, confidence=args.confidence)
    _print_table(f"a = {args.a} ({len(runs_a)} runs), b = {args.b} ({len(runs_b)} runs),"
                 f" {dataset}/{split}, reference set, {reference_set.num_segments} segments,"
                 f" {len(reference_set.sequence_ids)} road sequences, {args.bootstrap} resamples,"
                 f" {args.confidence:.0%} intervals",
                 evaluation_report.to_comparison_table(runs_a, runs_b, differences))
    _print_partially_undefined_intervals(
        "a - b", map_metric_values(lambda d: d.interval, differences), args.bootstrap)


def run_ensemble(args: argparse.Namespace) -> None:
    predictions_seq = _read_prediction_files(args.files)
    ensemble = ensemble_predictions(
        predictions_seq, MethodInfo(name=args.name, seed=args.method_seed),
        weights=args.weights, intersect_segments=args.intersect_segments)
    write_predictions(args.output, ensemble)
    print(f"Wrote the mean of {len(predictions_seq)} prediction files"
          f" ({ensemble.num_segments} segments) to {args.output}.")


def run_export_coding_table(args: argparse.Namespace) -> None:
    predictions = read_predictions(args.file)
    metadata = load_irap_metadata(args.metadata_dir)
    template = coding_tables.load_coding_table_template(args.template)
    table = coding_tables.predictions_to_coding_table(
        predictions, metadata, template,
        coder_name=args.coder_name or predictions.header.method.name,
        coding_date=args.coding_date)
    confidence_table = (coding_tables.predictions_to_confidence_table(predictions, metadata,
                                                                      template)
                        if args.confidence_table else None)
    for path in coding_tables.write_coding_table(args.output, table, confidence_table):
        print(f"Wrote {path}.")
    if unmapped := coding_tables.get_unmapped_attributes(template, predictions.attributes):
        print(f"Predicted attributes without a column (not exported): {unmapped}.")
    print(f"Blank columns: {coding_tables.get_unfilled_columns(template, predictions.attributes)}.")


# Argument parsing #################################################################################

def _add_scoring_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--attributes", choices=ATTRIBUTE_SELECTIONS, default="canonical",
                        help="canonical: the iRAP attribute subset, as in vidlu_irap_gaim."
                             " all: every attribute of the dataset. Attributes without a label"
                             " in the evaluation set are left out. Default: canonical.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed. Default: 0.")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level of the intervals. Default: 0.95.")


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="irap-eval", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="Check prediction files.")
    validate.add_argument("files", nargs="+", type=Path)
    validate.set_defaults(run=run_validate)

    evaluate = commands.add_parser(
        "evaluate", help="Score prediction files on the reference set and, if the context offsets"
                         " of a method are known, on its model-compatible set. Methods with several"
                         " runs also get the means over their runs.")
    evaluate.add_argument("metadata_dir", type=Path, help="Metadata directory of the release.")
    evaluate.add_argument("files", nargs="+", type=Path, help="Prediction files.")
    evaluate.add_argument("--out", type=Path, required=True, help="Output directory.")
    evaluate.add_argument("--context-offsets", nargs="*", default=[], metavar="NAME=OFFSETS",
                          help="Context offsets of methods, e.g. resnet=0,-1,-4. They override"
                               " the offsets in the file headers.")
    evaluate.add_argument("--bootstrap", type=int, default=0, metavar="NUM_RESAMPLES",
                          help="Number of bootstrap resamples for confidence intervals. The"
                               " intervals of a run cover the road sequences, those of a method"
                               " with several runs also the spread of its runs."
                               " Default: 0 (no intervals).")
    evaluate.add_argument("--table-metrics", nargs="+", choices=TABLE_METRICS,
                          default=[IRAP_MAIN_METRIC], metavar="METRIC",
                          help=f"Metrics of the per-attribute tables, which end with the attribute"
                               f" average, from {', '.join(TABLE_METRICS)}. The metrics of"
                               f" distributions need probs files. Default: {IRAP_MAIN_METRIC}.")
    evaluate.add_argument("--missing-attribute-policy", choices=MISSING_ATTRIBUTE_POLICIES,
                          default="error",
                          help="How a selected attribute that a file does not predict is scored."
                               " error: not at all, the command fails. invalid: as if every cell"
                               " of it were invalid (see --null-policy). Default: error.")
    evaluate.add_argument("--null-policy", choices=NULL_POLICIES, default="first_class",
                          help="How the results of the runs score invalid cells. first_class: as"
                               " class 0 in the metrics of the predicted classes and as the"
                               " uniform distribution in NLL and Brier. exclude: not scored, so"
                               " each run is scored on its own segments and no means over the"
                               " runs of a method are written (see"
                               " irap_evaluation.evaluation.NullPolicy). Default: first_class.")
    _add_scoring_arguments(evaluate)
    evaluate.set_defaults(run=run_evaluate)

    compare = commands.add_parser(
        "compare", help="Compare two methods, each with one or more runs, on the reference set,"
                        " with a bootstrap over road sequences and the spread of the runs."
                        " Invalid cells are scored with the null policy first_class.")
    compare.add_argument("metadata_dir", type=Path)
    compare.add_argument("files", nargs="+", type=Path,
                         help="The prediction files of the runs of both methods.")
    compare.add_argument("--a", required=True, metavar="METHOD", help="Name of method a.")
    compare.add_argument("--b", required=True, metavar="METHOD", help="Name of method b.")
    compare.add_argument("--metrics", nargs="+", default=None,
                         help="Metrics to compare, e.g. amF1 or the per-attribute mF1."
                              " Default: all attribute averages.")
    compare.add_argument("--bootstrap", type=int, default=1000, metavar="NUM_RESAMPLES",
                         help="Number of bootstrap resamples. Default: 1000.")
    _add_scoring_arguments(compare)
    compare.set_defaults(run=run_compare)

    ensemble = commands.add_parser(
        "ensemble", help="Combine prediction files into the weighted mean of their distributions.")
    ensemble.add_argument("files", nargs="+", type=Path)
    ensemble.add_argument("--name", required=True, help="Method name of the ensemble.")
    ensemble.add_argument("--method-seed", type=int, default=None,
                          help="Seed of the ensemble, if it is one run of a method with several"
                               " runs. Default: none.")
    ensemble.add_argument("-o", "--output", type=Path, required=True)
    ensemble.add_argument("--weights", type=float, nargs="+", default=None,
                          help="A positive weight per file. Default: equal weights.")
    ensemble.add_argument("--intersect-segments", action="store_true",
                          help="Combine the segments that all files predict, rather than"
                               " requiring the same segments.")
    ensemble.set_defaults(run=run_ensemble)

    export = commands.add_parser("export-coding-table",
                                 help="Write predictions as an iRAP coding table.")
    export.add_argument("metadata_dir", type=Path)
    export.add_argument("file", type=Path)
    export.add_argument("-o", "--output", type=Path, required=True, help="A .xlsx or .csv path.")
    export.add_argument("--coder-name", default=None, help="Default: the method name.")
    export.add_argument("--coding-date", default=date.today().isoformat(),
                        help="YYYY-MM-DD. Default: today.")
    export.add_argument("--confidence-table", action="store_true",
                        help="Also write the probability of each exported code (probs files), as"
                             " a second sheet of a .xlsx file or as <stem>.confidence.csv.")
    export.add_argument("--template", type=Path, default=None,
                        help="A coding-table template JSON. Default: the packaged iRAP layout.")
    export.set_defaults(run=run_export_coding_table)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_argument_parser().parse_args(argv)
    try:
        args.run(args)
    except (ValueError, FileNotFoundError) as e:  # including PredictionFormatError
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
