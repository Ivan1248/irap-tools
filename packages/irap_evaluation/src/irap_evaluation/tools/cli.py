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
from ..ensembling import MEMBER_COMBINATIONS, ensemble_predictions
from ..evaluation import (
    ATTRIBUTE_SELECTIONS,
    MISSING_ATTRIBUTE_POLICIES,
    NULL_POLICIES,
    AttributeSelection,
    EvaluationResult,
    MissingAttributePolicy,
    NullPolicy,
    check_metric_names,
    evaluate_predictions,
    get_default_metric_names,
    select_evaluated_attributes,
)
from ..evaluation_sets import (
    MISSING_SEGMENT_POLICIES,
    EvaluationSet,
    MissingSegmentPolicy,
    get_model_compatible_set,
    get_reference_set,
    get_shared_evaluation_sets,
    select_evaluation_sets,
)
from ..methods import check_prediction_file_models
from ..metric_names import (
    IRAP_MAIN_METRIC,
    add_per_attribute_metric_names,
    parse_metric_name,
)
from ..metrics import MetricValues, map_metric_values
from ..prediction_io import read_predictions, write_predictions
from ..predictions import (
    ModelInfo,
    Predictions,
    check_split_names,
    explain_split_use,
    get_common_dataset_and_split,
)
from ..reports import coding_tables, evaluation_report


# Helpers ##########################################################################################

def _parse_offsets(text: str) -> tuple[int, ...]:
    """Parses '0,-1,-4', or '' for no offsets."""
    return tuple(int(o) for o in text.split(",")) if text else ()


def _parse_method_offsets(items: T.Sequence[str]) -> dict[str, tuple[int, ...]]:
    """Parses `NAME=0,-1,-4` items, `NAME=` for no offsets."""
    def parse(item: str) -> tuple[str, tuple[int, ...]]:
        name, separator, offsets = item.rpartition("=")
        if not separator or not name:
            raise ValueError(f"Expected NAME=OFFSETS, e.g. resnet=0,-1,-4, got {item!r}.")
        return name, _parse_offsets(offsets)

    return dict(map(parse, items))


def _read_prediction_files(paths: T.Sequence[Path]) -> list[Predictions]:
    """Reads prediction files of models that `check_prediction_file_models` accepts."""
    predictions_seq = [read_predictions(p) for p in paths]
    check_prediction_file_models([p.header for p in predictions_seq])
    return predictions_seq


def _get_file_metric_names(metric_names: T.Sequence[str] | None, path: Path,
                           predictions: Predictions) -> tuple[str, ...]:
    """Returns the metrics to compute for a file: `metric_names` (`--metric-names`) with the
    per-attribute values of their aggregates, which the reports and the aggregates over attribute
    subsets need, or `get_default_metric_names` of its output kind if None.

    Raises:
        ValueError: As `check_metric_names` does, with the file and model in the message.
    """
    if metric_names is None:
        return get_default_metric_names(predictions.output_kind)
    file_metric_names = add_per_attribute_metric_names(metric_names)
    try:
        check_metric_names(file_metric_names, predictions.output_kind)
    except ValueError as e:
        raise ValueError(f"{path}, model {predictions.header.model.label!r}, --metric-names:"
                         f" {e}") from e
    return file_metric_names


def _check_table_metrics(table_metrics: T.Sequence[str],
                         file_metric_names: T.Iterable[T.Sequence[str]]) -> None:
    """Checks that the metrics of the per-attribute tables (`--table-metrics`) are computed per
    attribute for some file.

    Args:
        file_metric_names: The metrics that are computed for each file.

    Raises:
        ValueError: If one is not.
    """
    per_attribute = list(dict.fromkeys(
        n for names in file_metric_names for n in names
        if not parse_metric_name(n).is_attribute_aggregate))
    if unknown := [n for n in table_metrics if n not in per_attribute]:
        raise ValueError(f"--table-metrics {unknown} are not computed per attribute. The"
                         f" per-attribute metrics of the files: {', '.join(per_attribute)}. Add"
                         f" metrics with --metric-names.")


def _explain_split_use(model: ModelInfo, split: str) -> str | None:
    """A note on the scores of `model` on `split` if it used the split, or may have used it."""
    return explain_split_use(model.get_split_use(split), split)


def _format_training_splits(model: ModelInfo) -> str:
    training = model.training_splits
    return "unknown splits" if training is None else str(list(training))


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
        print(f"{path}: valid. Model {header.model.label!r}, {header.dataset}/{header.split},"
              f" trained on {_format_training_splits(header.model)},"
              f" early stopped on {list(header.model.early_stopping_splits)},"
              f" {predictions.output_kind}, {predictions.num_segments} segments,"
              f" {len(predictions.attributes)} attributes, {num_invalid} invalid attribute"
              f" predictions.")


def _score(predictions: Predictions, evaluation_set: EvaluationSet, *, null_policy: NullPolicy,
           attribute_selection: AttributeSelection,
           missing_attribute_policy: MissingAttributePolicy,
           missing_segment_policy: MissingSegmentPolicy, metric_names: T.Sequence[str],
           num_resamples: int, confidence: float, seed: int) -> EvaluationResult:
    """Scores predictions, with bootstrap intervals of the model if `num_resamples` > 0."""
    result = evaluate_predictions(
        predictions, evaluation_set, null_policy=null_policy,
        missing_attribute_policy=missing_attribute_policy,
        missing_segment_policy=missing_segment_policy,
        attributes=select_evaluated_attributes(evaluation_set, attribute_selection),
        metric_names=metric_names)
    if result.missing_attributes:
        print(f"Model {result.model.label!r}, {evaluation_set.name} set: scored as invalid, since"
              f" not predicted: {', '.join(result.missing_attributes)}.")
    if result.missing_segment_ids:
        print(f"Model {result.model.label!r}, {evaluation_set.name} set: segments without"
              f" predictions, scored as invalid: {len(result.missing_segment_ids)}.")
    if not num_resamples:
        return result
    return dc.replace(result, intervals=compute_bootstrap_intervals(
        [result], num_resamples=num_resamples, confidence=confidence, seed=seed))


def _select_model_evaluation_sets(
    predictions: Predictions,
    context_offsets: tuple[int, ...] | None,
    in_split_offsets: tuple[int, ...],
    shared_sets_f: T.Callable[[str, str], T.Mapping[str, EvaluationSet]],
    model_compatible_set_f: T.Callable[[str, str, tuple[int, ...], tuple[int, ...]],
                                       EvaluationSet],
    missing_segment_policy: MissingSegmentPolicy,
) -> tuple[EvaluationSet, ...]:
    """`select_evaluation_sets` for a model, with its notes printed, and a note if it used the
    split (`_explain_split_use`).

    Raises:
        ValueError: If there are in-split offsets without context offsets.
    """
    header = predictions.header
    name = header.model.label
    if context_offsets is None and in_split_offsets:
        raise ValueError(f"Model {name!r}: in-split offsets {in_split_offsets} without context"
                         f" offsets.")
    model_compatible_set = None if context_offsets is None else model_compatible_set_f(
        header.dataset, header.split, tuple(context_offsets), tuple(in_split_offsets))
    try:
        selection = select_evaluation_sets(
            predictions.segment_ids, shared_sets_f(header.dataset, header.split).values(),
            model_compatible_set, missing_segment_policy)
    except ValueError as e:
        raise ValueError(f"Model {name!r}: {e}") from e
    notes = list(selection.notes)
    if selection.evaluation_sets and (split_use_note := _explain_split_use(header.model,
                                                                           header.split)):
        notes.append(split_use_note)
    for note in notes:
        print(f"Model {name!r}: {note}")
    return selection.evaluation_sets


def _compute_method_average_tables(
    results: T.Sequence[EvaluationResult], *, num_resamples: int, confidence: float, seed: int,
) -> dict[tuple[str, str, str], pd.DataFrame]:
    """`evaluation_report.to_method_average_table` of each evaluation set with methods of
    several models, with intervals if `num_resamples` > 0, whose undefined resamples are
    printed."""
    def to_table(key: tuple[str, str, str],
                 method_to_models: T.Mapping[str, T.Sequence[EvaluationResult]]) -> pd.DataFrame:
        intervals = None if not num_resamples else {
            name: compute_bootstrap_intervals(models, num_resamples=num_resamples,
                                              confidence=confidence, seed=seed)
            for name, models in method_to_models.items()}
        dataset, split, evaluation_set = key
        for name, method_intervals in (intervals or {}).items():
            _print_partially_undefined_intervals(
                f"Method {name!r}, {dataset}/{split}, {evaluation_set} set", method_intervals,
                num_resamples)
        return evaluation_report.to_method_average_table(method_to_models, intervals)

    return {key: to_table(key, method_to_models)
            for key, method_to_models
            in evaluation_report.group_multi_model_methods(results).items()}


def run_evaluate(args: argparse.Namespace) -> None:
    predictions_seq = _read_prediction_files(args.files)
    evaluation_report.check_model_dir_names(p.header.model for p in predictions_seq)
    method_context_offsets = _parse_method_offsets(args.context_offsets)
    method_in_split_offsets = _parse_method_offsets(args.in_split_offsets)
    method_names = {p.header.model.method_name for p in predictions_seq}
    for option, method_offsets in [("--context-offsets", method_context_offsets),
                                   ("--in-split-offsets", method_in_split_offsets)]:
        if unknown := set(method_offsets) - method_names:
            raise ValueError(f"{option} names unknown methods: {sorted(unknown)}.")
    file_metric_names = [_get_file_metric_names(args.metric_names, path, p)
                         for path, p in zip(args.files, predictions_seq, strict=True)]
    _check_table_metrics(args.table_metrics, file_metric_names)
    metadata = load_irap_metadata(args.metadata_dir)
    for predictions in predictions_seq:
        check_split_names(predictions.header.model, metadata.splits)
    # Cached, since the shared sets are the same for all models of a release and split.
    shared_sets_f = functools.cache(functools.partial(get_shared_evaluation_sets, metadata))
    model_compatible_set_f = functools.cache(functools.partial(get_model_compatible_set, metadata))
    score = functools.partial(
        _score, null_policy=args.null_policy, attribute_selection=args.attributes,
        missing_attribute_policy=args.missing_attribute_policy,
        missing_segment_policy=args.missing_segment_policy, num_resamples=args.bootstrap,
        confidence=args.confidence, seed=args.seed)
    results = [
        score(predictions, evaluation_set, metric_names=metric_names)
        for predictions, metric_names in zip(predictions_seq, file_metric_names, strict=True)
        for evaluation_set in _select_model_evaluation_sets(
            predictions,
            method_context_offsets.get(predictions.header.model.method_name,
                                       predictions.header.context_offsets),
            method_in_split_offsets.get(predictions.header.model.method_name,
                                        predictions.header.in_split_offsets),
            shared_sets_f, model_compatible_set_f, args.missing_segment_policy)]
    if not results:
        raise ValueError("No model has an evaluation set with labels (see the notes above), so"
                         " nothing is scored.")
    for result in results:
        if result.intervals is not None:
            _print_partially_undefined_intervals(
                f"Model {result.model.label!r}, {result.evaluation_set.name} set",
                result.intervals, args.bootstrap)
    # Means over models need 'first_class' (NullPolicy).
    if args.null_policy == "first_class":
        method_average_tables = _compute_method_average_tables(
            results, num_resamples=args.bootstrap, confidence=args.confidence, seed=args.seed)
    else:
        method_average_tables = {}
        if evaluation_report.group_multi_model_methods(results):
            print(f"No means over the models of a method: the null policy {args.null_policy}"
                  f" scores each model on its own segments. Use --null-policy first_class for"
                  f" them.")

    evaluation_report.write_evaluation_reports(results, args.out, table_metrics=args.table_metrics,
                                               method_average_tables=method_average_tables)
    metric = args.table_metrics[0]
    evaluation_set_tables = evaluation_report.to_evaluation_set_tables(
        evaluation_report.to_long_table(results), [metric])
    for key, name_to_table in evaluation_set_tables.items():
        dataset, split, evaluation_set = key
        _print_table(f"Attribute averages and totals per model, {dataset}/{split},"
                     f" {evaluation_set} set", name_to_table["averages"])
        if key in method_average_tables:
            _print_table(f"Attribute averages and totals of the methods with several models"
                         f" (means over models), {dataset}/{split}, {evaluation_set} set",
                         method_average_tables[key])
        _print_table(f"{metric} per attribute, {dataset}/{split}, {evaluation_set} set",
                     name_to_table[f"per_attribute_{metric}"])
    print(f"\nWrote the results to {args.out}.")


def run_compare(args: argparse.Namespace) -> None:
    predictions_seq = _read_prediction_files(args.files)
    method_names = {p.header.model.method_name for p in predictions_seq}
    if missing := [n for n in (args.a, args.b) if n not in method_names]:
        raise ValueError(f"No files of the methods {missing}. The files are of"
                         f" {sorted(method_names)}.")
    if other := sorted(method_names - {args.a, args.b}):
        raise ValueError(f"Files of other methods than a and b: {other}.")
    dataset, split = get_common_dataset_and_split(p.header for p in predictions_seq)
    metadata = load_irap_metadata(args.metadata_dir)
    for predictions in predictions_seq:
        check_split_names(predictions.header.model, metadata.splits)
    reference_set = get_reference_set(metadata, dataset, split)
    attributes = select_evaluated_attributes(reference_set, args.attributes)

    def evaluate_models(method_name: str) -> list[EvaluationResult]:
        return [evaluate_predictions(p, reference_set, attributes=attributes,
                                     null_policy="first_class")
                for p in predictions_seq if p.header.model.method_name == method_name]

    models_a, models_b = evaluate_models(args.a), evaluate_models(args.b)
    differences = compare_methods(models_a, models_b, args.metrics, num_resamples=args.bootstrap,
                                  seed=args.seed, confidence=args.confidence)
    _print_table(f"a = {args.a} ({len(models_a)} models), b = {args.b} ({len(models_b)} models),"
                 f" {dataset}/{split}, reference set, {reference_set.num_segments} segments,"
                 f" {len(reference_set.sequence_ids)} road sequences, {args.bootstrap} resamples,"
                 f" {args.confidence:.0%} intervals",
                 evaluation_report.to_comparison_table(models_a, models_b, differences))
    for result in (models_a[0], models_b[0]):  # The models of a method use the same splits.
        if note := _explain_split_use(result.model, split):
            print(f"Method {result.model.method_name!r}: {note}")
    _print_partially_undefined_intervals(
        "a - b", map_metric_values(lambda d: d.interval, differences), args.bootstrap)


def run_ensemble(args: argparse.Namespace) -> None:
    predictions_seq = _read_prediction_files(args.files)
    ensemble = ensemble_predictions(
        predictions_seq, args.name, method_display_name=args.display_name, seed=args.model_seed,
        weights=args.weights, segment_combination=args.segment_combination,
        attribute_combination=args.attribute_combination)
    write_predictions(args.output, ensemble)
    print(f"Wrote the mean of {len(predictions_seq)} prediction files as the method"
          f" {ensemble.header.model.method_name!r}"
          f" ({ensemble.num_segments} segments, trained on"
          f" {_format_training_splits(ensemble.header.model)}) to {args.output}.")


def run_export_coding_table(args: argparse.Namespace) -> None:
    predictions_seq = [read_predictions(p) for p in args.files]
    metadata = load_irap_metadata(args.metadata_dir)
    for predictions in predictions_seq:
        check_split_names(predictions.header.model, metadata.splits)
    template = coding_tables.load_coding_table_template(args.template)
    table = coding_tables.predictions_to_coding_table(
        predictions_seq, metadata, template,
        coder_name=args.coder_name or predictions_seq[0].header.model.shown_method_name,
        coding_date=args.coding_date)
    confidence_table = (coding_tables.predictions_to_confidence_table(predictions_seq, metadata,
                                                                      template)
                        if args.confidence_table else None)
    for path in coding_tables.write_coding_table(args.output, table, confidence_table):
        print(f"Wrote {path}.")
    attributes = list(dict.fromkeys(a for p in predictions_seq for a in p.attributes))
    if unmapped := coding_tables.get_unmapped_attributes(template, attributes):
        print(f"Predicted attributes without a column (not exported): {unmapped}.")
    print(f"Blank columns: {coding_tables.get_unfilled_columns(template, attributes)}.")


# Argument parsing #################################################################################

def _add_scoring_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--attributes", choices=ATTRIBUTE_SELECTIONS, default="canonical",
                        help="canonical: the iRAP attribute subset."
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
        "evaluate", help="Score prediction files on the reference set, on the full set if they"
                         " predict all its segments, and, if the context offsets of a method are"
                         " known, on its model-compatible set. Methods with several models also"
                         " get the means over their models.")
    evaluate.add_argument("metadata_dir", type=Path, help="Metadata directory of the release.")
    evaluate.add_argument("files", nargs="+", type=Path, help="Prediction files.")
    evaluate.add_argument("--out", type=Path, required=True, help="Output directory.")
    evaluate.add_argument("--context-offsets", nargs="*", default=[], metavar="NAME=OFFSETS",
                          help="Context offsets of methods, e.g. resnet=0,-1,-4. They override"
                               " the offsets in the file headers.")
    evaluate.add_argument("--in-split-offsets", nargs="*", default=[], metavar="NAME=OFFSETS",
                          help="In-split offsets of methods, e.g. seq_enh=-1,0,1, or NAME= for"
                               " none: the context offsets whose segments a method uses"
                               " only within the split of the predicted segment. They override"
                               " the in-split offsets in the file headers.")
    evaluate.add_argument("--bootstrap", type=int, default=0, metavar="NUM_RESAMPLES",
                          help="Number of bootstrap resamples for confidence intervals. The"
                               " intervals of a model cover the road sequences, those of a method"
                               " with several models also the spread of its models."
                               " Default: 0 (no intervals).")
    evaluate.add_argument("--metric-names", nargs="+", default=None, metavar="METRIC",
                          help="Metrics to compute, e.g. amF1 aminR totnc_{R>0}, with the"
                               " per-attribute values of their averages (prefix a) and totals"
                               " (prefix tot) over attributes (see irap_evaluation.metric_names)."
                               " Quote names with {}, < or >. The metrics of distributions need"
                               " probs files. Default: for each file, the metrics of the iRAP"
                               " protocol for its output kind and, for probs files, aMRP90"
                               " (irap_evaluation.get_default_metric_names).")
    evaluate.add_argument("--table-metrics", nargs="+", default=[IRAP_MAIN_METRIC],
                          metavar="METRIC",
                          help=f"Per-attribute metrics of the per-attribute tables, which end with"
                               f" the attribute average and total of the metric if they are"
                               f" computed. They must be computed (see --metric-names), e.g. mF1"
                               f" for amF1. Default: {IRAP_MAIN_METRIC}.")
    evaluate.add_argument("--missing-attribute-policy", choices=MISSING_ATTRIBUTE_POLICIES,
                          default="error",
                          help="How a selected attribute that a file does not predict is scored."
                               " error: not at all, the command fails. invalid: as if every"
                               " attribute prediction of it were invalid (see --null-policy)."
                               " Default: error.")
    evaluate.add_argument("--missing-segment-policy", choices=MISSING_SEGMENT_POLICIES,
                          default="error",
                          help="How a segment of the model-compatible set that a file does not"
                               " predict is scored, e.g. of a file of an older metadata build."
                               " error: not at all, the command fails. invalid: as if every"
                               " attribute prediction of it were invalid (see --null-policy)."
                               " Default: error.")
    evaluate.add_argument("--null-policy", choices=NULL_POLICIES, default="first_class",
                          help="How the results of the models score invalid attribute"
                               " predictions. first_class:"
                               " as class 0 in the metrics of the predicted classes and as the"
                               " uniform distribution in NLL and Brier. exclude: not scored, so"
                               " each model is scored on its own segments and no means over the"
                               " models of a method are written (see"
                               " irap_evaluation.evaluation.NullPolicy). Default: first_class.")
    _add_scoring_arguments(evaluate)
    evaluate.set_defaults(run=run_evaluate)

    compare = commands.add_parser(
        "compare", help="Compare two methods, each with one or more models, on the reference set,"
                        " with a bootstrap over road sequences and the spread of the models."
                        " Invalid attribute predictions are scored with the null policy"
                        " first_class.")
    compare.add_argument("metadata_dir", type=Path)
    compare.add_argument("files", nargs="+", type=Path,
                         help="The prediction files of the models of both methods.")
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
        "ensemble", help="Combine prediction files into the weighted mean of their distributions."
                         " The ensemble's training splits are the union of the members' training"
                         " splits.")
    ensemble.add_argument("files", nargs="+", type=Path)
    ensemble.add_argument("--name", default=None,
                          help="Method name of the ensemble. Default: the sorted method names of"
                               " the members with their seeds, e.g. resnet-seq_seeds0-4+vit-l.")
    ensemble.add_argument("--display-name", default=None,
                          help="Method display name of the ensemble. Default: none.")
    ensemble.add_argument("--model-seed", type=int, default=None,
                          help="Seed of the ensemble, if it is one model of a method with several"
                               " models. Default: none.")
    ensemble.add_argument("-o", "--output", type=Path, required=True)
    ensemble.add_argument("--weights", type=float, nargs="+", default=None,
                          help="A positive weight per file. Default: equal weights.")
    for noun in ("segment", "attribute"):
        ensemble.add_argument(f"--{noun}-combination", choices=MEMBER_COMBINATIONS,
                              default="same",
                              help=f"Which {noun}s to combine if the files predict different ones."
                                   f" same: none, the command fails. intersection: those that"
                                   f" every file predicts. union: those that any file predicts,"
                                   f" where a file that does not predict one counts as invalid."
                                   f" Default: same.")
    ensemble.set_defaults(run=run_ensemble)

    export = commands.add_parser("export-coding-table",
                                 help="Write the predictions of a model as an iRAP coding table.")
    export.add_argument("metadata_dir", type=Path)
    export.add_argument("files", nargs="+", type=Path,
                        help="Prediction files of one model on disjoint segments, e.g. one per"
                             " split, which go into one table.")
    export.add_argument("-o", "--output", type=Path, required=True, help="A .xlsx or .csv path.")
    export.add_argument("--coder-name", default=None,
                        help="Default: the method display name, or the method name.")
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
