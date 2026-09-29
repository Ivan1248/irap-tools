"""The `irap-dataset-stats` command: class frequencies and other statistics of an IRAP release.

The statistics are computed from the metadata only (see `irap_data.dataset_statistics`). Run
`irap-dataset-stats <command> --help` for the options of each command.
"""

import argparse
import csv
import dataclasses as dc
import json
import sys
import typing as T
from datetime import date
from pathlib import Path

import numpy as np

from . import report_document as rd
from .dataset_statistics import (
    ATTRIBUTE_ORDERS,
    CLASS_ORDERS,
    SELECTION_DESCRIPTIONS,
    AttributeDistribution,
    DatasetStatistics,
    SelectionKind,
    SelectionStatistics,
    compute_dataset_statistics,
    compute_majority_accuracy,
    compute_selection_statistics,
    compute_total_variation_distance,
    select_segments,
    sort_distributions,
)
from .metadata import (
    DATASET_PRESETS,
    IRAPMetadata,
    get_dataset_preset,
    get_default_irap_paths,
    is_unlabeled_split,
    load_irap_metadata,
)

#: The columns of `to_class_frequency_rows`.
CLASS_FREQUENCY_COLUMNS = ("dataset", "selection", "split", "attribute", "class_index", "value",
                           "irap_code", "num_segments", "num_sequences", "share", "share_labeled")

#: The default plot format of each report format: HTML embeds an SVG plot, and Markdown links a
#: PDF plot, which many Markdown viewers do not render as an image.
_REPORT_FORMAT_TO_PLOT_FORMAT = {"html": "svg", "md": "pdf"}


# Formatting helpers ###############################################################################

def _format_context_offsets(context_offsets: T.Sequence[int]) -> str:
    """Formats consecutive ascending offsets as a range, e.g. '-4..4', and others as a list."""
    first, last = context_offsets[0], context_offsets[-1]
    if len(context_offsets) > 2 and list(context_offsets) == list(range(first, last + 1)):
        return f"{first}..{last}"
    return ",".join(map(str, context_offsets))


def _describe_selection(selection: SelectionKind,
                        context_offsets: T.Sequence[int] | None) -> str:
    """'labeled segments', or 'labeled segments with context [-4..0]' for a selection with a
    context."""
    if context_offsets is None:
        return f"{selection} segments"
    return f"labeled segments with context [{_format_context_offsets(context_offsets)}]"


def _get_report_stem(dataset: str, selection: SelectionKind,
                     context_offsets: T.Sequence[int] | None) -> str:
    """The file name stem of the report of a selection: 'vietnam_labeled_segments',
    'vietnam_labeled_segments_with_context' for the reference selection, and
    'vietnam_labeled_segments_with_context_<context_offsets>' for the model selection."""
    if selection == "labeled":
        return f"{dataset}_labeled_segments"
    stem = f"{dataset}_labeled_segments_with_context"
    if selection == "reference":
        return stem
    return f"{stem}_{_format_context_offsets(context_offsets)}"


def _format_count(n: int) -> str:
    return f"{n:,}"


def _format_share(x: float) -> str:
    return "" if np.isnan(x) else f"{x:.1%}"


def _format_ratio(x: float) -> str:
    if np.isnan(x):
        return ""
    return f"{x:,.0f}" if x >= 10 else f"{x:.1f}"


def _format_class_count(n: int, share: float | None = None) -> str:
    """A class count, in bold if zero, as those are the classes to notice."""
    if n == 0:
        return "**0**"
    return _format_count(n) if share is None else f"{_format_count(n)} ({_format_share(share)})"


def _to_json_float(x: float) -> float | None:
    return None if np.isnan(x) else float(x)


# Report ###########################################################################################

def compose_report(
    statistics: DatasetStatistics,
    selection: SelectionKind,
    *,
    report_date: str,
    rare_fraction: float,
    split_to_figure_path: T.Mapping[str, str] | None = None,
) -> list[rd.Block]:
    """Composes the report of one selection of segments, to be rendered with
    `report_document.to_html` or `report_document.to_markdown`.

    The class balance is measured on the base split, the first split of `statistics` with
    segments in the selection, and the other splits are compared to it.

    Args:
        report_date: The date of the report, YYYY-MM-DD.
        rare_fraction: The share of the labeled segments of an attribute in the base split below
            which a class is listed as rare.
        split_to_figure_path: Split -> the path of its class-frequency plot, relative to the
            report.
    """
    split_distributions = statistics.get_split_distributions(selection)
    blocks = [*_compose_header(statistics, selection, report_date),
              *_compose_overview(statistics, selection)]
    if not split_distributions:
        return [*blocks, rd.Paragraph("No split has segments in this selection.")]
    num_attributes = len(next(iter(split_distributions.values())))
    attribute_indices = [i for i in range(num_attributes)
                         if any(ds[i].is_labeled for ds in split_distributions.values())]
    return [
        *blocks,
        *_compose_label_coverage(split_distributions, attribute_indices),
        *_compose_class_balance(split_distributions, attribute_indices),
        *_compose_rare_classes(split_distributions, attribute_indices, rare_fraction),
        *_compose_missing_classes(split_distributions, attribute_indices),
        *_compose_class_frequencies(split_distributions, attribute_indices,
                                    split_to_figure_path or {}),
    ]


def _compose_header(statistics: DatasetStatistics, selection: SelectionKind,
                    report_date: str) -> list[rd.Block]:
    preset = statistics.preset
    description = _describe_selection(selection, statistics.get_context_offsets(selection))
    items = [
        f"Metadata directory: `{statistics.metadata_dir}`",
        f"Date: {report_date}",
        f"Selection: {selection}. {SELECTION_DESCRIPTIONS[selection]}",
        "A segment with a missing attribute code is "
        + ("kept without a label for the attribute." if preset.allow_missing_attributes
           else "left out."),
    ]
    if preset.use_ncontext_filter and selection != "labeled":
        items.append("The segments are restricted to the precomputed N-context subsets"
                     " (`seg_to_res`).")
    return [rd.Heading(f"Dataset statistics of {statistics.dataset}: {description}", 1),
            rd.BulletList(tuple(items))]


def _compose_overview(statistics: DatasetStatistics,
                      selection: SelectionKind) -> list[rd.Block]:
    selections = statistics.selections
    funnel = rd.make_table(
        ["split", "listed", "with image", *selections],
        [[s.split, _format_count(s.num_listed), _format_count(s.num_with_image),
          *(_format_count(s.selections[k].num_segments) if s.selections else ""
            for k in selections)]
         for s in statistics.splits],
        "l" + "r" * (2 + len(selections)))

    def format_segments_row(split: str, s: SelectionStatistics) -> list[str]:
        lengths = s.sequence_lengths
        return [split, _format_count(s.num_segments), _format_count(s.num_unlabeled),
                _format_count(s.num_sequences), _format_count(s.num_without_sequence),
                f"{lengths.min()} / {np.median(lengths):g} / {lengths.max()}" if lengths.size
                else "", f"{s.road_length_m / 1000:,.1f}"]

    segments = rd.make_table(
        ["split", "segments", "without any label", "sequences", "without sequence",
         "segments per sequence (min / median / max)", "road length (km)"],
        [format_segments_row(s.split, s.selections[selection])
         for s in statistics.splits if s.selections],
        "lrrrrrr")
    return [
        rd.Heading("Overview", 2),
        rd.Paragraph("The number of segments of each split: listed in `splits.json`, with an"
                     " image, and in each selection. Unlabeled splits are not in any selection."),
        funnel,
        rd.Paragraph(
            f"The {_describe_selection(selection, statistics.get_context_offsets(selection))} of"
            " each split. The road sequences are those of"
            " `road_id_to_segment_id_sequence.json`, and a segment that is in none is a sequence"
            " of its own. The road length is summed over the segments that have road data."),
        segments]


def _compose_label_coverage(
    split_distributions: T.Mapping[str, T.Sequence[AttributeDistribution]],
    attribute_indices: T.Sequence[int],
) -> list[rd.Block]:
    def format_cell(d: AttributeDistribution) -> str:
        if not d.num_missing:
            return _format_count(d.num_labeled)
        return (f"{_format_count(d.num_labeled)}"
                f" ({_format_share(d.num_missing / d.num_total)} missing)")

    distributions_seq = list(split_distributions.values())
    table = rd.make_table(
        ["attribute", *split_distributions],
        [[distributions_seq[0][i].attribute, *(format_cell(ds[i]) for ds in distributions_seq)]
         for i in attribute_indices],
        "l" + "r" * len(split_distributions))
    blocks = [rd.Heading("Label coverage", 2),
              rd.Paragraph("The number of labeled segments of each attribute, and the share of"
                           " the segments without a label."),
              table]
    if unlabeled := [d.attribute for i, d in enumerate(distributions_seq[0])
                     if i not in attribute_indices]:
        blocks.append(rd.Paragraph(f"Attributes without a label in any split, left out below"
                                   f" ({len(unlabeled)}): {', '.join(unlabeled)}."))
    return blocks


def _compose_class_balance(
    split_distributions: T.Mapping[str, T.Sequence[AttributeDistribution]],
    attribute_indices: T.Sequence[int],
) -> list[rd.Block]:
    base_split, *other_splits = split_distributions
    base = split_distributions[base_split]

    def format_row(i: int) -> list[str]:
        d = base[i]
        comparisons = [(compute_total_variation_distance(d, split_distributions[s][i]),
                        compute_majority_accuracy(d, split_distributions[s][i]))
                       for s in other_splits]
        return [d.attribute, str(d.num_classes), str(d.num_absent_classes),
                _format_share(d.majority_share), _format_ratio(d.imbalance_ratio),
                "" if np.isnan(d.effective_num_classes) else f"{d.effective_num_classes:.1f}",
                *(_format_share(x) for comparison in comparisons for x in comparison)]

    table = rd.make_table(
        ["attribute", "classes", "absent", "majority share", "max:min", "effective classes",
         *(f"{name} {s}" for s in other_splits for name in ("TVD", "majority accuracy"))],
        map(format_row, attribute_indices),
        "lrrrrr" + "rr" * len(other_splits))
    legend = (
        f"Measured on {base_split}. *Absent* is the number of classes without a segment."
        " *Majority share* is the share of the most frequent class, the accuracy of predicting"
        " it. *Max:min* is the ratio of the most to the least frequent class that has a segment."
        " *Effective classes* is the exponential of the class entropy: K for K equally frequent"
        " classes, and 1 for a single class.")
    if other_splits:
        legend += (
            f" *TVD* is the total variation distance between the class distributions of a split"
            f" and of {base_split}: the share of labels that would have to change class for them"
            f" to match. *Majority accuracy* is the accuracy on a split of predicting the most"
            f" frequent class of {base_split}.")
    return [rd.Heading("Class balance", 2), rd.Paragraph(legend), table]


def _compose_rare_classes(
    split_distributions: T.Mapping[str, T.Sequence[AttributeDistribution]],
    attribute_indices: T.Sequence[int],
    rare_fraction: float,
) -> list[rd.Block]:
    base_split = next(iter(split_distributions))
    base = split_distributions[base_split]
    distributions_seq = list(split_distributions.values())
    rare = sorted(((i, c) for i in attribute_indices for c in range(base[i].num_classes)
                   if 0 < base[i].labeled_shares[c] < rare_fraction),
                  key=lambda item: (base[item[0]].num_segments[item[1]], base[item[0]].attribute))
    table = rd.make_table(
        ["attribute", "class", "IRAP code", *(f"{s} segments" for s in split_distributions),
         f"{base_split} share", f"{base_split} sequences"],
        [[base[i].attribute, base[i].values[c], str(base[i].irap_codes[c]),
          *(_format_class_count(ds[i].num_segments[c]) for ds in distributions_seq),
          _format_share(base[i].labeled_shares[c]), _format_count(base[i].num_sequences[c])]
         for i, c in rare],
        "llr" + "r" * len(split_distributions) + "rr")
    return [
        rd.Heading("Rare classes", 2),
        rd.Paragraph(
            f"The classes with less than {rare_fraction:.1%} of the labeled segments of their"
            f" attribute in {base_split}, rarest first, with their number of segments in each"
            f" split. Classes without a segment are in the next section."),
        table if rare else rd.Paragraph("None.")]


def _compose_missing_classes(
    split_distributions: T.Mapping[str, T.Sequence[AttributeDistribution]],
    attribute_indices: T.Sequence[int],
) -> list[rd.Block]:
    base_split = next(iter(split_distributions))
    distributions_seq = list(split_distributions.values())
    rows = [[d.attribute, d.values[c], str(d.irap_codes[c]),
             *(_format_class_count(ds[i].num_segments[c]) for ds in distributions_seq)]
            for i in attribute_indices for d in [distributions_seq[0][i]]
            for c in range(d.num_classes)
            if any(ds[i].num_segments[c] == 0 for ds in distributions_seq)]
    table = rd.make_table(["attribute", "class", "IRAP code", *split_distributions], rows,
                          "llr" + "r" * len(split_distributions))
    return [
        rd.Heading("Missing classes", 2),
        rd.Paragraph(
            f"The classes without a segment in at least one split. A class that is missing from"
            f" {base_split} cannot be learned from it, and the recall of a class that is missing"
            f" from another split cannot be measured on it."),
        table if rows else rd.Paragraph("None.")]


def _compose_class_frequencies(
    split_distributions: T.Mapping[str, T.Sequence[AttributeDistribution]],
    attribute_indices: T.Sequence[int],
    split_to_figure_path: T.Mapping[str, str],
) -> list[rd.Block]:
    blocks = [rd.Heading("Class frequencies", 2),
              rd.Paragraph("The number of segments of each class, with its share of the labeled"
                           " segments of the attribute, and the number of road sequences that"
                           " contain the class.")]
    blocks += [rd.Figure(path, f"Class frequencies of {split}")
               for split, path in split_to_figure_path.items()]
    distributions_seq = list(split_distributions.values())
    for i in attribute_indices:
        first = distributions_seq[0][i]
        table = rd.make_table(
            ["class", "IRAP code",
             *(name for s in split_distributions for name in (s, f"{s} sequences"))],
            [[first.values[c], str(first.irap_codes[c]),
              *(cell for ds in distributions_seq
                for cell in (_format_class_count(ds[i].num_segments[c], ds[i].labeled_shares[c]),
                             _format_count(ds[i].num_sequences[c])))]
             for c in range(first.num_classes)],
            "lr" + "rr" * len(split_distributions))
        blocks += [rd.Heading(first.attribute, 3), table]
    return blocks


# JSON and CSV #####################################################################################

def to_json_dict(statistics: DatasetStatistics, *, report_date: str) -> dict[str, T.Any]:
    """A JSON-compatible document of all statistics, with NaN as None."""
    def distribution_to_json(d: AttributeDistribution) -> dict[str, T.Any]:
        return {
            "attribute": d.attribute, "num_labeled": d.num_labeled, "num_missing": d.num_missing,
            "num_absent_classes": d.num_absent_classes,
            "majority_share": _to_json_float(d.majority_share),
            "imbalance_ratio": _to_json_float(d.imbalance_ratio),
            "effective_num_classes": _to_json_float(d.effective_num_classes),
            "classes": [{"index": c, "value": d.values[c], "irap_code": d.irap_codes[c],
                         "num_segments": int(d.num_segments[c]),
                         "num_sequences": int(d.num_sequences[c]),
                         "share": _to_json_float(d.shares[c]),
                         "share_labeled": _to_json_float(d.labeled_shares[c])}
                        for c in range(d.num_classes)],
        }

    def selection_to_json(s: SelectionStatistics) -> dict[str, T.Any]:
        return {"num_segments": s.num_segments, "num_unlabeled": s.num_unlabeled,
                "num_without_sequence": s.num_without_sequence,
                "num_sequences": s.num_sequences, "road_length_m": s.road_length_m,
                "attributes": [distribution_to_json(d) for d in s.distributions]}

    def get_context_offsets(selection: SelectionKind) -> list[int] | None:
        offsets = statistics.get_context_offsets(selection)
        return None if offsets is None else list(offsets)

    return {
        "dataset": statistics.dataset,
        "metadata_dir": str(statistics.metadata_dir),
        "date": report_date,
        "preset": dc.asdict(statistics.preset),
        "selections": {k: {"description": SELECTION_DESCRIPTIONS[k],
                           "context_offsets": get_context_offsets(k)}
                       for k in statistics.selections},
        "splits": [{"split": s.split, "num_listed": s.num_listed,
                    "num_with_image": s.num_with_image,
                    "selections": {k: selection_to_json(v) for k, v in s.selections.items()}}
                   for s in statistics.splits],
    }


def to_class_frequency_rows(statistics: DatasetStatistics,
                            selection: SelectionKind) -> list[dict[str, T.Any]]:
    """One row per (split with segments in the selection, attribute, class), with the columns
    `CLASS_FREQUENCY_COLUMNS`. `share` is the share of all segments of the split, and
    `share_labeled` the share of its labeled segments of the attribute, None for an attribute
    without a label."""
    return [{"dataset": statistics.dataset, "selection": selection, "split": split,
             "attribute": d.attribute, "class_index": c, "value": d.values[c],
             "irap_code": d.irap_codes[c], "num_segments": int(d.num_segments[c]),
             "num_sequences": int(d.num_sequences[c]), "share": _to_json_float(d.shares[c]),
             "share_labeled": _to_json_float(d.labeled_shares[c])}
            for split, distributions in statistics.get_split_distributions(selection).items()
            for d in distributions for c in range(d.num_classes)]


# Commands #########################################################################################

def _get_metadata_dir(args: argparse.Namespace) -> Path:
    if args.metadata_dir is not None:
        return args.metadata_dir
    try:
        return get_default_irap_paths(args.dataset)[1]
    except RuntimeError as e:
        raise ValueError(f"{e} Alternatively, pass --metadata-dir.") from e


def _import_class_frequency_plot():
    try:
        from . import class_frequency_plot
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            f"Plots need {e.name!r}. Install the extra: pip install 'irap-data[report]'.") from e
    return class_frequency_plot


def _format_plot_title(dataset: str, split: str, selection_description: str,
                       num_segments: int) -> str:
    return (f"{dataset} / {split} – {selection_description} ({num_segments:,}), class"
            f" frequencies")


def _select_the_same_segments(metadata: IRAPMetadata, statistics: DatasetStatistics,
                              a: T.Sequence[int], b: T.Sequence[int]) -> bool:
    """Whether the context offsets `a` and `b` select the same segments in every labeled split
    of `statistics`."""
    def select(split: str, context_offsets: T.Sequence[int]) -> list[str]:
        return list(select_segments(metadata, statistics.dataset, split, "model", context_offsets))

    return all(select(s.split, a) == select(s.split, b) for s in statistics.splits if s.selections)


def run_report(args: argparse.Namespace) -> None:
    metadata = load_irap_metadata(_get_metadata_dir(args))
    statistics = compute_dataset_statistics(metadata, args.dataset, splits=args.splits,
                                            model_context_offsets=args.context_offsets)
    # Imported before anything is written, so that a missing matplotlib leaves no partial report.
    class_frequency_plot = _import_class_frequency_plot() if args.plots else None
    plot_format = args.plot_format or _REPORT_FORMAT_TO_PLOT_FORMAT[args.format]
    report_date = date.today().isoformat()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for selection in statistics.selections:
        context_offsets = statistics.get_context_offsets(selection)
        if selection == "model" and _select_the_same_segments(
                metadata, statistics, context_offsets, statistics.get_context_offsets("reference")):
            print("The model context offsets select the same segments as the reference ones, so"
                  " the model report would repeat the reference report and is not written.")
            continue
        stem = _get_report_stem(args.dataset, selection, context_offsets)
        description = _describe_selection(selection, context_offsets)
        split_to_figure_path = {}
        for split in statistics.splits if class_frequency_plot else ():
            selected = split.selections.get(selection)
            if selected is None or not any(d.is_labeled for d in selected.distributions):
                continue
            path = out_dir / f"{stem}_{split.split}_class_frequencies.{plot_format}"
            class_frequency_plot.write_class_frequency_plot(
                path, selected.distributions, dpi=args.dpi,
                title=_format_plot_title(args.dataset, split.split, description,
                                         selected.num_segments))
            split_to_figure_path[split.split] = path.name
            written.append(path)
        blocks = compose_report(statistics, selection, report_date=report_date,
                                rare_fraction=args.rare_fraction,
                                split_to_figure_path=split_to_figure_path)
        path = out_dir / f"{stem}.{args.format}"
        path.write_text(rd.to_html(blocks) if args.format == "html" else rd.to_markdown(blocks),
                        encoding="utf-8")
        written.append(path)
        path = out_dir / f"{stem}_class_frequencies.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CLASS_FREQUENCY_COLUMNS)
            writer.writeheader()
            writer.writerows(to_class_frequency_rows(statistics, selection))
        written.append(path)

    path = out_dir / f"{args.dataset}_statistics.json"
    path.write_text(json.dumps(to_json_dict(statistics, report_date=report_date), indent=2,
                               ensure_ascii=False, allow_nan=False), encoding="utf-8")
    written.append(path)
    print("\n".join(f"Wrote {p}." for p in written))


def run_plot(args: argparse.Namespace) -> None:
    class_frequency_plot = _import_class_frequency_plot()
    if is_unlabeled_split(args.split):
        raise ValueError(f"The split {args.split!r} has no labels.")
    metadata = load_irap_metadata(_get_metadata_dir(args))
    selected = compute_selection_statistics(metadata, select_segments(
        metadata, args.dataset, args.split, args.selection, args.context_offsets))
    context_offsets = (get_dataset_preset(args.dataset).reference_context_offsets
                       if args.selection == "reference" else args.context_offsets)
    distributions = sort_distributions(selected.distributions,
                                       attribute_order=args.sort_attributes,
                                       class_order=args.sort_classes)
    class_frequency_plot.write_class_frequency_plot(
        args.output, distributions, normalize=args.normalize, log=not args.linear, dpi=args.dpi,
        title=_format_plot_title(args.dataset, args.split, _describe_selection(
            args.selection, context_offsets), selected.num_segments))
    print(f"Wrote {args.output}.")


# Argument parsing #################################################################################

def _parse_context_offsets(text: str) -> tuple[int, ...]:
    return tuple(int(o) for o in text.split(","))


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dataset", choices=list(DATASET_PRESETS), help="The release.")
    parser.add_argument("--metadata-dir", type=Path, default=None,
                        help="Default: the metadata directory of the release under $IRAP_HOME or"
                             " $DATASETS_PATH.")
    parser.add_argument("--dpi", type=int, default=200,
                        help="The resolution of raster plots. Default: 200.")


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="irap-dataset-stats",
                                     description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    report = commands.add_parser(
        "report", help="Write an HTML or Markdown report and a CSV table of the class"
                       " frequencies of each selection of segments, with plots, and a JSON"
                       " document of all statistics.")
    _add_common_arguments(report)
    report.add_argument("-o", "--out", type=Path, required=True, help="Output directory.")
    report.add_argument("--format", choices=list(_REPORT_FORMAT_TO_PLOT_FORMAT), default="html",
                        help="The file format of the reports: an HTML page or GitHub-flavored"
                             " Markdown. Default: html.")
    report.add_argument("--splits", nargs="+", default=None,
                        help="Default: all splits in splits.json. The first split with segments"
                             " is the base of the comparisons.")
    report.add_argument("--context-offsets", type=_parse_context_offsets, default=None,
                        metavar="OFFSETS",
                        help="The context offsets of a model, e.g. 0,-1,-4, for an additional"
                             " 'model' selection.")
    report.add_argument("--rare-fraction", type=float, default=0.01,
                        help="A class is rare below this share of the labeled segments of its"
                             " attribute. Default: 0.01.")
    report.add_argument("--no-plots", dest="plots", action="store_false",
                        help="Leave out the plots, which need matplotlib.")
    report.add_argument("--plot-format", choices=("pdf", "png", "svg"), default=None,
                        help="The file format of the plots. A PDF plot is linked from the report"
                             " and the others are embedded. Default: svg for an HTML report and"
                             " pdf for a Markdown report.")
    report.set_defaults(run=run_report)

    plot = commands.add_parser("plot", help="Plot the class frequencies of one split.")
    _add_common_arguments(plot)
    plot.add_argument("-o", "--output", type=Path, required=True,
                      help="The image path. Its extension selects the format, e.g. .pdf or .png.")
    plot.add_argument("--split", default="train", help="Default: train.")
    plot.add_argument("--selection", choices=list(SELECTION_DESCRIPTIONS), default="labeled",
                      help="Default: labeled. 'model' needs --context-offsets.")
    plot.add_argument("--context-offsets", type=_parse_context_offsets, default=None,
                      metavar="OFFSETS", help="The context offsets of the 'model' selection.")
    plot.add_argument("--normalize", action="store_true",
                      help="Plot the share of the labeled segments of the attribute instead of"
                           " the number of segments.")
    plot.add_argument("--linear", action="store_true",
                      help="Use a linear value axis. Default: logarithmic, as the counts span"
                           " several orders of magnitude.")
    plot.add_argument("--sort-attributes", choices=ATTRIBUTE_ORDERS, default="schema",
                      help="The order of the attributes (see"
                           " irap_data.dataset_statistics.ATTRIBUTE_ORDERS). Default: schema.")
    plot.add_argument("--sort-classes", choices=CLASS_ORDERS, default="schema",
                      help="The order of the classes of an attribute (see"
                           " irap_data.dataset_statistics.CLASS_ORDERS). Default: schema.")
    plot.set_defaults(run=run_plot)
    return parser


def main(argv: T.Sequence[str] | None = None) -> int:
    args = make_argument_parser().parse_args(argv)
    try:
        args.run(args)
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
