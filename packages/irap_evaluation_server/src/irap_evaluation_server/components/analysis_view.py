"""The view of the analysis page: what it shows, as in its URL, and how the view is resolved
against the scored runs of its dataset split."""

import dataclasses as dc
import typing as T

import irap_evaluation as ie

from ..datasets import DatasetContext
from ..scoring import ScoredRun
from .attribute_filter import AttributeSubset
from .routes import ANALYSIS_PATH, ATTRIBUTE_PARAMETER, make_query_path
from .view_queries import parse_dataset, parse_evaluation_set


@dc.dataclass(frozen=True)
class AnalysisView:
    """What the analysis page shows, as in its URL. None and '' mean the default, which
    `resolve_view` chooses.

    Attributes:
        run_id: The submission id of run A, which the confusion matrix is of.
        compare_id: The submission id of run B, which A is compared with on the map.
        attribute_name: The analysed attribute.
        cell: (label index, predicted index) of the selected cell of the confusion matrix, with
            the predicted index K for invalid predictions.
    """

    dataset: str
    split: str
    evaluation_set: str = ie.REFERENCE_SET_NAME
    run_id: int | None = None
    compare_id: int | None = None
    attribute_name: str = ""
    cell: tuple[int, int] | None = None
    segment_id: str = ""

    def to_path(self, attributes: T.Sequence[str] = ()) -> str:
        """The URL path, without the default values.

        Args:
            attributes: The attribute subset (`AttributeSubset.to_query`).
        """
        return make_query_path(ANALYSIS_PATH, {
            "dataset": self.dataset, "split": self.split,
            "set": "" if self.evaluation_set == ie.REFERENCE_SET_NAME else self.evaluation_set,
            "run": "" if self.run_id is None else str(self.run_id),
            "compare": "" if self.compare_id is None else str(self.compare_id),
            "attr": self.attribute_name,
            "cell": "" if self.cell is None else f"{self.cell[0]},{self.cell[1]}",
            "segment": self.segment_id, ATTRIBUTE_PARAMETER: attributes})

    @classmethod
    def from_query(cls, query: T.Mapping[str, str],
                   dataset_contexts: T.Mapping[str, DatasetContext]) -> T.Self:
        """The view of the query parameters of `to_path`, with the dataset and split filled in.

        Raises:
            ValueError: For an unknown dataset or evaluation set, a split that is not an
                analysis split, or a malformed submission number or cell.
        """
        dataset = parse_dataset(query, dataset_contexts,
                                default=next(iter(list_analysis_datasets(dataset_contexts)), None))
        analysis_splits = dataset_contexts[dataset].config.analysis_splits
        if not analysis_splits:
            raise ValueError(f"The dataset {dataset!r} has no analysis splits (`analysis_splits`"
                             f" in the configuration of the server).")
        split = query.get("split") or analysis_splits[0]
        if split not in analysis_splits:
            raise ValueError(f"{split!r} is not an analysis split of {dataset!r}. Its analysis"
                             f" splits: {', '.join(analysis_splits)}.")
        return cls(dataset=dataset, split=split, evaluation_set=parse_evaluation_set(query),
                   run_id=_parse_submission_id(query.get("run", ""), "run"),
                   compare_id=_parse_submission_id(query.get("compare", ""), "compared run"),
                   attribute_name=query.get("attr", ""), cell=parse_cell(query.get("cell", "")),
                   segment_id=query.get("segment", ""))


def _parse_submission_id(text: str, name: str) -> int | None:
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"The {name} must be a submission number, got {text!r}.") from None


def parse_cell(text: str) -> tuple[int, int] | None:
    """The cell of `AnalysisView.to_path`, e.g. '2,3', None for ''.

    Raises:
        ValueError: If it is not two integers.
    """
    if not text:
        return None
    try:
        label_index, predicted_index = map(int, text.split(","))
    except ValueError:
        raise ValueError(f"A matrix cell is two indices, e.g. '2,3', got {text!r}.") from None
    return label_index, predicted_index


def list_analysis_datasets(dataset_contexts: T.Mapping[str, DatasetContext]) -> list[str]:
    """The datasets that have analysis splits."""
    return [name for name, c in dataset_contexts.items() if c.config.analysis_splits]


def list_compared_runs(runs: T.Iterable[ScoredRun], scored_run: ScoredRun) -> list[ScoredRun]:
    """The runs that can be compared with `scored_run`: the others of its evaluation set."""
    fingerprint = scored_run.scores.evaluation_set_fingerprint
    return [r for r in runs if r.submission.id != scored_run.submission.id
            and r.scores.evaluation_set_fingerprint == fingerprint]


def select_attribute_options(subset: AttributeSubset, scored_run: ScoredRun) -> list[str]:
    """The attributes that can be analysed: those of the subset that the run is scored on."""
    return [a for a in subset.selected if a in scored_run.scores.attributes]


@dc.dataclass(frozen=True)
class ResolvedView:
    """A view with its runs.

    Attributes:
        view: With the submission id of run A and the attribute.
        evaluation_set: The evaluation set of run A (and B).
    """

    view: AnalysisView
    run_a: ScoredRun
    run_b: ScoredRun | None
    evaluation_set: ie.EvaluationSet


def resolve_view(view: AnalysisView, runs: T.Sequence[ScoredRun], subset: AttributeSubset,
                 get_evaluation_set: T.Callable[[ScoredRun], ie.EvaluationSet]) -> ResolvedView:
    """Chooses the defaults of a view and checks it against the runs.

    Args:
        runs: The scored runs of the dataset split on the evaluation set of the view, at least
            one. By default, run A is the first.
        subset: The attribute subset of the page. By default, the attribute is the first of it
            that run A is scored on.
        get_evaluation_set: The evaluation set of a run (`scoring.get_run_evaluation_set`).

    Raises:
        ValueError: If run A or B is not among the runs, B cannot be compared with A, the
            attribute is not scored or not in the subset, or the cell or the segment is not in
            the matrix or the evaluation set.
    """
    run_a = (runs[0] if view.run_id is None
             else next((r for r in runs if r.submission.id == view.run_id), None))
    if run_a is None:
        raise ValueError(f"Submission #{view.run_id} is not a scored run of"
                         f" {view.dataset}/{view.split} on this evaluation set, e.g. since it is"
                         f" deleted.")
    run_b = None
    if view.compare_id is not None:
        run_b = next((r for r in list_compared_runs(runs, run_a)
                      if r.submission.id == view.compare_id), None)
        if run_b is None:
            raise ValueError(f"Submission #{view.compare_id} cannot be compared with"
                             f" #{run_a.submission.id}: it is the same run, it is not scored, or"
                             f" its evaluation set differs.")
    evaluation_set = get_evaluation_set(run_a)
    options = select_attribute_options(subset, run_a)
    attribute = view.attribute_name or next(iter(options), "")
    errors = []
    if view.attribute_name and view.attribute_name not in run_a.scores.attributes:
        errors.append(f"The run is not scored on the attribute {view.attribute_name!r}.")
    elif attribute not in options:
        errors.append(f"The attribute {view.attribute_name!r} is not in the attribute subset."
                      if attribute else
                      "The run is scored on none of the attributes of the subset.")
    elif view.cell is not None:
        num_classes = len(evaluation_set.vocabulary.get_irap_codes(attribute))
        label_index, predicted_index = view.cell
        if not (0 <= label_index < num_classes and 0 <= predicted_index <= num_classes):
            errors.append(f"The matrix cell {view.cell} is outside the matrix of {attribute!r}"
                          f" ({num_classes} classes).")
    if view.segment_id and view.segment_id not in evaluation_set.segment_ids:
        errors.append(f"The segment {view.segment_id!r} is not in the evaluation set.")
    if errors:
        raise ValueError(" ".join(errors))
    return ResolvedView(
        view=dc.replace(view, run_id=run_a.submission.id, attribute_name=attribute),
        run_a=run_a, run_b=run_b, evaluation_set=evaluation_set)
