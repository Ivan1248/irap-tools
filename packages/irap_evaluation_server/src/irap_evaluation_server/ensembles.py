"""Ensembles of stored runs (`irap_evaluation.ensemble_predictions`): the weighted mean of their
predicted distributions, stored as new submissions, one for each split where every member has a
run.

The action log records each ensemble with its member submissions and weights (action
'ensemble'), and the prediction header records the member methods (see
`irap_evaluation/docs/prediction_format.md`).
"""

import dataclasses as dc
import math
import typing as T

import irap_evaluation as ie

from .archive import NewSubmission, Submission, SubmissionArchive, check_actor
from .datasets import DatasetContext


@dc.dataclass(frozen=True)
class EnsembleMember:
    """A member of an ensemble: a run, which can have submissions of several splits, and its
    weight.

    Attributes:
        method: The name and seed of the run, as `Submission.method`.
    """

    method: ie.MethodInfo
    weight: float = 1.0


@dc.dataclass(frozen=True)
class MemberRun:
    """A run that can be an ensemble member, with the splits that it has submissions of.

    Attributes:
        method: The name and seed of the run, as `Submission.method`.
    """

    method: ie.MethodInfo
    splits: tuple[str, ...]


def list_member_runs(submissions: T.Iterable[Submission], dataset: str,
                     split_names: T.Sequence[str]) -> list[MemberRun]:
    """The runs of a dataset in the submissions that are not deleted, by run label.

    Args:
        split_names: The splits of the dataset, which give the order of `MemberRun.splits`.
    """
    method_to_splits: dict[ie.MethodInfo, set[str]] = {}
    for s in submissions:
        if s.dataset == dataset and not s.is_deleted:
            method_to_splits.setdefault(s.method, set()).add(s.split)
    runs = [MemberRun(method, tuple(s for s in split_names if s in splits))
            for method, splits in method_to_splits.items()]
    return sorted(runs, key=lambda r: r.method.run_label)


@dc.dataclass(frozen=True)
class EnsemblePlan:
    """The submissions that an ensemble is made of, by split.

    Attributes:
        split_to_submissions: Split -> the submission of each member, in the order of `members`,
            for the splits where every member has a run.
        split_to_missing: Split -> the run labels of the members without a run on it, for the
            other splits that some member has a run on.
    """

    dataset: str
    members: tuple[EnsembleMember, ...]
    split_to_submissions: T.Mapping[str, tuple[Submission, ...]]
    split_to_missing: T.Mapping[str, tuple[str, ...]]


def plan_ensemble(submissions: T.Iterable[Submission], dataset: str,
                  members: T.Sequence[EnsembleMember],
                  split_names: T.Sequence[str]) -> EnsemblePlan:
    """
    Args:
        submissions: E.g. all submissions. Deleted ones are not used.
        split_names: The splits of the dataset, in the order of the plan.

    Raises:
        ValueError: If there are fewer than 2 members, a member is given twice, a weight is not
            positive and finite, or no split has a run of every member.
    """
    if len(members) < 2:
        raise ValueError("An ensemble needs at least 2 members.")
    if len({m.method for m in members}) < len(members):
        raise ValueError("Each run can be a member only once.")
    if bad := [m.method.run_label for m in members
               if not (math.isfinite(m.weight) and m.weight > 0)]:
        raise ValueError(f"The weights of {', '.join(bad)} must be positive numbers.")
    split_and_method_to_submission = {(s.split, s.method): s for s in submissions
                                      if s.dataset == dataset and not s.is_deleted}
    split_to_submissions, split_to_missing = {}, {}
    for split in split_names:
        found = [split_and_method_to_submission.get((split, m.method)) for m in members]
        if all(s is not None for s in found):
            split_to_submissions[split] = tuple(found)
        elif any(s is not None for s in found):
            split_to_missing[split] = tuple(m.method.run_label for m, s in zip(members, found)
                                            if s is None)
    if not split_to_submissions:
        raise ValueError("No split has a run of every member.")
    return EnsemblePlan(dataset=dataset, members=tuple(members),
                        split_to_submissions=split_to_submissions,
                        split_to_missing=split_to_missing)


def create_ensemble(
    archive: SubmissionArchive,
    context: DatasetContext,
    plan: EnsemblePlan,
    *,
    method: ie.MethodInfo,
    intersect_segments: bool,
    submitter: str,
    description: str,
) -> list[Submission]:
    """Makes the ensemble of each split of a plan and stores them, all or none.

    Each ensemble is checked as an upload (`DatasetContext.check_new_predictions`).

    Args:
        context: The dataset of the plan.
        method: The method name and seed of the ensemble runs.
        intersect_segments: Whether the ensemble predicts only the segments that all members
            predict, e.g. for members with different context offsets (see
            `irap_evaluation.ensemble_predictions`).

    Raises:
        irap_evaluation.PredictionFormatError: If the members cannot be combined, e.g. they
            predict other segments and `intersect_segments` is False.
        ValueError: If `submitter` is empty, the run cannot be stored (e.g. a run with the same
            name and seed exists), or `irap_evaluation.select_evaluation_sets` refuses an
            ensemble.
    """
    # The cheap checks come before reading the files.
    check_actor(submitter)
    for split in plan.split_to_submissions:
        archive.check_new_run(plan.dataset, split, method)
    weights = [m.weight for m in plan.members]
    new_submissions = []
    upload_paths = []
    try:
        for split, submissions in plan.split_to_submissions.items():
            ensemble = ie.ensemble_predictions(
                [ie.read_predictions(archive.get_predictions_path(s.id)) for s in submissions],
                method, weights=weights, intersect_segments=intersect_segments)
            notes = context.check_new_predictions(ensemble)
            path = archive.make_upload_path("ensemble.predictions.parquet")
            upload_paths.append(path)
            ie.write_predictions(path, ensemble)
            new_submissions.append(NewSubmission.from_predictions(
                ensemble, path, action="ensemble", details={
                    "members": [{"submission_id": s.id, "run_label": s.run_label, "weight": w}
                                for s, w in zip(submissions, weights, strict=True)],
                    "intersect_segments": intersect_segments}, notes=notes))
        return archive.add_submissions(new_submissions, submitter=submitter,
                                       description=description)
    finally:
        # Those that are stored are moved into the archive.
        for path in upload_paths:
            path.unlink(missing_ok=True)
