"""What the Models page lists: the models of a dataset with the active submission of each split,
grouped by method."""

import dataclasses as dc
import typing as T
from datetime import datetime

import irap_evaluation as ie

from .archive import Model, Submission


@dc.dataclass(frozen=True)
class ModelSummary:
    """A model with its active submissions.

    The active submissions of a model have the same `archive.MODEL_FILE_FIELDS`, so the
    properties take them from any one.

    Attributes:
        split_to_submission: Split -> the active submission of the model.
    """

    model: Model
    split_to_submission: T.Mapping[str, Submission]

    @classmethod
    def from_submissions(cls, model: Model, submissions: T.Iterable[Submission]) -> T.Self:
        """Args:
            submissions: Submissions of the model, also replaced ones, which are left out.
        """
        return cls(model, {s.split: s for s in submissions if s.is_active})

    @property
    def submissions(self) -> list[Submission]:
        return list(self.split_to_submission.values())

    @property
    def output_kind(self) -> ie.OutputKind | None:
        """None if the model has no active submission."""
        return next((s.output_kind for s in self.submissions), None)

    @property
    def context_offsets(self) -> tuple[int, ...] | None:
        """None if they are unknown or the model has no active submission."""
        return next((s.context_offsets for s in self.submissions), None)

    @property
    def model_info(self) -> ie.ModelInfo | None:
        """The model of the files, with the training splits, None if the model has no active
        submission."""
        return next((s.model_info for s in self.submissions), None)

    @property
    def num_segments(self) -> int:
        """The number of segments of all active submissions."""
        return sum(s.num_segments for s in self.submissions)

    @property
    def attribute_range(self) -> tuple[int, int] | None:
        """The fewest and most attributes of an active submission, None without one."""
        counts = [s.num_attributes for s in self.submissions]
        return (min(counts), max(counts)) if counts else None

    @property
    def updated_at(self) -> datetime:
        """The time of the latest upload of an active submission, or of the creation of the
        model."""
        return max([self.model.created_at, *(s.uploaded_at for s in self.submissions)])


@dc.dataclass(frozen=True)
class MethodGroup:
    """The models of a method, by seed (a model without a seed first)."""

    method_name: str
    models: tuple[ModelSummary, ...]

    @property
    def updated_at(self) -> datetime:
        return max(m.updated_at for m in self.models)

    @property
    def shown_method_name(self) -> str:
        """`archive.Model.shown_method_name`, which the models of a method share."""
        return self.models[0].model.shown_method_name

    @property
    def model_info(self) -> ie.ModelInfo | None:
        """The `irap_evaluation.ModelInfo` of a model of the method, whose training and early
        stopping splits are those of all its models, or None if none has an active
        submission."""
        return next((m.model_info for m in self.models if m.model_info is not None), None)


def summarize_models_by_method(models: T.Iterable[Model],
                               submissions: T.Iterable[Submission]) -> list[MethodGroup]:
    """Summarizes models and groups them by method, the latest updated method first, and methods
    updated at the same time by shown name.

    Args:
        models: E.g. the models of a dataset.
        submissions: Submissions, also replaced ones, e.g. all of the dataset. Those of other
            models and the replaced ones are left out.
    """
    model_id_to_submissions: dict[int, list[Submission]] = {}
    for s in submissions:
        model_id_to_submissions.setdefault(s.model.id, []).append(s)
    method_to_summaries: dict[str, list[ModelSummary]] = {}
    for model in models:
        method_to_summaries.setdefault(model.method_name, []).append(
            ModelSummary.from_submissions(model, model_id_to_submissions.get(model.id, ())))
    groups = sorted((MethodGroup(name, tuple(sorted(summaries, key=lambda m: (
                         m.model.seed is not None, m.model.seed or 0))))
                     for name, summaries in method_to_summaries.items()),
                    key=lambda g: (g.shown_method_name, g.method_name))
    return sorted(groups, key=lambda g: g.updated_at, reverse=True)  # Stable: ties by name.
