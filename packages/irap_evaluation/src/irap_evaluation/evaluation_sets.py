"""Evaluation sets: the segments of a split that predictions are scored on, with their labels.

`docs/evaluation.md` defines the reference set and the model set.
"""

import dataclasses as dc
import functools
import typing as T

import numpy as np

from irap_data.metadata import (
    IGNORE_LABEL_INDEX,
    ClassVocabulary,
    IRAPMetadata,
    compute_label_matrix,
    get_dataset_preset,
    select_preset_split_segments,
)


@dc.dataclass(frozen=True)
class EvaluationSet:
    """The labels of the segments that predictions are scored on.

    Attributes:
        dataset: The iRAP release, a key of `irap_data.DATASET_PRESETS`.
        split: The split, e.g. 'val'.
        name: Identifies the set in reports, e.g. 'reference' (`get_reference_set`) or 'model'
            for a model set.
        context_offsets: The context offsets that selected the segments, also of a subset
            (see `select_evaluation_subset`).
        vocabulary: The attributes and classes of the dataset.
        segment_ids: (N,) the evaluated segments.
        class_indices: Attribute -> (N,) int64 class indices in `vocabulary` order,
            `IGNORE_LABEL_INDEX` where the segment has no label.
        segment_sequence_ids: (N,) the road sequence of each segment. Segments of one sequence
            are correlated, so the bootstrap resamples whole sequences.
    """

    dataset: str
    split: str
    name: str
    context_offsets: tuple[int, ...]
    vocabulary: ClassVocabulary
    segment_ids: tuple[str, ...]
    class_indices: T.Mapping[str, np.ndarray]
    segment_sequence_ids: tuple[str, ...]

    @property
    def num_segments(self) -> int:
        return len(self.segment_ids)

    @functools.cached_property
    def sequence_ids(self) -> tuple[str, ...]:
        """(G,) the sorted distinct road sequences of the segments."""
        return tuple(sorted(set(self.segment_sequence_ids)))

    @functools.cached_property
    def segment_sequence_indices(self) -> np.ndarray:
        """(N,) int64 index into `sequence_ids` of the road sequence of each segment."""
        index_of = {sequence_id: i for i, sequence_id in enumerate(self.sequence_ids)}
        return np.array([index_of[s] for s in self.segment_sequence_ids], dtype=np.int64)

    @property
    def attr_to_num_labeled(self) -> dict[str, int]:
        return {a: int((v != IGNORE_LABEL_INDEX).sum()) for a, v in self.class_indices.items()}


def get_evaluation_set(metadata: IRAPMetadata, dataset: str, split: str,
                       context_offsets: T.Sequence[int], name: str) -> EvaluationSet:
    """The evaluation set that `context_offsets` select in a split.

    It has the segments with a complete context window for `context_offsets`, selected with
    `irap_data.select_preset_split_segments`. The offsets of a model give its model set, and
    (0,) selects every segment of the split that has an image and labels.

    Args:
        name: See `EvaluationSet.name`.
    """
    selection = select_preset_split_segments(metadata, dataset, split, context_offsets)
    vocabulary = metadata.vocabulary
    labels = compute_label_matrix(selection.segment_id_to_labels, selection.segment_ids,
                                  len(vocabulary.attribute_names))
    return EvaluationSet(
        dataset=dataset, split=split, name=name,
        context_offsets=tuple(context_offsets), vocabulary=vocabulary,
        segment_ids=selection.segment_ids,
        class_indices={a: labels[:, i] for i, a in enumerate(vocabulary.attribute_names)},
        segment_sequence_ids=tuple(metadata.sequence_index[sid][0]
                                   for sid in selection.segment_ids))


def get_reference_set(metadata: IRAPMetadata, dataset: str, split: str) -> EvaluationSet:
    """The reference set of a split, named 'reference', the same for every model.

    It is the evaluation set of the reference offsets of the release
    (`irap_data.DatasetPreset.reference_context_offsets`).
    """
    return get_evaluation_set(metadata, dataset, split,
                              get_dataset_preset(dataset).reference_context_offsets, "reference")


def select_evaluation_subset(evaluation_set: EvaluationSet, is_selected: np.ndarray,
                             name: str) -> EvaluationSet:
    """A subset of an evaluation set, e.g. the segments of one region.

    The segments keep their road sequences, so that the bootstrap resamples them as in the
    whole set.

    Args:
        is_selected: (N,) bool, True for the segments of the subset.
        name: The name of the subset (see `EvaluationSet.name`).

    Raises:
        ValueError: If `is_selected` is not an (N,) bool array.
    """
    is_selected = np.asarray(is_selected)
    if is_selected.dtype != bool or is_selected.shape != (evaluation_set.num_segments,):
        raise ValueError(f"is_selected must be a bool array of shape"
                         f" {(evaluation_set.num_segments,)}, got {is_selected.dtype}"
                         f" {is_selected.shape}.")
    rows = np.flatnonzero(is_selected)
    return dc.replace(
        evaluation_set, name=name,
        segment_ids=tuple(evaluation_set.segment_ids[i] for i in rows),
        class_indices={a: v[rows] for a, v in evaluation_set.class_indices.items()},
        segment_sequence_ids=tuple(evaluation_set.segment_sequence_ids[i] for i in rows))


def select_evaluation_sets(
    predicted_segment_ids: T.Collection[str],
    reference_set: EvaluationSet,
    model_set: EvaluationSet | None,
) -> tuple[list[EvaluationSet], list[str]]:
    """The evaluation sets to score a model on: the reference set, if the model predicts all its
    segments, and the model set, if it is known and has other segments than the reference set.

    A set without labels, e.g. of an unlabeled split (`irap_data.is_unlabeled_split`), is left
    out, but the predictions must still cover its segments as above.

    Args:
        predicted_segment_ids: The segments that the model predicts.
        reference_set: See `get_reference_set`.
        model_set: The evaluation set of the model's context offsets (see
            `get_evaluation_set`), or None if they are unknown.

    Returns:
        The evaluation sets, and notes on why a set is left out.

    Raises:
        ValueError: If segments of `model_set` have no prediction, or reference segments have
            none and `model_set` is None.
    """
    predicted = set(predicted_segment_ids)

    def count_missing(evaluation_set: EvaluationSet) -> int:
        return sum(sid not in predicted for sid in evaluation_set.segment_ids)

    if model_set is not None and (num_missing_model := count_missing(model_set)):
        raise ValueError(
            f"No predictions for {num_missing_model} of the {model_set.num_segments} segments of"
            f" the model set (context offsets {model_set.context_offsets}), so the predictions"
            f" are incomplete or of another metadata build.")
    num_missing = count_missing(reference_set)
    missing_note = (f"No predictions for {num_missing} of the {reference_set.num_segments}"
                    f" reference segments")
    if num_missing and model_set is None:
        raise ValueError(
            f"{missing_note} and no context offsets, so the model cannot be scored. If its"
            f" context window reaches beyond the reference window"
            f" {reference_set.context_offsets}, give its context offsets. Otherwise, the file is"
            f" incomplete or of another metadata build.")
    evaluation_sets, notes = [], []
    if num_missing:
        notes.append(f"{missing_note}, so the model is not scored on the reference set (context"
                     f" offsets {reference_set.context_offsets}).")
    else:
        evaluation_sets.append(reference_set)
    if model_set is None:
        notes.append("No context offsets, so the model is not scored on its own set.")
    elif model_set.segment_ids == reference_set.segment_ids:
        notes.append(f"The context offsets {model_set.context_offsets} select the segments of the"
                     f" reference set, so the model set is left out.")
    else:
        evaluation_sets.append(model_set)
    labeled_sets = []
    for evaluation_set in evaluation_sets:
        if any(evaluation_set.attr_to_num_labeled.values()):
            labeled_sets.append(evaluation_set)
        else:
            notes.append(f"The {evaluation_set.name} set has no labels, so the model is not"
                         f" scored on it.")
    return labeled_sets, notes
