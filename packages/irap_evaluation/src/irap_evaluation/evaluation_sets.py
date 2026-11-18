"""Evaluation sets: the segments of a split that predictions are scored on, with their labels.

`docs/evaluation.md` defines the reference set, the full set and the model-compatible set.
"""

import dataclasses as dc
import functools
import hashlib
import json
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

from .predictions import format_examples

REFERENCE_SET_NAME = "reference"
FULL_SET_NAME = "full"
MODEL_COMPATIBLE_SET_NAME = "compatible"
#: The names of the sets that are the same for every model (`get_shared_evaluation_sets`).
SHARED_EVALUATION_SET_NAMES = (REFERENCE_SET_NAME, FULL_SET_NAME)

#: How a segment of an evaluation set that the predictions lack is scored:
#: - 'error': not at all. `select_evaluation_sets` refuses predictions that lack segments of their
#:   model-compatible set, and `evaluation.evaluate_predictions` raises an error.
#: - 'invalid': as if every attribute prediction of it were invalid, e.g. for predictions of an
#:   older metadata build, in which the split had other segments.
#: `docs/evaluation.md#missing-segments` gives the reasons.
MissingSegmentPolicy = T.Literal["error", "invalid"]
MISSING_SEGMENT_POLICIES: tuple[str, ...] = T.get_args(MissingSegmentPolicy)


@dc.dataclass(frozen=True)
class EvaluationSet:
    """The labels of the segments that predictions are scored on.

    Attributes:
        dataset: The iRAP release, a key of `irap_data.DATASET_PRESETS`.
        split: The split, e.g. 'val'.
        name: Identifies the set in reports, e.g. `REFERENCE_SET_NAME` (`get_reference_set`),
            `FULL_SET_NAME` (`get_full_set`) or `MODEL_COMPATIBLE_SET_NAME`
            (`get_model_compatible_set`).
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

    @functools.cached_property
    def fingerprint(self) -> str:
        """A SHA-256 of everything that scores on the set depend on: its segments, their labels
        and road sequences, and the classes. It changes e.g. with a new build of the metadata,
        so it tells whether stored scores are of this set."""
        digest = hashlib.sha256()
        description = {
            "dataset": self.dataset, "split": self.split, "name": self.name,
            "context_offsets": self.context_offsets, "segment_ids": self.segment_ids,
            "segment_sequence_ids": self.segment_sequence_ids,
            "attribute_to_irap_codes": self.vocabulary.attribute_to_irap_codes,
            "attributes": list(self.class_indices)}
        digest.update(json.dumps(description).encode())
        for class_indices in self.class_indices.values():
            digest.update(class_indices.astype("<i8").tobytes())
        return digest.hexdigest()


def get_evaluation_set(metadata: IRAPMetadata, dataset: str, split: str,
                       context_offsets: T.Sequence[int], name: str) -> EvaluationSet:
    """The evaluation set that `context_offsets` select in a split.

    It has the segments with a complete context window for `context_offsets`, selected with
    `irap_data.select_preset_split_segments`. The offsets of a model give its model-compatible
    set (`get_model_compatible_set`), and (0,) selects every segment of the split that has an
    image and labels (`get_full_set`).

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
    """The reference set of a split, named `REFERENCE_SET_NAME`, the same for every model.

    It is the evaluation set of the reference offsets of the release
    (`irap_data.DatasetPreset.reference_context_offsets`).
    """
    return get_evaluation_set(metadata, dataset, split,
                              get_dataset_preset(dataset).reference_context_offsets,
                              REFERENCE_SET_NAME)


def get_full_set(metadata: IRAPMetadata, dataset: str, split: str) -> EvaluationSet:
    """The full set of a split, named `FULL_SET_NAME`, the same for every model: every segment of
    the split that has an image and labels, the evaluation set of the context offsets (0,).

    It contains every other evaluation set of the split.
    """
    return get_evaluation_set(metadata, dataset, split, (0,), FULL_SET_NAME)


def get_shared_evaluation_sets(metadata: IRAPMetadata, dataset: str,
                               split: str) -> dict[str, EvaluationSet]:
    """The evaluation sets of a split that are the same for every model, so that models can be
    compared on them, by name (`SHARED_EVALUATION_SET_NAMES`): the reference set and the full
    set."""
    return {REFERENCE_SET_NAME: get_reference_set(metadata, dataset, split),
            FULL_SET_NAME: get_full_set(metadata, dataset, split)}


def get_model_compatible_set(metadata: IRAPMetadata, dataset: str, split: str,
                             context_offsets: T.Sequence[int]) -> EvaluationSet:
    """The model-compatible set of a split for a model's context offsets, named
    `MODEL_COMPATIBLE_SET_NAME`."""
    return get_evaluation_set(metadata, dataset, split, context_offsets,
                              MODEL_COMPATIBLE_SET_NAME)


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


@dc.dataclass(frozen=True)
class EvaluationSetSelection:
    """The evaluation sets to score a model on (`select_evaluation_sets`).

    Attributes:
        evaluation_sets: The shared sets that the model is scored on, then its model-compatible
            set if it is scored on it.
        model_compatible_set_name: The name of the set of `evaluation_sets` that has the
            segments of the model-compatible set: `MODEL_COMPATIBLE_SET_NAME`, or the name of
            the shared set that equals it. None if the context offsets are unknown or the set has
            no labels.
        notes: Why sets are left out, and which segments the predictions lack or have in excess.
    """

    evaluation_sets: tuple[EvaluationSet, ...]
    model_compatible_set_name: str | None
    notes: tuple[str, ...]


def select_evaluation_sets(
    predicted_segment_ids: T.Collection[str],
    shared_sets: T.Collection[EvaluationSet],
    model_compatible_set: EvaluationSet | None,
    missing_segment_policy: MissingSegmentPolicy = "error",
) -> EvaluationSetSelection:
    """Selects the evaluation sets to score a model on: its model-compatible set, if it is known
    and has other segments than each shared set, and each shared set of which the model predicts
    every segment, or, with `missing_segment_policy='invalid'`, every segment outside the
    model-compatible set.

    So with 'invalid', the segments of its model-compatible set that a model lacks, e.g. those
    that a new metadata build adds, are scored as invalid, while a shared set whose context
    window the model's context offsets do not cover is left out as with 'error'.

    A set without labels, e.g. of an unlabeled split (`irap_data.is_unlabeled_split`), is left
    out, but the predictions must still cover its segments as above.

    Args:
        predicted_segment_ids: The segments that the model predicts.
        shared_sets: See `get_shared_evaluation_sets`.
        model_compatible_set: See `get_model_compatible_set`. None if the model's context
            offsets are unknown.
        missing_segment_policy: See `MissingSegmentPolicy`.

    Raises:
        ValueError: For an unknown `missing_segment_policy`, if segments of
            `model_compatible_set` have no prediction and `missing_segment_policy` is 'error', or
            if `model_compatible_set` is None and segments of each shared set have no prediction.
    """
    if missing_segment_policy not in MISSING_SEGMENT_POLICIES:
        raise ValueError(f"missing_segment_policy must be one of {MISSING_SEGMENT_POLICIES},"
                         f" got {missing_segment_policy!r}.")
    predicted = set(predicted_segment_ids)
    notes = []
    # The segments that the model is expected to predict, so that they are scored as invalid.
    expected = (set(model_compatible_set.segment_ids)
                if model_compatible_set is not None and missing_segment_policy == "invalid"
                else set())
    if model_compatible_set is not None and (missing := [
            s for s in model_compatible_set.segment_ids if s not in predicted]):
        description = (f"No predictions for {len(missing)} of the"
                       f" {model_compatible_set.num_segments} segments of the model-compatible set"
                       f" (context offsets {model_compatible_set.context_offsets})")
        if missing_segment_policy == "error":
            raise ValueError(f"{description}, so the predictions are incomplete or of another"
                             f" metadata build.")
        notes.append(f"{description}, e.g. {format_examples(missing)}. They are scored as"
                     f" invalid predictions. The predictions are incomplete or of another"
                     f" metadata build.")
    evaluation_sets = []
    for shared_set in shared_sets:
        if unscored := [s for s in shared_set.segment_ids
                        if s not in predicted and s not in expected]:
            outside = " outside the model-compatible set" if expected else ""
            notes.append(f"No predictions for {len(unscored)} of the {shared_set.num_segments}"
                         f" segments of the {shared_set.name} set (context offsets"
                         f" {shared_set.context_offsets}){outside}, so the model is not scored"
                         f" on it.")
        else:
            evaluation_sets.append(shared_set)
    model_compatible_set_name = None
    if model_compatible_set is None:
        if not evaluation_sets:
            raise ValueError(" ".join([
                *notes, "Without context offsets, the model cannot be scored. If its context"
                " window reaches beyond those of these sets, give its context offsets."
                " Otherwise, the file is incomplete or of another metadata build."]))
        notes.append("No context offsets, so the model is not scored on its model-compatible"
                     " set.")
    # A shared set with the segments of the model-compatible set is in `evaluation_sets`, since
    # its unpredicted segments are those of the model-compatible set.
    elif equal_set_names := [s.name for s in evaluation_sets
                             if s.segment_ids == model_compatible_set.segment_ids]:
        model_compatible_set_name = equal_set_names[0]
        notes.append(f"The context offsets {model_compatible_set.context_offsets} select the"
                     f" segments of the {model_compatible_set_name} set, so the scores on the"
                     f" model-compatible set are those on the {model_compatible_set_name} set.")
    else:
        evaluation_sets.append(model_compatible_set)
        model_compatible_set_name = MODEL_COMPATIBLE_SET_NAME
    labeled_sets = []
    for evaluation_set in evaluation_sets:
        if any(evaluation_set.attr_to_num_labeled.values()):
            labeled_sets.append(evaluation_set)
        else:
            notes.append(f"The {evaluation_set.name} set has no labels, so the model is not"
                         f" scored on it.")
    if model_compatible_set_name not in {s.name for s in labeled_sets}:
        model_compatible_set_name = None
    candidate_sets = [*shared_sets, *([] if model_compatible_set is None
                                      else [model_compatible_set])]
    in_some_set = {s for evaluation_set in candidate_sets for s in evaluation_set.segment_ids}
    if extra := [s for s in predicted_segment_ids if s not in in_some_set]:
        notes.append(f"{len(extra)} predicted segments are in no evaluation set of the split, e.g."
                     f" {format_examples(extra)}, so they are not scored. The predictions may be"
                     f" of another metadata build.")
    return EvaluationSetSelection(tuple(labeled_sets), model_compatible_set_name, tuple(notes))
