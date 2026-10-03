"""The analysis of the predictions of runs on an evaluation set, as on the analysis page: the
outcome of each segment for an attribute, the confusion matrix, the segments of a matrix cell,
and the details of a segment.

A run is analysed as `irap_evaluation.evaluate_predictions` scores it, with the scoring settings
(`align_run`), so that the counts agree with its scores.
"""

import collections
import dataclasses as dc
import functools
import threading
import typing as T

import irap_evaluation as ie
import numpy as np
from irap_data.metadata import IGNORE_LABEL_INDEX, IRAPMetadata

from .archive import Submission, SubmissionArchive
from .scoring import ScoringSettings, select_scored_attributes

#: The outcomes of the segments for one run, by outcome code (the `RUN_*` constants).
RUN_OUTCOME_NAMES = ("correct", "wrong", "invalid", "unlabeled")
RUN_CORRECT, RUN_WRONG, RUN_INVALID, RUN_UNLABELED = range(len(RUN_OUTCOME_NAMES))
#: The outcomes of the segments for two runs, A and B, by outcome code (the `COMPARISON_*`
#: constants). A run is correct on a segment if its prediction is valid and equals the label.
COMPARISON_OUTCOME_NAMES = ("both correct", "only A correct", "only B correct", "neither correct",
                            "unlabeled")
(COMPARISON_BOTH_CORRECT, COMPARISON_ONLY_A_CORRECT, COMPARISON_ONLY_B_CORRECT,
 COMPARISON_NEITHER_CORRECT, COMPARISON_UNLABELED) = range(len(COMPARISON_OUTCOME_NAMES))


@dc.dataclass(frozen=True)
class AlignedRun:
    """The predictions of a run on an evaluation set (see `align_run`).

    Attributes:
        predictions: Of the scored attributes and the segments of the set, with the classes and
            segments in the order of the set.
        predicted_indices: Attribute -> (N,) int16 predicted class index of each segment of the
            set, `IGNORE_LABEL_INDEX` where the prediction is invalid.
    """

    submission: Submission
    evaluation_set: ie.EvaluationSet
    predictions: ie.Predictions
    predicted_indices: T.Mapping[str, np.ndarray]


def align_run(submission: Submission, predictions: ie.Predictions,
              evaluation_set: ie.EvaluationSet, settings: ScoringSettings) -> AlignedRun:
    """
    Args:
        predictions: The contents of the file of `submission`.

    Raises:
        irap_evaluation.PredictionFormatError: See `irap_evaluation.align_to_evaluation_set`.
    """
    aligned = ie.align_to_evaluation_set(
        predictions, evaluation_set, select_scored_attributes(evaluation_set, settings),
        missing_attribute_policy=settings.missing_attribute_policy)
    return AlignedRun(submission, evaluation_set, aligned,
                      {a: v.astype(np.int16) for a, v in ie.to_class_indices(aligned).items()})


class AlignedRunCache:
    """The aligned runs that pages show, read from the prediction files on first use.

    It keeps the `max_size` most recently used ones, since a file of Vietnam train takes about
    0.5 s to read and 10 MB of memory. Open pages also keep the runs that they show. The cache is
    shared by the threads of the pages, and two threads that miss the same run both read it.
    """

    def __init__(self, archive: SubmissionArchive, settings: ScoringSettings, max_size: int = 8):
        self.archive = archive
        self.settings = settings
        self.max_size = max_size
        self._key_to_run: collections.OrderedDict[tuple, AlignedRun] = collections.OrderedDict()
        self._lock = threading.Lock()

    def get(self, submission: Submission, evaluation_set: ie.EvaluationSet) -> AlignedRun:
        """
        Raises:
            irap_evaluation.PredictionFormatError: See `align_run`.
        """
        key = (submission.id, submission.file_sha256, evaluation_set.fingerprint)
        with self._lock:
            if (run := self._key_to_run.get(key)) is not None:
                self._key_to_run.move_to_end(key)
                return run
        predictions = ie.read_predictions(self.archive.get_predictions_path(submission.id))
        run = align_run(submission, predictions, evaluation_set, self.settings)
        with self._lock:
            self._key_to_run[key] = run
            while len(self._key_to_run) > self.max_size:
                self._key_to_run.popitem(last=False)
        return run


# Outcomes #########################################################################################

@dc.dataclass(frozen=True)
class AttributeOutcomes:
    """The outcomes of the segments of an evaluation set for one attribute.

    Attributes:
        outcome_names: `RUN_OUTCOME_NAMES`, or `COMPARISON_OUTCOME_NAMES` for two runs.
        segment_outcomes: (N,) int8 index into `outcome_names` of each segment of the set.
        confusion_matrix: (K, K + 1) int64 counts of the labeled segments for the first run:
            rows are labels, columns are predicted classes, and the last column counts invalid
            predictions. Added to the first column, the last one gives the matrix of
            `null_policy='first_class'`, which scores an invalid cell as class 0.
    """

    outcome_names: tuple[str, ...]
    segment_outcomes: np.ndarray
    confusion_matrix: np.ndarray


def _get_num_classes(evaluation_set: ie.EvaluationSet, attribute: str) -> int:
    """K, the number of classes of an attribute, also of the aligned predictions."""
    return len(evaluation_set.vocabulary.get_irap_codes(attribute))


def _get_predicted_columns(run: AlignedRun, attribute: str) -> np.ndarray:
    """(N,) the confusion-matrix column of each segment: the predicted class index, or K for an
    invalid prediction."""
    num_classes = _get_num_classes(run.evaluation_set, attribute)
    predicted = run.predicted_indices[attribute]
    return np.where(predicted == IGNORE_LABEL_INDEX, num_classes, predicted)


def compute_attribute_outcomes(run_a: AlignedRun, attribute: str,
                               run_b: AlignedRun | None = None) -> AttributeOutcomes:
    """
    Raises:
        ValueError: If `run_b` is of another evaluation set.
    """
    evaluation_set = run_a.evaluation_set
    labels = evaluation_set.class_indices[attribute]
    is_labeled = labels != IGNORE_LABEL_INDEX
    predicted_a = run_a.predicted_indices[attribute]
    is_correct_a = predicted_a == labels  # Invalid predictions are IGNORE_LABEL_INDEX.
    if run_b is None:
        outcome_names = RUN_OUTCOME_NAMES
        outcomes = np.select([~is_labeled, predicted_a == IGNORE_LABEL_INDEX, is_correct_a],
                             [RUN_UNLABELED, RUN_INVALID, RUN_CORRECT], RUN_WRONG)
    else:
        if run_b.evaluation_set.fingerprint != evaluation_set.fingerprint:
            raise ValueError("The runs are of different evaluation sets.")
        is_correct_b = run_b.predicted_indices[attribute] == labels
        outcome_names = COMPARISON_OUTCOME_NAMES
        outcomes = np.select(
            [~is_labeled, is_correct_a & is_correct_b, is_correct_a, is_correct_b],
            [COMPARISON_UNLABELED, COMPARISON_BOTH_CORRECT, COMPARISON_ONLY_A_CORRECT,
             COMPARISON_ONLY_B_CORRECT], COMPARISON_NEITHER_CORRECT)
    num_columns = _get_num_classes(evaluation_set, attribute) + 1
    cells = labels[is_labeled] * num_columns + _get_predicted_columns(run_a, attribute)[is_labeled]
    confusion_matrix = np.bincount(cells, minlength=(num_columns - 1) * num_columns).reshape(
        num_columns - 1, num_columns)
    return AttributeOutcomes(outcome_names=outcome_names,
                             segment_outcomes=outcomes.astype(np.int8),
                             confusion_matrix=confusion_matrix)


def select_cell_segments(run: AlignedRun, attribute: str, label_index: int,
                         predicted_index: int) -> np.ndarray:
    """(M,) int64 indices of the segments of the set in a cell of the confusion matrix
    (`AttributeOutcomes.confusion_matrix`), whose `predicted_index` K is the invalid column.

    For probabilistic predictions, the most confident ones come first, so that a cell of
    errors lists the confident errors first. Otherwise, and for invalid ones, the segments are in
    the order of the set.
    """
    labels = run.evaluation_set.class_indices[attribute]
    indices = np.flatnonzero((labels == label_index)
                             & (_get_predicted_columns(run, attribute) == predicted_index))
    if (run.predictions.output_kind == "probs"
            and predicted_index < _get_num_classes(run.evaluation_set, attribute)):
        confidences = run.predictions.probs[attribute][indices, predicted_index]
        indices = indices[np.argsort(-confidences, kind="stable")]
    return indices


# Segments #########################################################################################

@dc.dataclass(frozen=True)
class MapPoints:
    """The segments of an evaluation set that have a location
    (`irap_data.metadata.IRAPMetadata.segment_coordinates`).

    Attributes:
        segment_indices: (P,) int64 indices into the segments of the set.
        coordinates: (P, 2) float64 latitude and longitude in degrees.
        num_segments: The number of segments of the set.
    """

    segment_indices: np.ndarray
    coordinates: np.ndarray
    num_segments: int

    @property
    def num_unlocated(self) -> int:
        return self.num_segments - len(self.segment_indices)

    @functools.cached_property
    def segment_to_point(self) -> np.ndarray:
        """(N,) int64 the point of each segment of the set, -1 if it has no location."""
        segment_to_point = np.full(self.num_segments, -1, dtype=np.int64)
        segment_to_point[self.segment_indices] = np.arange(len(self.segment_indices))
        return segment_to_point


def compute_map_points(metadata: IRAPMetadata, evaluation_set: ie.EvaluationSet) -> MapPoints:
    segment_coordinates = metadata.segment_coordinates
    pairs = [(i, segment_coordinates[s]) for i, s in enumerate(evaluation_set.segment_ids)
             if s in segment_coordinates]
    return MapPoints(segment_indices=np.array([i for i, _ in pairs], dtype=np.int64),
                     coordinates=np.array([c for _, c in pairs], dtype=np.float64).reshape(-1, 2),
                     num_segments=evaluation_set.num_segments)


def get_context_segment_ids(metadata: IRAPMetadata, segment_id: str,
                            context_offsets: T.Sequence[int]) -> list[str | None]:
    """The segments at the context offsets in the road sequence of a segment, None outside the
    sequence."""
    if segment_id not in metadata.sequence_index:
        return [segment_id if offset == 0 else None for offset in context_offsets]
    road_id, position = metadata.sequence_index[segment_id]
    sequence = metadata.road_id_to_segment_id_sequence[road_id]
    return [sequence[position + o] if 0 <= position + o < len(sequence) else None
            for o in context_offsets]


@dc.dataclass(frozen=True)
class RunPrediction:
    """
    Attributes:
        probs: (K,) float32 distribution in the class order of the evaluation set, None if the
            prediction is invalid.
    """

    run_label: str
    probs: np.ndarray | None

    @property
    def predicted_index(self) -> int | None:
        return None if self.probs is None else int(self.probs.argmax())


@dc.dataclass(frozen=True)
class SegmentDetail:
    """A segment of an evaluation set with its label and the predictions of runs for one
    attribute.

    Attributes:
        road_id: None if the segment is in no road sequence.
        position: The position in the road sequence, None if it is in none.
        coordinates: Latitude and longitude in degrees, None if the segment has no location.
        label_index: None if the segment has no label.
        context_segment_ids: (offset, segment id or None outside the road sequence), by offset,
            for the context offsets of the first run, or (0,) if they are unknown.
    """

    segment_id: str
    road_id: str | None
    position: int | None
    coordinates: tuple[float, float] | None
    class_names: tuple[str, ...]
    irap_codes: tuple[int, ...]
    label_index: int | None
    run_predictions: tuple[RunPrediction, ...]
    context_segment_ids: tuple[tuple[int, str | None], ...]


def get_segment_detail(metadata: IRAPMetadata, runs: T.Sequence[AlignedRun], segment_index: int,
                       attribute: str) -> SegmentDetail:
    """
    Args:
        runs: Of the same evaluation set, at least one.
        segment_index: Into the segments of the set.
    """
    evaluation_set = runs[0].evaluation_set
    segment_id = evaluation_set.segment_ids[segment_index]
    road_id, position = metadata.sequence_index.get(segment_id, (None, None))
    label_index = int(evaluation_set.class_indices[attribute][segment_index])
    offsets = sorted(runs[0].submission.context_offsets or (0,))
    return SegmentDetail(
        segment_id=segment_id, road_id=road_id, position=position,
        coordinates=metadata.segment_coordinates.get(segment_id),
        class_names=evaluation_set.vocabulary.get_values(attribute),
        irap_codes=evaluation_set.vocabulary.get_irap_codes(attribute),
        label_index=None if label_index == IGNORE_LABEL_INDEX else label_index,
        run_predictions=tuple(
            RunPrediction(run.submission.run_label,
                          run.predictions.probs[attribute][segment_index]
                          if run.predictions.is_valid[attribute][segment_index] else None)
            for run in runs),
        context_segment_ids=tuple(zip(offsets, get_context_segment_ids(metadata, segment_id,
                                                                       offsets))))
