"""The candidate labeling errors of an evaluation set, as on the Analysis page: the labeled
segments whose label models disagree with, ranked by the models' mean probability of the label.

A model disagrees with the label of a segment if its prediction is valid and is another class.
An invalid prediction says nothing about the label. The ranking is the self-confidence of
confident learning (Northcutt et al., 2021), averaged over the models with a valid prediction.

The ranking does not depend on the reviews of the labels (`label_reviews`). A page that hides
the reviewed labels lists the ranks that are left (`LabelErrorRanking.find_ranks`).
"""

import dataclasses as dc
import functools
import typing as T

import irap_evaluation as ie
import numpy as np
from irap_data.metadata import IGNORE_LABEL_INDEX

from .prediction_analysis import AlignedPredictions, ModelPrediction, get_model_predictions


@dc.dataclass(frozen=True)
class LabelErrorCandidate:
    """A labeled segment whose label at least one model disagrees with, for one attribute.

    Attributes:
        rank: Its index in the ranking, 0 for the least probable label.
        segment_index: Into the segments of the evaluation set.
        mean_label_probability: The mean probability of the label over the models with a valid
            prediction. For hard predictions, the proportion of them that predict the label.
        model_predictions: Of each model, in the order of the models of the ranking.
    """

    rank: int
    attribute: str
    segment_index: int
    label_index: int
    mean_label_probability: float
    model_predictions: tuple[ModelPrediction, ...]


@dc.dataclass(frozen=True)
class LabelErrorRanking:
    """All candidate labeling errors of models on an evaluation set, the least probable labels
    first (see `rank_label_errors`). A candidate is identified by its rank, its index in the
    arrays.

    Attributes:
        labeled_aligned_seq: (`ModelPrediction.label`, predictions) of the models.
        attributes: The attributes whose labels are ranked.
        attribute_positions: (C,) int64 index into `attributes` of each candidate.
        segment_indices: (C,) int64 index into the segments of the set of each candidate.
        mean_label_probabilities: (C,) float64 `LabelErrorCandidate.mean_label_probability` of
            each candidate, ascending.
    """

    labeled_aligned_seq: tuple[tuple[str, AlignedPredictions], ...]
    attributes: tuple[str, ...]
    attribute_positions: np.ndarray
    segment_indices: np.ndarray
    mean_label_probabilities: np.ndarray

    def __len__(self) -> int:
        return len(self.segment_indices)

    @property
    def evaluation_set(self) -> ie.EvaluationSet:
        return self.labeled_aligned_seq[0][1].evaluation_set

    def get_candidates(self, ranks: T.Iterable[int]) -> tuple[LabelErrorCandidate, ...]:
        class_indices = self.evaluation_set.class_indices

        def get_candidate(rank: int) -> LabelErrorCandidate:
            attribute = self.attributes[self.attribute_positions[rank]]
            segment_index = int(self.segment_indices[rank])
            return LabelErrorCandidate(
                rank=rank, attribute=attribute, segment_index=segment_index,
                label_index=int(class_indices[attribute][segment_index]),
                mean_label_probability=float(self.mean_label_probabilities[rank]),
                model_predictions=get_model_predictions(self.labeled_aligned_seq, attribute,
                                                        segment_index))

        return tuple(get_candidate(int(rank)) for rank in ranks)

    def find_ranks(self, labels: T.Iterable[tuple[int, str]]) -> np.ndarray:
        """Finds the candidates among labels, e.g. among the reviewed ones.

        Args:
            labels: (segment index, attribute) of labels of the set.

        Returns:
            (M,) int64 ascending ranks of the labels that are candidates.
        """
        num_segments = self.evaluation_set.num_segments
        attribute_to_position = {a: p for p, a in enumerate(self.attributes)}
        label_keys = np.array(
            [attribute_to_position[attribute] * num_segments + segment_index
             for segment_index, attribute in labels if attribute in attribute_to_position],
            dtype=np.int64)
        sorted_keys, sorted_key_ranks = self._sorted_candidate_keys
        positions = np.searchsorted(sorted_keys, label_keys)
        is_found = positions < len(sorted_keys)
        is_found[is_found] = sorted_keys[positions[is_found]] == label_keys[is_found]
        return np.unique(sorted_key_ranks[positions[is_found]])

    @functools.cached_property
    def _sorted_candidate_keys(self) -> tuple[np.ndarray, np.ndarray]:
        """(C,) int64 ascending keys of the candidates, `attribute position * number of segments
        + segment index`, and (C,) int64 ranks of the candidates of the keys, so that
        `find_ranks`, which runs on each click, takes O(M log C) time."""
        keys = self.attribute_positions * self.evaluation_set.num_segments + self.segment_indices
        order = np.argsort(keys)
        return keys[order], order


def rank_label_errors(labeled_aligned_seq: T.Sequence[tuple[str, AlignedPredictions]],
                      attributes: T.Sequence[str], *,
                      require_all_models: bool) -> LabelErrorRanking:
    """Ranks the labeled segments that any model disagrees with, or all models with
    `require_all_models`, by the mean probability of the label, lowest first. Ties are in the
    order of `attributes`, then of the segments of the set.

    Args:
        labeled_aligned_seq: (`ModelPrediction.label`, predictions) of models on the same
            evaluation set, at least one.
        attributes: Scored attributes of the predictions.

    Raises:
        ValueError: If there are no models, or they are on different evaluation sets.
    """
    if not labeled_aligned_seq:
        raise ValueError("Labeling errors need the predictions of at least one model.")
    alignments = [a for _, a in labeled_aligned_seq]
    evaluation_set = alignments[0].evaluation_set
    if any(a.evaluation_set.fingerprint != evaluation_set.fingerprint for a in alignments[1:]):
        raise ValueError("The predictions are of different evaluation sets.")
    # Empty arrays first, so that there is something to concatenate without attributes.
    attribute_positions = [np.empty(0, dtype=np.int64)]
    segment_indices = [np.empty(0, dtype=np.int64)]
    mean_probabilities = [np.empty(0, dtype=np.float64)]
    for position, attribute in enumerate(attributes):
        labels = evaluation_set.class_indices[attribute]
        predicted = np.stack([a.predicted_indices[attribute] for a in alignments])  # (M, N)
        is_valid = predicted != IGNORE_LABEL_INDEX
        does_disagree = is_valid & (predicted != labels)
        is_candidate = (labels != IGNORE_LABEL_INDEX) & (
            does_disagree.all(axis=0) if require_all_models else does_disagree.any(axis=0))
        indices = np.flatnonzero(is_candidate)
        candidate_labels = labels[indices]
        is_valid = is_valid[:, indices]
        # Invalid predictions are NaN in `probs`, so they are masked before the sum.
        label_probabilities = np.stack([a.predictions.probs[attribute][indices, candidate_labels]
                                        for a in alignments]).astype(np.float64)
        mean_probabilities.append(np.where(is_valid, label_probabilities, 0).sum(axis=0)
                                  / is_valid.sum(axis=0))
        segment_indices.append(indices)
        attribute_positions.append(np.full(len(indices), position, dtype=np.int64))
    attribute_positions, segment_indices, mean_probabilities = (
        np.concatenate(v) for v in (attribute_positions, segment_indices, mean_probabilities))
    # The candidates are in the order of the attributes, then of the segments, which a stable
    # sort keeps for ties.
    order = np.argsort(mean_probabilities, kind="stable")
    return LabelErrorRanking(
        labeled_aligned_seq=tuple(labeled_aligned_seq), attributes=tuple(attributes),
        attribute_positions=attribute_positions[order], segment_indices=segment_indices[order],
        mean_label_probabilities=mean_probabilities[order])


def find_adjacent_rank(ranks: np.ndarray, rank: int | None, *, is_forward: bool) -> int | None:
    """Finds the next or the previous of the listed ranks, e.g. of the candidates whose labels
    are not reviewed.

    Args:
        ranks: (M,) ascending ranks.
        rank: A rank, which need not be listed, e.g. that of a label that was reviewed and is
            hidden since, or None for a place before the first rank.

    Returns:
        The nearest of `ranks` above `rank`, or below it, None if there is none.
    """
    if rank is None:
        position = 0 if is_forward else -1
    elif is_forward:
        position = int(np.searchsorted(ranks, rank, side="right"))
    else:
        position = int(np.searchsorted(ranks, rank, side="left")) - 1
    return int(ranks[position]) if 0 <= position < len(ranks) else None


def find_listed_position(ranks: np.ndarray, rank: int | None) -> int | None:
    """Finds the position of `rank` in the (M,) ascending `ranks`, None if it is not listed or
    is None."""
    if rank is None:
        return None
    position = int(np.searchsorted(ranks, rank))
    return position if position < len(ranks) and ranks[position] == rank else None
