"""Combining the predictions of several methods or models."""

import typing as T

import numpy as np

from .predictions import (
    ModelInfo,
    PredictionFormatError,
    PredictionHeader,
    Predictions,
    align_classes,
    get_common_dataset_and_split,
    select_segments,
)


def _get_common_segment_ids(predictions_seq: T.Sequence[Predictions],
                            intersect_segments: bool) -> tuple[str, ...]:
    first = predictions_seq[0].segment_ids
    segment_sets = [set(p.segment_ids) for p in predictions_seq]
    if intersect_segments:
        common = set.intersection(*segment_sets)
        return tuple(sid for sid in first if sid in common)
    if any(s != segment_sets[0] for s in segment_sets[1:]):
        raise PredictionFormatError("The predictions cover different segments. Pass"
                                    " intersect_segments=True to combine the common ones.")
    return first


def _get_union_of_context_offsets(
        headers: T.Iterable[PredictionHeader]) -> tuple[int, ...] | None:
    offsets = [h.context_offsets for h in headers]
    if any(o is None for o in offsets):
        return None
    return tuple(sorted(set().union(*offsets), reverse=True))


def make_ensemble_model(members: T.Iterable[ModelInfo], method_name: str, *,
                        seed: int | None = None,
                        details: T.Mapping[str, T.Any] | None = None) -> ModelInfo:
    """Makes the model of an ensemble, which uses every split that a member uses:
    - Its training splits are the union of the members' training splits, or None if those of a
      member are unknown.
    - Its early stopping splits are the union of the members' early stopping splits that are not
      training splits of the ensemble: a member that was fitted on a split makes the ensemble
      fitted on it.
    """
    members = list(members)
    training = (None if any(m.training_splits is None for m in members)
                else set().union(*(m.training_splits for m in members)))
    early_stopping = set().union(*(m.early_stopping_splits for m in members)) - (training or set())
    return ModelInfo(method_name=method_name, training_splits=training,
                     early_stopping_splits=early_stopping, seed=seed, details=details)


def _compute_weighted_mean(probs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """The weighted mean of (M, N, K) distributions with (M, N) weights, 0 for invalid cells.

    Returns:
        (N, K) distributions, NaN in rows with no weight.
    """
    probs = np.nan_to_num(probs.astype(np.float64), nan=0.0, copy=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.einsum("mn,mnk->nk", weights, probs) / weights.sum(0)[:, None]


def ensemble_predictions(
    predictions_seq: T.Sequence[Predictions],
    method_name: str,
    *,
    seed: int | None = None,
    details: T.Mapping[str, T.Any] | None = None,
    weights: T.Sequence[float] | None = None,
    intersect_segments: bool = False,
) -> Predictions:
    """Computes the weighted mean of the distributions of several members (methods or models) of
    the same attributes, release and split. For hard predictions, it gives weighted vote shares.

    Classes are matched by iRAP code, and the output uses the class order of the first member.
    A cell that is invalid in a member is left out of that cell's mean, and it is invalid in the
    ensemble only if it is invalid in every member. The ensemble uses whatever its members use:
    - Its context offsets are the union of the members' offsets, since it reads every frame a
      member reads, or None if the offsets of a member are unknown.
    - Its training and early stopping splits are those of `make_ensemble_model`.

    Args:
        predictions_seq: The members.
        method_name: The method name of the ensemble's model.
        seed: The seed of the ensemble's model.
        details: The details of the ensemble's model. The members with their context offsets and
            weights are added under 'ensemble'.
        weights: A finite positive weight per member. None weights them equally.
        intersect_segments: Whether to combine only the segments that all members predict,
            rather than requiring the same segments.

    Raises:
        PredictionFormatError: If the members differ in release, split, attributes or iRAP codes,
            or in segments without `intersect_segments`.
        ValueError: For no members or invalid weights.
    """
    if not predictions_seq:
        raise ValueError("At least one member is required.")
    weights = np.ones(len(predictions_seq)) if weights is None else np.asarray(weights, float)
    if weights.shape != (len(predictions_seq),) or not (np.isfinite(weights) & (weights > 0)).all():
        raise ValueError(f"Expected {len(predictions_seq)} finite positive weights, got {weights}.")
    headers = [p.header for p in predictions_seq]
    dataset, split = get_common_dataset_and_split(headers)
    if len({frozenset(p.attributes) for p in predictions_seq}) > 1:
        raise PredictionFormatError("The members predict different attributes.")

    attribute_to_irap_codes = predictions_seq[0].attribute_to_irap_codes
    segment_ids = _get_common_segment_ids(predictions_seq, intersect_segments)
    members = [align_classes(select_segments(p, segment_ids), attribute_to_irap_codes)
               for p in predictions_seq]

    def combine_attribute(attr: str) -> np.ndarray:
        cell_weights = weights[:, None] * np.stack([m.is_valid[attr] for m in members])
        mean = _compute_weighted_mean(np.stack([m.probs[attr] for m in members]), cell_weights)
        return mean.astype(np.float32)

    ensemble = {"members": [{"model": h.model.to_json_dict(),
                             "context_offsets": (None if h.context_offsets is None
                                                 else list(h.context_offsets)),
                             "weight": float(w)}
                            for h, w in zip(headers, weights)]}
    model = make_ensemble_model([h.model for h in headers], method_name, seed=seed,
                                details={**(details or {}), "ensemble": ensemble})
    header = PredictionHeader(dataset=dataset, split=split, model=model,
                              context_offsets=_get_union_of_context_offsets(headers))
    return Predictions(header, "probs", attribute_to_irap_codes, segment_ids,
                       {attr: combine_attribute(attr) for attr in attribute_to_irap_codes})
