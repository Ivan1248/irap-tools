"""Model predictions for IRAP road segments: the `Predictions` datatype and conversions.

Classes are identified by IRAP code, not by class index, so predictions of models trained with
differently ordered metadata can be compared and combined (see `align_classes`).
"""

import dataclasses as dc
import functools
import typing as T

import numpy as np

from irap_data.metadata import IGNORE_LABEL_INDEX, validate_irap_codes

#: 'probs': predicted class distributions. 'hard': predicted classes without probabilities, e.g.
#: parsed from generated text, for which the probabilistic metrics are not computed.
OutputKind = T.Literal["probs", "hard"]
OUTPUT_KINDS: tuple[str, ...] = T.get_args(OutputKind)

#: The IRAP code of an invalid prediction in `Predictions.from_irap_codes` and `to_irap_codes`.
INVALID_IRAP_CODE = -1

#: Tolerance of the sum of a stored float32 distribution.
PROBABILITY_SUM_EPS = 1e-4


class PredictionFormatError(ValueError):
    """Predictions that do not satisfy the invariants of `Predictions` or the file format, or
    that do not match the other predictions or the evaluation set they are used with."""


@dc.dataclass(frozen=True)
class MethodInfo:
    """What produced the predictions: a method and, optionally, which run of it.

    Attributes:
        name: A short name that identifies the method in reports. Files with the same name are
            runs of one method, whose spread the bootstrap includes (see `bootstrap`).
        seed: Identifies the run among the runs of the method, e.g. the training seed, or None
            for a method with a single run.
        details: JSON-compatible free-form details, e.g. the checkpoint, the code commit or the
            training configuration.
    """

    name: str
    seed: int | None = None
    details: T.Mapping[str, T.Any] | None = None

    @property
    def run_label(self) -> str:
        return self.name if self.seed is None else f"{self.name}/seed{self.seed}"


@dc.dataclass(frozen=True)
class PredictionHeader:
    """The origin of a set of predictions: the method and the segments it was run on.

    Attributes:
        dataset: The IRAP release, a key of `irap_data.DATASET_PRESETS` ('bh', 'vietnam').
        split: The split the segments come from, e.g. 'val'.
        method: See `MethodInfo`.
        context_offsets: The distinct positions of the segments whose images the model reads,
            relative to the predicted segment in its road sequence, e.g. (0, -1, -4), where -1
            is the previous segment in driving order. With the dataset and split, they define the
            model's own evaluation set. None if unknown.

    Raises:
        PredictionFormatError: If `context_offsets` has duplicates.
    """

    dataset: str
    split: str
    method: MethodInfo
    context_offsets: tuple[int, ...] | None = None

    def __post_init__(self):
        if self.context_offsets is not None:
            offsets = tuple(self.context_offsets)
            if len(set(offsets)) != len(offsets):
                raise PredictionFormatError(f"The context offsets {offsets} have duplicates.")
            object.__setattr__(self, "context_offsets", offsets)


def get_common_dataset_and_split(headers: T.Iterable[PredictionHeader]) -> tuple[str, str]:
    """The (dataset, split) that all headers share.

    Raises:
        PredictionFormatError: If they differ.
    """
    dataset_splits = {(h.dataset, h.split) for h in headers}
    if len(dataset_splits) != 1:
        raise PredictionFormatError(f"The predictions are of different releases or splits:"
                                    f" {sorted(dataset_splits)}.")
    return dataset_splits.pop()


@dc.dataclass(frozen=True)
class Predictions:
    """Per-attribute class distributions predicted for road segments.

    A cell is the prediction of one attribute for one segment, a row of `probs[attribute]`. It is
    invalid if the model gave no usable prediction, e.g. an unparseable VLM answer. The row of an
    invalid cell is NaN.

    Attributes:
        header: See `PredictionHeader`.
        output_kind: See `OutputKind`.
        attribute_to_irap_codes: Attribute -> the IRAP codes of its classes, in the column order
            of `probs`, which need not match the dataset's order (see
            `ClassVocabulary.attribute_to_irap_codes`). Its keys are the predicted attributes.
        segment_ids: (N,) unique segment IDs, the row order of `probs`.
        probs: Attribute -> (N, K) float32 distributions over the classes of
            `attribute_to_irap_codes`, one-hot for hard predictions. Stored as read-only views,
            so that they stay consistent with the cached `is_valid`.

    Raises:
        PredictionFormatError: If there is no attribute, the arrays do not match
            `attribute_to_irap_codes` and `segment_ids`, or a row is neither NaN nor a
            distribution (a one-hot one for hard predictions).
    """

    header: PredictionHeader
    output_kind: OutputKind
    attribute_to_irap_codes: T.Mapping[str, tuple[int, ...]]
    segment_ids: tuple[str, ...]
    probs: T.Mapping[str, np.ndarray]

    def __post_init__(self):
        object.__setattr__(self, "attribute_to_irap_codes",
                           {attr: tuple(codes)
                            for attr, codes in self.attribute_to_irap_codes.items()})
        object.__setattr__(self, "segment_ids", tuple(self.segment_ids))
        object.__setattr__(self, "probs", _to_read_only_views(self.probs))
        _validate_predictions(self)

    @property
    def attributes(self) -> tuple[str, ...]:
        return tuple(self.attribute_to_irap_codes)

    @property
    def num_segments(self) -> int:
        return len(self.segment_ids)

    @functools.cached_property
    def is_valid(self) -> dict[str, np.ndarray]:
        """Attribute -> (N,) read-only bool, False for an invalid cell."""
        return _to_read_only_views({attr: ~np.isnan(self.probs[attr][:, 0])
                                    for attr in self.attributes})

    @classmethod
    def from_probabilities(
        cls,
        header: PredictionHeader,
        attribute_to_irap_codes: T.Mapping[str, T.Sequence[int]],
        segment_ids: T.Sequence[str],
        probs: T.Mapping[str, np.ndarray],
        is_valid: T.Mapping[str, np.ndarray] | None = None,
    ) -> "Predictions":
        """Builds 'probs' predictions from distributions of any float type.

        Args:
            attribute_to_irap_codes: See `Predictions`.
            probs: Attribute -> (N, K) distributions in the class order of
                `attribute_to_irap_codes`.
            is_valid: Attribute -> (N,) bool, or None if all predictions are valid. The rows of
                invalid cells are ignored. A NaN in a valid row is an error, so that a numerical
                failure of the model does not pass for an invalid cell.
        """
        _check_attributes(probs, "probs", attribute_to_irap_codes)
        if is_valid is not None:
            _check_attributes(is_valid, "is_valid", attribute_to_irap_codes)

        def get_probs(attr: str) -> np.ndarray:
            values = np.asarray(probs[attr])
            attr_is_valid = (np.ones(len(values), dtype=bool) if is_valid is None
                             else np.asarray(is_valid[attr], dtype=bool))
            if np.isnan(values[attr_is_valid]).any():
                raise PredictionFormatError(f"{attr!r}: the distribution of a valid cell contains"
                                            f" NaN.")
            return _fill_invalid_rows(values, attr_is_valid)

        return cls(header, "probs", attribute_to_irap_codes, tuple(segment_ids),
                   {attr: get_probs(attr) for attr in probs})

    @classmethod
    def from_logits(
        cls,
        header: PredictionHeader,
        attribute_to_irap_codes: T.Mapping[str, T.Sequence[int]],
        segment_ids: T.Sequence[str],
        logits: T.Mapping[str, np.ndarray],
        is_valid: T.Mapping[str, np.ndarray] | None = None,
    ) -> "Predictions":
        """Builds 'probs' predictions from (N, K) logits per attribute, as
        `from_probabilities`."""
        return cls.from_probabilities(header, attribute_to_irap_codes, segment_ids,
                                      {attr: _softmax(v) for attr, v in logits.items()},
                                      is_valid)

    @classmethod
    def from_class_indices(
        cls,
        header: PredictionHeader,
        attribute_to_irap_codes: T.Mapping[str, T.Sequence[int]],
        segment_ids: T.Sequence[str],
        class_indices: T.Mapping[str, np.ndarray],
    ) -> "Predictions":
        """Builds 'hard' predictions from (N,) class indices per attribute.

        Args:
            attribute_to_irap_codes: See `Predictions`.
            class_indices: Attribute -> (N,) integer indices into the classes of
                `attribute_to_irap_codes`, `IGNORE_LABEL_INDEX` for an invalid prediction.

        Raises:
            PredictionFormatError: If the indices are not integers or not class indices.
        """
        _check_attributes(class_indices, "class_indices", attribute_to_irap_codes)

        def get_probs(attr: str) -> np.ndarray:
            num_classes = len(attribute_to_irap_codes[attr])
            indices = _to_integer_array(attr, class_indices[attr], "class indices")
            is_valid = indices != IGNORE_LABEL_INDEX
            if ((indices < 0) & is_valid).any() or (indices >= num_classes).any():
                raise PredictionFormatError(f"{attr!r}: class indices must be in"
                                            f" [0, {num_classes}) or {IGNORE_LABEL_INDEX}.")
            one_hot = np.eye(num_classes, dtype=np.float32)[np.where(is_valid, indices, 0)]
            return _fill_invalid_rows(one_hot, is_valid)

        return cls(header, "hard", attribute_to_irap_codes, tuple(segment_ids),
                   {attr: get_probs(attr) for attr in class_indices})

    @classmethod
    def from_irap_codes(
        cls,
        header: PredictionHeader,
        attribute_to_irap_codes: T.Mapping[str, T.Sequence[int]],
        segment_ids: T.Sequence[str],
        irap_codes: T.Mapping[str, np.ndarray],
    ) -> "Predictions":
        """Builds 'hard' predictions from (N,) IRAP codes per attribute.

        Args:
            attribute_to_irap_codes: See `Predictions`. The classes the model chooses from.
            irap_codes: Attribute -> (N,) integer IRAP codes of the predicted classes,
                `INVALID_IRAP_CODE` for an invalid prediction.

        Raises:
            PredictionFormatError: If the codes are not integers, a code is not a class of its
                attribute, or `INVALID_IRAP_CODE` is the code of a class.
        """
        _check_attributes(irap_codes, "irap_codes", attribute_to_irap_codes)

        def get_class_indices(attr: str) -> np.ndarray:
            code_to_index = {code: i for i, code in enumerate(attribute_to_irap_codes[attr])}
            if INVALID_IRAP_CODE in code_to_index:
                raise PredictionFormatError(f"{attr!r}: {INVALID_IRAP_CODE} is the code of a"
                                            f" class, so it cannot mark invalid predictions.")
            distinct_codes, inverse = np.unique(
                _to_integer_array(attr, irap_codes[attr], "IRAP codes"), return_inverse=True)
            distinct_codes = distinct_codes.tolist()
            if unknown := sorted(set(distinct_codes) - set(code_to_index) - {INVALID_IRAP_CODE}):
                raise PredictionFormatError(f"{attr!r}: IRAP codes {unknown} are not among the"
                                            f" codes of the attribute, {sorted(code_to_index)}.")
            distinct_indices = [code_to_index.get(c, IGNORE_LABEL_INDEX) for c in distinct_codes]
            return np.array(distinct_indices, dtype=np.int64)[inverse.reshape(-1)]

        return cls.from_class_indices(header, attribute_to_irap_codes, segment_ids,
                                      {attr: get_class_indices(attr) for attr in irap_codes})


def _to_read_only_views(attribute_to_array: T.Mapping[str, T.Any]) -> dict[str, np.ndarray]:
    """Read-only views of the arrays. The arrays themselves stay writeable."""
    def to_read_only_view(array: T.Any) -> np.ndarray:
        view = np.asarray(array).view()
        view.flags.writeable = False
        return view

    return {attr: to_read_only_view(array) for attr, array in attribute_to_array.items()}


def _check_attributes(attribute_to_values: T.Mapping[str, T.Any], name: str,
                      attribute_to_irap_codes: T.Mapping[str, T.Any]) -> None:
    if set(attribute_to_values) != set(attribute_to_irap_codes):
        raise PredictionFormatError(
            f"The attributes of {name} differ from those of attribute_to_irap_codes: missing"
            f" {sorted(set(attribute_to_irap_codes) - set(attribute_to_values))}, extra"
            f" {sorted(set(attribute_to_values) - set(attribute_to_irap_codes))}.")


def _validate_predictions(predictions: Predictions) -> None:
    """Checks the invariants documented in `Predictions`."""
    p = predictions
    if p.output_kind not in OUTPUT_KINDS:
        raise PredictionFormatError(f"output_kind must be one of {OUTPUT_KINDS},"
                                    f" got {p.output_kind!r}.")
    if not p.attribute_to_irap_codes:
        raise PredictionFormatError("Predictions need at least one attribute.")
    num_segments = len(p.segment_ids)
    if not all(isinstance(sid, str) for sid in p.segment_ids):
        raise PredictionFormatError("Segment IDs must be strings.")
    if len(set(p.segment_ids)) != num_segments:
        raise PredictionFormatError("Segment IDs must be unique.")
    _check_attributes(p.probs, "probs", p.attribute_to_irap_codes)
    for attr, codes in p.attribute_to_irap_codes.items():
        try:
            validate_irap_codes(attr, codes)
        except ValueError as e:
            raise PredictionFormatError(str(e)) from e
        probs = p.probs[attr]
        if probs.dtype != np.float32 or probs.shape != (num_segments, len(codes)):
            raise PredictionFormatError(
                f"{attr!r}: expected float32 probabilities of shape"
                f" {(num_segments, len(codes))}, got {probs.dtype} {probs.shape}.")
        is_nan = np.isnan(probs)
        is_invalid = is_nan.all(1)
        if (is_nan.any(1) & ~is_invalid).any():
            raise PredictionFormatError(f"{attr!r}: a row must be either NaN (an invalid cell) or"
                                        f" a distribution without NaN.")
        valid_probs = probs[~is_invalid]
        if not (np.isfinite(valid_probs).all() and (valid_probs >= 0).all()):
            raise PredictionFormatError(f"{attr!r}: probabilities must be finite and >= 0.")
        if not np.all(np.abs(valid_probs.sum(1, dtype=np.float64) - 1) <= PROBABILITY_SUM_EPS):
            raise PredictionFormatError(f"{attr!r}: each distribution must sum to 1.")
        if p.output_kind == "hard" and not (
                np.isin(valid_probs, (0, 1)).all() and (valid_probs.sum(1) == 1).all()):
            raise PredictionFormatError(f"{attr!r}: hard predictions must be one-hot.")


def _replace_unvalidated(predictions: Predictions, **changes) -> Predictions:
    """`dataclasses.replace` without `_validate_predictions`, for changes that keep the
    invariants: selecting segments or attributes and reordering classes. The checks scan every
    probability, which takes about a second for 100k segments and 50 attributes."""
    if "probs" in changes:
        changes = {**changes, "probs": _to_read_only_views(changes["probs"])}
    replaced = object.__new__(Predictions)
    for field in dc.fields(Predictions):
        object.__setattr__(replaced, field.name,
                           changes.get(field.name, getattr(predictions, field.name)))
    return replaced


# Conversions ######################################################################################

def _fill_invalid_rows(probs: np.ndarray, is_valid: np.ndarray) -> np.ndarray:
    return np.where(is_valid[:, None], probs, np.float32(np.nan)).astype(np.float32, copy=False)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax over the last axis in float64."""
    logits = np.asarray(logits, dtype=np.float64)
    exp = np.exp(logits - logits.max(-1, keepdims=True))
    return exp / exp.sum(-1, keepdims=True)


def _to_integer_array(attr: str, values: T.Any, name: str) -> np.ndarray:
    """`values` as an int64 array.

    Raises:
        PredictionFormatError: If `values` is not empty and not of an integer type, since casting,
            e.g. of 2.7, would silently give another value.
    """
    array = np.asarray(values)
    if array.size and not np.issubdtype(array.dtype, np.integer):
        raise PredictionFormatError(f"{attr!r}: {name} must be integers, got {array.dtype}.")
    return array.astype(np.int64)


def to_class_indices(predictions: Predictions) -> dict[str, np.ndarray]:
    """Attribute -> (N,) int64 argmax class indices, `IGNORE_LABEL_INDEX` where invalid."""
    return {attr: np.where(predictions.is_valid[attr],
                           np.nan_to_num(predictions.probs[attr], nan=0).argmax(1),
                           IGNORE_LABEL_INDEX)
            for attr in predictions.attributes}


def to_irap_codes(predictions: Predictions) -> dict[str, np.ndarray]:
    """Attribute -> (N,) int64 IRAP codes of the argmax classes, `INVALID_IRAP_CODE` where
    invalid."""
    return {attr: np.where(indices == IGNORE_LABEL_INDEX, INVALID_IRAP_CODE,
                           np.asarray(predictions.attribute_to_irap_codes[attr])[indices])
            for attr, indices in to_class_indices(predictions).items()}


def align_classes(predictions: Predictions,
                  attribute_to_irap_codes: T.Mapping[str, T.Sequence[int]]) -> Predictions:
    """The predictions of the attributes of `attribute_to_irap_codes`, in its class order.

    Classes are matched by IRAP code. `ClassVocabulary.attribute_to_irap_codes` gives the class
    codes of a vocabulary.

    Raises:
        PredictionFormatError: If an attribute is not predicted, or its predicted IRAP codes
            differ from those of `attribute_to_irap_codes`.
    """
    predicted = predictions.attribute_to_irap_codes
    if not attribute_to_irap_codes:
        raise PredictionFormatError("Predictions need at least one attribute.")
    if missing := [a for a in attribute_to_irap_codes if a not in predicted]:
        raise PredictionFormatError(f"Attributes not predicted: {missing}.")

    def get_column_order(attr: str) -> np.ndarray:
        code_to_index = {code: i for i, code in enumerate(predicted[attr])}
        codes = attribute_to_irap_codes[attr]
        if len(codes) != len(code_to_index) or set(codes) != set(code_to_index):
            raise PredictionFormatError(
                f"{attr!r}: the predicted IRAP codes {sorted(code_to_index)} differ from"
                f" {sorted(codes)}.")
        return np.array([code_to_index[c] for c in codes])

    return _replace_unvalidated(
        predictions,
        attribute_to_irap_codes={attr: tuple(codes)
                                 for attr, codes in attribute_to_irap_codes.items()},
        probs={attr: predictions.probs[attr][:, get_column_order(attr)]
               for attr in attribute_to_irap_codes})


def select_segments(predictions: Predictions, segment_ids: T.Sequence[str]) -> Predictions:
    """The predictions of `segment_ids`, in that order.

    Raises:
        PredictionFormatError: If a segment has no prediction, or `segment_ids` has duplicates.
    """
    segment_ids = tuple(segment_ids)
    row_of = {sid: i for i, sid in enumerate(predictions.segment_ids)}
    if missing := [sid for sid in segment_ids if sid not in row_of]:
        raise PredictionFormatError(f"{len(missing)} of {len(segment_ids)} segments have no"
                                    f" prediction, e.g. {missing[:5]}.")
    if len(set(segment_ids)) != len(segment_ids):
        raise PredictionFormatError("Segment IDs must be unique.")
    rows = np.array([row_of[sid] for sid in segment_ids], dtype=np.int64)
    return _replace_unvalidated(predictions, segment_ids=segment_ids,
                                probs={attr: v[rows] for attr, v in predictions.probs.items()})
