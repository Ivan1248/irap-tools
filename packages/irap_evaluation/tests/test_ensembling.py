import numpy as np
import pytest

from irap_data.metadata import IGNORE_LABEL_INDEX
from irap_evaluation.ensembling import ensemble_predictions
from irap_evaluation.predictions import (
    MethodInfo,
    PredictionFormatError,
    PredictionHeader,
    Predictions,
)

CLASS_CODES = {"a": (1, 2)}
REVERSED_CLASS_CODES = {"a": (2, 1)}


def _probs(attribute_to_irap_codes, values, name, is_valid=None, offsets=(0,)):
    header = PredictionHeader("vietnam", "val", MethodInfo(name=name), context_offsets=offsets)
    return Predictions.from_probabilities(header, attribute_to_irap_codes, ["s1", "s2"],
                                          {"a": np.array(values)},
                                          None if is_valid is None else {"a": np.array(is_valid)})


def test_mean_aligns_classes_by_code_and_skips_invalid_cells():
    first = _probs(CLASS_CODES, [[0.8, 0.2], [0.6, 0.4]], "m1", offsets=(0, -1))
    # The same distributions over (x, y) written in the order (y, x), and s2 invalid.
    second = _probs(REVERSED_CLASS_CODES, [[0.6, 0.4], [0.5, 0.5]], "m2", is_valid=[True, False],
                    offsets=(0, -4))
    ensemble = ensemble_predictions([first, second], MethodInfo(name="e"), weights=[1, 3])
    np.testing.assert_allclose(ensemble.probs["a"][0], [0.25 * 0.8 + 0.75 * 0.4,
                                                        0.25 * 0.2 + 0.75 * 0.6], rtol=1e-6)
    np.testing.assert_allclose(ensemble.probs["a"][1], [0.6, 0.4], rtol=1e-6)
    assert ensemble.header.context_offsets == (0, -1, -4)
    members = ensemble.header.method.details["ensemble"]["members"]
    assert [(m["method"]["name"], m["context_offsets"], m["weight"]) for m in members] == \
           [("m1", [0, -1], 1), ("m2", [0, -4], 3)]


def test_mean_of_hard_predictions_gives_vote_shares():
    votes = [Predictions.from_class_indices(
                 PredictionHeader("vietnam", "val", MethodInfo(name=f"h{i}")),
                 CLASS_CODES, ["s1", "s2"], {"a": np.array(v)})
             for i, v in enumerate([[0, 1], [1, IGNORE_LABEL_INDEX], [1, IGNORE_LABEL_INDEX]])]
    ensemble = ensemble_predictions(votes, MethodInfo(name="e"))
    np.testing.assert_allclose(ensemble.probs["a"], [[1 / 3, 2 / 3], [0, 1]], rtol=1e-6)
    assert ensemble.output_kind == "probs"


@pytest.mark.parametrize("weights", [[1, 0], [1, -1], [1, np.inf], [1, np.nan], [1]])
def test_weights_must_be_finite_and_positive_per_member(weights):
    members = [_probs(CLASS_CODES, [[0.8, 0.2], [0.6, 0.4]], f"m{i}") for i in range(2)]
    with pytest.raises(ValueError, match="finite positive weights"):
        ensemble_predictions(members, MethodInfo(name="e"), weights=weights)


def test_members_must_cover_the_same_segments_unless_intersected():
    first = _probs(CLASS_CODES, [[0.8, 0.2], [0.6, 0.4]], "m1")
    header = PredictionHeader("vietnam", "val", MethodInfo(name="m2"))
    other = Predictions.from_probabilities(header, CLASS_CODES, ["s2", "s3"],
                                           {"a": np.array([[0.5, 0.5], [0.1, 0.9]])})
    with pytest.raises(PredictionFormatError, match="different segments"):
        ensemble_predictions([first, other], MethodInfo(name="e"))
    ensemble = ensemble_predictions([first, other], MethodInfo(name="e"), intersect_segments=True)
    assert ensemble.segment_ids == ("s2",)
