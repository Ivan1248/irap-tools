import irap_evaluation as ie
import numpy as np
import pytest
from irap_data.metadata import IGNORE_LABEL_INDEX
from run_helpers import add_model, make_model
from synthetic_vietnam import SPLIT, invalidate_attribute_predictions

from irap_evaluation_server.label_errors import (
    find_adjacent_rank,
    find_listed_position,
    rank_label_errors,
)
from irap_evaluation_server.prediction_analysis import align_predictions
from irap_evaluation_server.scoring import SCORING_SETTINGS

ATTRIBUTES = ["Lane width", "Curvature", "Number of lanes"]


def _align_models(archive, dataset_contexts, evaluation_set, is_hard=False):
    """Aligns two models, of which A has invalid predictions of 'Lane width'."""
    model_a = make_model(dataset_contexts, "a", is_hard=is_hard)
    predictions_seq = [
        invalidate_attribute_predictions(model_a, "Lane width",
                                         np.arange(model_a.num_segments) % 3 != 0),
        make_model(dataset_contexts, "b", random_seed=1, is_hard=is_hard)]
    return [(label, align_predictions(add_model(archive, dataset_contexts, p), p, evaluation_set,
                                      SCORING_SETTINGS))
            for label, p in zip(("A", "B"), predictions_seq, strict=True)]


def _list_expected(labeled_aligned_seq, attributes, require_all_models):
    """Lists (attribute, segment index, mean label probability) of the candidates, in their
    order, segment by segment."""
    evaluation_set = labeled_aligned_seq[0][1].evaluation_set
    expected = []
    for position, attribute in enumerate(attributes):
        labels = evaluation_set.class_indices[attribute]
        for i, label in enumerate(labels.tolist()):
            if label == IGNORE_LABEL_INDEX:
                continue
            predicted = [int(a.predicted_indices[attribute][i]) for _, a in labeled_aligned_seq]
            disagreements = [p != IGNORE_LABEL_INDEX and p != label for p in predicted]
            if not (all if require_all_models else any)(disagreements):
                continue
            label_probabilities = [float(a.predictions.probs[attribute][i, label])
                                   for (_, a), p in zip(labeled_aligned_seq, predicted)
                                   if p != IGNORE_LABEL_INDEX]
            expected.append((np.mean(label_probabilities), position, i))
    return [(attributes[position], i, mean) for mean, position, i in sorted(expected)]


def _get_all_candidates(ranking):
    return ranking.get_candidates(range(len(ranking)))


def _to_tuples(candidates):
    return [(c.attribute, c.segment_index, c.mean_label_probability) for c in candidates]


@pytest.mark.parametrize("is_hard", [False, True])
@pytest.mark.parametrize("num_models", [1, 2])
@pytest.mark.parametrize("require_all_models", [False, True])
def test_candidates_are_ranked_by_the_mean_label_probability(
        archive, dataset_contexts, reference_set, is_hard, num_models, require_all_models):
    labeled_aligned_seq = _align_models(archive, dataset_contexts, reference_set,
                                        is_hard)[:num_models]
    ranking = rank_label_errors(labeled_aligned_seq, ATTRIBUTES,
                                require_all_models=require_all_models)
    candidates = _get_all_candidates(ranking)
    expected = _list_expected(labeled_aligned_seq, ATTRIBUTES, require_all_models)
    assert len(ranking) == len(expected) > 0
    assert _to_tuples(candidates) == pytest.approx(expected)
    assert [c.rank for c in candidates] == list(range(len(ranking)))
    for candidate in candidates:
        assert candidate.label_index == reference_set.class_indices[candidate.attribute][
            candidate.segment_index]
        assert [p.label for p in candidate.model_predictions] == ["A", "B"][:num_models]
        for prediction, (_, aligned) in zip(candidate.model_predictions, labeled_aligned_seq,
                                            strict=True):
            predicted = aligned.predicted_indices[candidate.attribute][candidate.segment_index]
            assert prediction.predicted_index == (
                None if predicted == IGNORE_LABEL_INDEX else predicted)


def test_any_model_includes_segments_with_one_invalid_prediction(archive, dataset_contexts,
                                                                 reference_set):
    labeled_aligned_seq = _align_models(archive, dataset_contexts, reference_set)
    candidates = _get_all_candidates(rank_label_errors(labeled_aligned_seq, ["Lane width"],
                                                       require_all_models=False))
    with_invalid = [c for c in candidates if c.model_predictions[0].probs is None]
    assert with_invalid
    for candidate in with_invalid:  # The mean is B's probability alone.
        b_probs = candidate.model_predictions[1].probs
        assert candidate.mean_label_probability == pytest.approx(
            float(b_probs[candidate.label_index]))
    all_candidates = _get_all_candidates(rank_label_errors(labeled_aligned_seq, ["Lane width"],
                                                           require_all_models=True))
    assert all(c.model_predictions[0].probs is not None for c in all_candidates)


def test_candidates_of_ranks_are_those_of_the_ranking(archive, dataset_contexts, reference_set):
    labeled_aligned_seq = _align_models(archive, dataset_contexts, reference_set)[:1]
    ranking = rank_label_errors(labeled_aligned_seq, ATTRIBUTES, require_all_models=False)
    candidates = _get_all_candidates(ranking)
    selected = ranking.get_candidates(np.array([4, 1]))
    assert [c.rank for c in selected] == [4, 1]
    assert _to_tuples(selected) == _to_tuples([candidates[4], candidates[1]])
    assert ranking.get_candidates([]) == ()


def test_ranks_of_labels_are_found(archive, dataset_contexts, reference_set):
    labeled_aligned_seq = _align_models(archive, dataset_contexts, reference_set)
    ranking = rank_label_errors(labeled_aligned_seq, ATTRIBUTES, require_all_models=True)
    candidates = _get_all_candidates(ranking)
    np.testing.assert_array_equal(
        ranking.find_ranks((c.segment_index, c.attribute) for c in candidates),
        np.arange(len(ranking)))
    candidate_labels = {(c.segment_index, c.attribute) for c in candidates}
    # A label that the models agree with, and one of an attribute that is not ranked.
    other_label = next((i, "Curvature") for i in range(reference_set.num_segments)
                       if (i, "Curvature") not in candidate_labels)
    unranked_label = (candidates[0].segment_index, "Unranked attribute")
    found = ranking.find_ranks([(candidates[5].segment_index, candidates[5].attribute),
                                other_label, unranked_label,
                                (candidates[2].segment_index, candidates[2].attribute)])
    np.testing.assert_array_equal(found, [2, 5])
    assert len(ranking.find_ranks([])) == 0


def test_candidates_are_symmetric_in_the_models(archive, dataset_contexts, reference_set):
    labeled_aligned_seq = _align_models(archive, dataset_contexts, reference_set)
    for require_all_models in (False, True):
        candidates, swapped = (
            _get_all_candidates(rank_label_errors(seq, ATTRIBUTES,
                                                  require_all_models=require_all_models))
            for seq in (labeled_aligned_seq, labeled_aligned_seq[::-1]))
        assert _to_tuples(candidates) == pytest.approx(_to_tuples(swapped))
        for candidate, swapped_candidate in zip(candidates, swapped, strict=True):
            assert ([(p.label, p.predicted_index) for p in candidate.model_predictions]
                    == [(p.label, p.predicted_index)
                        for p in swapped_candidate.model_predictions[::-1]])


def test_no_attributes_give_no_candidates(archive, dataset_contexts, reference_set):
    labeled_aligned_seq = _align_models(archive, dataset_contexts, reference_set)
    ranking = rank_label_errors(labeled_aligned_seq, [], require_all_models=False)
    assert len(ranking) == 0
    assert len(ranking.find_ranks([(0, "Curvature")])) == 0


def test_models_of_different_sets_or_none_are_refused(archive, dataset_contexts, reference_set):
    full_set = dataset_contexts["vietnam"].get_shared_evaluation_sets(SPLIT)[ie.FULL_SET_NAME]
    [a] = _align_models(archive, dataset_contexts, reference_set)[:1]
    [_, b] = _align_models(archive, dataset_contexts, full_set)
    assert a[1].evaluation_set.fingerprint != b[1].evaluation_set.fingerprint
    with pytest.raises(ValueError, match="different evaluation sets"):
        rank_label_errors([a, b], ATTRIBUTES, require_all_models=False)
    with pytest.raises(ValueError, match="at least one model"):
        rank_label_errors([], ATTRIBUTES, require_all_models=False)


@pytest.mark.parametrize("rank, is_forward, expected", [
    (5, True, 9), (5, False, 2),  # From a listed rank.
    (6, True, 9), (6, False, 5),  # From a rank that is not listed, e.g. of a reviewed label.
    (9, True, None), (2, False, None),  # At the ends.
    (None, True, 2), (None, False, None),
])
def test_adjacent_rank(rank, is_forward, expected):
    assert find_adjacent_rank(np.array([2, 5, 9]), rank, is_forward=is_forward) == expected
    assert find_adjacent_rank(np.array([], dtype=np.int64), rank, is_forward=is_forward) is None


@pytest.mark.parametrize("rank, expected", [(2, 0), (9, 2), (6, None), (10, None), (None, None)])
def test_listed_position(rank, expected):
    assert find_listed_position(np.array([2, 5, 9]), rank) == expected
    assert find_listed_position(np.array([], dtype=np.int64), rank) is None
