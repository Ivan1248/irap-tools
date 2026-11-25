import dataclasses as dc

import numpy as np
import pytest

from irap_data.metadata import IGNORE_LABEL_INDEX
from irap_evaluation.evaluation import evaluate_predictions
from irap_evaluation.evaluation_sets import (
    SHARED_EVALUATION_SET_NAMES,
    get_evaluation_set,
    get_model_compatible_set,
    get_shared_evaluation_sets,
    select_evaluation_sets,
    select_evaluation_subset,
)

from synthetic_vietnam import NUM_ROADS, ROAD_LENGTH, SPLIT


def test_evaluation_sets_are_nested(metadata, reference_set, full_set):
    model_compatible_set = get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -1), ())
    assert ((reference_set.name, full_set.name, model_compatible_set.name)
            == ("reference", "full", "compatible"))
    assert (tuple(get_shared_evaluation_sets(metadata, "vietnam", SPLIT))
            == SHARED_EVALUATION_SET_NAMES == ("reference", "full"))
    # The -4..0 window leaves out the first 4 segments of each road, (0, -1) only the first, and
    # (0,) none.
    assert reference_set.num_segments == NUM_ROADS * (ROAD_LENGTH - 4)
    assert model_compatible_set.num_segments == NUM_ROADS * (ROAD_LENGTH - 1)
    assert full_set.num_segments == NUM_ROADS * ROAD_LENGTH
    assert (set(reference_set.segment_ids) <= set(model_compatible_set.segment_ids)
            <= set(full_set.segment_ids))
    assert reference_set.sequence_ids == tuple(f"road{r}" for r in range(NUM_ROADS))
    np.testing.assert_array_equal(
        np.array(reference_set.sequence_ids)[reference_set.segment_sequence_indices],
        reference_set.segment_sequence_ids)


def test_model_compatible_set_keeps_in_split_offsets_in_the_split(metadata):
    # The second segment of road 0 moves to the training split, so the third has its previous
    # segment in another split.
    in_val = [s for s in metadata.splits[SPLIT] if s != "0001"]
    metadata = dc.replace(metadata, splits={SPLIT: in_val, "train": ["0001"]})
    any_split = get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -1), ())
    in_split = get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -1), (-1,))
    assert set(any_split.segment_ids) - set(in_split.segment_ids) == {"0002"}
    assert (in_split.in_split_offsets, any_split.in_split_offsets) == ((-1,), ())
    assert in_split.context_window_description == ("context offsets (0, -1), in-split offsets"
                                                   " (-1,)")
    # The in-split offsets select the segments, so they are part of the fingerprint.
    assert in_split.fingerprint != dc.replace(in_split, in_split_offsets=()).fingerprint
    with pytest.raises(ValueError, match="not among the context offsets"):
        get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -1), (1,))


def test_evaluation_set_fingerprint(metadata, reference_set):
    fingerprint = reference_set.fingerprint
    assert get_evaluation_set(metadata, "vietnam", SPLIT, reference_set.context_offsets,
                              reference_set.name,
                              in_split_offsets=reference_set.in_split_offsets
                              ).fingerprint == fingerprint
    attr = next(iter(reference_set.class_indices))
    labels = reference_set.class_indices[attr].copy()
    labels[0] = (labels[0] + 1) % 2
    changed = dc.replace(reference_set,
                         class_indices={**reference_set.class_indices, attr: labels})
    assert changed.fingerprint != fingerprint
    assert (get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -1), ()).fingerprint
            != fingerprint)


def test_select_evaluation_sets(metadata, reference_set, full_set):
    model_compatible_set = get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -1), ())
    # The window of (0, -5) reaches beyond the -4..0 reference window, so that the fifth
    # segment of each road is in the reference set but not in this model-compatible set.
    wide_model_compatible_set = get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -5), ())
    all_segments = full_set.segment_ids
    compatible_segments = model_compatible_set.segment_ids
    without_a_reference_segment = [s for s in compatible_segments
                                   if s != reference_set.segment_ids[0]]
    full_set_note = (f"No predictions for {NUM_ROADS} of the {NUM_ROADS * ROAD_LENGTH} segments of"
                     f" the full set (context offsets (0,)), so the model is not scored on it.")

    def select(segment_ids, model_compatible_set, missing_segment_policy="error"):
        selection = select_evaluation_sets(segment_ids, (reference_set, full_set),
                                           model_compatible_set, missing_segment_policy)
        return ([s.name for s in selection.evaluation_sets],
                selection.model_compatible_set_name, list(selection.notes))

    assert select(all_segments, model_compatible_set) == (
        ["reference", "full", "compatible"], "compatible", [])
    assert select(compatible_segments, model_compatible_set) == (
        ["reference", "compatible"], "compatible", [full_set_note])
    names, compatible_name, notes = select(all_segments, None)
    assert names == ["reference", "full"] and compatible_name is None
    assert "not scored on its model-compatible set" in notes[0]
    names, _, notes = select(wide_model_compatible_set.segment_ids, wide_model_compatible_set)
    assert names == ["compatible"] and "segments of the reference set" in notes[0]
    names, compatible_name, notes = select(all_segments,
                                           dc.replace(reference_set, name="compatible"))
    assert names == ["reference", "full"] and compatible_name == "reference"
    assert "the model-compatible set are those on the reference set" in notes[0]
    names, compatible_name, notes = select(
        all_segments, get_model_compatible_set(metadata, "vietnam", SPLIT, (0,), ()))
    assert names == ["reference", "full"] and compatible_name == "full"
    assert "the model-compatible set are those on the full set" in notes[0]
    with pytest.raises(ValueError, match="cannot be scored"):
        select(without_a_reference_segment, None)
    # Incomplete predictions, also when the model-compatible set has the segments of the
    # reference set.
    for incomplete_set in (model_compatible_set, dc.replace(reference_set, name="compatible")):
        with pytest.raises(ValueError, match="of the model-compatible set .* incomplete"):
            select(without_a_reference_segment, incomplete_set)
    with pytest.raises(ValueError, match="missing_segment_policy"):
        select(all_segments, model_compatible_set, "skip")


def test_select_evaluation_sets_with_missing_segments(metadata, reference_set, full_set):
    model_compatible_set = get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -1), ())
    missing = reference_set.segment_ids[:2]
    # Not in the model-compatible set: the first segment of each road.
    beyond_window = [s for s in full_set.segment_ids if s not in model_compatible_set.segment_ids]

    def select(segment_ids, model_compatible_set):
        return select_evaluation_sets(segment_ids, (reference_set, full_set),
                                      model_compatible_set, "invalid")

    selection = select([s for s in full_set.segment_ids if s not in missing], model_compatible_set)
    assert [s.name for s in selection.evaluation_sets] == ["reference", "full", "compatible"]
    assert selection.model_compatible_set_name == "compatible"
    [note] = selection.notes
    assert note.startswith(f"No predictions for 2 of the {model_compatible_set.num_segments}"
                           f" segments of the model-compatible set")
    assert "scored as invalid" in note
    # A missing segment outside the model-compatible set leaves the full set out.
    selection = select([s for s in full_set.segment_ids if s != beyond_window[0]],
                       model_compatible_set)
    assert [s.name for s in selection.evaluation_sets] == ["reference", "compatible"]
    assert selection.notes == (
        f"No predictions for 1 of the {full_set.num_segments} segments of the full set (context"
        f" offsets (0,)) outside the model-compatible set, so the model is not scored on it.",)
    # Without context offsets, the shared sets must be complete.
    with pytest.raises(ValueError, match="cannot be scored"):
        select([s for s in full_set.segment_ids if s not in missing], None)
    # Equal to the reference set, the model-compatible set takes its scores.
    selection = select([s for s in reference_set.segment_ids if s not in missing],
                       dc.replace(reference_set, name="compatible"))
    assert [s.name for s in selection.evaluation_sets] == ["reference"]
    assert selection.model_compatible_set_name == "reference"


def test_predicted_segments_outside_the_sets_are_noted(reference_set, full_set):
    selection = select_evaluation_sets([*full_set.segment_ids, "x1", "x2"],
                                       (reference_set, full_set), None)
    assert [s.name for s in selection.evaluation_sets] == ["reference", "full"]
    assert ("2 predicted segments are in no evaluation set of the split, e.g. x1, x2, so they are"
            " not scored. The predictions may be of another metadata build." in selection.notes)


def test_sets_without_labels_are_left_out(metadata, reference_set, full_set):
    def without_labels(evaluation_set):
        return dc.replace(evaluation_set, class_indices={
            a: np.full_like(v, IGNORE_LABEL_INDEX) for a, v in evaluation_set.class_indices.items()})

    model_compatible_set = get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -1), ())
    segment_ids = full_set.segment_ids
    selection = select_evaluation_sets(
        segment_ids, (without_labels(reference_set), full_set), model_compatible_set)
    assert [s.name for s in selection.evaluation_sets] == ["full", "compatible"]
    assert selection.notes == ("The reference set has no labels, so the model is not scored on"
                               " it.",)
    unlabeled_shared_sets = (without_labels(reference_set), without_labels(full_set))
    selection = select_evaluation_sets(segment_ids, unlabeled_shared_sets,
                                       without_labels(model_compatible_set))
    assert selection.evaluation_sets == () and len(selection.notes) == 3
    assert selection.model_compatible_set_name is None
    with pytest.raises(ValueError, match="of the model-compatible set .* incomplete"):
        select_evaluation_sets(model_compatible_set.segment_ids[1:], unlabeled_shared_sets,
                               without_labels(model_compatible_set))


def test_subsets_keep_their_road_sequences(reference_set, predictions):
    is_selected = np.isin(reference_set.segment_sequence_ids, ["road1", "road3"])
    subset = select_evaluation_subset(reference_set, is_selected, "roads 1 and 3")
    assert subset.name == "roads 1 and 3"
    assert subset.context_offsets == reference_set.context_offsets
    assert subset.sequence_ids == ("road1", "road3")
    np.testing.assert_array_equal(subset.class_indices["Curvature"],
                                  reference_set.class_indices["Curvature"][is_selected])
    result = evaluate_predictions(predictions, subset)
    assert result.metrics.per_attribute["n"]["Lane width"] == int(is_selected.sum())
    with pytest.raises(ValueError, match="bool array"):
        select_evaluation_subset(reference_set, np.flatnonzero(is_selected), "x")
