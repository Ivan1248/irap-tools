from datetime import UTC, datetime

import irap_evaluation as ie
import numpy as np
import pytest
from irap_data.metadata import IGNORE_LABEL_INDEX
from synthetic_vietnam import SPLIT

from irap_evaluation_server.label_reviews import (
    LABEL_REVIEW_METHOD_NAME,
    LabelReviewStore,
    get_label_irap_code,
    list_reviewed_labels,
    make_review_predictions,
    review_label,
)


def test_a_correction_replaces_the_correction_of_the_label(tmp_path):
    store = LabelReviewStore(tmp_path)
    store.set_review("vietnam", "s1", "Curvature", "class", 2, 1, " wrong ", "Al")
    store.set_review("vietnam", "s1", "Lane width", "unclear", None, 3, "", "Al")
    store.set_review("vietnam", "s2", "Curvature", "class", 1, 1, "", "Al")
    store.set_review("bh", "s1", "Curvature", "class", 1, 1, "", "Al")
    # Replaced and withdrawn, and the withdrawal of a label without a correction.
    store.set_review("vietnam", "s1", "Curvature", "class", 1, 1, "", "Bo")
    store.remove_review("vietnam", "s2", "Curvature")
    store.remove_review("vietnam", "s3", "Curvature")
    reviews = LabelReviewStore(tmp_path).get_reviews("vietnam")  # Opened again.
    assert reviews.keys() == {("s1", "Curvature"), ("s1", "Lane width")}
    review = reviews["s1", "Curvature"]
    assert (review.correction, review.irap_code, review.reviewer) == ("class", 1, "Bo")
    assert not review.is_outdated(1) and review.is_outdated(2)
    assert LabelReviewStore(tmp_path).get_reviews("bh").keys() == {("s1", "Curvature")}


def test_comments_are_stripped(tmp_path):
    review = LabelReviewStore(tmp_path).set_review("vietnam", "s1", "Curvature", "class", 2, 1,
                                                   " a curve \n", "Al")
    assert review.comment == "a curve"


@pytest.mark.parametrize("correction, irap_code, message", [
    ("class", None, "needs a class"),
    ("unclear", 1, "needs a class"),
    ("withdrawn", None, "Unknown correction"),
])
def test_inconsistent_corrections_are_refused(tmp_path, correction, irap_code, message):
    store = LabelReviewStore(tmp_path)
    with pytest.raises(ValueError, match=message):
        store.set_review("vietnam", "s1", "Curvature", correction, irap_code, 1, "", "Al")
    assert store.get_reviews("vietnam") == {}


def test_reviewed_labels_have_a_correction_of_the_current_label(tmp_path, reference_set):
    store = LabelReviewStore(tmp_path)
    segment_ids = reference_set.segment_ids
    labels = [get_label_irap_code(reference_set, "Curvature", i) for i in range(3)]
    old_label = max(reference_set.vocabulary.get_irap_codes("Curvature")) + 1
    store.set_review("vietnam", segment_ids[2], "Curvature", "unclear", None, labels[2], "", "Al")
    store.set_review("vietnam", segment_ids[0], "Curvature", "class", 1, labels[0], "", "Al")
    # On an earlier label, of a segment outside the set, and of an attribute that it lacks.
    store.set_review("vietnam", segment_ids[1], "Curvature", "class", 1, old_label, "", "Al")
    store.set_review("vietnam", "elsewhere", "Curvature", "class", 1, 1, "", "Al")
    store.set_review("vietnam", segment_ids[0], "Removed", "class", 1, 1, "", "Al")
    reviewed = list_reviewed_labels(store.get_reviews("vietnam"), reference_set)
    assert sorted(reviewed) == [(0, "Curvature"), (2, "Curvature")]


def test_review_of_a_label_of_a_set(tmp_path, reference_set):
    store = LabelReviewStore(tmp_path)
    codes = reference_set.vocabulary.get_irap_codes("Curvature")
    review = review_label(store, reference_set, 1, "Curvature", "class", codes[0], "", "Al")
    assert review.segment_id == reference_set.segment_ids[1]
    assert review.label_irap_code == get_label_irap_code(reference_set, "Curvature", 1)
    assert not review.is_outdated(review.label_irap_code)
    assert store.get_reviews("vietnam") == {review.key: review}
    with pytest.raises(ValueError, match="not a class"):
        review_label(store, reference_set, 1, "Curvature", "class", max(codes) + 1, "", "Al")
    with pytest.raises(ValueError, match="not an attribute"):
        review_label(store, reference_set, 1, "Removed", "unclear", None, "", "Al")
    assert store.get_reviews("vietnam") == {review.key: review}


def test_export_as_predictions(tmp_path, reference_set):
    store = LabelReviewStore(tmp_path)
    segment_ids = reference_set.segment_ids
    lane_width_codes = reference_set.vocabulary.get_irap_codes("Lane width")
    # Of the second and the first segment of the set, and of a segment outside the set.
    store.set_review("vietnam", segment_ids[1], "Lane width", "class", lane_width_codes[2], None,
                     "", "Al")
    store.set_review("vietnam", segment_ids[1], "Curvature", "unclear", None, None, "", "Al")
    store.set_review("vietnam", segment_ids[0], "Curvature", "class", 2, None, "", "Al")
    store.set_review("vietnam", "elsewhere", "Number of lanes", "class", 1, None, "", "Al")
    # Of an attribute and of a class that are not in the vocabulary, e.g. after a metadata update.
    store.set_review("vietnam", segment_ids[2], "Removed", "unclear", None, None, "", "Al")
    store.set_review("vietnam", segment_ids[3], "Lane width", "class",
                     max(lane_width_codes) + 1, None, "", "Al")
    exported_at = datetime(2026, 11, 16, 12, tzinfo=UTC)
    predictions = make_review_predictions(store.get_reviews("vietnam"), reference_set,
                                          exported_at)
    path = tmp_path / "reviews.predictions.parquet"
    ie.write_predictions(path, predictions)
    read = ie.read_predictions(path)
    assert read.output_kind == "hard"
    assert (read.header.dataset, read.header.split) == ("vietnam", SPLIT)
    assert read.header.model.method_name == LABEL_REVIEW_METHOD_NAME
    assert read.segment_ids == segment_ids[:2]
    assert list(read.attribute_to_irap_codes) == ["Lane width", "Curvature"]  # Vocabulary order.
    class_indices = ie.to_class_indices(read)
    np.testing.assert_array_equal(class_indices["Lane width"], [IGNORE_LABEL_INDEX, 2])
    curvature_codes = reference_set.vocabulary.get_irap_codes("Curvature")
    np.testing.assert_array_equal(class_indices["Curvature"],
                                  [curvature_codes.index(2), IGNORE_LABEL_INDEX])
    assert read.header.model.details == {"label_review": {
        "exported_at": "2026-11-16T12:00:00+00:00", "unclear": {"Curvature": [segment_ids[1]]},
        "skipped": {"Removed": [segment_ids[2]], "Lane width": [segment_ids[3]]}}}


def test_export_without_reviews_of_the_set_is_refused(tmp_path, reference_set):
    store = LabelReviewStore(tmp_path)
    exported_at = datetime(2026, 11, 16, tzinfo=UTC)
    with pytest.raises(LookupError, match="has a label review"):
        make_review_predictions({}, reference_set, exported_at)
    store.set_review("vietnam", "elsewhere", "Curvature", "class", 1, None, "", "Al")
    with pytest.raises(LookupError, match="has a label review"):
        make_review_predictions(store.get_reviews("vietnam"), reference_set, exported_at)
    # Only a correction that is left out.
    store.set_review("vietnam", reference_set.segment_ids[0], "Removed", "unclear", None, None,
                     "", "Al")
    with pytest.raises(LookupError, match="apart from 1 "):
        make_review_predictions(store.get_reviews("vietnam"), reference_set, exported_at)
