import dataclasses as dc
import hashlib

import irap_evaluation as ie
import numpy as np
import pytest
from irap_data.metadata import IGNORE_LABEL_INDEX
from run_helpers import add_run, make_run
from synthetic_vietnam import ROAD_LENGTH, SPLIT

from irap_evaluation_server.map_libraries import LibraryFile, ensure_map_libraries
from irap_evaluation_server.prediction_analysis import (
    COMPARISON_OUTCOME_NAMES,
    RUN_OUTCOME_NAMES,
    AlignedRunCache,
    align_run,
    compute_attribute_outcomes,
    compute_map_points,
    get_context_segment_ids,
    get_segment_detail,
    select_cell_segments,
)
from irap_evaluation_server.scoring import SCORING_SETTINGS, score_submission


@pytest.fixture
def reference_set(dataset_contexts):
    return dataset_contexts["vietnam"].get_reference_set(SPLIT)


def _invalidate_every_third(predictions: ie.Predictions, attribute: str) -> ie.Predictions:
    is_valid = np.arange(predictions.num_segments) % 3 != 0
    probs = np.where(is_valid[:, None], predictions.probs[attribute], np.nan).astype(np.float32)
    return dc.replace(predictions, probs={**predictions.probs, attribute: probs})


def _align(archive, dataset_contexts, predictions, evaluation_set):
    submission = add_run(archive, dataset_contexts, predictions)
    return align_run(submission, predictions, evaluation_set, SCORING_SETTINGS)


@pytest.mark.parametrize("is_hard", [False, True])
def test_confusion_matrix_is_that_of_the_scores(archive, dataset_contexts, reference_set,
                                                is_hard):
    predictions = _invalidate_every_third(
        make_run(dataset_contexts, drop_attribute="Curvature", is_hard=is_hard), "Lane width")
    run = _align(archive, dataset_contexts, predictions, reference_set)
    result = ie.evaluate_predictions(predictions, reference_set, null_policy="first_class",
                                     missing_attribute_policy="invalid")
    class_metrics = ie.compute_class_metrics(result, ["F1"])
    _, reports = score_submission(archive, dataset_contexts["vietnam"], run.submission,
                                  SCORING_SETTINGS)
    for attribute in result.attributes:
        matrix = compute_attribute_outcomes(run, attribute).confusion_matrix
        merged = matrix[:, :-1].copy()
        merged[:, 0] += matrix[:, -1]
        np.testing.assert_array_equal(merged, class_metrics[attribute].confusion_matrix)
        np.testing.assert_array_equal(
            merged, reports[ie.REFERENCE_SET_NAME]["classes"][attribute]["confusion_matrix"])
        assert matrix[:, -1].sum() == result.num_invalid[attribute]
    assert compute_attribute_outcomes(run, "Lane width").confusion_matrix[:, -1].sum() > 0
    # A missing attribute is invalid in every labeled segment.
    curvature = compute_attribute_outcomes(run, "Curvature").confusion_matrix
    assert curvature[:, :-1].sum() == 0 and curvature[:, -1].sum() > 0


def test_outcomes_of_one_and_two_runs(archive, dataset_contexts, reference_set):
    run_a = _align(archive, dataset_contexts,
                   _invalidate_every_third(make_run(dataset_contexts, "a"), "Curvature"),
                   reference_set)
    run_b = _align(archive, dataset_contexts, make_run(dataset_contexts, "b", random_seed=1),
                   reference_set)
    labels = reference_set.class_indices["Curvature"]
    predicted_a = run_a.predicted_indices["Curvature"]
    predicted_b = run_b.predicted_indices["Curvature"]
    single = compute_attribute_outcomes(run_a, "Curvature")
    assert single.outcome_names == RUN_OUTCOME_NAMES
    expected = np.where(labels == IGNORE_LABEL_INDEX, "unlabeled",
                        np.where(predicted_a == IGNORE_LABEL_INDEX, "invalid",
                                 np.where(predicted_a == labels, "correct", "wrong")))
    np.testing.assert_array_equal(np.array(RUN_OUTCOME_NAMES)[single.segment_outcomes], expected)
    assert {"correct", "wrong", "invalid", "unlabeled"} <= set(expected)

    compared = compute_attribute_outcomes(run_a, "Curvature", run_b)
    assert compared.outcome_names == COMPARISON_OUTCOME_NAMES
    is_correct_a, is_correct_b = predicted_a == labels, predicted_b == labels
    names = np.array(COMPARISON_OUTCOME_NAMES)[compared.segment_outcomes]
    is_labeled = labels != IGNORE_LABEL_INDEX
    np.testing.assert_array_equal(names == "unlabeled", ~is_labeled)
    np.testing.assert_array_equal(names == "both correct", is_correct_a & is_correct_b)
    np.testing.assert_array_equal(names == "only A correct", is_correct_a & ~is_correct_b)
    np.testing.assert_array_equal(names == "only B correct", ~is_correct_a & is_correct_b)
    np.testing.assert_array_equal(names == "neither correct",
                                  is_labeled & ~is_correct_a & ~is_correct_b)
    # The matrix is of run A.
    np.testing.assert_array_equal(compared.confusion_matrix, single.confusion_matrix)


def test_cell_segments_list_confident_predictions_first(archive, dataset_contexts,
                                                        reference_set):
    run = _align(archive, dataset_contexts,
                 _invalidate_every_third(make_run(dataset_contexts), "Lane width"), reference_set)
    matrix = compute_attribute_outcomes(run, "Lane width").confusion_matrix
    labels = reference_set.class_indices["Lane width"]
    num_classes = matrix.shape[0]
    for label_index in range(num_classes):
        for predicted_index in range(num_classes + 1):
            segments = select_cell_segments(run, "Lane width", label_index, predicted_index)
            assert len(segments) == matrix[label_index, predicted_index]
            assert (labels[segments] == label_index).all()
            if predicted_index == num_classes:
                assert not run.predictions.is_valid["Lane width"][segments].any()
                assert (np.diff(segments) > 0).all()
            else:
                assert (run.predicted_indices["Lane width"][segments] == predicted_index).all()
                confidences = run.predictions.probs["Lane width"][segments, predicted_index]
                assert (np.diff(confidences) <= 0).all()


def test_aligned_run_cache_reads_each_run_once(archive, dataset_contexts, reference_set,
                                               monkeypatch):
    submissions = [add_run(archive, dataset_contexts, make_run(dataset_contexts, "m", seed, seed))
                   for seed in range(3)]
    read_paths = []
    read_predictions = ie.read_predictions
    monkeypatch.setattr(ie, "read_predictions",
                        lambda path: read_paths.append(path) or read_predictions(path))
    cache = AlignedRunCache(archive, SCORING_SETTINGS, max_size=2)
    run = cache.get(submissions[0], reference_set)
    assert run.predictions.segment_ids == reference_set.segment_ids
    assert cache.get(submissions[0], reference_set) is run
    cache.get(submissions[1], reference_set)
    cache.get(submissions[2], reference_set)  # Drops the least recently used, submission 0.
    cache.get(submissions[2], reference_set)
    cache.get(submissions[0], reference_set)
    assert read_paths == [archive.get_predictions_path(submissions[i].id) for i in (0, 1, 2, 0)]


def test_map_points_and_segment_detail(archive, dataset_contexts, reference_set):
    metadata = dataset_contexts["vietnam"].metadata
    points = compute_map_points(metadata, reference_set)
    assert points.num_unlocated == 0
    np.testing.assert_array_equal(points.segment_indices, np.arange(reference_set.num_segments))
    np.testing.assert_array_equal(points.segment_to_point, np.arange(reference_set.num_segments))
    first = reference_set.segment_ids[0]
    np.testing.assert_allclose(points.coordinates[0], metadata.segment_coordinates[first])

    predictions = _invalidate_every_third(make_run(dataset_contexts, "a"), "Lane width")
    run_a = _align(archive, dataset_contexts, predictions, reference_set)
    run_b = _align(archive, dataset_contexts, make_run(dataset_contexts, "b"), reference_set)
    # A segment that is invalid in A, since every third one is.
    index = next(i for i, s in enumerate(reference_set.segment_ids)
                 if predictions.segment_ids.index(s) % 3 == 0)
    segment_id = reference_set.segment_ids[index]
    detail = get_segment_detail(metadata, [run_a, run_b], index, "Lane width")
    road_id, position = metadata.sequence_index[segment_id]
    assert (detail.segment_id, detail.road_id, detail.position) == (segment_id, road_id,
                                                                    position)
    assert detail.coordinates == metadata.segment_coordinates[segment_id]
    assert detail.irap_codes == metadata.vocabulary.get_irap_codes("Lane width")
    assert detail.label_index == reference_set.class_indices["Lane width"][index]
    assert [p.run_label for p in detail.run_predictions] == ["a", "b"]
    assert detail.run_predictions[0].probs is None
    assert detail.run_predictions[1].predicted_index == \
           run_b.predicted_indices["Lane width"][index]
    # The context offsets (0, -1) of the runs, by offset.
    sequence = metadata.road_id_to_segment_id_sequence[road_id]
    assert detail.context_segment_ids == ((-1, sequence[position - 1]), (0, segment_id))


def test_context_segments_outside_the_road_are_none(dataset_contexts):
    metadata = dataset_contexts["vietnam"].metadata
    sequence = metadata.road_id_to_segment_id_sequence["road0"]
    assert get_context_segment_ids(metadata, sequence[1], (0, -1, -4)) == [
        sequence[1], sequence[0], None]
    assert get_context_segment_ids(metadata, sequence[-1], (1,)) == [None]
    assert len(sequence) == ROAD_LENGTH


def test_segment_images_must_be_in_the_images_directory(dataset_contexts, tmp_path):
    context = dataset_contexts["vietnam"]
    segment_id = context.metadata.splits[SPLIT][0]
    images_dir = tmp_path / "images"
    path = images_dir / context.metadata.segment_id_to_data_paths_rel[segment_id]["rgb"]
    assert context.find_segment_image(segment_id) is None  # No images directory.
    context = dc.replace(context, config=dc.replace(context.config, images_dir=images_dir))
    assert context.find_segment_image(segment_id) is None
    path.parent.mkdir(parents=True)
    path.write_bytes(b"png")
    assert context.find_segment_image(segment_id) == path.resolve()
    (tmp_path / "outside.png").write_bytes(b"png")
    context.metadata.segment_id_to_data_paths_rel[segment_id] = {"rgb": "../outside.png"}
    assert context.find_segment_image(segment_id) is None


def test_ensure_map_libraries_checks_the_hashes(tmp_path):
    url_to_data = {"https://a/x.js": b"x", "https://a/y.css": b"y"}
    files = [LibraryFile(f"lib/{url[-4:]}", url, hashlib.sha256(data).hexdigest())
             for url, data in url_to_data.items()]
    fetched_urls = []

    def fetch_url(url: str) -> bytes:
        fetched_urls.append(url)
        return url_to_data[url]

    ensure_map_libraries(tmp_path, files, fetch_url)
    assert (tmp_path / "lib/x.js").read_bytes() == b"x"
    ensure_map_libraries(tmp_path, files, fetch_url)  # The files are present.
    assert fetched_urls == list(url_to_data)
    (tmp_path / "lib/x.js").write_bytes(b"changed")
    url_to_data["https://a/x.js"] = b"tampered"
    with pytest.raises(ValueError, match="SHA-256"):
        ensure_map_libraries(tmp_path, files, fetch_url)
    assert (tmp_path / "lib/x.js").read_bytes() == b"changed"
