import dataclasses as dc
import json
import math

import irap_evaluation as ie
import numpy as np
import pytest
from irap_data.metadata import MetaFiles
from irap_evaluation.reports.evaluation_report import to_json_dict, to_labeled_metric_values
from run_helpers import add_run, make_run
from synthetic_vietnam import SPLIT

from irap_evaluation_server.config import DatasetConfig
from irap_evaluation_server.datasets import load_dataset_contexts
from irap_evaluation_server.scoring import (
    SCORING_SETTINGS,
    ScoreStore,
    ScoringSettings,
    compute_method_scores,
    compute_requested_intervals,
    evaluate_run,
    group_scored_runs_by_method,
    is_run_scoring_current,
    make_interval_request,
    select_scored_runs,
)
from irap_evaluation_server.scoring_worker import ScoringWorker


def assert_close(a, b, label=""):
    """Equality of metric values, intervals or floats, with NaN equal to NaN."""
    if isinstance(a, ie.MetricValues):
        a, b = to_labeled_metric_values(a), to_labeled_metric_values(b)
        assert a.keys() == b.keys()
        for key in a:
            assert_close(a[key], b[key], key)
    elif dc.is_dataclass(a):
        for field in dc.fields(a):
            assert_close(getattr(a, field.name), getattr(b, field.name), f"{label}.{field.name}")
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b), label
    else:
        assert a == pytest.approx(b, rel=1e-12, abs=0), label


@pytest.fixture
def settings():
    # Fewer resamples than the server, for speed.
    return ScoringSettings(num_resamples=50)


@pytest.fixture
def store(archive):
    return ScoreStore(archive.database_path)


@pytest.fixture
def worker(archive, store, dataset_contexts, settings):
    return ScoringWorker(archive, store, dataset_contexts, settings)


def score_and_get_runs(worker, submission_ids, evaluation_set="reference"):
    worker.update_submissions(submission_ids)
    worker.run_pending()
    return select_scored_runs(worker.archive.list_submissions(), worker.get_current_scorings(),
                              evaluation_set)


def evaluate_submission(worker, submission, evaluation_set):
    return evaluate_run(ie.read_predictions(worker.archive.get_predictions_path(submission.id)),
                        evaluation_set, worker.settings)


def test_scoring_equals_irap_evaluation(archive, dataset_contexts, store, worker):
    submission = add_run(archive, dataset_contexts, make_run(dataset_contexts, "m"))
    [run] = score_and_get_runs(worker, [submission.id])

    scoring = store.get_run_scoring(submission.id)
    assert (scoring.status, scoring.message) == ("scored", "")
    assert is_run_scoring_current(scoring, submission, dataset_contexts["vietnam"],
                                  worker.settings)
    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    result = evaluate_submission(worker, submission, reference_set)
    assert run.scores.attributes == result.attributes
    assert_close(run.scores.metrics, result.metrics)
    assert run.scores.num_invalid == result.num_invalid
    # The JSON round trip keeps undefined values as None, like the report.
    assert store.get_score_report(submission.id, "reference") == json.loads(
        json.dumps(to_json_dict(result)))
    # The context offsets (0, -1) select more segments than the reference window (-4..0).
    model_compatible_set = dataset_contexts["vietnam"].get_model_compatible_set(SPLIT, (0, -1))
    assert model_compatible_set.num_segments > reference_set.num_segments
    assert scoring.notes == ()
    model_scores = scoring.get_scores("model")
    assert model_scores.context_offsets == (0, -1)
    assert_close(model_scores.metrics,
                 evaluate_submission(worker, submission, model_compatible_set).metrics)

    # The default intervals of the run are computed after the scoring.
    request = make_interval_request([run], run.scores.attributes, worker.settings)
    cached = worker.get_intervals(request)
    assert cached.intervals is not None
    assert_close(cached.intervals, ie.compute_bootstrap_intervals(
        [result], num_resamples=worker.settings.num_resamples, seed=0))


def test_method_scores_and_intervals_over_a_subset(archive, dataset_contexts, worker):
    submissions = [
        add_run(archive, dataset_contexts, make_run(dataset_contexts, "m", 1, random_seed=1)),
        add_run(archive, dataset_contexts, make_run(dataset_contexts, "m", 2, random_seed=2)),
        add_run(archive, dataset_contexts, make_run(dataset_contexts, "other", random_seed=3))]
    runs = score_and_get_runs(worker, [s.id for s in submissions])
    method_to_runs = group_scored_runs_by_method(runs)
    assert [r.submission.id for r in method_to_runs["m"]] == [s.id for s in submissions[:2]]
    subset = method_to_runs["m"][0].scores.attributes[1:]

    method_scores = compute_method_scores("m", method_to_runs["m"], subset)
    assert method_scores.error is None and method_scores.num_runs == 2
    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    results = [ie.select_result_attributes(evaluate_submission(worker, s, reference_set), subset)
               for s in submissions[:2]]
    assert_close(method_scores.metrics, ie.compute_method_metrics(results))

    # Intervals over the subset are computed on request.
    request = make_interval_request(method_to_runs["m"], subset, worker.settings)
    assert worker.request_intervals(request) is None
    assert worker.is_intervals_pending(request)
    worker.run_pending()
    assert not worker.is_intervals_pending(request)
    assert_close(worker.request_intervals(request).intervals, ie.compute_bootstrap_intervals(
        results, num_resamples=worker.settings.num_resamples, seed=0))
    # The order of the subset does not matter.
    assert make_interval_request(method_to_runs["m"], subset[::-1],
                                 worker.settings).key == request.key


def test_missing_attribute(archive, dataset_contexts, worker):
    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    attr = ie.select_evaluated_attributes(reference_set)[0]
    submission = add_run(archive, dataset_contexts,
                         make_run(dataset_contexts, "m", drop_attribute=attr))
    [run] = score_and_get_runs(worker, [submission.id])
    assert run.scores.missing_attributes == (attr,)
    assert compute_method_scores("m", [run], run.scores.attributes).missing_attributes == (
        attr,)
    assert compute_method_scores("m", [run], run.scores.attributes[1:]).missing_attributes \
        == ()


def test_runs_that_cannot_be_combined(archive, dataset_contexts, worker):
    submissions = [
        add_run(archive, dataset_contexts, make_run(dataset_contexts, "m", 1)),
        add_run(archive, dataset_contexts, make_run(dataset_contexts, "m", 2, is_hard=True))]
    runs = score_and_get_runs(worker, [s.id for s in submissions])
    method_scores = compute_method_scores("m", runs, runs[0].scores.attributes)
    assert method_scores.metrics is None
    assert "differ in metrics" in method_scores.error


def test_deletion_updates_the_method_intervals(archive, dataset_contexts, worker):
    submissions = [add_run(archive, dataset_contexts, make_run(dataset_contexts, "m", seed))
                   for seed in (1, 2)]
    runs = score_and_get_runs(worker, [s.id for s in submissions])
    attributes = runs[0].scores.attributes
    assert worker.get_intervals(make_interval_request(runs, attributes, worker.settings))

    archive.set_submission_deleted(submissions[0].id, True, actor="Bo")
    [remaining] = score_and_get_runs(worker, [submissions[0].id])
    assert remaining.submission.id == submissions[1].id
    assert worker.get_intervals(make_interval_request([remaining], attributes, worker.settings))

    # A restored run is not scored again, since its scoring is current.
    archive.set_submission_deleted(submissions[0].id, False, actor="Bo")
    worker.update_submissions([submissions[0].id])
    assert not worker.is_scoring_pending(submissions[0].id)


def test_out_of_date_scorings_are_scored_again(archive, dataset_contexts, metadata_dir, store,
                                                worker):
    submission = add_run(archive, dataset_contexts, make_run(dataset_contexts, "m"))
    score_and_get_runs(worker, [submission.id])
    first = store.get_run_scoring(submission.id)

    # Other settings.
    other_worker = ScoringWorker(archive, store, dataset_contexts,
                                 ScoringSettings(num_resamples=20))
    assert other_worker.get_current_scorings() == {}
    other_worker.update_all_submissions()
    assert other_worker.is_scoring_pending(submission.id)
    other_worker.run_pending()
    assert set(other_worker.get_current_scorings()) == {submission.id}
    assert worker.get_current_scorings() == {}

    # A new metadata build with another label of a reference segment.
    path = metadata_dir / MetaFiles.SEGMENT_ID_TO_ROAD_DATA
    road_data = json.loads(path.read_text(encoding="utf-8"))
    segment_id = dataset_contexts["vietnam"].get_reference_set(SPLIT).segment_ids[0]
    codes = road_data[segment_id]["required_attributes"]
    codes["Number of lanes"] = 1 if codes["Number of lanes"] != 1 else 2
    path.write_text(json.dumps(road_data), encoding="utf-8")
    new_contexts = load_dataset_contexts({"vietnam": DatasetConfig(
        metadata_dir=metadata_dir, images_dir=None, analysis_splits=(SPLIT,))})
    new_worker = ScoringWorker(archive, store, new_contexts, ScoringSettings(num_resamples=20))
    assert new_worker.get_current_scorings() == {}
    new_worker.update_all_submissions()
    new_worker.run_pending()
    rescored = store.get_run_scoring(submission.id)
    assert rescored.status == "scored"
    assert (rescored.get_scores("reference").evaluation_set_fingerprint
            != first.get_scores("reference").evaluation_set_fingerprint)


def test_scoring_errors(archive, dataset_contexts, store, worker):
    submissions = [add_run(archive, dataset_contexts, make_run(dataset_contexts, "m", seed))
                   for seed in (1, 2)]
    # An invalid file fails, and is not scored again while it is current.
    archive.get_predictions_path(submissions[0].id).write_bytes(b"not parquet")
    # A missing file is an unexpected error, which is scored again.
    archive.get_predictions_path(submissions[1].id).unlink()
    worker.update_submissions([s.id for s in submissions])
    worker.run_pending()
    failed, error = (store.get_run_scoring(s.id) for s in submissions)
    assert failed.status == "failed" and failed.message
    assert error.status == "error" and "FileNotFoundError" in error.message
    assert set(worker.get_current_scorings()) == {submissions[0].id}
    worker.update_all_submissions()
    assert worker.is_scoring_pending(submissions[1].id)
    assert not worker.is_scoring_pending(submissions[0].id)
    worker.run_pending()


def test_compute_requested_intervals_refuses_changed_files(archive, dataset_contexts, worker):
    submission = add_run(archive, dataset_contexts, make_run(dataset_contexts, "m"))
    [run] = score_and_get_runs(worker, [submission.id])
    request = make_interval_request([run], run.scores.attributes, worker.settings)
    with pytest.raises(ValueError, match="other scoring settings"):
        compute_requested_intervals(request, archive, dataset_contexts, SCORING_SETTINGS)
    changed = dc.replace(request, runs=((submission.id, "0" * 64),))
    with pytest.raises(ValueError, match="another file"):
        compute_requested_intervals(changed, archive, dataset_contexts, worker.settings)


def test_worker_thread(archive, dataset_contexts, store, worker):
    submission = add_run(archive, dataset_contexts, make_run(dataset_contexts, "m"))
    worker.start()
    try:
        worker.update_submissions([submission.id])
        assert worker.wait_until_idle(timeout_s=60)
        assert store.get_run_scoring(submission.id).status == "scored"
    finally:
        worker.stop(timeout_s=60)


def test_store_keeps_undefined_and_infinite_values(archive, dataset_contexts, store, worker):
    submission = add_run(archive, dataset_contexts, make_run(dataset_contexts, "m"))
    score_and_get_runs(worker, [submission.id])
    scoring = store.get_run_scoring(submission.id)
    scores = scoring.get_scores("reference")
    metrics = dc.replace(scores.metrics, averages={**scores.metrics.averages,
                                                   "amF1": math.nan, "aNLL": math.inf})
    store.set_run_scoring(
        dc.replace(scoring, evaluation_set_scores=(dc.replace(scores, metrics=metrics),)),
        {"reference": {}})
    stored = ScoreStore(archive.database_path).get_run_scoring(submission.id)
    averages = stored.get_scores("reference").metrics.averages
    assert math.isnan(averages["amF1"]) and averages["aNLL"] == math.inf
    assert np.isfinite(averages["aA"])
