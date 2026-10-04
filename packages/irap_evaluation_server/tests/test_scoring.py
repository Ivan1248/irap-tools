import dataclasses as dc
import json
import math

import irap_evaluation as ie
import numpy as np
import pytest
from irap_data.metadata import MetaFiles
from irap_evaluation.reports.evaluation_report import to_json_dict
from run_helpers import (
    add_model,
    assert_close,
    make_model,
    score_and_get_models,
    score_submission_again,
)
from synthetic_vietnam import SPLIT

from irap_evaluation_server import scoring_worker as scoring_worker_module
from irap_evaluation_server.config import DatasetConfig
from irap_evaluation_server.datasets import load_dataset_contexts
from irap_evaluation_server.scoring import (
    SCORING_SETTINGS,
    ScoreStore,
    ScoringSettings,
    compute_method_scores,
    group_scored_models_by_method,
    is_scoring_current,
    make_interval_request,
)
from irap_evaluation_server.scoring_worker import ScoringWorker, TaskPriority


def test_scoring_equals_irap_evaluation(archive, dataset_contexts, store, worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    [model] = score_and_get_models(worker, [submission.id])

    scoring = store.get_scoring(submission.id)
    assert (scoring.status, scoring.message) == ("scored", "")
    assert is_scoring_current(scoring, submission, dataset_contexts["vietnam"], worker.settings)
    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    result = score_submission_again(worker, submission, reference_set)
    assert model.scores.attributes == result.attributes
    assert_close(model.scores.metrics, result.metrics)
    assert model.scores.num_invalid == result.num_invalid
    # The JSON round trip keeps undefined values as None, like the report.
    assert store.get_score_report(submission.id, "reference") == json.loads(
        json.dumps(to_json_dict(result)))
    # The context offsets (0, -1) select more segments than the reference window (-4..0).
    model_compatible_set = dataset_contexts["vietnam"].get_model_compatible_set(SPLIT, (0, -1))
    assert model_compatible_set.num_segments > reference_set.num_segments
    assert scoring.notes == ()
    compatible_scores = scoring.get_scores(ie.MODEL_COMPATIBLE_SET_NAME)
    assert compatible_scores.context_offsets == (0, -1)
    assert_close(compatible_scores.metrics,
                 score_submission_again(worker, submission, model_compatible_set).metrics)

    # The default intervals of the model are computed after the scoring.
    request = make_interval_request([model], model.scores.attributes, worker.settings)
    cached = worker.get_result(request)
    assert cached.value is not None
    assert_close(cached.value, ie.compute_bootstrap_intervals(
        [result], num_resamples=worker.settings.num_resamples, seed=0))


def test_method_scores_and_intervals_over_a_subset(archive, dataset_contexts, worker):
    submissions = [
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "m", 1, random_seed=1)),
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "m", 2, random_seed=2)),
        add_model(archive, dataset_contexts,
                  make_model(dataset_contexts, "other", random_seed=3))]
    models = score_and_get_models(worker, [s.id for s in submissions])
    method_to_models = group_scored_models_by_method(models)
    assert [m.submission.id for m in method_to_models["m"]] == [s.id for s in submissions[:2]]
    subset = method_to_models["m"][0].scores.attributes[1:]

    method_scores = compute_method_scores("m", method_to_models["m"], subset)
    assert method_scores.error is None and method_scores.num_models == 2
    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    results = [ie.select_result_attributes(score_submission_again(worker, s, reference_set),
                                           subset)
               for s in submissions[:2]]
    assert_close(method_scores.metrics, ie.compute_method_metrics(results))

    # Intervals over the subset are computed on request.
    request = make_interval_request(method_to_models["m"], subset, worker.settings)
    assert worker.request(request) is None
    assert worker.is_pending(request)
    worker.run_pending()
    assert not worker.is_pending(request)
    assert_close(worker.request(request).value, ie.compute_bootstrap_intervals(
        results, num_resamples=worker.settings.num_resamples, seed=0))
    # The order of the subset does not matter.
    assert make_interval_request(method_to_models["m"], subset[::-1],
                                 worker.settings).key == request.key


def test_set_requests_withdraws_left_subsets(archive, dataset_contexts, worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    [model] = score_and_get_models(worker, [submission.id])
    attributes = model.scores.attributes
    first, second, default = (make_interval_request([model], subset, worker.settings)
                              for subset in (attributes[:2], attributes[1:3], attributes[2:4]))

    worker.set_requests("page 1", [first])
    assert worker.is_pending(first)
    # The page leaves the first subset.
    worker.set_requests("page 1", [second])
    assert not worker.is_pending(first) and worker.is_pending(second)
    # A request that another page waits for is kept.
    worker.set_requests("page 2", [second])
    worker.set_requests("page 1", [])
    assert worker.is_pending(second)
    # A withdrawn request that is also queued otherwise stays queued.
    worker.request(default, TaskPriority.DEFAULT_INTERVALS)
    worker.set_requests("page 1", [default])
    worker.set_requests("page 1", [])
    assert worker.is_pending(default)
    worker.run_pending()
    assert worker.get_result(second).value is not None
    assert worker.get_result(default).value is not None
    assert worker.get_result(first) is None


def test_missing_attribute(archive, dataset_contexts, worker):
    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    attr = ie.select_evaluated_attributes(reference_set)[0]
    submission = add_model(archive, dataset_contexts,
                           make_model(dataset_contexts, "m", drop_attribute=attr))
    [model] = score_and_get_models(worker, [submission.id])
    assert model.scores.missing_attributes == (attr,)
    assert compute_method_scores("m", [model], model.scores.attributes).missing_attributes == (
        attr,)
    assert compute_method_scores("m", [model],
                                 model.scores.attributes[1:]).missing_attributes == ()


def test_models_that_cannot_be_combined(archive, dataset_contexts, worker):
    submissions = [
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "m", 1)),
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "m", 2, is_hard=True))]
    models = score_and_get_models(worker, [s.id for s in submissions])
    method_scores = compute_method_scores("m", models, models[0].scores.attributes)
    assert method_scores.metrics is None
    assert "differ in metrics" in method_scores.error


def test_deletion_updates_the_method_intervals(archive, dataset_contexts, worker):
    submissions = [add_model(archive, dataset_contexts, make_model(dataset_contexts, "m", seed))
                   for seed in (1, 2)]
    models = score_and_get_models(worker, [s.id for s in submissions])
    attributes = models[0].scores.attributes
    assert worker.get_result(make_interval_request(models, attributes, worker.settings))

    archive.set_model_deleted(submissions[0].model.id, True, actor="Bo")
    [remaining] = score_and_get_models(worker, [submissions[0].id])
    assert remaining.submission.id == submissions[1].id
    assert worker.get_result(make_interval_request([remaining], attributes, worker.settings))

    # A restored model is not scored again, since its scoring is current.
    archive.set_model_deleted(submissions[0].model.id, False, actor="Bo")
    worker.update_submissions([archive.get_submission(submissions[0].id)])
    assert not worker.is_scoring_pending(submissions[0].id)


def test_a_current_scoring_is_not_scored_again(archive, dataset_contexts, store, worker,
                                               monkeypatch):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts))
    score_and_get_models(worker, [submission.id])

    def fail(*args):
        raise RuntimeError("Scored again.")

    monkeypatch.setattr(scoring_worker_module, "score_submission", fail)
    worker._score(submission.id)  # E.g. queued again while it was being scored.
    assert store.get_scoring(submission.id).status == "scored"  # Not an error of `fail`.


def test_out_of_date_scorings_are_scored_again(archive, dataset_contexts, metadata_dir, store,
                                                worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    score_and_get_models(worker, [submission.id])
    first = store.get_scoring(submission.id)

    # Other settings.
    other_worker = ScoringWorker(archive, store, dataset_contexts,
                                 ScoringSettings(num_resamples=20))
    assert other_worker.get_current_scorings([submission]) == {}
    other_worker.update_all_submissions()
    assert other_worker.is_scoring_pending(submission.id)
    other_worker.run_pending()
    assert set(other_worker.get_current_scorings([submission])) == {submission.id}
    assert worker.get_current_scorings([submission]) == {}

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
    assert new_worker.get_current_scorings([submission]) == {}
    new_worker.update_all_submissions()
    new_worker.run_pending()
    rescored = store.get_scoring(submission.id)
    assert rescored.status == "scored"
    assert (rescored.get_scores("reference").evaluation_set_fingerprint
            != first.get_scores("reference").evaluation_set_fingerprint)


def test_scoring_errors(archive, dataset_contexts, store, worker):
    submissions = [add_model(archive, dataset_contexts, make_model(dataset_contexts, "m", seed))
                   for seed in (1, 2)]
    # An invalid file fails, and is not scored again while it is current.
    archive.get_predictions_path(submissions[0].id).write_bytes(b"not parquet")
    # A missing file is an unexpected error, which is scored again.
    archive.get_predictions_path(submissions[1].id).unlink()
    worker.update_submissions(submissions)
    worker.run_pending()
    failed, error = (store.get_scoring(s.id) for s in submissions)
    assert failed.status == "failed" and failed.message
    assert error.status == "error" and "FileNotFoundError" in error.message
    assert set(worker.get_current_scorings(submissions)) == {submissions[0].id}
    worker.update_all_submissions()
    assert worker.is_scoring_pending(submissions[1].id)
    assert not worker.is_scoring_pending(submissions[0].id)
    worker.run_pending()


def test_requested_intervals_refuse_changed_files(archive, dataset_contexts, worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    [model] = score_and_get_models(worker, [submission.id])
    request = make_interval_request([model], model.scores.attributes, worker.settings)
    with pytest.raises(ValueError, match="other scoring settings"):
        request.compute(archive, dataset_contexts, SCORING_SETTINGS)
    changed = dc.replace(request, submissions=((submission.id, "0" * 64),))
    with pytest.raises(ValueError, match="other contents than requested"):
        changed.compute(archive, dataset_contexts, worker.settings)
    changed = dc.replace(request, evaluation_set_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="has changed since it was scored"):
        changed.compute(archive, dataset_contexts, worker.settings)


def test_worker_thread(archive, dataset_contexts, store, worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    worker.start()
    try:
        worker.update_submissions([submission])
        assert worker.wait_until_idle(timeout_s=60)
        assert store.get_scoring(submission.id).status == "scored"
    finally:
        worker.stop(timeout_s=60)


def test_store_keeps_undefined_and_infinite_values(archive, dataset_contexts, store, worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    score_and_get_models(worker, [submission.id])
    scoring = store.get_scoring(submission.id)
    scores = scoring.get_scores("reference")
    metrics = dc.replace(scores.metrics, averages={**scores.metrics.averages,
                                                   "amF1": math.nan, "aNLL": math.inf})
    store.set_scoring(
        dc.replace(scoring, evaluation_set_scores=(dc.replace(scores, metrics=metrics),)),
        {"reference": {}})
    stored = ScoreStore(archive.database_path).get_scoring(submission.id)
    averages = stored.get_scores("reference").metrics.averages
    assert math.isnan(averages["amF1"]) and averages["aNLL"] == math.inf
    assert np.isfinite(averages["aA"])
