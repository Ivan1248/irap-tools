import dataclasses as dc
import json
import math
from datetime import datetime, timezone

import irap_evaluation as ie
import numpy as np
import pytest
from irap_data.metadata import MetaFiles, get_dataset_preset
from irap_evaluation.reports.evaluation_report import to_json_dict
from run_helpers import (
    add_model,
    assert_close,
    make_model,
    score_and_get_models,
    score_submission_again,
    with_context_offsets,
)
from synthetic_vietnam import SPLIT

from irap_evaluation_server import scoring as scoring_module
from irap_evaluation_server import scoring_worker as scoring_worker_module
from irap_evaluation_server.config import DatasetConfig
from irap_evaluation_server.database import connect
from irap_evaluation_server.datasets import load_dataset_contexts
from irap_evaluation_server.method_scores import (
    compute_method_scores,
    find_common_evaluation_set,
    get_combination_error,
)
from irap_evaluation_server.scoring import (
    ADDED_AGGREGATE_METRIC_NAMES,
    SCORE_CACHE_VERSION,
    ScoreStore,
    ScoringSettings,
    SubmissionScoring,
    compute_scoring_code_fingerprint,
    get_scored_metric_names,
    get_unscored_reason,
    is_scoring_current,
    remove_class_scores,
    score_predictions,
)
from irap_evaluation_server.scoring_worker import ScoringWorker


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
    model_compatible_set = dataset_contexts["vietnam"].get_model_compatible_set(SPLIT, (0, -1),
                                                                                ())
    assert model_compatible_set.num_segments > reference_set.num_segments
    assert scoring.notes == ()
    compatible_scores = scoring.get_scores(ie.MODEL_COMPATIBLE_SET_NAME)
    assert compatible_scores.context_offsets == (0, -1)
    assert_close(compatible_scores.metrics,
                 score_submission_again(worker, submission, model_compatible_set).metrics)


def test_the_added_metrics_are_scored_and_resampled(archive, dataset_contexts, store, worker):
    submissions = [
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "probs")),
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "hard", is_hard=True))]
    method_to_model = {m.submission.model.method_name: m
                       for m in score_and_get_models(worker, [s.id for s in submissions])}
    for method_name, expected in [
            ("probs", ADDED_AGGREGATE_METRIC_NAMES),
            ("hard", ("amF1_supp1", "amF1_supp100", "totnc_{R>0}", "aminR"))]:
        model = method_to_model[method_name]
        metrics = model.scores.metrics
        assert ([n for n in ADDED_AGGREGATE_METRIC_NAMES if n in metrics.aggregates]
                == list(expected))
        per_attribute = metrics.per_attribute["nc_{R>0}"]
        assert metrics.aggregates["totnc_{R>0}"] == sum(per_attribute.values())
        assert isinstance(metrics.aggregates["totnc_{R>0}"], int)
        resamples = store.get_resamples(model.submission.id, "reference",
                                        model.scores.evaluation_set_fingerprint)
        for name in expected:
            assert name in resamples.aggregates
            assert ie.parse_metric_name(name).name in resamples.per_attribute
    # The scored metrics of hard predictions are those of probabilistic ones without the metrics
    # of distributions.
    assert set(get_scored_metric_names("hard")) == {
        n for n in get_scored_metric_names("probs")
        if not ie.parse_metric_name(n).needs_distributions}


def test_missing_segments_are_scored_as_invalid(archive, dataset_contexts, store, worker):
    context = dataset_contexts["vietnam"]
    reference_set = context.get_reference_set(SPLIT)
    missing = reference_set.segment_ids[:2]
    complete = make_model(dataset_contexts, "m")
    submission = add_model(archive, dataset_contexts, ie.select_segments(
        complete, [s for s in complete.segment_ids if s not in missing]))
    [model] = score_and_get_models(worker, [submission.id])

    assert model.scores.missing_segment_ids == missing
    without_predictions = ie.select_segments(ie.add_invalid_segments(
        ie.select_segments(complete, [s for s in complete.segment_ids if s not in missing]),
        missing), complete.segment_ids)
    assert_close(model.scores.metrics, score_predictions(
        without_predictions, reference_set, worker.settings).metrics)
    scoring = store.get_scoring(submission.id)
    assert any("scored as invalid" in note for note in scoring.notes)
    assert scoring.get_scores(ie.MODEL_COMPATIBLE_SET_NAME).missing_segment_ids == missing
    # The full set has segments outside the model-compatible set, which are all predicted.
    assert scoring.get_scores(ie.FULL_SET_NAME).missing_segment_ids == missing


def test_model_compatible_sets_include_models_whose_set_is_a_shared_one(archive,
                                                                       dataset_contexts, worker):
    reference_offsets = tuple(get_dataset_preset("vietnam").reference_context_offsets)
    submissions = [add_model(archive, dataset_contexts,
                             with_context_offsets(make_model(dataset_contexts, name), offsets))
                   for name, offsets in (("m", (0, -1)), ("r", reference_offsets))]
    models = score_and_get_models(worker, [s.id for s in submissions],
                                  ie.MODEL_COMPATIBLE_SET_NAME)
    assert sorted((m.submission.model.method_name, m.scores.evaluation_set) for m in models) == [
        ("m", ie.MODEL_COMPATIBLE_SET_NAME), ("r", ie.REFERENCE_SET_NAME)]
    scorings = worker.load_split_scorings("vietnam", SPLIT)
    assert scorings.current[submissions[1].id].model_compatible_set == ie.REFERENCE_SET_NAME
    # The Model and Method pages show the scores of a set once.
    assert {n: sorted(m.submission.model.method_name for m in models)
            for n, models in scorings.group_scored_models_by_set().items()} == {
        ie.REFERENCE_SET_NAME: ["m", "r"], ie.FULL_SET_NAME: ["m", "r"],
        ie.MODEL_COMPATIBLE_SET_NAME: ["m"]}


def test_in_split_offsets_select_the_model_compatible_set(archive, store,
                                                          split_boundary_contexts, settings):
    context = split_boundary_contexts["vietnam"]
    worker = ScoringWorker(archive, store, split_boundary_contexts, settings)
    submissions = [
        add_model(archive, split_boundary_contexts, with_context_offsets(
            make_model(split_boundary_contexts, name), (0, -1), in_split_offsets))
        for name, in_split_offsets in (("m", ()), ("s", (-1,)))]
    assert [s.in_split_offsets for s in submissions] == [(), (-1,)]
    models = score_and_get_models(worker, [s.id for s in submissions],
                                  ie.MODEL_COMPATIBLE_SET_NAME)
    method_to_model = {m.submission.model.method_name: m for m in models}
    assert method_to_model["s"].scores.in_split_offsets == (-1,)
    in_split_set = context.get_model_compatible_set(SPLIT, (0, -1), (-1,))
    any_split_set = context.get_model_compatible_set(SPLIT, (0, -1), ())
    assert set(any_split_set.segment_ids) - set(in_split_set.segment_ids) == {"0002"}
    for model, evaluation_set in [(method_to_model["m"], any_split_set),
                                  (method_to_model["s"], in_split_set)]:
        assert find_common_evaluation_set(context, [model]) is evaluation_set
        assert_close(model.scores.metrics, score_submission_again(
            worker, model.submission, evaluation_set).metrics)
    # The same context offsets with other in-split offsets select other segments.
    assert find_common_evaluation_set(context, models) is None
    assert "s/seed0: -1, 0 (in the split: -1)" in get_combination_error(models)


def test_remove_class_scores(archive, dataset_contexts, store, worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    score_and_get_models(worker, [submission.id])
    report = store.get_score_report(submission.id, "reference")
    original = json.loads(json.dumps(report))

    reduced = remove_class_scores(report)
    assert report["classes"] and reduced["classes"] is None
    assert {k: v for k, v in reduced.items() if k != "classes"} == {
        k: v for k, v in report.items() if k != "classes"}
    assert report == original


def test_deletion_deletes_the_cached_values(archive, dataset_contexts, store, worker):
    submissions = [add_model(archive, dataset_contexts, make_model(dataset_contexts, "m", seed))
                   for seed in (1, 2)]
    models = score_and_get_models(worker, [s.id for s in submissions])
    attributes = models[0].scores.attributes
    deleted = models[0].submission
    row = compute_method_scores("m", models, attributes)
    assert worker.get_method_resamples(row, ["amF1"]) is not None

    archive.delete_model(deleted.model.id, actor="Bo")
    worker.update_model_submissions(deleted.model)  # As the pages do after a change.
    assert store.get_scoring(deleted.id) is None
    assert store.get_score_report(deleted.id, "reference") is None
    with pytest.raises(LookupError):
        store.get_resamples(deleted.id, "reference", models[0].scores.evaluation_set_fingerprint)
    # A page that read the scores before the deletion, with a cold cache.
    new_worker = ScoringWorker(archive, store, dataset_contexts, worker.settings)
    assert new_worker.get_method_resamples(row, ["amF1"]) is None


def test_values_of_deleted_submissions_are_not_stored(archive, dataset_contexts, store, worker):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    score_and_get_models(worker, [submission.id])
    scoring = store.get_scoring(submission.id)
    resamples = {s.evaluation_set: store.get_resamples(submission.id, s.evaluation_set,
                                                       s.evaluation_set_fingerprint)
                 for s in scoring.evaluation_set_scores}
    archive.delete_model(submission.model.id, actor="Bo")

    # E.g. values computed while the submission was deleted.
    store.set_scoring(scoring, {s.evaluation_set: {} for s in scoring.evaluation_set_scores},
                      resamples)
    assert store.get_scoring(submission.id) is None
    worker._score(submission.id)  # Queued before the deletion.
    assert store.get_scoring(submission.id) is None


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
        dataset_dir=metadata_dir, metadata_dir=metadata_dir, analysis_splits=(SPLIT,))})
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


def test_get_unscored_reason():
    scoring = SubmissionScoring(submission_id=1, status="scored", message="", notes=(),
                                model_compatible_set=None, settings_fingerprint="s",
                                evaluation_sets_fingerprint="e",
                                scored_at=datetime(2026, 10, 3, tzinfo=timezone.utc))
    assert get_unscored_reason(scoring, is_current=True, is_pending=False) is None
    assert get_unscored_reason(scoring, True, is_pending=True) == "Scoring…"
    assert "current settings" in get_unscored_reason(None, False, False)
    failed = dc.replace(scoring, status="failed", message="No predictions for 3 segments.")
    assert get_unscored_reason(failed, True, False) == failed.message
    error = dc.replace(scoring, status="error", message="Unexpected error.")
    assert "scored again" in get_unscored_reason(error, False, False)


def test_scorings_of_other_scoring_code_are_out_of_date(archive, dataset_contexts, worker,
                                                        monkeypatch):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    score_and_get_models(worker, [submission.id])
    compute_uncached = compute_scoring_code_fingerprint.__wrapped__
    assert compute_uncached(worker.settings) == compute_uncached(worker.settings)

    resample_result = scoring_module.resample_result
    monkeypatch.setattr(scoring_module, "resample_result", lambda result, settings: (
        ie.map_metric_values(lambda v: v + 1, resample_result(result, settings))))
    compute_scoring_code_fingerprint.cache_clear()
    try:
        # A new instance, since the fingerprint is cached.
        new_worker = ScoringWorker(archive, worker.store, dataset_contexts,
                                   dc.replace(worker.settings))
        assert new_worker.get_current_scorings([submission]) == {}
    finally:
        compute_scoring_code_fingerprint.cache_clear()


def test_a_score_cache_of_another_version_is_created_again(archive, dataset_contexts, store,
                                                           worker, caplog):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts, "m"))
    score_and_get_models(worker, [submission.id])
    with connect(store.database_path) as connection:
        connection.execute("PRAGMA user_version = 99")

    new_store = ScoreStore(archive.data_dir)
    assert (f"version 99, so it is created again with version {SCORE_CACHE_VERSION}"
            in caplog.text)
    assert new_store.get_scoring(submission.id) is None
    new_worker = ScoringWorker(archive, new_store, dataset_contexts, worker.settings)
    new_worker.update_all_submissions()
    new_worker.run_pending()
    assert new_store.get_scoring(submission.id).status == "scored"
    # The cache of the current version is kept.
    caplog.clear()
    assert ScoreStore(archive.data_dir).get_scoring(submission.id).status == "scored"
    assert not caplog.text

    # A cache whose tables changed without a new version.
    with connect(store.database_path) as connection:
        connection.execute("ALTER TABLE submission_scores ADD COLUMN extra TEXT")
    with pytest.raises(ValueError, match="submission_scores'] of the score cache"):
        ScoreStore(archive.data_dir)



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
    metrics = dc.replace(scores.metrics, aggregates={**scores.metrics.aggregates,
                                                     "amF1": math.nan, "aNLL": math.inf})
    resamples = store.get_resamples(submission.id, "reference", scores.evaluation_set_fingerprint)
    special = np.resize([math.nan, math.inf, -math.inf, 0.25], len(resamples.aggregates["aNLL"]))
    resamples = dc.replace(resamples, aggregates={**resamples.aggregates, "aNLL": special})
    store.set_scoring(
        dc.replace(scoring, evaluation_set_scores=(dc.replace(scores, metrics=metrics),)),
        {"reference": {}}, {"reference": resamples})
    new_store = ScoreStore(archive.data_dir)
    aggregates = new_store.get_scoring(submission.id).get_scores("reference").metrics.aggregates
    assert math.isnan(aggregates["amF1"]) and aggregates["aNLL"] == math.inf
    assert np.isfinite(aggregates["aA"])
    # A total over attributes stays an int.
    assert isinstance(aggregates["totnc_{R>0}"], int)
    stored = new_store.get_resamples(submission.id, "reference",
                                     scores.evaluation_set_fingerprint)
    np.testing.assert_equal(dc.asdict(stored), dc.asdict(resamples))
