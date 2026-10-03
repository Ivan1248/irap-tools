"""Synthetic runs, their uploads and scoring, for the tests."""

import dataclasses as dc
import math
from pathlib import Path

import irap_evaluation as ie
import pytest
from irap_evaluation.reports.evaluation_report import to_labeled_metric_values
from synthetic_vietnam import SPLIT, make_predictions

from irap_evaluation_server.archive import SubmissionArchive, add_uploaded_submission
from irap_evaluation_server.scoring import evaluate_run, select_scored_runs

#: The second split of the `two_split_contexts` fixture.
OTHER_SPLIT = "test"


def with_split(predictions: ie.Predictions, split: str) -> ie.Predictions:
    return dc.replace(predictions, header=dc.replace(predictions.header, split=split))


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


def score_and_get_runs(worker, submission_ids, evaluation_set="reference"):
    """Scores the submissions with the worker. Returns the scored runs of all submissions."""
    worker.update_submissions(submission_ids)
    worker.run_pending()
    return select_scored_runs(worker.archive.list_submissions(), worker.get_current_scorings(),
                              evaluation_set)


def evaluate_submission(worker, submission, evaluation_set):
    return evaluate_run(ie.read_predictions(worker.archive.get_predictions_path(submission.id)),
                        evaluation_set, worker.settings)


def make_run(dataset_contexts, name="m", method_seed=None, random_seed=0,
             drop_attribute=None, is_hard=False) -> ie.Predictions:
    """Predictions of a run on `SPLIT`, without `drop_attribute`, and hard if `is_hard`."""
    metadata = dataset_contexts["vietnam"].metadata
    predictions = make_predictions(metadata.vocabulary, list(metadata.splits[SPLIT]), name=name,
                                   seed=random_seed, method_seed=method_seed)
    if drop_attribute is not None:
        predictions = dc.replace(
            predictions,
            attribute_to_irap_codes={a: c for a, c in predictions.attribute_to_irap_codes.items()
                                     if a != drop_attribute},
            probs={a: p for a, p in predictions.probs.items() if a != drop_attribute})
    if is_hard:
        predictions = ie.Predictions.from_class_indices(
            predictions.header, predictions.attribute_to_irap_codes, predictions.segment_ids,
            ie.to_class_indices(predictions))
    return predictions


def write_upload(archive: SubmissionArchive, predictions: ie.Predictions) -> Path:
    path = archive.make_upload_path("run.predictions.parquet")
    ie.write_predictions(path, predictions)
    return path


def add_run(archive, dataset_contexts, predictions):
    """Uploads and stores a run. Returns its submission."""
    return add_uploaded_submission(archive, dataset_contexts, write_upload(archive, predictions),
                                   file_name="run.parquet", submitter="Ana", description="")[0]
