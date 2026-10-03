"""Synthetic runs and their uploads, for the tests."""

import dataclasses as dc
from pathlib import Path

import irap_evaluation as ie
from synthetic_vietnam import SPLIT, make_predictions

from irap_evaluation_server.archive import SubmissionArchive, add_uploaded_submission


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
