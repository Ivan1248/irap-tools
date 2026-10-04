import dataclasses as dc
import hashlib
import sqlite3

import irap_evaluation as ie
import pytest
from run_helpers import (
    OTHER_SPLIT,
    add_model,
    make_model,
    with_model,
    with_split,
    write_upload,
)
from synthetic_vietnam import SPLIT

from irap_evaluation_server import archive as archive_module
from irap_evaluation_server.archive import ModelArchive, NewSubmission
from irap_evaluation_server.uploads import apply_upload, plan_upload


def plan(archive, dataset_contexts, predictions, **kwargs):
    return plan_upload(archive, dataset_contexts, write_upload(archive, predictions),
                       file_name="m.predictions.parquet", **kwargs)


def test_upload_creates_a_model(archive, dataset_contexts):
    predictions = make_model(dataset_contexts, model_seed=1, training_splits=())
    planned = plan(archive, dataset_contexts, predictions, description=" first model ")
    assert planned.update.model is None and planned.update.confirmations == []
    uploaded_sha256 = hashlib.sha256(planned.uploaded_path.read_bytes()).hexdigest()
    [submission] = apply_upload(archive, planned, submitter="Ana")

    assert not planned.uploaded_path.exists()
    stored_path = archive.get_predictions_path(submission.id)
    assert hashlib.sha256(stored_path.read_bytes()).hexdigest() == uploaded_sha256
    model = submission.model
    assert (model.dataset, model.method_name, model.seed, model.label, model.description) == (
        "vietnam", "m", 1, "m/seed1", "first model")
    assert (submission.split, submission.file_sha256, submission.deleted_at) == (
        SPLIT, uploaded_sha256, None)
    assert submission.is_in_use
    assert submission.context_offsets == (0, -1)
    assert (submission.training_splits, submission.early_stopping_splits) == ((), ())
    assert submission.model_info == ie.ModelInfo("m", (), (), 1)
    assert submission.output_kind == "probs"
    assert submission.num_segments == predictions.num_segments
    assert submission.num_attributes == len(predictions.attributes)
    assert submission.submitter == "Ana"
    assert archive.list_models() == [model]
    assert archive.list_submissions() == [submission]
    assert archive.get_submission(submission.id) == submission

    [action] = archive.list_actions()
    assert (action.actor, action.action, action.model_id, action.submission_id) == (
        "Ana", "upload", model.id, submission.id)
    assert action.details == {"file_name": "m.predictions.parquet", "notes": []}


def test_reupload_replaces_the_split(archive, dataset_contexts):
    first = add_model(archive, dataset_contexts, make_model(dataset_contexts, random_seed=1))
    planned = plan(archive, dataset_contexts, make_model(dataset_contexts, random_seed=2))
    assert planned.update.model == first.model
    assert planned.update.replaced == {SPLIT: first}
    [confirmation] = planned.update.confirmations
    assert f"Replaces the {SPLIT} file #{first.id} of m" in confirmation
    [second] = apply_upload(archive, planned, submitter="Bo")

    assert second.model == first.model
    replaced = archive.get_submission(first.id)
    assert replaced.is_deleted and archive.get_predictions_path(first.id).exists()
    assert archive.list_submissions() == [second]
    assert archive.list_model_submissions(first.model.id) == [second, replaced]
    assert archive.list_actions()[0].details["replaced_submission_id"] == first.id


def test_description_change_needs_confirmation(archive, dataset_contexts):
    predictions = make_model(dataset_contexts)
    first = add_model(archive, dataset_contexts, predictions, description="first")
    assert first.model.description == "first"
    planned = plan(archive, dataset_contexts, predictions, description="second")
    assert any("Changes the description of m from 'first' to 'second'" in c
               for c in planned.update.confirmations)
    second = apply_upload(archive, planned, submitter="Ana")[0]
    assert second.model.description == "second"
    assert archive.list_actions()[1].action == "edit_description"
    # A blank description keeps it.
    third = add_model(archive, dataset_contexts, predictions, description=None)
    assert third.model.description == "second"

    edited = archive.set_model_description(first.model.id, " third ", actor="Bo")
    assert edited.description == "third"
    with pytest.raises(ValueError, match="unchanged"):
        archive.set_model_description(first.model.id, "third", actor="Bo")


def test_seed_override_adds_another_model(archive, dataset_contexts):
    first = add_model(archive, dataset_contexts, make_model(dataset_contexts, model_seed=1))
    planned = plan(archive, dataset_contexts, make_model(dataset_contexts, model_seed=1), seed=2)
    uploaded_sha256 = hashlib.sha256(planned.uploaded_path.read_bytes()).hexdigest()
    assert planned.update.model is None
    [second] = apply_upload(archive, planned, submitter="Ana")

    assert second.model.label == "m/seed2" and second.model != first.model
    stored = ie.read_predictions(archive.get_predictions_path(second.id))
    assert stored.header.model.seed == 2
    assert second.file_sha256 != uploaded_sha256
    details = archive.list_actions()[0].details
    assert (details["original_seed"], details["uploaded_file_sha256"]) == (1, uploaded_sha256)
    assert not list(archive.uploads_dir.iterdir())


def test_models_of_a_method_need_seeds(archive, dataset_contexts):
    add_model(archive, dataset_contexts, make_model(dataset_contexts, model_seed=None))
    path = write_upload(archive, make_model(dataset_contexts, model_seed=1))
    with pytest.raises(ValueError, match=r"each needs a seed.*on vietnam: m, m/seed1\.$"):
        plan_upload(archive, dataset_contexts, path, file_name="m")
    assert list(archive.uploads_dir.iterdir()) == [path]  # Kept for a retry.
    add_model(archive, dataset_contexts, make_model(dataset_contexts, name="other"))


def test_models_of_a_method_need_the_same_training_splits(archive, dataset_contexts):
    add_model(archive, dataset_contexts,
              make_model(dataset_contexts, model_seed=0, training_splits=()))
    with pytest.raises(ValueError, match="different training or early stopping splits"):
        plan(archive, dataset_contexts,
             make_model(dataset_contexts, model_seed=1, training_splits=(SPLIT,)))
    with pytest.raises(ValueError, match="different training or early stopping splits"):
        plan(archive, dataset_contexts,
             make_model(dataset_contexts, model_seed=1, training_splits=None))
    add_model(archive, dataset_contexts,
              make_model(dataset_contexts, model_seed=1, training_splits=()))


def test_unknown_split_names_are_refused(archive, dataset_contexts):
    path = write_upload(archive, make_model(dataset_contexts, training_splits=("trian",)))
    with pytest.raises(ie.PredictionFormatError, match=r"unknown splits \['trian'\]"):
        plan_upload(archive, dataset_contexts, path, file_name="m")
    assert path.exists()


def test_files_of_a_model_must_agree(archive, two_split_contexts):
    first = add_model(archive, two_split_contexts, make_model(two_split_contexts))
    hard = with_split(make_model(two_split_contexts, is_hard=True), OTHER_SPLIT)
    with pytest.raises(ValueError, match=r"would differ in output kind.*replace all its files"):
        plan(archive, two_split_contexts, hard)
    model = with_split(make_model(two_split_contexts), OTHER_SPLIT)
    other_offsets = dc.replace(model, header=dc.replace(model.header, context_offsets=(0,)))
    with pytest.raises(ValueError, match="would differ in context offsets"):
        plan(archive, two_split_contexts, other_offsets)
    trained = with_split(make_model(two_split_contexts, training_splits=(SPLIT,)), OTHER_SPLIT)
    with pytest.raises(ValueError, match="would differ in training splits"):
        plan(archive, two_split_contexts, trained)

    # Replacing the only file may change them.
    second = add_model(archive, two_split_contexts, make_model(two_split_contexts, is_hard=True))
    assert (second.model, second.output_kind) == (first.model, "hard")


def test_a_change_after_the_planning_is_refused(archive, dataset_contexts):
    planned = plan(archive, dataset_contexts, make_model(dataset_contexts, random_seed=1))
    other = plan(archive, dataset_contexts, make_model(dataset_contexts, random_seed=2))
    apply_upload(archive, other, submitter="Bo")
    with pytest.raises(ValueError, match="model m has changed since the files were checked"):
        apply_upload(archive, planned, submitter="Ana")
    assert planned.uploaded_path.exists()  # Kept for a retry.

    replanned = plan_upload(archive, dataset_contexts, planned.uploaded_path, file_name="m")
    newer = plan(archive, dataset_contexts, make_model(dataset_contexts, random_seed=3))
    apply_upload(archive, newer, submitter="Bo")
    with pytest.raises(ValueError, match=f"{SPLIT} file of m has changed"):
        apply_upload(archive, replanned, submitter="Ana")


def test_delete_and_restore_a_model(archive, dataset_contexts):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts))
    deleted = archive.set_model_deleted(submission.model.id, True, actor="Bo")
    assert deleted.is_deleted
    assert archive.list_models() == [] and archive.list_submissions() == []
    assert archive.list_models(include_deleted=True) == [deleted]
    # The file stays active, but it is not in use.
    stored = archive.get_submission(submission.id)
    assert not stored.is_deleted and not stored.is_in_use
    with pytest.raises(ValueError, match="already deleted"):
        archive.set_model_deleted(submission.model.id, True, actor="Bo")

    # A deleted model does not constrain the seeds of its method, until it is restored.
    add_model(archive, dataset_contexts, make_model(dataset_contexts, model_seed=1))
    with pytest.raises(ValueError, match="each needs a seed"):
        archive.set_model_deleted(submission.model.id, False, actor="Bo")

    # An upload of a deleted model restores it.
    other = add_model(archive, dataset_contexts, make_model(dataset_contexts, name="other"))
    archive.set_model_deleted(other.model.id, True, actor="Bo")
    planned = plan(archive, dataset_contexts, make_model(dataset_contexts, name="other"))
    assert "Restores the deleted model other." in planned.update.confirmations
    restored = apply_upload(archive, planned, submitter="Ana")[0]
    assert not restored.model.is_deleted and restored.is_in_use
    assert [a.action for a in archive.list_actions(model_id=other.model.id)] == [
        "upload", "restore_model", "delete_model", "upload"]


def test_delete_a_submission(archive, two_split_contexts):
    predictions = make_model(two_split_contexts)
    submission = add_model(archive, two_split_contexts, predictions)
    other = add_model(archive, two_split_contexts, with_split(predictions, OTHER_SPLIT))
    deleted = archive.delete_submission(submission.id, actor="Bo")
    assert deleted.is_deleted and archive.list_submissions() == [other]
    assert archive.list_model_submissions(deleted.model.id) == [other, deleted]
    with pytest.raises(ValueError, match="already deleted or replaced"):
        archive.delete_submission(submission.id, actor="Bo")
    assert [a.action for a in archive.list_actions()] == ["delete", "upload", "upload"]

    # A replaced file is not deleted, e.g. after another user replaced the file that a user
    # wanted to delete.
    replacing = add_model(archive, two_split_contexts,
                          with_split(make_model(two_split_contexts, random_seed=1), OTHER_SPLIT))
    with pytest.raises(ValueError, match="already deleted or replaced"):
        archive.delete_submission(other.id, actor="Bo")
    assert archive.get_submission(replacing.id).is_in_use


def test_deleting_the_last_file_deletes_the_model(archive, dataset_contexts):
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts))
    deleted = archive.delete_submission(submission.id, actor="Bo")
    assert deleted.model.is_deleted and archive.list_models() == []
    assert [(a.action, a.details) for a in archive.list_actions()[:2]] == [
        ("delete_model", {"by": "delete"}), ("delete", {})]
    with pytest.raises(ValueError, match="has no active files. Upload a file"):
        archive.set_model_deleted(submission.model.id, False, actor="Bo")

    # An upload restores it.
    uploaded = add_model(archive, dataset_contexts, make_model(dataset_contexts))
    assert uploaded.model.id == submission.model.id and uploaded.is_in_use


def test_list_submissions_of_deleted_models(archive, dataset_contexts):
    add_model(archive, dataset_contexts, make_model(dataset_contexts, random_seed=1))
    second = add_model(archive, dataset_contexts, make_model(dataset_contexts, random_seed=2))
    archive.set_model_deleted(second.model.id, True, actor="Bo")
    second = archive.get_submission(second.id)
    assert archive.list_submissions() == []
    # Without the replaced first submission.
    assert archive.list_submissions(include_deleted_models=True) == [second]


def test_actions_need_an_actor(archive, dataset_contexts):
    planned = plan(archive, dataset_contexts, make_model(dataset_contexts))
    with pytest.raises(ValueError, match="actor of an action"):
        apply_upload(archive, planned, submitter=" ")
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts))
    for change in (lambda: archive.delete_submission(submission.id, actor=""),
                   lambda: archive.set_model_deleted(submission.model.id, True, actor=""),
                   lambda: archive.set_model_description(submission.model.id, "x", actor="")):
        with pytest.raises(ValueError, match="actor of an action"):
            change()
    with pytest.raises(LookupError):
        archive.get_submission(submission.id + 1)
    with pytest.raises(LookupError):
        archive.get_model(submission.model.id + 1)


def test_failed_storing_moves_back_all_files(archive, two_split_contexts, monkeypatch):
    new_submissions = []
    for split in (SPLIT, OTHER_SPLIT):
        predictions = with_split(make_model(two_split_contexts), split)
        path = write_upload(archive, predictions)
        new_submissions.append(NewSubmission.from_predictions(predictions, path, details={},
                                                              notes=()))
    update = archive.plan_model_update(new_submissions, action="upload", description=None)

    def fail(*args):
        raise RuntimeError("after the moves")

    # Called after the files are moved, in the transaction.
    monkeypatch.setattr(archive_module, "_get_submission", fail)
    with pytest.raises(RuntimeError, match="after the moves"):
        archive.apply_model_update(update, submitter="Ana")
    assert all(new.file_path.exists() for new in new_submissions)
    assert not archive.get_predictions_path(1).exists()
    assert not archive.get_predictions_path(2).exists()
    assert archive.list_models(include_deleted=True) == []
    assert archive.list_actions() == []


def test_updates_are_of_one_model(archive, two_split_contexts):
    def make_new(predictions):
        return NewSubmission.from_predictions(predictions, write_upload(archive, predictions),
                                              details={}, notes=())

    predictions = make_model(two_split_contexts)
    with pytest.raises(ValueError, match="several models"):
        archive.plan_model_update(
            [make_new(predictions), make_new(with_model(with_split(predictions, OTHER_SPLIT),
                                                        seed=3))],
            action="upload", description=None)
    with pytest.raises(ValueError, match="a split twice"):
        archive.plan_model_update([make_new(predictions)] * 2, action="upload",
                                  description=None)


@pytest.mark.parametrize("model_seed, seed", [(2**63, None), (None, 2**63)])
def test_seed_must_fit_64_bits(archive, dataset_contexts, model_seed, seed):
    path = write_upload(archive, make_model(dataset_contexts, model_seed=model_seed))
    with pytest.raises(ValueError, match="64-bit"):
        plan_upload(archive, dataset_contexts, path, file_name="m", seed=seed)
    assert path.exists()


def test_a_database_of_another_version_is_refused(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with sqlite3.connect(data_dir / "archive.sqlite3") as connection:
        connection.execute("CREATE TABLE submissions (id INTEGER PRIMARY KEY)")
    with pytest.raises(ValueError, match="another version of the archive.*new data directory"):
        ModelArchive(data_dir)
    ModelArchive(tmp_path / "new")
    ModelArchive(tmp_path / "new")  # Opening it again.
