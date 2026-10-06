import dataclasses as dc
import hashlib
import io
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
    [submission] = apply_upload(archive, planned, submitter="Al")

    assert not planned.uploaded_path.exists()
    stored_path = archive.get_predictions_path(submission.id)
    assert hashlib.sha256(stored_path.read_bytes()).hexdigest() == uploaded_sha256
    model = submission.model
    assert (model.dataset, model.method_name, model.seed, model.label, model.description) == (
        "vietnam", "m", 1, "m/seed1", "first model")
    assert (submission.split, submission.file_sha256, submission.replaced_at) == (
        SPLIT, uploaded_sha256, None)
    assert submission.is_active
    assert submission.context_offsets == (0, -1)
    assert (submission.training_splits, submission.early_stopping_splits) == ((), ())
    assert submission.model_info == ie.ModelInfo("m", (), (), 1)
    assert submission.output_kind == "probs"
    assert submission.num_segments == predictions.num_segments
    assert submission.num_attributes == len(predictions.attributes)
    assert submission.submitter == "Al"
    assert archive.list_models() == [model]
    assert archive.list_submissions() == [submission]
    assert archive.get_submission(submission.id) == submission

    [action] = archive.list_actions()
    assert (action.actor, action.action, action.model_id, action.model_label,
            action.submission_id) == ("Al", "upload", model.id, "m/seed1", submission.id)
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
    assert not replaced.is_active and archive.get_predictions_path(first.id).exists()
    assert archive.list_submissions() == [second]
    assert archive.list_model_submissions(first.model.id) == [second, replaced]
    assert archive.list_actions()[0].details["replaced_submission_id"] == first.id


def test_description_change_needs_confirmation(archive, dataset_contexts):
    predictions = make_model(dataset_contexts)
    first = add_model(archive, dataset_contexts, predictions, description="first")
    assert first.model.description == "first"
    planned = plan(archive, dataset_contexts, predictions, description="second")
    assert any("Changes the description of m/seed0 from 'first' to 'second'" in c
               for c in planned.update.confirmations)
    second = apply_upload(archive, planned, submitter="Al")[0]
    assert second.model.description == "second"
    assert archive.list_actions()[1].action == "edit_description"
    # A blank description keeps it.
    third = add_model(archive, dataset_contexts, predictions, description=None)
    assert third.model.description == "second"

    edited = archive.set_model_description(first.model.id, " third ", actor="Bo")
    assert edited.description == "third"
    with pytest.raises(ValueError, match="unchanged"):
        archive.set_model_description(first.model.id, "third", actor="Bo")


def test_files_set_the_method_display_name(archive, dataset_contexts):
    first = add_model(archive, dataset_contexts, with_model(
        make_model(dataset_contexts, model_seed=1), method_display_name=" Model M "))
    assert (first.model.method_display_name, first.label, first.shown_label) == (
        "Model M", "m/seed1", "Model M/seed1")
    [action] = [a for a in archive.list_actions() if a.action == "edit_method_display_name"]
    assert action.details == {"old_display_name": None, "display_name": "Model M", "by": "upload"}
    # A file without one keeps it, and the models of the method share it.
    second = add_model(archive, dataset_contexts, make_model(dataset_contexts, model_seed=2))
    assert second.model.method_display_name == "Model M"
    other = add_model(archive, dataset_contexts, make_model(dataset_contexts, name="other"))
    assert other.model.method_display_name is None

    planned = plan(archive, dataset_contexts, with_model(
        make_model(dataset_contexts, model_seed=3), method_display_name="M"))
    assert any("Changes the display name of the method 'm' on vietnam from 'Model M' to 'M'" in c
               for c in planned.update.confirmations)
    apply_upload(archive, planned, submitter="Al")
    assert {m.label: m.method_display_name for m in archive.list_models()} == {
        "m/seed1": "M", "m/seed2": "M", "m/seed3": "M", "other/seed0": None}


def test_an_entered_method_display_name_replaces_the_files(archive, dataset_contexts):
    predictions = with_model(make_model(dataset_contexts), method_display_name="From file")
    [first] = apply_upload(archive, plan(archive, dataset_contexts, predictions,
                                         method_display_name=" Entered "), submitter="Al")
    assert first.model.method_display_name == "Entered"
    stored = ie.read_predictions(archive.get_predictions_path(first.id))
    assert stored.header.model.method_display_name == "From file"
    with pytest.raises(ValueError, match="must not be blank"):
        plan(archive, dataset_contexts, predictions, method_display_name=" ")


def test_inherited_method_display_name(archive, dataset_contexts):
    predictions = make_model(dataset_contexts)
    named = with_model(predictions, method_display_name="From file").header
    assert archive.get_inherited_method_display_name([predictions.header]) is None
    first = add_model(archive, dataset_contexts, predictions)
    archive.set_method_display_name(first.model.id, "Stored", actor="Bo")
    assert archive.get_inherited_method_display_name([predictions.header]) == "Stored"
    assert archive.get_inherited_method_display_name([predictions.header, named]) == "From file"
    with pytest.raises(ValueError, match="different method display names"):
        archive.get_inherited_method_display_name(
            [named, with_model(predictions, method_display_name="Other").header])


def test_set_method_display_name(archive, dataset_contexts):
    first = add_model(archive, dataset_contexts, make_model(dataset_contexts, model_seed=1))
    second = add_model(archive, dataset_contexts, make_model(dataset_contexts, model_seed=2))
    edited = archive.set_method_display_name(first.model.id, " Model M ", actor="Bo")
    assert (edited.method_display_name, edited.shown_label) == ("Model M", "Model M/seed1")
    assert archive.get_submission(second.id).shown_label == "Model M/seed2"
    with pytest.raises(ValueError, match="unchanged"):
        archive.set_method_display_name(second.model.id, "Model M", actor="Bo")
    cleared = archive.set_method_display_name(second.model.id, " ", actor="Bo")
    assert (cleared.method_display_name, cleared.shown_label) == (None, "m/seed2")
    assert [(a.model_id, a.details) for a in archive.list_actions()
            if a.action == "edit_method_display_name"] == [
        (second.model.id, {"old_display_name": "Model M", "display_name": None}),
        (first.model.id, {"old_display_name": None, "display_name": "Model M"})]


def test_a_display_name_change_after_the_planning_is_refused(archive, dataset_contexts):
    first = add_model(archive, dataset_contexts, make_model(dataset_contexts, model_seed=1))
    # A new model of the method.
    planned = plan(archive, dataset_contexts, make_model(dataset_contexts, model_seed=2))
    archive.set_method_display_name(first.model.id, "M", actor="Bo")
    with pytest.raises(ValueError, match="display name of the method 'm' has changed"):
        apply_upload(archive, planned, submitter="Al")


def test_seed_override_adds_another_model(archive, dataset_contexts):
    first = add_model(archive, dataset_contexts, make_model(dataset_contexts, model_seed=1))
    planned = plan(archive, dataset_contexts, make_model(dataset_contexts, model_seed=1), seed=2)
    uploaded_sha256 = hashlib.sha256(planned.uploaded_path.read_bytes()).hexdigest()
    assert planned.update.model is None
    [second] = apply_upload(archive, planned, submitter="Al")

    assert second.model.label == "m/seed2" and second.model != first.model
    stored = ie.read_predictions(archive.get_predictions_path(second.id))
    assert stored.header.model.seed == 2
    assert second.file_sha256 != uploaded_sha256
    details = archive.list_actions()[0].details
    assert (details["original_seed"], details["uploaded_file_sha256"]) == (1, uploaded_sha256)
    assert not list(archive.uploads_dir.iterdir())


def test_a_model_needs_a_seed(archive, dataset_contexts):
    path = write_upload(archive, make_model(dataset_contexts, model_seed=None))
    with pytest.raises(ValueError, match="The model has no seed. Enter one"):
        plan_upload(archive, dataset_contexts, path, file_name="m")
    assert list(archive.uploads_dir.iterdir()) == [path]  # Kept for a retry.
    planned = plan_upload(archive, dataset_contexts, path, file_name="m", seed=0)
    [submission] = apply_upload(archive, planned, submitter="Al")
    assert submission.model.seed == 0


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
    with pytest.raises(ValueError, match="model m/seed0 has changed since the files were checked"):
        apply_upload(archive, planned, submitter="Al")
    assert planned.uploaded_path.exists()  # Kept for a retry.

    replanned = plan_upload(archive, dataset_contexts, planned.uploaded_path, file_name="m")
    newer = plan(archive, dataset_contexts, make_model(dataset_contexts, random_seed=3))
    apply_upload(archive, newer, submitter="Bo")
    with pytest.raises(ValueError, match=f"{SPLIT} file of m/seed0 has changed"):
        apply_upload(archive, replanned, submitter="Al")


def test_delete_a_submission(archive, two_split_contexts):
    predictions = make_model(two_split_contexts)
    replaced = add_model(archive, two_split_contexts, predictions)
    active = add_model(archive, two_split_contexts, make_model(two_split_contexts, random_seed=1))
    other = add_model(archive, two_split_contexts, with_split(predictions, OTHER_SPLIT))

    # A replaced file is deleted with its file.
    assert archive.delete_submission(replaced.id, actor="Bo").id == replaced.id
    assert not archive.get_predictions_path(replaced.id).parent.exists()
    with pytest.raises(LookupError):
        archive.get_submission(replaced.id)
    with pytest.raises(LookupError):
        archive.delete_submission(replaced.id, actor="Bo")

    archive.delete_submission(active.id, actor="Bo")
    assert archive.list_submissions() == [other]
    with pytest.raises(ValueError, match="last active file of m/seed0. Delete the model instead"):
        archive.delete_submission(other.id, actor="Bo")
    assert [(a.action, a.model_label, a.submission_id, a.details)
            for a in archive.list_actions()[:2]] == [
        ("delete", "m/seed0", active.id, {"split": SPLIT}),
        ("delete", "m/seed0", replaced.id, {"split": SPLIT})]


def test_delete_a_model(archive, dataset_contexts):
    first = add_model(archive, dataset_contexts, with_model(
        make_model(dataset_contexts, model_seed=1), method_display_name="M"))
    replacing = add_model(archive, dataset_contexts,
                          make_model(dataset_contexts, model_seed=1, random_seed=1))
    other = add_model(archive, dataset_contexts, make_model(dataset_contexts, model_seed=2))
    model = replacing.model
    deleted = archive.delete_model(model.id, actor="Bo")
    assert [s.id for s in deleted] == [replacing.id, first.id]
    assert archive.list_models() == [other.model]
    assert [p.name for p in archive.submissions_dir.iterdir()] == [str(other.id)]
    assert [(a.action, a.model_label, a.submission_id, a.details)
            for a in archive.list_actions(model_id=model.id)[:3]] == [
        ("delete_model", "m/seed1", None, {}),
        ("delete", "m/seed1", first.id, {"split": SPLIT, "by": "delete_model"}),
        ("delete", "m/seed1", replacing.id, {"split": SPLIT, "by": "delete_model"})]
    with pytest.raises(LookupError):
        archive.delete_model(model.id, actor="Bo")

    # The method keeps its display name until its last model is deleted.
    assert archive.get_model(other.model.id).method_display_name == "M"
    archive.delete_model(other.model.id, actor="Bo")
    # An upload creates the model again, with a new id.
    again = add_model(archive, dataset_contexts, make_model(dataset_contexts, model_seed=1))
    assert again.model.method_display_name is None
    assert again.model.id not in {model.id, other.model.id}


def test_download_and_upload_the_files_of_a_model(archive, two_split_contexts):
    predictions = make_model(two_split_contexts)
    submissions = [add_model(archive, two_split_contexts, predictions),
                   add_model(archive, two_split_contexts, with_split(predictions, OTHER_SPLIT))]
    model = submissions[0].model
    path = archive.make_upload_path("m.zip")
    with path.open("wb") as file:
        archive.write_model_predictions_zip(model.id, file)
    archive.delete_model(model.id, actor="Bo")
    with pytest.raises(LookupError):
        archive.write_model_predictions_zip(model.id, io.BytesIO())

    planned = plan_upload(archive, two_split_contexts, path, file_name="m.zip")
    uploaded = apply_upload(archive, planned, submitter="Al")
    assert ({(s.split, s.file_sha256) for s in uploaded}
            == {(s.split, s.file_sha256) for s in submissions})


def test_actions_need_an_actor(archive, dataset_contexts):
    planned = plan(archive, dataset_contexts, make_model(dataset_contexts))
    with pytest.raises(ValueError, match="actor of an action"):
        apply_upload(archive, planned, submitter=" ")
    submission = add_model(archive, dataset_contexts, make_model(dataset_contexts))
    for change in (lambda: archive.delete_submission(submission.id, actor=""),
                   lambda: archive.delete_model(submission.model.id, actor=""),
                   lambda: archive.set_model_description(submission.model.id, "x", actor=""),
                   lambda: archive.set_method_display_name(submission.model.id, "x", actor="")):
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
        archive.apply_model_update(update, submitter="Al")
    assert all(new.file_path.exists() for new in new_submissions)
    assert not archive.get_predictions_path(1).exists()
    assert not archive.get_predictions_path(2).exists()
    assert archive.list_models() == []
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
    with pytest.raises(ValueError, match="different method display names"):
        archive.plan_model_update(
            [make_new(with_model(predictions, method_display_name="A")),
             make_new(with_model(with_split(predictions, OTHER_SPLIT), method_display_name="B"))],
            action="upload", description=None)


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
