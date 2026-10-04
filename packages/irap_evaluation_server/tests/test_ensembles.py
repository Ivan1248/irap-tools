import dataclasses as dc

import irap_evaluation as ie
import numpy as np
import pytest
from run_helpers import OTHER_SPLIT, add_model, make_model, with_split
from synthetic_vietnam import SPLIT

from irap_evaluation_server.archive import discard_model_update
from irap_evaluation_server.ensembles import (
    EnsembleMember,
    list_member_models,
    plan_ensemble,
    prepare_ensemble,
)

SPLITS = (SPLIT, OTHER_SPLIT)


@pytest.fixture
def submissions(archive, two_split_contexts):
    """Models a/seed1 and b on both splits, c only on `SPLIT`, and a deleted model d."""
    added = {}
    for name, seed, splits in [("a", 1, SPLITS), ("b", None, SPLITS), ("c", None, (SPLIT,)),
                               ("d", None, SPLITS)]:
        predictions = make_model(two_split_contexts, name, seed, random_seed=len(added))
        for split in splits:
            added[(name, split)] = add_model(archive, two_split_contexts,
                                             with_split(predictions, split))
    archive.set_model_deleted(added[("d", SPLIT)].model.id, True, actor="Bo")
    return added


def member(submissions, name, weight=1.0):
    return EnsembleMember(submissions[(name, SPLIT)].model, weight)


def test_member_models_and_plan(archive, submissions):
    members = list_member_models(archive.list_submissions(include_deleted_models=True),
                                 "vietnam", SPLITS)
    assert [(m.model.label, m.splits) for m in members] == [
        ("a/seed1", SPLITS), ("b", SPLITS), ("c", (SPLIT,))]

    members = [member(submissions, "a", 2.0), member(submissions, "b"),
               member(submissions, "c")]
    plan = plan_ensemble(archive.list_submissions(), "vietnam", members, SPLITS)
    assert plan.split_to_submissions == {SPLIT: tuple(submissions[(n, SPLIT)] for n in "abc")}
    assert plan.split_to_missing == {OTHER_SPLIT: ("c",)}
    for wrong_members, message in [
            (members[:1], "at least 2"), ([members[0], members[0]], "only once"),
            ([dc.replace(members[0], weight=0.0), members[1]], "positive"),
            ([members[0], member(submissions, "d")], "No split")]:
        with pytest.raises(ValueError, match=message):
            plan_ensemble(archive.list_submissions(), "vietnam", wrong_members, SPLITS)


def test_create_ensemble(archive, two_split_contexts, submissions):
    members = [member(submissions, "a", 2.0), member(submissions, "b")]
    plan = plan_ensemble(archive.list_submissions(), "vietnam", members, SPLITS)
    update = prepare_ensemble(archive, two_split_contexts["vietnam"], plan, method_name="ens",
                              seed=7, intersect_segments=False, description="test")
    assert update.confirmations == []
    created = archive.apply_model_update(update, submitter="Bo")
    assert [(s.split, s.label, s.model.description) for s in created] == [
        (SPLIT, "ens/seed7", "test"), (OTHER_SPLIT, "ens/seed7", "test")]
    for submission in created:
        stored = ie.read_predictions(archive.get_predictions_path(submission.id))
        expected = ie.ensemble_predictions(
            [ie.read_predictions(archive.get_predictions_path(submissions[(n, submission.split)]
                                                              .id)) for n in "ab"],
            "ens", seed=7, weights=[2.0, 1.0])
        assert stored.header == expected.header
        assert stored.header.model.training_splits == ()
        for attribute in expected.attributes:
            np.testing.assert_array_equal(stored.probs[attribute], expected.probs[attribute])
        [action] = [a for a in archive.list_actions() if a.submission_id == submission.id]
        assert action.action == "ensemble"
        assert action.details["members"] == [
            {"submission_id": submissions[("a", submission.split)].id,
             "model_id": submissions[("a", SPLIT)].model.id, "label": "a/seed1", "weight": 2.0},
            {"submission_id": submissions[("b", submission.split)].id,
             "model_id": submissions[("b", SPLIT)].model.id, "label": "b", "weight": 1.0}]
    assert not list(archive.uploads_dir.iterdir())

    # The model exists now, so another ensemble with its name and seed replaces its files.
    update = prepare_ensemble(archive, two_split_contexts["vietnam"], plan, method_name="ens",
                              seed=7, intersect_segments=False, description=None)
    assert len(update.confirmations) == 2
    discard_model_update(update)
    assert not list(archive.uploads_dir.iterdir())


def _prepare(archive, two_split_contexts, submissions, names):
    plan = plan_ensemble(archive.list_submissions(), "vietnam",
                         [member(submissions, n) for n in names], SPLITS)
    return prepare_ensemble(archive, two_split_contexts["vietnam"], plan, method_name="ens",
                            seed=None, intersect_segments=False, description=None)


def test_an_ensemble_replaces_all_files_of_its_model(archive, two_split_contexts, submissions):
    [first, second] = archive.apply_model_update(
        _prepare(archive, two_split_contexts, submissions, "ab"), submitter="Bo")
    # c has no file of OTHER_SPLIT, so the ensemble of a and c has only SPLIT.
    update = _prepare(archive, two_split_contexts, submissions, "ac")
    assert (list(update.replaced), list(update.deleted)) == ([SPLIT], [OTHER_SPLIT])
    assert any(f"Deletes the {OTHER_SPLIT} file #{second.id}" in c for c in update.confirmations)
    [third] = archive.apply_model_update(update, submitter="Bo")
    assert archive.list_submissions() == [third, *archive.list_submissions()[1:]]
    assert [s.split for s in archive.list_model_submissions(third.model.id) if s.is_in_use] == [
        SPLIT]
    assert archive.get_submission(second.id).is_deleted
    assert archive.list_actions(model_id=third.model.id)[0].details == {"by": "ensemble"}


def test_an_ensemble_is_refused_if_a_member_changes(archive, two_split_contexts, submissions):
    update = _prepare(archive, two_split_contexts, submissions, "ab")
    archive.delete_submission(submissions[("b", OTHER_SPLIT)].id, actor="Al")
    with pytest.raises(ValueError, match=f"{OTHER_SPLIT} file #.* of b, which the new files are"
                                         f" made of, or its model has been deleted"):
        archive.apply_model_update(update, submitter="Bo")
    discard_model_update(update)


def test_an_ensemble_is_refused_if_its_model_gets_another_split(archive, two_split_contexts,
                                                                 submissions):
    archive.apply_model_update(_prepare(archive, two_split_contexts, submissions, "ac"),
                               submitter="Bo")
    update = _prepare(archive, two_split_contexts, submissions, "ac")
    # Another user adds a file of the other split meanwhile.
    other = _prepare(archive, two_split_contexts, submissions, "ab")
    archive.apply_model_update(archive.plan_model_update(
        [n for n in other.new_submissions if n.split == OTHER_SPLIT], action="upload",
        description=None), submitter="Al")
    discard_model_update(other)
    with pytest.raises(ValueError, match="files of ens have changed"):
        archive.apply_model_update(update, submitter="Bo")
    discard_model_update(update)


def test_ensemble_files_are_removed_if_it_fails(archive, two_split_contexts, submissions,
                                                monkeypatch):
    plan = plan_ensemble(archive.list_submissions(), "vietnam",
                         [member(submissions, "a"), member(submissions, "b")], SPLITS)
    ensemble_predictions = ie.ensemble_predictions

    def fail_on_second_split(predictions_seq, method_name, **kwargs):
        if predictions_seq[0].header.split == OTHER_SPLIT:
            raise ie.PredictionFormatError("Refused.")
        return ensemble_predictions(predictions_seq, method_name, **kwargs)

    monkeypatch.setattr(ie, "ensemble_predictions", fail_on_second_split)
    with pytest.raises(ie.PredictionFormatError, match="Refused"):
        prepare_ensemble(archive, two_split_contexts["vietnam"], plan, method_name="ens",
                         seed=None, intersect_segments=False, description=None)
    assert not list(archive.uploads_dir.iterdir())
