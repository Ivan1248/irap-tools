import dataclasses as dc

import irap_evaluation as ie
import numpy as np
import pytest
from run_helpers import OTHER_SPLIT, add_run, make_run, with_split
from synthetic_vietnam import SPLIT

from irap_evaluation_server.ensembles import (
    EnsembleMember,
    create_ensemble,
    list_member_runs,
    plan_ensemble,
)

SPLITS = (SPLIT, OTHER_SPLIT)


def member(method_name, seed=None, weight=1.0):
    return EnsembleMember(ie.MethodInfo(method_name, seed), weight)


@pytest.fixture
def submissions(archive, two_split_contexts):
    """Runs a/seed1 and b on both splits, c only on `SPLIT`, and a deleted run d."""
    added = {}
    for name, seed, splits in [("a", 1, SPLITS), ("b", None, SPLITS), ("c", None, (SPLIT,)),
                               ("d", None, SPLITS)]:
        predictions = make_run(two_split_contexts, name, seed, random_seed=len(added))
        for split in splits:
            added[(name, split)] = add_run(archive, two_split_contexts,
                                           with_split(predictions, split))
    for split in SPLITS:
        archive.set_submission_deleted(added[("d", split)].id, True, actor="Bo")
    return added


def test_member_runs_and_plan(archive, submissions):
    runs = list_member_runs(archive.list_submissions(include_deleted=True), "vietnam", SPLITS)
    assert [(r.method.run_label, r.splits) for r in runs] == [
        ("a/seed1", SPLITS), ("b", SPLITS), ("c", (SPLIT,))]

    members = [member("a", 1, 2.0), member("b", None), member("c", None)]
    plan = plan_ensemble(archive.list_submissions(), "vietnam", members, SPLITS)
    assert plan.split_to_submissions == {SPLIT: tuple(submissions[(n, SPLIT)] for n in "abc")}
    assert plan.split_to_missing == {OTHER_SPLIT: ("c",)}
    for wrong_members, message in [
            (members[:1], "at least 2"), ([members[0], members[0]], "only once"),
            ([dc.replace(members[0], weight=0.0), members[1]], "positive"),
            ([members[0], member("d", None)], "No split")]:
        with pytest.raises(ValueError, match=message):
            plan_ensemble(archive.list_submissions(), "vietnam", wrong_members, SPLITS)


def test_create_ensemble(archive, two_split_contexts, submissions):
    members = [member("a", 1, 2.0), member("b", None)]
    plan = plan_ensemble(archive.list_submissions(), "vietnam", members, SPLITS)
    method = ie.MethodInfo("ens", 7)
    created = create_ensemble(archive, two_split_contexts["vietnam"], plan, method=method,
                              intersect_segments=False, submitter="Bo", description="test")
    assert [(s.split, s.run_label) for s in created] == [(SPLIT, "ens/seed7"),
                                                          (OTHER_SPLIT, "ens/seed7")]
    for submission in created:
        stored = ie.read_predictions(archive.get_predictions_path(submission.id))
        expected = ie.ensemble_predictions(
            [ie.read_predictions(archive.get_predictions_path(submissions[(n, submission.split)]
                                                              .id)) for n in "ab"],
            method, weights=[2.0, 1.0])
        assert stored.header == expected.header
        for attribute in expected.attributes:
            np.testing.assert_array_equal(stored.probs[attribute], expected.probs[attribute])
        [action] = archive.list_actions(submission_id=submission.id)
        assert action.action == "ensemble"
        assert action.details["members"] == [
            {"submission_id": submissions[("a", submission.split)].id, "run_label": "a/seed1",
             "weight": 2.0},
            {"submission_id": submissions[("b", submission.split)].id, "run_label": "b",
             "weight": 1.0}]
    assert not list(archive.uploads_dir.iterdir())

    # The run exists now, so another ensemble with its name and seed is refused.
    with pytest.raises(ValueError, match="seed"):
        create_ensemble(archive, two_split_contexts["vietnam"], plan, method=method,
                        intersect_segments=False, submitter="Bo", description="")


def test_ensemble_is_stored_entirely_or_not_at_all(archive, two_split_contexts, submissions,
                                                   monkeypatch):
    plan = plan_ensemble(archive.list_submissions(), "vietnam",
                         [member("a", 1), member("b", None)], SPLITS)
    ensemble_predictions = ie.ensemble_predictions

    def fail_on_second_split(predictions_seq, method, **kwargs):
        if predictions_seq[0].header.split == OTHER_SPLIT:
            raise ie.PredictionFormatError("Refused.")
        return ensemble_predictions(predictions_seq, method, **kwargs)

    monkeypatch.setattr(ie, "ensemble_predictions", fail_on_second_split)
    num_submissions = len(archive.list_submissions(include_deleted=True))
    with pytest.raises(ie.PredictionFormatError, match="Refused"):
        create_ensemble(archive, two_split_contexts["vietnam"], plan, method=ie.MethodInfo("ens"),
                        intersect_segments=False, submitter="Bo", description="")
    assert len(archive.list_submissions(include_deleted=True)) == num_submissions
    assert not list(archive.uploads_dir.iterdir())
    with pytest.raises(ValueError, match="Enter your name"):
        create_ensemble(archive, two_split_contexts["vietnam"], plan, method=ie.MethodInfo("ens"),
                        intersect_segments=False, submitter=" ", description="")
