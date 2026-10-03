import concurrent.futures
import dataclasses as dc
import hashlib
import json
import os
import time
import zipfile

import irap_evaluation as ie
import pytest
from irap_data.metadata import MetaFiles
from run_helpers import make_run, write_upload
from synthetic_vietnam import SPLIT, make_predictions

from irap_evaluation_server import archive as archive_module
from irap_evaluation_server.archive import (
    NewSubmission,
    add_uploaded_archive,
    add_uploaded_submission,
)
from irap_evaluation_server.config import DatasetConfig
from irap_evaluation_server.datasets import DatasetContext, load_dataset_contexts


def upload(archive, dataset_contexts, predictions, submitter="Ana", seed=None):
    path = write_upload(archive, predictions)
    return path, lambda: add_uploaded_submission(
        archive, dataset_contexts, path, file_name="m.predictions.parquet", submitter=submitter,
        description=" first run ", seed=seed)


def add(archive, dataset_contexts, predictions, **kwargs):
    return upload(archive, dataset_contexts, predictions, **kwargs)[1]()[0]


OTHER_SPLIT = "test"


@pytest.fixture
def two_split_contexts(metadata_dir):
    """The dataset with the split `OTHER_SPLIT`, which has the segments of `SPLIT`."""
    splits_path = metadata_dir / MetaFiles.SPLITS
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    splits[OTHER_SPLIT] = splits[SPLIT]
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    return load_dataset_contexts({"vietnam": DatasetConfig(
        metadata_dir=metadata_dir, images_dir=None, analysis_splits=())})


def with_split(predictions, split):
    return dc.replace(predictions, header=dc.replace(predictions.header, split=split))


def make_run_files(dataset_contexts, splits=(SPLIT, OTHER_SPLIT), **kwargs):
    """Archive member name -> the predictions of the run on each split."""
    predictions = make_run(dataset_contexts, **kwargs)
    return {f"run/m.{split}.predictions.parquet": with_split(predictions, split)
            for split in splits}


def write_archive(path, member_name_to_contents, tmp_path):
    """Writes a .zip archive of predictions, or of texts for other files."""
    with zipfile.ZipFile(path, "w") as zip_file:
        zip_file.writestr("run/", "")  # A directory entry, which is skipped.
        for member_name, contents in member_name_to_contents.items():
            if isinstance(contents, str):
                zip_file.writestr(member_name, contents)
                continue
            member_path = tmp_path / "member.parquet"
            ie.write_predictions(member_path, contents)
            zip_file.write(member_path, member_name)
    return path


def upload_archive(archive, dataset_contexts, member_name_to_contents, tmp_path, seed=None):
    path = write_archive(archive.make_upload_path("m.zip"), member_name_to_contents, tmp_path)
    return path, lambda: add_uploaded_archive(
        archive, dataset_contexts, path, file_name="m.zip", submitter="Ana",
        description="all splits", seed=seed)


def read_member_sha256s(path):
    with zipfile.ZipFile(path) as zip_file:
        return {i.filename: hashlib.sha256(zip_file.read(i)).hexdigest()
                for i in zip_file.infolist() if not i.is_dir()}


def test_add_uploaded_submission(archive, dataset_contexts):
    predictions = make_run(dataset_contexts, method_seed=1)
    path, add_upload = upload(archive, dataset_contexts, predictions)
    uploaded_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    submission, notes = add_upload()

    assert not path.exists()
    stored_path = archive.get_predictions_path(submission.id)
    assert hashlib.sha256(stored_path.read_bytes()).hexdigest() == uploaded_sha256
    assert submission.file_sha256 == uploaded_sha256
    assert (submission.dataset, submission.split, submission.run_label) == (
        "vietnam", SPLIT, "m/seed1")
    assert submission.context_offsets == (0, -1)
    assert submission.output_kind == "probs"
    assert submission.num_segments == predictions.num_segments
    assert submission.num_attributes == len(predictions.attributes)
    assert (submission.submitter, submission.description) == ("Ana", "first run")
    assert not submission.is_deleted
    assert archive.list_submissions() == [submission]
    assert archive.get_submission(submission.id) == submission

    [action] = archive.list_actions()
    assert (action.actor, action.action, action.submission_id) == ("Ana", "upload", submission.id)
    assert action.details == {"file_name": "m.predictions.parquet", "notes": notes}


def test_duplicate_run_is_refused_and_can_get_another_seed(archive, dataset_contexts):
    first = add(archive, dataset_contexts, make_run(dataset_contexts, method_seed=1))
    path, add_upload = upload(archive, dataset_contexts, make_run(dataset_contexts, method_seed=1))
    with pytest.raises(ValueError, match=rf"same seed.*#{first.id} \(m/seed1\)"):
        add_upload()
    assert path.exists()  # Kept for a retry.

    uploaded_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    second, _ = add_uploaded_submission(archive, dataset_contexts, path, file_name="m2",
                                        submitter="Ana", description="", seed=2)
    assert not path.exists()
    assert second.run_label == "m/seed2"
    stored = ie.read_predictions(archive.get_predictions_path(second.id))
    assert stored.header.method.seed == 2
    assert second.file_sha256 != uploaded_sha256
    assert archive.list_actions()[0].details["original_seed"] == 1
    assert archive.list_actions()[0].details["uploaded_file_sha256"] == uploaded_sha256
    assert not list(archive.uploads_dir.iterdir())


def test_runs_of_a_method_need_seeds(archive, dataset_contexts):
    add(archive, dataset_contexts, make_run(dataset_contexts, method_seed=None))
    with pytest.raises(ValueError, match="each needs a seed.*Delete it first"):
        add(archive, dataset_contexts, make_run(dataset_contexts, method_seed=1))
    add(archive, dataset_contexts, make_run(dataset_contexts, name="other", method_seed=None))


def test_delete_and_restore(archive, dataset_contexts):
    first = add(archive, dataset_contexts, make_run(dataset_contexts, method_seed=1))
    deleted = archive.set_submission_deleted(first.id, True, actor="Bo")
    assert deleted.is_deleted
    assert archive.list_submissions() == []
    assert archive.list_submissions(include_deleted=True) == [deleted]
    with pytest.raises(ValueError, match="already deleted"):
        archive.set_submission_deleted(first.id, True, actor="Bo")

    # A deleted run does not block the same run, which then blocks restoring the deleted one.
    second = add(archive, dataset_contexts, make_run(dataset_contexts, method_seed=1))
    with pytest.raises(ValueError, match="same seed"):
        archive.set_submission_deleted(first.id, False, actor="Bo")
    archive.set_submission_deleted(second.id, True, actor="Bo")
    restored = archive.set_submission_deleted(first.id, False, actor="Bo")
    assert not restored.is_deleted
    assert [a.action for a in archive.list_actions(submission_id=first.id)] == [
        "restore", "delete", "upload"]


def test_actions_need_an_actor(archive, dataset_contexts):
    _, add_upload = upload(archive, dataset_contexts, make_run(dataset_contexts), submitter=" ")
    with pytest.raises(ValueError, match="Enter your name"):
        add_upload()
    submission = add(archive, dataset_contexts, make_run(dataset_contexts))
    with pytest.raises(ValueError, match="Enter your name"):
        archive.set_submission_deleted(submission.id, True, actor="")
    with pytest.raises(LookupError):
        archive.get_submission(submission.id + 1)


def test_uploads_are_checked(archive, dataset_contexts):
    predictions = make_run(dataset_contexts)
    other_dataset = dc.replace(predictions, header=dc.replace(predictions.header, dataset="bh"))
    with pytest.raises(ValueError, match="dataset 'bh'"):
        add(archive, dataset_contexts, other_dataset)

    incomplete = ie.select_segments(predictions, predictions.segment_ids[:-3])
    with pytest.raises(ValueError, match="No predictions for 3"):
        add(archive, dataset_contexts, incomplete)

    wrong_codes = dc.replace(predictions, attribute_to_irap_codes={
        **predictions.attribute_to_irap_codes, "Curvature": (1, 3)})
    with pytest.raises(ie.PredictionFormatError, match="Curvature"):
        add(archive, dataset_contexts, wrong_codes)
    assert archive.list_submissions() == []


def test_unlabeled_split_is_not_scored(metadata_dir):
    splits_path = metadata_dir / MetaFiles.SPLITS
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    splits["unlabeled_val"] = splits[SPLIT][:15]
    splits_path.write_text(json.dumps(splits), encoding="utf-8")
    # Unlabeled segments have no road data.
    road_data_path = metadata_dir / MetaFiles.SEGMENT_ID_TO_ROAD_DATA
    road_data = json.loads(road_data_path.read_text(encoding="utf-8"))
    for segment_id in splits["unlabeled_val"]:
        del road_data[segment_id]
    road_data_path.write_text(json.dumps(road_data), encoding="utf-8")
    context = load_dataset_contexts({"vietnam": DatasetConfig(
        metadata_dir=metadata_dir, images_dir=None, analysis_splits=())})["vietnam"]
    predictions = make_predictions(context.metadata.vocabulary, splits["unlabeled_val"],
                                   name="m", seed=0)
    predictions = dc.replace(predictions,
                             header=dc.replace(predictions.header, split="unlabeled_val"))
    evaluation_sets, notes = context.select_evaluation_sets(predictions)
    assert evaluation_sets == []
    assert len(notes) == 2 and all("has no labels" in note for note in notes)
    with pytest.raises(ValueError, match="No predictions for 1"):
        context.select_evaluation_sets(ie.select_segments(predictions,
                                                          predictions.segment_ids[:-1]))


def test_failed_storing_keeps_the_upload(archive, dataset_contexts, monkeypatch):
    path, add_upload = upload(archive, dataset_contexts, make_run(dataset_contexts))

    def fail(*args):
        raise RuntimeError("after the move")

    # Called after the file is moved, in the transaction.
    monkeypatch.setattr(archive_module, "_get_submission", fail)
    with pytest.raises(RuntimeError, match="after the move"):
        add_upload()
    assert path.exists()
    assert not archive.get_predictions_path(1).exists()
    assert archive.list_submissions(include_deleted=True) == []
    assert archive.list_actions() == []


@pytest.mark.parametrize("method_seed, seed", [(2**63, None), (None, 2**63)])
def test_seed_must_fit_64_bits(archive, dataset_contexts, method_seed, seed):
    path, add_upload = upload(archive, dataset_contexts,
                              make_run(dataset_contexts, method_seed=method_seed), seed=seed)
    with pytest.raises(ValueError, match="64-bit"):
        add_upload()
    assert path.exists()


def test_duplicate_run_is_refused_before_the_file_is_checked(archive, dataset_contexts,
                                                             monkeypatch):
    add(archive, dataset_contexts, make_run(dataset_contexts, method_seed=1))
    _, add_upload = upload(archive, dataset_contexts, make_run(dataset_contexts, method_seed=2),
                           seed=1)

    def fail(*args, **kwargs):
        raise AssertionError("The file is checked or rewritten.")

    monkeypatch.setattr(archive_module, "_replace_method", fail)
    monkeypatch.setattr(DatasetContext, "select_evaluation_sets", fail)
    monkeypatch.setattr(ie, "write_predictions", fail)
    with pytest.raises(ValueError, match="same seed"):
        add_upload()


def test_evaluation_sets_are_computed_once(dataset_contexts, monkeypatch):
    context = dataset_contexts["vietnam"]
    num_calls = 0
    get_reference_set = ie.get_reference_set

    def get_reference_set_slowly(*args):
        nonlocal num_calls
        num_calls += 1
        time.sleep(0.1)
        return get_reference_set(*args)

    monkeypatch.setattr(ie, "get_reference_set", get_reference_set_slowly)
    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        reference_sets = list(executor.map(context.get_reference_set, [SPLIT] * 4))
    assert num_calls == 1
    assert all(s is reference_sets[0] for s in reference_sets)


def test_remove_uploads(archive):
    archive.make_upload_path("m.parquet").write_bytes(b"x")
    assert archive.remove_uploads() == 1
    assert not list(archive.uploads_dir.iterdir())

    old_path = archive.make_upload_path("old.parquet")
    new_path = archive.make_upload_path("new.parquet")
    old_path.write_bytes(b"x")
    new_path.write_bytes(b"x")
    old_time_s = time.time() - 3600
    os.utime(old_path, (old_time_s, old_time_s))
    assert archive.remove_uploads(older_than_s=600) == 1
    assert list(archive.uploads_dir.iterdir()) == [new_path]


def test_upload_paths(archive):
    file_path = archive.make_upload_path("run.parquet")
    archive_path = archive.make_upload_path("RUN.ZIP")
    assert not archive_module.is_archive_upload(file_path)
    assert archive_module.is_archive_upload(archive_path)
    for path in (file_path, archive_path):
        assert path.parent == archive.uploads_dir
        assert archive.get_upload_path(path.name) == path
    for name in ("../archive.sqlite3", f"../{file_path.name}", "x.zip", file_path.stem):
        assert archive.get_upload_path(name) is None


def test_add_uploaded_archive(archive, two_split_contexts, tmp_path):
    path, add_upload = upload_archive(
        archive, two_split_contexts, make_run_files(two_split_contexts, method_seed=1), tmp_path)
    member_sha256s = read_member_sha256s(path)
    results = add_upload()

    assert not list(archive.uploads_dir.iterdir())
    submissions = [s for s, _ in results]
    assert [(s.split, s.run_label, s.description) for s in submissions] == [
        (SPLIT, "m/seed1", "all splits"), (OTHER_SPLIT, "m/seed1", "all splits")]
    assert archive.list_submissions() == submissions[::-1]
    actions = archive.list_actions()[::-1]
    assert [a.submission_id for a in actions] == [s.id for s in submissions]
    for action, (submission, notes), (member_name, sha256) in zip(
            actions, results, member_sha256s.items(), strict=True):
        assert action.details == {"file_name": "m.zip", "archive_member": member_name,
                                  "notes": notes}
        assert submission.file_sha256 == sha256
        assert hashlib.sha256(archive.get_predictions_path(submission.id).read_bytes()
                              ).hexdigest() == sha256


def test_archive_seed_replaces_the_seeds_of_all_files(archive, two_split_contexts, tmp_path):
    _, add_upload = upload_archive(
        archive, two_split_contexts, make_run_files(two_split_contexts, method_seed=1), tmp_path,
        seed=5)
    results = add_upload()
    assert [s.run_label for s, _ in results] == ["m/seed5", "m/seed5"]
    for submission, _ in results:
        stored = ie.read_predictions(archive.get_predictions_path(submission.id))
        assert stored.header.method.seed == 5
    assert all(a.details["original_seed"] == 1 for a in archive.list_actions())
    assert not list(archive.uploads_dir.iterdir())


def test_archive_is_stored_entirely_or_not_at_all(archive, two_split_contexts, tmp_path):
    existing = add(archive, two_split_contexts,
                   with_split(make_run(two_split_contexts, method_seed=1), OTHER_SPLIT))
    path, add_upload = upload_archive(
        archive, two_split_contexts, make_run_files(two_split_contexts, method_seed=1), tmp_path)
    with pytest.raises(ValueError, match=rf"^run/m\.{OTHER_SPLIT}\.predictions\.parquet: .*same"
                                         rf" seed.*#{existing.id}"):
        add_upload()
    assert archive.list_submissions() == [existing]
    assert list(archive.uploads_dir.iterdir()) == [path]  # Kept for a retry.

    results = add_uploaded_archive(archive, two_split_contexts, path, file_name="m.zip",
                                   submitter="Ana", description="", seed=2)
    assert [s.run_label for s, _ in results] == ["m/seed2", "m/seed2"]
    assert not list(archive.uploads_dir.iterdir())


def _make_mixed_runs(dataset_contexts):
    files = make_run_files(dataset_contexts, splits=(SPLIT,), method_seed=1)
    other = make_run_files(dataset_contexts, splits=(OTHER_SPLIT,), method_seed=2)
    return files | other


def _make_one_split_twice(dataset_contexts):
    [predictions] = make_run_files(dataset_contexts, splits=(SPLIT,)).values()
    return {"a.parquet": predictions, "b.parquet": predictions}


@pytest.mark.parametrize("make_member_name_to_contents, message", [
    (_make_mixed_runs, (r"test\.predictions\.parquet: It is of the run m/seed2 on vietnam with"
                        r" the context offsets 0, -1, but .* of m/seed1")),
    (_make_one_split_twice, r"^b\.parquet: It is of the split 'val', like a\.parquet"),
    (lambda contexts: {**make_run_files(contexts), "run/notes.txt": "x"},
     r"only prediction files .* Other files: run/notes\.txt\.$"),
    (lambda contexts: {}, r"has no files"),
])
def test_archive_contents_are_checked(archive, two_split_contexts, tmp_path,
                                      make_member_name_to_contents, message):
    path, add_upload = upload_archive(
        archive, two_split_contexts, make_member_name_to_contents(two_split_contexts), tmp_path)
    with pytest.raises(ValueError, match=message):
        add_upload()
    assert archive.list_submissions(include_deleted=True) == []
    assert list(archive.uploads_dir.iterdir()) == [path]


def test_upload_that_is_not_a_zip_archive_is_refused(archive, dataset_contexts):
    path = archive.make_upload_path("m.zip")
    path.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match=r"m\.zip is not a \.zip archive"):
        add_uploaded_archive(archive, dataset_contexts, path, file_name="m.zip",
                             submitter="Ana", description="")
    assert path.exists()


def test_macos_metadata_in_an_archive_is_skipped(archive, two_split_contexts, tmp_path):
    files = make_run_files(two_split_contexts)
    _, add_upload = upload_archive(
        archive, two_split_contexts,
        {**files, **{f"__MACOSX/{name.replace('/', '/._')}": "x" for name in files}}, tmp_path)
    assert len(add_upload()) == 2


def test_encrypted_archive_member_is_refused(archive, dataset_contexts):
    path = archive.make_upload_path("m.zip")
    with zipfile.ZipFile(path, "w") as zip_file:
        zip_file.writestr("m.parquet", b"x")
    data = bytearray(path.read_bytes())
    data[data.rfind(b"PK\x01\x02") + 8] |= 0x01  # The encryption flag of the central directory.
    path.write_bytes(data)
    with pytest.raises(ValueError, match=r"^m\.parquet: It cannot be extracted: .*encrypted"):
        add_uploaded_archive(archive, dataset_contexts, path, file_name="m.zip",
                             submitter="Ana", description="")
    assert path.exists()


def test_failed_storing_moves_back_all_files(archive, two_split_contexts, monkeypatch):
    new_submissions = []
    for split in (SPLIT, OTHER_SPLIT):
        predictions = with_split(make_run(two_split_contexts), split)
        path = write_upload(archive, predictions)
        new_submissions.append(NewSubmission.from_predictions(predictions, path, {}))

    def fail(*args):
        raise RuntimeError("after the moves")

    # Called after the files are moved, in the transaction.
    monkeypatch.setattr(archive_module, "_get_submission", fail)
    with pytest.raises(RuntimeError, match="after the moves"):
        archive.add_submissions(new_submissions, submitter="Ana", description="")
    assert all(new.file_path.exists() for new in new_submissions)
    assert not archive.get_predictions_path(1).exists()
    assert not archive.get_predictions_path(2).exists()
    assert archive.list_submissions(include_deleted=True) == []
    assert archive.list_actions() == []


def test_load_dataset_contexts_checks_the_analysis_splits(metadata_dir):
    with pytest.raises(ValueError, match=r"analysis splits \['test'\]"):
        load_dataset_contexts({"vietnam": DatasetConfig(
            metadata_dir=metadata_dir, images_dir=None, analysis_splits=("test",))})
