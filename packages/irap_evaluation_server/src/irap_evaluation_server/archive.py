"""The submissions archive: prediction files on disk, an SQLite index of them and an action log.

The data directory holds:

- `archive.sqlite3`: the index of the submissions and the action log,
- `submissions/<id>/predictions.parquet`: the stored files, which are not modified,
- `uploads/`: uploaded files and .zip archives that are not yet stored, and the files extracted
  from archives while they are checked.

Submissions are not removed: a deleted one is marked and can be restored. There are no logins, so
each action records a free-text name of who did it.
"""

import dataclasses as dc
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import time
import typing as T
import uuid
import zipfile
import zlib
from datetime import datetime
from pathlib import Path, PurePosixPath

import irap_evaluation as ie

from .database import begin_write, connect, get_utc_now
from .datasets import DatasetContext

#: The actions that add a submission: an upload, or an ensemble of stored runs (see `ensembles`).
AddingActionKind = T.Literal["upload", "ensemble"]
ADDING_ACTION_KINDS: tuple[str, ...] = T.get_args(AddingActionKind)
ActionKind = AddingActionKind | T.Literal["delete", "restore"]

#: SQLite stores integers as signed 64-bit values.
_SQLITE_INTEGER_RANGE = range(-2**63, 2**63)

_PREDICTIONS_SUFFIX = ".predictions.parquet"
_ARCHIVE_SUFFIX = ".zip"
#: The names of `SubmissionArchive.make_upload_path`.
_UPLOAD_NAME_PATTERN = re.compile(
    rf"[0-9a-f]{{32}}({re.escape(_PREDICTIONS_SUFFIX)}|{re.escape(_ARCHIVE_SUFFIX)})")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY,
    file_sha256 TEXT NOT NULL,
    dataset TEXT NOT NULL,
    split TEXT NOT NULL,
    method_name TEXT NOT NULL,
    method_seed INTEGER,
    context_offsets TEXT,
    output_kind TEXT NOT NULL,
    num_segments INTEGER NOT NULL,
    num_attributes INTEGER NOT NULL,
    submitter TEXT NOT NULL,
    description TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    deleted_at TEXT
);
CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY,
    time TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    submission_id INTEGER REFERENCES submissions (id),
    details TEXT NOT NULL
);
"""


@dc.dataclass(frozen=True)
class Submission:
    """A stored prediction file: one run of a method on one split of a dataset.

    The header fields are copies of the header of the file, so that listing does not read the
    files.

    Attributes:
        method: The name and seed of the run. The details are only in the file.
        submitter: The free-text name of who uploaded it.
        uploaded_at: The time of the upload, in UTC.
        deleted_at: The time of the deletion, in UTC, or None if it is not deleted.
    """

    id: int
    file_sha256: str
    dataset: str
    split: str
    method: ie.MethodInfo
    context_offsets: tuple[int, ...] | None
    output_kind: ie.OutputKind
    num_segments: int
    num_attributes: int
    submitter: str
    description: str
    uploaded_at: datetime
    deleted_at: datetime | None

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None

    @property
    def run_label(self) -> str:
        return self.method.run_label


@dc.dataclass(frozen=True)
class ActionLogEntry:
    """An action on the archive.

    Attributes:
        time: In UTC.
        actor: The free-text name of who did it.
        details: JSON-compatible details, e.g. the name of an uploaded file.
    """

    id: int
    time: datetime
    actor: str
    action: ActionKind
    submission_id: int | None
    details: T.Mapping[str, T.Any]


@dc.dataclass(frozen=True)
class NewSubmission:
    """A checked prediction file to store with `SubmissionArchive.add_submissions`.

    It holds only what the index needs, so that the predictions of several files need not be in
    memory together.

    Attributes:
        file_path: The file, in the same file system as the data directory, e.g. at
            `SubmissionArchive.make_upload_path`.
        action: The action that adds it, in the action log.
        details: The details of the action in the action log, e.g. the name of the uploaded
            file.
        notes: The notes of `DatasetContext.check_new_predictions`, which the action log
            records with the details (as 'notes').
    """

    header: ie.PredictionHeader
    output_kind: ie.OutputKind
    num_segments: int
    num_attributes: int
    file_path: Path
    action: AddingActionKind
    details: T.Mapping[str, T.Any]
    notes: tuple[str, ...]

    @classmethod
    def from_predictions(cls, predictions: ie.Predictions, file_path: Path, *,
                         action: AddingActionKind, details: T.Mapping[str, T.Any],
                         notes: T.Sequence[str]) -> T.Self:
        """Args:
            predictions: The contents of the file, e.g. from `irap_evaluation.read_predictions`.
        """
        return cls(header=predictions.header, output_kind=predictions.output_kind,
                   num_segments=predictions.num_segments,
                   num_attributes=len(predictions.attributes), file_path=file_path,
                   action=action, details=details, notes=tuple(notes))


def _compute_file_sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _to_submission(row: sqlite3.Row) -> Submission:
    return Submission(
        id=row["id"], file_sha256=row["file_sha256"], dataset=row["dataset"], split=row["split"],
        method=ie.MethodInfo(name=row["method_name"], seed=row["method_seed"]),
        context_offsets=(None if row["context_offsets"] is None
                         else tuple(json.loads(row["context_offsets"]))),
        output_kind=row["output_kind"], num_segments=row["num_segments"],
        num_attributes=row["num_attributes"], submitter=row["submitter"],
        description=row["description"], uploaded_at=datetime.fromisoformat(row["uploaded_at"]),
        deleted_at=None if row["deleted_at"] is None else datetime.fromisoformat(row["deleted_at"]))


def _to_action_log_entry(row: sqlite3.Row) -> ActionLogEntry:
    return ActionLogEntry(id=row["id"], time=datetime.fromisoformat(row["time"]),
                          actor=row["actor"], action=row["action"],
                          submission_id=row["submission_id"], details=json.loads(row["details"]))


def check_actor(actor: str) -> str:
    """The free-text name of who does an action, stripped.

    Raises:
        ValueError: If it is empty.
    """
    if not actor.strip():
        raise ValueError("Enter your name, so that the action log shows who did this.")
    return actor.strip()


def _get_submission(connection: sqlite3.Connection, submission_id: int) -> Submission:
    row = connection.execute("SELECT * FROM submissions WHERE id = ?",
                             (submission_id,)).fetchone()
    if row is None:
        raise LookupError(f"There is no submission #{submission_id}.")
    return _to_submission(row)


def _log_action(connection: sqlite3.Connection, actor: str, action: ActionKind,
                submission_id: int | None, details: T.Mapping[str, T.Any]) -> None:
    connection.execute(
        "INSERT INTO actions (time, actor, action, submission_id, details)"
        " VALUES (?, ?, ?, ?, ?)",
        (get_utc_now().isoformat(), actor, action, submission_id, json.dumps(details)))


def _check_seed_storable(seed: int | None) -> None:
    if seed is not None and seed not in _SQLITE_INTEGER_RANGE:
        raise ValueError(f"The seed {seed} does not fit a signed 64-bit integer.")


def parse_seed(text: str) -> int | None:
    """The seed that a user entered, None for a blank text.

    Raises:
        ValueError: If it is not an integer.
    """
    if not text.strip():
        return None
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"The seed must be an integer, not {text.strip()!r}.") from None


def _check_run_distinct(connection: sqlite3.Connection, dataset: str, split: str,
                        method: ie.MethodInfo, excluded_id: int | None = None) -> None:
    """Checks that the run is distinct from the runs of its method that are not deleted (see
    `irap_evaluation.check_runs_distinct`), apart from `excluded_id`, e.g. a submission being
    restored.

    Raises:
        ValueError: If it is not.
    """
    rows = connection.execute(
        "SELECT * FROM submissions WHERE deleted_at IS NULL AND dataset = ? AND split = ?"
        " AND method_name = ? AND id IS NOT ?", (dataset, split, method.name, excluded_id))
    runs = [_to_submission(row) for row in rows]
    try:
        ie.check_runs_distinct([*(s.method for s in runs), method])
    except ValueError as e:
        existing = ", ".join(f"#{s.id} ({s.run_label})" for s in runs)
        # Another seed of the new run does not help if an existing run has none.
        remedy = ("Delete it first." if any(s.method.seed is None for s in runs)
                  else "Delete one of them, or set another seed.")
        raise ValueError(f"{e} Runs of {method.name!r} on {dataset}/{split}: {existing}."
                         f" {remedy}") from e


class SubmissionArchive:
    """The archive in a data directory (see the module docstring), created on first use.

    Each operation opens its own database connection, so the archive can be used from several
    threads.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "submissions").mkdir(exist_ok=True)
        with connect(self.database_path) as connection:
            connection.executescript(_SCHEMA)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "archive.sqlite3"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def get_predictions_path(self, submission_id: int) -> Path:
        return self.data_dir / "submissions" / str(submission_id) / "predictions.parquet"

    def make_upload_path(self, file_name: str) -> Path:
        """A new path in `uploads_dir` for an uploaded file, a .zip archive (see
        `is_archive_upload`) if `file_name` has that suffix, otherwise a prediction file.

        Args:
            file_name: The name of the file on the uploader's computer.
        """
        suffix = (_ARCHIVE_SUFFIX if file_name.lower().endswith(_ARCHIVE_SUFFIX)
                  else _PREDICTIONS_SUFFIX)
        return self.uploads_dir / f"{uuid.uuid4().hex}{suffix}"

    def get_upload_path(self, upload_name: str) -> Path | None:
        """The path of an upload by the name of its file, None if `make_upload_path` does not
        make such names, e.g. if a browser sent a name that points outside `uploads_dir`."""
        if _UPLOAD_NAME_PATTERN.fullmatch(upload_name) is None:
            return None
        return self.uploads_dir / upload_name

    def remove_uploads(self, *, older_than_s: float | None = None) -> int:
        """Removes the files in `uploads_dir`, e.g. those left when the server stopped, or those
        of abandoned uploads.

        Args:
            older_than_s: If not None, only the files last modified more than this many seconds
                ago are removed.

        Returns:
            The number of removed files.
        """
        max_mtime_s = math.inf if older_than_s is None else time.time() - older_than_s
        num_removed = 0
        for path in self.uploads_dir.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < max_mtime_s:
                    path.unlink()
                    num_removed += 1
            except FileNotFoundError:  # Removed by a concurrent call.
                pass
        return num_removed

    def list_submissions(self, *, include_deleted: bool = False) -> list[Submission]:
        """The submissions, newest first."""
        where = "" if include_deleted else "WHERE deleted_at IS NULL"
        with connect(self.database_path) as connection:
            rows = connection.execute(f"SELECT * FROM submissions {where} ORDER BY id DESC")
            return [_to_submission(row) for row in rows]

    def get_submission(self, submission_id: int) -> Submission:
        """The submission with this id, also if it is deleted.

        Raises:
            LookupError: If there is no such submission.
        """
        with connect(self.database_path) as connection:
            return _get_submission(connection, submission_id)

    def list_actions(self, *, submission_id: int | None = None) -> list[ActionLogEntry]:
        """The actions, newest first, optionally only those on one submission."""
        query, parameters = "SELECT * FROM actions", ()
        if submission_id is not None:
            query, parameters = query + " WHERE submission_id = ?", (submission_id,)
        with connect(self.database_path) as connection:
            rows = connection.execute(query + " ORDER BY id DESC", parameters)
            return [_to_action_log_entry(row) for row in rows]

    def check_new_run(self, dataset: str, split: str, method: ie.MethodInfo) -> None:
        """Checks that a run can be stored: its seed fits SQLite's INTEGER, and the run is
        distinct from the runs of its method that are not deleted (see
        `irap_evaluation.check_runs_distinct`).

        `add_submissions` checks this again in its transaction. Calling it first only refuses a
        run before an expensive check of its file.

        Raises:
            ValueError: If the run cannot be stored.
        """
        _check_seed_storable(method.seed)
        with connect(self.database_path) as connection:
            _check_run_distinct(connection, dataset, split, method)

    def add_submissions(self, new_submissions: T.Sequence[NewSubmission], *, submitter: str,
                        description: str) -> list[Submission]:
        """Stores prediction files, moved from their `file_path`, and logs the action that adds
        each of them (`NewSubmission.action`).

        Either all are stored or none. Files that are not stored stay at their `file_path`.

        Raises:
            ValueError: If `submitter` is empty, or `check_new_run` refuses a run, also because
                of a run stored before it in the same call.
        """
        submitter = check_actor(submitter)
        for new in new_submissions:
            _check_seed_storable(new.header.method.seed)
        file_sha256s = [_compute_file_sha256(new.file_path) for new in new_submissions]
        moved_paths: list[tuple[Path, Path]] = []  # (target, source)
        try:
            with begin_write(self.database_path) as connection:
                submission_ids = []
                for new, file_sha256 in zip(new_submissions, file_sha256s, strict=True):
                    header = new.header
                    _check_run_distinct(connection, header.dataset, header.split, header.method)
                    offsets = header.context_offsets
                    cursor = connection.execute(
                        "INSERT INTO submissions (file_sha256, dataset, split, method_name,"
                        " method_seed, context_offsets, output_kind, num_segments,"
                        " num_attributes, submitter, description, uploaded_at)"
                        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (file_sha256, header.dataset, header.split, header.method.name,
                         header.method.seed,
                         None if offsets is None else json.dumps(list(offsets)),
                         new.output_kind, new.num_segments, new.num_attributes, submitter,
                         description.strip(), get_utc_now().isoformat()))
                    submission_id = cursor.lastrowid
                    submission_ids.append(submission_id)
                    _log_action(connection, submitter, new.action, submission_id,
                                {**new.details, "notes": list(new.notes)})
                    target_path = self.get_predictions_path(submission_id)
                    # A directory left by a crash between the move and COMMIT is overwritten.
                    target_path.parent.mkdir(exist_ok=True)
                    os.replace(new.file_path, target_path)
                    moved_paths.append((target_path, new.file_path))
                submissions = [_get_submission(connection, i) for i in submission_ids]
        except BaseException:
            # The transaction is rolled back, also if COMMIT failed.
            for target_path, source_path in reversed(moved_paths):
                os.replace(target_path, source_path)
            raise
        return submissions

    def set_submission_deleted(self, submission_id: int, is_deleted: bool, *,
                               actor: str) -> Submission:
        """Deletes or restores a submission and logs it.

        Raises:
            LookupError: If there is no such submission.
            ValueError: If `actor` is empty, the submission is already in that state, or a
                restored run is no longer distinct from the runs of its method.
        """
        actor = check_actor(actor)
        with begin_write(self.database_path) as connection:
            submission = _get_submission(connection, submission_id)
            if submission.is_deleted == is_deleted:
                state = "deleted" if is_deleted else "not deleted"
                raise ValueError(f"Submission #{submission_id} is already {state}.")
            if not is_deleted:
                _check_run_distinct(connection, submission.dataset, submission.split,
                                    submission.method, excluded_id=submission_id)
            connection.execute("UPDATE submissions SET deleted_at = ? WHERE id = ?",
                               (get_utc_now().isoformat() if is_deleted else None,
                                submission_id))
            _log_action(connection, actor, "delete" if is_deleted else "restore",
                        submission_id, {})
            return _get_submission(connection, submission_id)


def _override_seed(method: ie.MethodInfo, seed: int | None) -> ie.MethodInfo:
    """The method with the seed given at the upload, if it is not None."""
    return method if seed is None else dc.replace(method, seed=seed)


def _replace_method(predictions: ie.Predictions, method: ie.MethodInfo) -> ie.Predictions:
    return dc.replace(predictions, header=dc.replace(predictions.header, method=method))


def _get_rewritten_path(uploaded_path: Path) -> Path:
    """Where `_prepare_upload` writes an uploaded file with a replaced seed."""
    return uploaded_path.with_name(f"{uploaded_path.name}.rewritten")


def _prepare_upload(
    archive: SubmissionArchive,
    dataset_contexts: T.Mapping[str, DatasetContext],
    uploaded: ie.Predictions,
    uploaded_path: Path,
    *,
    seed: int | None,
    details: T.Mapping[str, T.Any],
) -> NewSubmission:
    """Checks the contents of an uploaded prediction file, read from `uploaded_path` (see
    `add_uploaded_submission`).

    If the seed is replaced, the file is rewritten to `_get_rewritten_path(uploaded_path)`, and
    the new submission is of that file.

    Args:
        details: The details of the upload in the action log, without the notes and seeds.
    """
    header = uploaded.header
    method = _override_seed(header.method, seed)
    is_seed_replaced = method != header.method
    if header.dataset not in dataset_contexts:
        raise ValueError(f"The predictions are of the dataset {header.dataset!r}, which the"
                         f" server does not have. Its datasets: {', '.join(dataset_contexts)}.")
    archive.check_new_run(header.dataset, header.split, method)
    notes = dataset_contexts[header.dataset].check_new_predictions(uploaded)
    # After the checks, since it validates the predictions again.
    predictions = _replace_method(uploaded, method) if is_seed_replaced else uploaded
    path = uploaded_path
    if is_seed_replaced:
        details = {**details, "original_seed": header.method.seed,
                   "uploaded_file_sha256": _compute_file_sha256(uploaded_path)}
        path = _get_rewritten_path(uploaded_path)
        ie.write_predictions(path, predictions)
    return NewSubmission.from_predictions(predictions, path, action="upload", details=details,
                                          notes=notes)


def add_uploaded_submission(
    archive: SubmissionArchive,
    dataset_contexts: T.Mapping[str, DatasetContext],
    uploaded_path: Path,
    *,
    file_name: str,
    submitter: str,
    description: str,
    seed: int | None = None,
) -> tuple[Submission, list[str]]:
    """Checks an uploaded prediction file and stores it in the archive.

    The file must be of a configured dataset, predict attributes of its vocabulary, and be
    accepted by `irap_evaluation.select_evaluation_sets`. If it is stored, the uploaded file is
    moved into the archive or removed. Otherwise it is kept, so that the upload can be retried,
    e.g. with another seed.

    Args:
        uploaded_path: The uploaded file, at `archive.make_upload_path`.
        file_name: The name of the file on the uploader's computer, for the action log.
        seed: If not None, it replaces the seed in the header (`MethodInfo.seed`), e.g. to make
            the run distinct from another run of its method. The file is then rewritten, and
            the action log records the original seed and the hash of the uploaded file.

    Returns:
        The submission and the notes of `select_evaluation_sets`, e.g. that the model is not
        scored on the reference set.

    Raises:
        irap_evaluation.PredictionFormatError: If the file is not a valid prediction file, or
            its attributes or classes differ from the dataset's.
        ValueError: If `submitter` is empty, the dataset is not configured, or the file is
            refused by `select_evaluation_sets` or by `SubmissionArchive.check_new_run`.
    """
    # The cheap checks come first, so that a refused retry does not wait for the others.
    submitter = check_actor(submitter)
    uploaded = ie.read_predictions(uploaded_path)
    try:
        new = _prepare_upload(archive, dataset_contexts, uploaded, uploaded_path, seed=seed,
                              details={"file_name": file_name})
        [submission] = archive.add_submissions([new], submitter=submitter,
                                               description=description)
    finally:
        _get_rewritten_path(uploaded_path).unlink(missing_ok=True)
    if new.file_path != uploaded_path:  # Otherwise it is moved into the archive.
        uploaded_path.unlink()
    return submission, list(new.notes)


def _is_macos_metadata(info: zipfile.ZipInfo) -> bool:
    """Whether a member is metadata that macOS adds to archives (AppleDouble files)."""
    return (info.filename.startswith("__MACOSX/")
            or PurePosixPath(info.filename).name.startswith("._"))


def _list_archive_members(zip_file: zipfile.ZipFile, file_name: str) -> list[zipfile.ZipInfo]:
    members = [info for info in zip_file.infolist()
               if not info.is_dir() and not _is_macos_metadata(info)]
    if not members:
        raise ValueError(f"The archive {file_name} has no files.")
    if others := [m.filename for m in members if not m.filename.lower().endswith(".parquet")]:
        raise ValueError(f"The archive {file_name} may hold only prediction files (.parquet)."
                         f" Other files: {', '.join(others)}.")
    return members


@dc.dataclass(frozen=True)
class _ArchiveRun:
    """What the files of an archive must share."""

    dataset: str
    method_name: str
    method_seed: int | None
    context_offsets: tuple[int, ...] | None

    @classmethod
    def from_header(cls, header: ie.PredictionHeader, seed: int | None) -> T.Self:
        """Args:
            seed: The seed given at the upload (see `_override_seed`).
        """
        method = _override_seed(header.method, seed)
        return cls(dataset=header.dataset, method_name=method.name, method_seed=method.seed,
                   context_offsets=header.context_offsets)

    def __str__(self) -> str:
        run_label = ie.MethodInfo(name=self.method_name, seed=self.method_seed).run_label
        offsets = self.context_offsets
        return (f"{run_label} on {self.dataset} with the context offsets"
                f" {'unknown' if offsets is None else ', '.join(map(str, offsets))}")


def _extract_archive_member(zip_file: zipfile.ZipFile, member: zipfile.ZipInfo,
                            path: Path) -> None:
    """Extracts a member to `path`.

    Raises:
        ValueError: If the member cannot be extracted, e.g. it is encrypted, corrupt, or
            compressed with an unsupported method.
    """
    try:
        with zip_file.open(member) as source, path.open("wb") as target:
            shutil.copyfileobj(source, target)
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, zlib.error, EOFError) as e:
        raise ValueError(f"It cannot be extracted: {e}") from e


def add_uploaded_archive(
    archive: SubmissionArchive,
    dataset_contexts: T.Mapping[str, DatasetContext],
    uploaded_path: Path,
    *,
    file_name: str,
    submitter: str,
    description: str,
    seed: int | None = None,
) -> list[tuple[Submission, list[str]]]:
    """Checks an uploaded .zip archive with the prediction files of one run, one per split, and
    stores them in the archive.

    Each file is checked as in `add_uploaded_submission`. All must have the same dataset, method
    name, seed (after `seed` replaces it) and context offsets, and distinct splits. Either all
    files are stored or none. If they are stored, the uploaded archive is removed. Otherwise it
    is kept, so that the upload can be retried, e.g. with another seed.

    Args:
        uploaded_path: The uploaded archive, at `archive.make_upload_path`.
        file_name: The name of the archive on the uploader's computer, for the action log, which
            also records the name of each file in it.
        seed: If not None, it replaces the seed in each header, as in `add_uploaded_submission`.

    Returns:
        The submissions and their notes, in the order of the files in the archive.

    Raises:
        ValueError: If `submitter` is empty, the file is not a .zip archive, or it has no files
            or files without the suffix '.parquet', or a file is refused as in
            `add_uploaded_submission`, of another run than the first file, or of the split of an
            earlier file. The message of a refused file starts with its name.
    """
    submitter = check_actor(submitter)
    try:
        zip_file = zipfile.ZipFile(uploaded_path)
    except zipfile.BadZipFile as e:
        raise ValueError(f"{file_name} is not a .zip archive: {e}") from e
    extracted_paths: list[Path] = []
    try:
        prepared: list[NewSubmission] = []
        with zip_file:
            first_run_and_member_name: tuple[_ArchiveRun, str] | None = None
            split_to_member_name: dict[str, str] = {}
            for member in _list_archive_members(zip_file, file_name):
                path = archive.make_upload_path(member.filename)
                extracted_paths.append(path)
                try:
                    _extract_archive_member(zip_file, member, path)
                    uploaded = ie.read_predictions(path)
                    header = uploaded.header
                    run = _ArchiveRun.from_header(header, seed)
                    if first_run_and_member_name is None:
                        first_run_and_member_name = (run, member.filename)
                    elif run != first_run_and_member_name[0]:
                        first_run, first_member_name = first_run_and_member_name
                        raise ValueError(f"It is of the run {run}, but {first_member_name} is"
                                         f" of {first_run}. An archive holds the files of one"
                                         f" run.")
                    if header.split in split_to_member_name:
                        raise ValueError(f"It is of the split {header.split!r}, like"
                                         f" {split_to_member_name[header.split]}. An archive"
                                         f" holds one file per split.")
                    split_to_member_name[header.split] = member.filename
                    prepared.append(_prepare_upload(
                        archive, dataset_contexts, uploaded, path, seed=seed,
                        details={"file_name": file_name, "archive_member": member.filename}))
                except ValueError as e:
                    raise ValueError(f"{member.filename}: {e}") from e
        submissions = archive.add_submissions(prepared, submitter=submitter,
                                              description=description)
    finally:
        # Those that are stored are moved into the archive.
        for path in extracted_paths:
            path.unlink(missing_ok=True)
            _get_rewritten_path(path).unlink(missing_ok=True)
    uploaded_path.unlink()
    return [(s, list(new.notes)) for s, new in zip(submissions, prepared, strict=True)]


def is_archive_upload(uploaded_path: Path) -> bool:
    """Whether an upload at `SubmissionArchive.make_upload_path` is a .zip archive."""
    return uploaded_path.name.endswith(_ARCHIVE_SUFFIX)


def add_upload(
    archive: SubmissionArchive,
    dataset_contexts: T.Mapping[str, DatasetContext],
    uploaded_path: Path,
    *,
    file_name: str,
    submitter: str,
    description: str,
    seed: int | None = None,
) -> list[tuple[Submission, list[str]]]:
    """`add_uploaded_archive` for a .zip archive (`is_archive_upload`), otherwise
    `add_uploaded_submission`, with the same arguments, errors and the submissions in a list."""
    kwargs = dict(file_name=file_name, submitter=submitter, description=description, seed=seed)
    if is_archive_upload(uploaded_path):
        return add_uploaded_archive(archive, dataset_contexts, uploaded_path, **kwargs)
    return [add_uploaded_submission(archive, dataset_contexts, uploaded_path, **kwargs)]
