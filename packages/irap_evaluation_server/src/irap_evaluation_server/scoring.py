"""Scores of the submissions, computed with irap_evaluation and cached in a database
(`ScoreStore`).

A submission is scored once on all its evaluation sets and all scored attributes
(`SubmissionScoring`), and again only when its scoring is out of date. The scoring also stores
the metrics of the submission in each bootstrap resample of road sequences (`resample_result`).
Means over the models of a method (`compute_method_scores`), and their resampled values, from
which intervals and paired comparisons follow (`combine_method_resamples`), are computed from
these when they are shown, also over a subset of the attributes.

Scorings carry fingerprints of what they depend on: the scoring settings and code
(`ScoringSettings.fingerprint`), and the evaluation sets
(`irap_evaluation.EvaluationSet.fingerprint`). A scoring whose fingerprints differ from the current
ones is out of date.

`scoring_worker` computes the scores in the background.
"""

import dataclasses as dc
import functools
import hashlib
import json
import logging
import sqlite3
import typing as T
from datetime import datetime
from pathlib import Path

import irap_evaluation as ie
import numpy as np
from irap_data.attrs import get_attrs_to_include
from irap_data.metadata import IGNORE_LABEL_INDEX, ClassVocabulary
from irap_evaluation.metrics import parse_metric_name
from irap_evaluation.reports.evaluation_report import to_json_dict

from .archive import DATABASE_FILE_NAME as ARCHIVE_DATABASE_FILE_NAME
from .archive import ModelArchive, Submission
from .database import begin_write, connect, get_utc_now, initialize_schema
from .datasets import DatasetContext

_logger = logging.getLogger(__name__)

#: 'scored': the scores are stored (none for a split without labels, see
#: `SubmissionScoring.notes`).
#: 'failed': the file cannot be scored, e.g. it lacks segments of a new metadata build. It is
#: scored again only when the settings or the evaluation sets change.
#: 'error': an unexpected error, e.g. a missing file. It is scored again at the next start.
ScoringStatus = T.Literal["scored", "failed", "error"]

#: The attribute averages of the iRAP protocol, amF1 first. Hard predictions lack the
#: probabilistic ones.
IRAP_AVERAGE_METRIC_NAMES = tuple(n for n in ie.get_irap_metric_names(output_kind="probs")
                                  if parse_metric_name(n).is_averaged)
#: The metrics whose resampled values are stored (`resample_result`): `IRAP_AVERAGE_METRIC_NAMES`
#: and their per-attribute values, from which the averages over a subset of the attributes are
#: computed.
RESAMPLED_METRIC_NAMES = ie.add_per_attribute_metric_names(IRAP_AVERAGE_METRIC_NAMES)

#: The file of the score cache in the data directory, apart from the archive, so that a migration
#: of the archive does not copy it to its backup.
DATABASE_FILE_NAME = "scores.sqlite3"
#: The version of the tables of the score cache (`_SCHEMA`, `PRAGMA user_version`). A cache of
#: another version, e.g. of an older server, is created again, and the files are scored again,
#: since the cache is computed from the files (`ScoreStore`).
SCORE_CACHE_VERSION = 1

_SCHEMA = """
CREATE TABLE submission_scorings (
    submission_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    notes TEXT NOT NULL,
    settings_fingerprint TEXT NOT NULL,
    evaluation_sets_fingerprint TEXT NOT NULL,
    scored_at TEXT NOT NULL
);
CREATE TABLE submission_scores (
    submission_id INTEGER NOT NULL REFERENCES submission_scorings (submission_id)
        ON DELETE CASCADE,
    evaluation_set TEXT NOT NULL,
    context_offsets TEXT NOT NULL,
    evaluation_set_fingerprint TEXT NOT NULL,
    attributes TEXT NOT NULL,
    metrics TEXT NOT NULL,
    num_invalid TEXT NOT NULL,
    missing_attributes TEXT NOT NULL,
    report TEXT NOT NULL,
    -- `resamples_to_bytes` of `resample_result`.
    resamples BLOB NOT NULL,
    PRIMARY KEY (submission_id, evaluation_set)
);
"""


def compute_json_sha256(value: T.Any) -> str:
    """Computes the SHA-256 of a JSON-compatible value, independent of the order of dict keys,
    e.g. for a fingerprint or a cache key."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# Settings and scoring #############################################################################

@dc.dataclass(frozen=True)
class ScoringSettings:
    """How the server scores submissions and computes intervals: the defaults of
    `irap-eval evaluate`, but with 1000 bootstrap resamples, and with missing attributes counted
    as invalid cells, so that all models are scored on the same attributes."""

    attribute_selection: ie.AttributeSelection = "canonical"
    null_policy: ie.NullPolicy = "first_class"
    missing_attribute_policy: ie.MissingAttributePolicy = "invalid"
    num_resamples: int = 1000
    confidence: float = 0.95
    bootstrap_seed: int = 0

    @functools.cached_property
    def fingerprint(self) -> str:
        """A SHA-256 of the settings, of what they select from the libraries, e.g. the canonical
        attributes and the default metrics, and of the output of the scoring code
        (`compute_scoring_code_fingerprint`), so that a library update that changes them makes
        stored scorings out of date."""
        return compute_json_sha256({
            **dc.asdict(self),
            "canonical_attributes": list(get_attrs_to_include()),
            "metric_names": {kind: list(ie.get_default_metric_names(kind))
                             for kind in T.get_args(ie.OutputKind)},
            "scoring_code": compute_scoring_code_fingerprint(self)})


SCORING_SETTINGS = ScoringSettings()


def select_scored_attributes(evaluation_set: ie.EvaluationSet,
                             settings: ScoringSettings) -> tuple[str, ...]:
    return ie.select_evaluated_attributes(evaluation_set, settings.attribute_selection)


def score_predictions(predictions: ie.Predictions, evaluation_set: ie.EvaluationSet,
                      settings: ScoringSettings) -> ie.EvaluationResult:
    return ie.evaluate_predictions(
        predictions, evaluation_set, attributes=select_scored_attributes(evaluation_set, settings),
        null_policy=settings.null_policy,
        missing_attribute_policy=settings.missing_attribute_policy)


def resample_result(result: ie.EvaluationResult,
                    settings: ScoringSettings) -> ie.MetricValues[np.ndarray]:
    """Computes the metrics of `RESAMPLED_METRIC_NAMES` that a result has in each bootstrap
    resample (`irap_evaluation.resample_model_metrics`): (num_resamples,) float32 arrays, which
    are half the size of float64 ones and precise enough for intervals and differences."""
    names = [n for n in RESAMPLED_METRIC_NAMES if n in result.metrics.names]
    return _to_float32(ie.resample_model_metrics(result, names,
                                                 num_resamples=settings.num_resamples,
                                                 seed=settings.bootstrap_seed))


def _to_float32(values: ie.MetricValues[np.ndarray]) -> ie.MetricValues[np.ndarray]:
    """Converts resampled values to float32, in which they are stored (`resample_result`)."""
    return ie.map_metric_values(lambda v: v.astype(np.float32), values)


def _make_code_probe_predictions(
        evaluation_set: ie.EvaluationSet) -> tuple[ie.Predictions, ie.Predictions]:
    """Makes random probabilistic and hard predictions of a method for the segments of
    `evaluation_set` (`compute_scoring_code_fingerprint`)."""
    rng = np.random.default_rng(0)
    header = ie.PredictionHeader(
        dataset=evaluation_set.dataset, split=evaluation_set.split,
        model=ie.ModelInfo("probe", training_splits=(), early_stopping_splits=(), seed=0),
        context_offsets=evaluation_set.context_offsets)
    attribute_to_irap_codes = evaluation_set.vocabulary.attribute_to_irap_codes
    probabilistic = ie.Predictions.from_logits(
        header, attribute_to_irap_codes, evaluation_set.segment_ids,
        {a: 3 * rng.normal(size=(evaluation_set.num_segments, len(codes)))
         for a, codes in attribute_to_irap_codes.items()})
    hard = ie.Predictions.from_class_indices(header, attribute_to_irap_codes,
                                             evaluation_set.segment_ids,
                                             ie.to_class_indices(probabilistic))
    return probabilistic, hard


@functools.cache
def compute_scoring_code_fingerprint(settings: ScoringSettings) -> str:
    """Computes a SHA-256 of the scores and resampled values (`resample_result`, with 20
    resamples) of random predictions on a small synthetic evaluation set, which changes if the
    code of scoring or resampling, e.g. of irap_evaluation or of NumPy's random generator, gives
    other values.

    The values are rounded to 9 significant digits, so that rounding differences, e.g. of
    another number of threads, do not change it. If they did, the scorings would only be scored
    again.
    """
    rng = np.random.default_rng(0)
    attributes = get_attrs_to_include()[:2]
    num_sequences, sequence_length = 6, 4
    segment_ids = tuple(f"{i:03d}" for i in range(num_sequences * sequence_length))
    vocabulary = ClassVocabulary({a: {f"{a} {code}": code for code in (1, 2, 3)}
                                  for a in attributes})
    class_indices = {a: rng.integers(0, 3, len(segment_ids)) for a in attributes}
    class_indices[attributes[1]][::5] = IGNORE_LABEL_INDEX
    evaluation_set = ie.EvaluationSet(
        dataset="vietnam", split="val", name=ie.REFERENCE_SET_NAME, context_offsets=(0,),
        vocabulary=vocabulary, segment_ids=segment_ids, class_indices=class_indices,
        segment_sequence_ids=tuple(f"sequence{i // sequence_length}"
                                   for i in range(len(segment_ids))))
    probe_settings = dc.replace(settings, num_resamples=20)
    outputs = []
    for predictions in _make_code_probe_predictions(evaluation_set):
        result = score_predictions(predictions, evaluation_set, probe_settings)
        outputs += [result.metrics, resample_result(result, probe_settings)]
    return compute_json_sha256([
        dc.asdict(ie.map_metric_values(
            lambda v: np.char.mod("%.9g", np.asarray(v, dtype=np.float64)).tolist(), values))
        for values in outputs])


def _combine_evaluation_set_fingerprints(evaluation_sets: T.Iterable[ie.EvaluationSet]) -> str:
    return compute_json_sha256([s.fingerprint for s in evaluation_sets])


def get_evaluation_sets_fingerprint(context: DatasetContext, split: str,
                                    context_offsets: T.Sequence[int] | None) -> str:
    """Computes a SHA-256 of the fingerprints of the evaluation sets that a submission can be
    scored on (see `DatasetContext.get_submission_evaluation_sets`).

    Raises:
        ValueError: If the dataset has no such split.
    """
    return _combine_evaluation_set_fingerprints(
        context.get_submission_evaluation_sets(split, context_offsets).values())


# Cached values ####################################################################################

@dc.dataclass(frozen=True)
class EvaluationSetScores:
    """The scores of a submission on one of its evaluation sets, over all scored attributes.

    Attributes:
        evaluation_set: The name of the set (`datasets.EVALUATION_SET_NAMES`).
        context_offsets: The context offsets that select the segments of the set.
        evaluation_set_fingerprint: See `irap_evaluation.EvaluationSet.fingerprint`.
        attributes: The scored attributes, in the order of the result.
        metrics: The default metrics (`irap_evaluation.get_default_metric_names`), which
            include the per-attribute value of each average.
        num_invalid: Attribute -> the number of labeled segments with an invalid prediction,
            including those of missing attributes.
        missing_attributes: See `irap_evaluation.EvaluationResult.missing_attributes`.
    """

    evaluation_set: str
    context_offsets: tuple[int, ...]
    evaluation_set_fingerprint: str
    attributes: tuple[str, ...]
    metrics: ie.MetricValues[float]
    num_invalid: T.Mapping[str, int]
    missing_attributes: tuple[str, ...]

    @classmethod
    def from_result(cls, result: ie.EvaluationResult) -> T.Self:
        evaluation_set = result.evaluation_set
        return cls(evaluation_set=evaluation_set.name,
                   context_offsets=evaluation_set.context_offsets,
                   evaluation_set_fingerprint=evaluation_set.fingerprint,
                   attributes=result.attributes, metrics=result.metrics,
                   num_invalid=dict(result.num_invalid),
                   missing_attributes=result.missing_attributes)


@dc.dataclass(frozen=True)
class SubmissionScoring:
    """The outcome of scoring a submission.

    Attributes:
        message: Why the scoring failed, empty if it did not.
        notes: The notes of `irap_evaluation.select_evaluation_sets`, e.g. why a set is left out.
        settings_fingerprint: `ScoringSettings.fingerprint` of the settings it was scored with.
        evaluation_sets_fingerprint: `get_evaluation_sets_fingerprint` when it was scored.
        evaluation_set_scores: The scores on each evaluation set, empty unless 'scored'.
    """

    submission_id: int
    status: ScoringStatus
    message: str
    notes: tuple[str, ...]
    settings_fingerprint: str
    evaluation_sets_fingerprint: str
    scored_at: datetime
    evaluation_set_scores: tuple[EvaluationSetScores, ...] = ()

    def get_scores(self, evaluation_set: str) -> EvaluationSetScores | None:
        """Returns the scores on the evaluation set with this name, None if it is not scored on
        it."""
        return next((s for s in self.evaluation_set_scores if s.evaluation_set == evaluation_set),
                    None)


def make_unscored_scoring(submission_id: int, status: T.Literal["failed", "error"],
                          message: str, settings: ScoringSettings,
                          evaluation_sets_fingerprint: str = "") -> SubmissionScoring:
    return SubmissionScoring(submission_id=submission_id, status=status, message=message,
                             notes=(), settings_fingerprint=settings.fingerprint,
                             evaluation_sets_fingerprint=evaluation_sets_fingerprint,
                             scored_at=get_utc_now())


def metric_values_to_json(values: ie.MetricValues[float]) -> str:
    """Encodes metric values as the JSON of the cache, which keeps NaN and infinite values,
    unlike standard JSON and `evaluation_report.to_json_dict`."""
    return json.dumps(dc.asdict(values))


def metric_values_from_json(text: str) -> ie.MetricValues[float]:
    """Decodes the JSON of `metric_values_to_json`."""
    values = json.loads(text)
    return ie.MetricValues(averages=values["averages"], per_attribute=values["per_attribute"])


def resamples_to_bytes(resamples: ie.MetricValues[np.ndarray]) -> bytes:
    """Encodes resampled values (`resample_result`) as a line of JSON with the metric names and
    attributes, followed by the (K, B) little-endian float32 values in their order."""
    names = {"averages": list(resamples.averages),
             "per_attribute": {n: list(a) for n, a in resamples.per_attribute.items()}}
    return (json.dumps(names).encode() + b"\n"
            + np.stack(list(resamples.iterate_values())).astype("<f4").tobytes())


def resamples_from_bytes(data: bytes) -> ie.MetricValues[np.ndarray]:
    """Decodes `resamples_to_bytes` to read-only float32 arrays."""
    header, values_data = data.split(b"\n", 1)
    names = json.loads(header)
    num_values = len(names["averages"]) + sum(len(a) for a in names["per_attribute"].values())
    values = iter(np.frombuffer(values_data, dtype="<f4").reshape(num_values, -1))
    return ie.MetricValues(
        averages={n: next(values) for n in names["averages"]},
        per_attribute={n: {a: next(values) for a in attributes}
                       for n, attributes in names["per_attribute"].items()})


def _to_evaluation_set_scores(row: sqlite3.Row) -> EvaluationSetScores:
    return EvaluationSetScores(
        evaluation_set=row["evaluation_set"],
        context_offsets=tuple(json.loads(row["context_offsets"])),
        evaluation_set_fingerprint=row["evaluation_set_fingerprint"],
        attributes=tuple(json.loads(row["attributes"])),
        metrics=metric_values_from_json(row["metrics"]),
        num_invalid=json.loads(row["num_invalid"]),
        missing_attributes=tuple(json.loads(row["missing_attributes"])))


def _to_scoring(row: sqlite3.Row,
                evaluation_set_scores: T.Iterable[EvaluationSetScores]) -> SubmissionScoring:
    return SubmissionScoring(
        submission_id=row["submission_id"], status=row["status"], message=row["message"],
        notes=tuple(json.loads(row["notes"])), settings_fingerprint=row["settings_fingerprint"],
        evaluation_sets_fingerprint=row["evaluation_sets_fingerprint"],
        scored_at=datetime.fromisoformat(row["scored_at"]),
        evaluation_set_scores=tuple(evaluation_set_scores))


#: The columns of `submission_scores` except the large report.
_SCORES_COLUMNS = ("submission_id, evaluation_set, context_offsets, evaluation_set_fingerprint,"
                   " attributes, metrics, num_invalid, missing_attributes")


class ScoreStore:
    """The cached scores of the submissions of the archive in a data directory, in their own
    database (`DATABASE_FILE_NAME`).

    A cache of another version than `SCORE_CACHE_VERSION` is created again, so that
    `scoring_worker.ScoringWorker.update_all_submissions` scores all files again.

    The scorings of deleted submissions are deleted by
    `delete_scorings_of_deleted_submissions`, since a foreign key cannot refer to the archive
    database.

    Raises:
        ValueError: If the tables of the cache differ from those of its version
            (`database.check_schema`), e.g. since `_SCHEMA` was changed without a new version.
    """

    def __init__(self, data_dir: Path):
        self.database_path = data_dir / DATABASE_FILE_NAME
        self.archive_database_path = data_dir / ARCHIVE_DATABASE_FILE_NAME
        if self.database_path.exists():
            # Reading the version also rolls back an unfinished write, whose journal file would
            # otherwise be applied to the new database.
            with connect(self.database_path) as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version != SCORE_CACHE_VERSION:
                self.database_path.unlink()
                _logger.warning("The score cache %s was of version %d, so it is created again"
                                " with version %d, and all files are scored again.",
                                self.database_path, version, SCORE_CACHE_VERSION)
        initialize_schema(self.database_path, _SCHEMA, SCORE_CACHE_VERSION, "score cache", {})

    def delete_scorings_of_deleted_submissions(self) -> None:
        """Deletes the scorings of the submissions that are no longer in the archive, with their
        scores.

        One statement reads the archive and deletes, so a scoring that is stored before a
        deletion of its submission is deleted either now or by a call after the deletion.
        """
        with connect(self.database_path) as connection:
            connection.execute("ATTACH DATABASE ? AS archive", (str(self.archive_database_path),))
            connection.execute("DELETE FROM submission_scorings WHERE submission_id NOT IN"
                               " (SELECT id FROM archive.submissions)")

    def get_scorings(self, submission_ids: T.Iterable[int]) -> dict[int, SubmissionScoring]:
        """Returns submission id -> its stored scoring, for those of `submission_ids` that have
        one."""
        where = "WHERE submission_id IN (SELECT value FROM json_each(?))"
        parameters = (json.dumps(list(submission_ids)),)
        with connect(self.database_path) as connection:
            # One read transaction, so that a scoring and its scores are of the same write.
            connection.execute("BEGIN")
            submission_id_to_scores: dict[int, list[EvaluationSetScores]] = {}
            for row in connection.execute(
                    f"SELECT {_SCORES_COLUMNS} FROM submission_scores {where}"
                    f" ORDER BY submission_id, evaluation_set DESC", parameters):
                submission_id_to_scores.setdefault(row["submission_id"], []).append(
                    _to_evaluation_set_scores(row))
            submission_id_to_scoring = {
                row["submission_id"]: _to_scoring(
                    row, submission_id_to_scores.get(row["submission_id"], ()))
                for row in connection.execute(f"SELECT * FROM submission_scorings {where}",
                                              parameters)}
            connection.execute("COMMIT")
            return submission_id_to_scoring

    def get_scoring(self, submission_id: int) -> SubmissionScoring | None:
        return self.get_scorings([submission_id]).get(submission_id)

    def set_scoring(self, scoring: SubmissionScoring,
                    reports: T.Mapping[str, T.Mapping[str, T.Any]],
                    resamples: T.Mapping[str, ie.MetricValues[np.ndarray]]) -> None:
        """Stores a scoring, replacing the previous one of the submission. If the submission has
        been deleted, e.g. while it was scored, the scoring is deleted again
        (`delete_scorings_of_deleted_submissions`).

        Args:
            reports: Evaluation set name -> its `evaluation_report.to_json_dict` document, for
                each of `scoring.evaluation_set_scores`.
            resamples: Evaluation set name -> its resampled values (`resample_result`), the same.
        """
        set_names = {s.evaluation_set for s in scoring.evaluation_set_scores}
        if set(reports) != set_names or set(resamples) != set_names:
            raise ValueError("Each evaluation set of the scores needs a report and resamples.")
        with begin_write(self.database_path) as connection:
            connection.execute("DELETE FROM submission_scores WHERE submission_id = ?",
                               (scoring.submission_id,))
            connection.execute(
                "INSERT OR REPLACE INTO submission_scorings (submission_id, status, message,"
                " notes, settings_fingerprint, evaluation_sets_fingerprint, scored_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (scoring.submission_id, scoring.status, scoring.message,
                 json.dumps(list(scoring.notes)), scoring.settings_fingerprint,
                 scoring.evaluation_sets_fingerprint, scoring.scored_at.isoformat()))
            for scores in scoring.evaluation_set_scores:
                connection.execute(
                    "INSERT INTO submission_scores (submission_id, evaluation_set,"
                    " context_offsets, evaluation_set_fingerprint, attributes, metrics,"
                    " num_invalid, missing_attributes, report, resamples)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (scoring.submission_id, scores.evaluation_set,
                     json.dumps(list(scores.context_offsets)),
                     scores.evaluation_set_fingerprint, json.dumps(list(scores.attributes)),
                     metric_values_to_json(scores.metrics), json.dumps(scores.num_invalid),
                     json.dumps(list(scores.missing_attributes)),
                     json.dumps(reports[scores.evaluation_set]),
                     resamples_to_bytes(resamples[scores.evaluation_set])))
        self.delete_scorings_of_deleted_submissions()

    def get_score_report(self, submission_id: int,
                         evaluation_set: str) -> dict[str, T.Any] | None:
        """Returns the `evaluation_report.to_json_dict` document of a submission on an evaluation
        set, None if it is not scored on it."""
        with connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT report FROM submission_scores WHERE submission_id = ? AND"
                " evaluation_set = ?", (submission_id, evaluation_set)).fetchone()
        return None if row is None else json.loads(row["report"])

    def get_resamples(self, submission_id: int, evaluation_set: str,
                      evaluation_set_fingerprint: str) -> ie.MetricValues[np.ndarray]:
        """Returns the resampled values (`resample_result`) of a submission on an evaluation set.

        Raises:
            LookupError: If they are not stored for the set with this fingerprint, e.g. since the
                submission has been deleted or scored again since its scores were read.
        """
        with connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT resamples FROM submission_scores WHERE submission_id = ? AND"
                " evaluation_set = ? AND evaluation_set_fingerprint = ?",
                (submission_id, evaluation_set, evaluation_set_fingerprint)).fetchone()
        if row is None:
            raise LookupError(f"File #{submission_id} has no resamples on the {evaluation_set}"
                              f" set that it was scored on.")
        return resamples_from_bytes(row["resamples"])


def remove_class_scores(report: T.Mapping[str, T.Any]) -> dict[str, T.Any]:
    """Returns a copy of a score report (`ScoreStore.get_score_report`) with its per-class scores
    and confusion matrices (`classes`) set to None, e.g. for a protected split
    (`config.DatasetConfig.is_protected_split`)."""
    return {**report, "classes": None}


def is_scoring_current(scoring: SubmissionScoring, submission: Submission,
                       context: DatasetContext, settings: ScoringSettings) -> bool:
    """Checks whether a scoring holds for the current settings and evaluation sets, so that the
    submission need not be scored again. An 'error' scoring is not current."""
    if scoring.status == "error" or scoring.settings_fingerprint != settings.fingerprint:
        return False
    try:
        sets_fingerprint = get_evaluation_sets_fingerprint(context, submission.split,
                                                           submission.context_offsets)
    except ValueError:  # The split is no longer in the metadata.
        return False
    return scoring.evaluation_sets_fingerprint == sets_fingerprint


def score_submission(
        archive: ModelArchive, context: DatasetContext, submission: Submission,
        settings: ScoringSettings
) -> tuple[SubmissionScoring, dict[str, dict[str, T.Any]], dict[str, ie.MetricValues[np.ndarray]]]:
    """Scores a stored submission on its evaluation sets.

    A `ValueError`, e.g. of a file that lacks segments of a new metadata build or of a split that
    is not in the metadata, gives a 'failed' scoring with its message. Other errors are raised.

    Returns:
        The scoring, the reports and the resampled values for `ScoreStore.set_scoring`.
    """
    scored_at = get_utc_now()
    try:
        sets_fingerprint = get_evaluation_sets_fingerprint(context, submission.split,
                                                           submission.context_offsets)
    except ValueError as e:  # The split is not in the metadata, so the scoring is not current.
        return make_unscored_scoring(submission.id, "failed", str(e), settings), {}, {}
    try:
        predictions = ie.read_predictions(archive.get_predictions_path(submission.id))
        evaluation_sets, notes = context.select_evaluation_sets(predictions)
        results = [score_predictions(predictions, s, settings) for s in evaluation_sets]
        resamples = {r.evaluation_set.name: resample_result(r, settings) for r in results}
    except ValueError as e:  # Also irap_evaluation.PredictionFormatError.
        return make_unscored_scoring(submission.id, "failed", str(e), settings,
                                     sets_fingerprint), {}, {}
    scoring = SubmissionScoring(
        submission_id=submission.id, status="scored", message="", notes=tuple(notes),
        settings_fingerprint=settings.fingerprint, evaluation_sets_fingerprint=sets_fingerprint,
        scored_at=scored_at,
        evaluation_set_scores=tuple(EvaluationSetScores.from_result(r) for r in results))
    return scoring, {r.evaluation_set.name: to_json_dict(r) for r in results}, resamples


# Methods ##########################################################################################

@dc.dataclass(frozen=True)
class ScoredModel:
    """A model with the scores of its submission of one split on one evaluation set."""

    submission: Submission
    scores: EvaluationSetScores


def get_model_evaluation_set(context: DatasetContext,
                             scored_model: ScoredModel) -> ie.EvaluationSet:
    """Returns the evaluation set that the scores of a model are of.

    Raises:
        ValueError: If the submission no longer has the set, or the set has changed since the
            scoring (its fingerprint differs).
    """
    submission, scores = scored_model.submission, scored_model.scores
    name_to_set = context.get_submission_evaluation_sets(submission.split,
                                                         submission.context_offsets)
    if scores.evaluation_set not in name_to_set:
        raise ValueError(f"File #{submission.id} of {submission.label} has no evaluation set"
                         f" {scores.evaluation_set!r}.")
    if name_to_set[scores.evaluation_set].fingerprint != scores.evaluation_set_fingerprint:
        raise ValueError(f"The {scores.evaluation_set} set of file #{submission.id} of"
                         f" {submission.label} has changed since it was scored.")
    return name_to_set[scores.evaluation_set]


@dc.dataclass(frozen=True)
class MethodScores:
    """The scores of a method on an evaluation set over a subset of the attributes: the means
    over its models (`irap_evaluation.compute_mean_metrics`).

    Attributes:
        models: In the order of their submission ids.
        metrics: None if there is an `error`.
        error: Why the means cannot be computed, e.g. the models differ in output kind or are not
            scored on an attribute of the subset, or None.
        missing_attributes: The attributes of the subset that some model does not predict,
            in the order of the subset.
    """

    method_name: str
    models: tuple[ScoredModel, ...]
    metrics: ie.MetricValues[float] | None
    error: str | None
    missing_attributes: tuple[str, ...]

    @property
    def num_models(self) -> int:
        return len(self.models)

    @property
    def shown_method_name(self) -> str:
        """The shown method name of the models (`archive.Model.shown_method_name`), which is the
        same for the models of a method, or `method_name` if there are no models."""
        return (self.models[0].submission.model.shown_method_name if self.models
                else self.method_name)


def select_scored_models(submissions: T.Iterable[Submission],
                         submission_id_to_scoring: T.Mapping[int, SubmissionScoring],
                         evaluation_set: str) -> list[ScoredModel]:
    """Selects the submissions that are scored on the evaluation set with this name, with those
    scores.

    Args:
        submissions: Of one split, so that the submissions of a method are its models.
        submission_id_to_scoring: The current scorings (see `is_scoring_current`).
    """
    models = []
    for submission in submissions:
        scoring = submission_id_to_scoring.get(submission.id)
        if scoring is not None and (scores := scoring.get_scores(evaluation_set)) is not None:
            models.append(ScoredModel(submission, scores))
    return models


def group_scored_models_by_method(
        models: T.Iterable[ScoredModel]) -> dict[str, list[ScoredModel]]:
    """Groups models by method name, in the order of their submission ids."""
    method_to_models: dict[str, list[ScoredModel]] = {}
    for model in sorted(models, key=lambda m: m.submission.id):
        method_to_models.setdefault(model.submission.model.method_name, []).append(model)
    return method_to_models


def are_scored_on_same_set(models: T.Iterable[ScoredModel]) -> bool:
    """Checks whether models are scored on the same evaluation set (the same fingerprint), so
    that their resampled values are paired: of the same resamples of the same road sequences."""
    return len({m.scores.evaluation_set_fingerprint for m in models}) <= 1


def get_combination_error(models: T.Sequence[ScoredModel]) -> str | None:
    """Returns why the models cannot be combined in means (`compute_method_scores`) or resampled
    values (`combine_method_resamples`), or None if they can."""
    if not models:
        return "At least one model is required."
    if not are_scored_on_same_set(models):
        offsets = ", ".join(f"{m.submission.label}: {m.scores.context_offsets}" for m in models)
        return (f"The models are scored on different segments (context offsets {offsets}), so"
                f" they cannot be combined.")
    if len({(m.scores.metrics.names, m.scores.attributes) for m in models}) > 1:
        return ("The models differ in metrics or attributes, e.g. hard and probabilistic"
                " predictions, so they cannot be combined.")
    return None


def compute_method_scores(method_name: str, models: T.Sequence[ScoredModel],
                          attributes: T.Sequence[str]) -> MethodScores:
    """Computes the means over the models of a method of their metrics over `attributes`.

    Errors of combining the models, and attributes that they are not scored on, are in
    `MethodScores.error`, so that a page can show them next to the other methods.

    Raises:
        ValueError: If `attributes` is empty.
    """
    if not attributes:
        raise ValueError("At least one attribute is required.")
    models = tuple(sorted(models, key=lambda m: m.submission.id))
    unpredicted = {a for m in models for a in m.scores.missing_attributes}
    error = get_combination_error(models)
    if error is None and (unscored := [a for a in attributes
                                       if a not in models[0].scores.attributes]):
        error = (f"The models are not scored on {', '.join(unscored)}, e.g. since their"
                 f" evaluation set has no labels of them.")
    metrics = None if error is not None else ie.compute_mean_metrics(
        [ie.select_metric_attributes(m.scores.metrics, attributes) for m in models])
    return MethodScores(method_name=method_name, models=models, metrics=metrics, error=error,
                        missing_attributes=tuple(a for a in attributes if a in unpredicted))


# Resampled values #################################################################################

def combine_method_resamples(method_name: str,
                             model_resamples: T.Sequence[ie.MetricValues[np.ndarray]],
                             attributes: T.Sequence[str], metric_names: T.Collection[str],
                             seed: int) -> ie.MetricValues[np.ndarray]:
    """Computes the resampled values of a method over `attributes` from those of its models
    (`ScoreStore.get_resamples`): their averages over the attributes
    (`irap_evaluation.select_metric_attributes`), combined with the variation between the models
    (`irap_evaluation.combine_model_resamples`). The values are float32, like the stored ones.

    Args:
        metric_names: The metrics to compute, e.g. those that a page shows, since all metrics of
            all attributes take tens of milliseconds per method. Those that the models lack are
            left out.
        seed: `ScoringSettings.bootstrap_seed`.

    Raises:
        ValueError: If `metric_names` has metrics that are not stored (`RESAMPLED_METRIC_NAMES`),
            `attributes` is empty or has attributes without values, or the models differ in
            metrics or attributes.
    """
    if unstored := [n for n in metric_names if n not in RESAMPLED_METRIC_NAMES]:
        raise ValueError(f"The resampled values of {unstored} are not stored.")
    # The averages need their per-attribute values.
    needed_names = ie.add_per_attribute_metric_names(list(metric_names))
    return _to_float32(ie.combine_model_resamples(
        [ie.select_metric_names(
            ie.select_metric_attributes(ie.select_metric_names(r, needed_names), attributes),
            metric_names)
         for r in model_resamples], method_name, seed=seed))
