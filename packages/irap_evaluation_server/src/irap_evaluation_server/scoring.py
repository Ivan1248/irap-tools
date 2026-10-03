"""Scores of the submissions, computed with irap_evaluation and cached in the database.

A run is scored once, after its upload, on all its evaluation sets and all scored attributes
(`RunScoring`). Means over the runs of a method, also over a subset of the attributes, are
computed from these scores when they are shown (`compute_method_scores`). Bootstrap intervals
need the per-sequence statistics, which are too large to cache (about 11 MB per run on Vietnam
train), so they are computed from the prediction files and cached per set of runs, evaluation set
and subset of attributes (`IntervalRequest`, a `WorkRequest`).

Cached values carry fingerprints of what they depend on: the scoring settings
(`ScoringSettings.fingerprint`), the evaluation sets (`irap_evaluation.EvaluationSet.fingerprint`),
and the prediction files (their SHA-256). A cached value whose fingerprints differ from the
current ones is out of date.

`scoring_worker` computes the scores and intervals in the background.
"""

import dataclasses as dc
import functools
import hashlib
import json
import sqlite3
import typing as T
from datetime import datetime
from pathlib import Path

import irap_evaluation as ie
from irap_data.attrs import get_attrs_to_include
from irap_evaluation.reports.evaluation_report import to_json_dict

from .archive import Submission, SubmissionArchive
from .database import begin_write, connect, get_utc_now
from .datasets import DatasetContext

V = T.TypeVar("V")

#: 'scored': the scores are stored (none for a split without labels, see `RunScoring.notes`).
#: 'failed': the file cannot be scored, e.g. it lacks segments of a new metadata build. It is
#: scored again only when the settings or the evaluation sets change.
#: 'error': an unexpected error, e.g. a missing file. It is scored again at the next start.
RunScoringStatus = T.Literal["scored", "failed", "error"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_scorings (
    submission_id INTEGER PRIMARY KEY REFERENCES submissions (id),
    status TEXT NOT NULL,
    message TEXT NOT NULL,
    notes TEXT NOT NULL,
    settings_fingerprint TEXT NOT NULL,
    evaluation_sets_fingerprint TEXT NOT NULL,
    scored_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_scores (
    submission_id INTEGER NOT NULL REFERENCES submissions (id),
    evaluation_set TEXT NOT NULL,
    context_offsets TEXT NOT NULL,
    evaluation_set_fingerprint TEXT NOT NULL,
    attributes TEXT NOT NULL,
    metrics TEXT NOT NULL,
    num_invalid TEXT NOT NULL,
    missing_attributes TEXT NOT NULL,
    report TEXT NOT NULL,
    PRIMARY KEY (submission_id, evaluation_set)
);
CREATE TABLE IF NOT EXISTS result_cache (
    key TEXT PRIMARY KEY,
    value TEXT,
    message TEXT NOT NULL,
    computed_at TEXT NOT NULL
);
"""


def compute_json_sha256(value: T.Any) -> str:
    """A SHA-256 of a JSON-compatible value, independent of the order of dict keys, e.g. a
    fingerprint or a cache key."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# Settings and scoring #############################################################################

@dc.dataclass(frozen=True)
class ScoringSettings:
    """How the server scores runs and computes intervals: the defaults of `irap-eval evaluate`,
    except that missing attributes count as invalid cells, so that all runs are scored on
    the same attributes."""

    attribute_selection: ie.AttributeSelection = "canonical"
    null_policy: ie.NullPolicy = "first_class"
    missing_attribute_policy: ie.MissingAttributePolicy = "invalid"
    num_resamples: int = 1000
    confidence: float = 0.95
    bootstrap_seed: int = 0

    @functools.cached_property
    def fingerprint(self) -> str:
        """A SHA-256 of the settings and of what they select from the libraries, e.g. the
        canonical attributes and the default metrics, so that a library update that changes them
        makes cached values out of date."""
        return compute_json_sha256({
            **dc.asdict(self),
            "canonical_attributes": list(get_attrs_to_include()),
            "metric_names": {kind: list(ie.get_default_metric_names(kind))
                             for kind in T.get_args(ie.OutputKind)}})


SCORING_SETTINGS = ScoringSettings()


def select_scored_attributes(evaluation_set: ie.EvaluationSet,
                             settings: ScoringSettings) -> tuple[str, ...]:
    return ie.select_evaluated_attributes(evaluation_set, settings.attribute_selection)


def evaluate_run(predictions: ie.Predictions, evaluation_set: ie.EvaluationSet,
                 settings: ScoringSettings) -> ie.EvaluationResult:
    return ie.evaluate_predictions(
        predictions, evaluation_set, attributes=select_scored_attributes(evaluation_set, settings),
        null_policy=settings.null_policy,
        missing_attribute_policy=settings.missing_attribute_policy)


def _combine_evaluation_set_fingerprints(evaluation_sets: T.Iterable[ie.EvaluationSet]) -> str:
    return compute_json_sha256([s.fingerprint for s in evaluation_sets])


def get_evaluation_sets_fingerprint(context: DatasetContext, split: str,
                                    context_offsets: T.Sequence[int] | None) -> str:
    """A SHA-256 of the fingerprints of the evaluation sets that a run can be scored on (see
    `DatasetContext.get_run_evaluation_sets`).

    Raises:
        ValueError: If the dataset has no such split.
    """
    return _combine_evaluation_set_fingerprints(
        context.get_run_evaluation_sets(split, context_offsets).values())


# Cached values ####################################################################################

@dc.dataclass(frozen=True)
class EvaluationSetScores:
    """The scores of a run on one of its evaluation sets, over all scored attributes.

    Attributes:
        evaluation_set: The name of the set, 'reference' or 'model'.
        context_offsets: The context offsets that select the segments of the set.
        evaluation_set_fingerprint: See `irap_evaluation.EvaluationSet.fingerprint`.
        attributes: The scored attributes, in the order of the result.
        metrics: With the per-attribute value of each average (see
            `irap_evaluation.get_default_metric_names`).
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
class RunScoring:
    """The outcome of scoring a submission.

    Attributes:
        message: Why the scoring failed, empty if it did not.
        notes: The notes of `irap_evaluation.select_evaluation_sets`, e.g. why a set is left out.
        settings_fingerprint: `ScoringSettings.fingerprint` of the settings it was scored with.
        evaluation_sets_fingerprint: `get_evaluation_sets_fingerprint` when it was scored.
        evaluation_set_scores: The scores on each evaluation set, empty unless 'scored'.
    """

    submission_id: int
    status: RunScoringStatus
    message: str
    notes: tuple[str, ...]
    settings_fingerprint: str
    evaluation_sets_fingerprint: str
    scored_at: datetime
    evaluation_set_scores: tuple[EvaluationSetScores, ...] = ()

    def get_scores(self, evaluation_set: str) -> EvaluationSetScores | None:
        """The scores on the evaluation set with this name, None if it is not scored on it."""
        return next((s for s in self.evaluation_set_scores if s.evaluation_set == evaluation_set),
                    None)


def make_unscored_run_scoring(submission_id: int, status: T.Literal["failed", "error"],
                              message: str, settings: ScoringSettings,
                              evaluation_sets_fingerprint: str = "") -> RunScoring:
    return RunScoring(submission_id=submission_id, status=status, message=message, notes=(),
                      settings_fingerprint=settings.fingerprint,
                      evaluation_sets_fingerprint=evaluation_sets_fingerprint,
                      scored_at=get_utc_now())


@dc.dataclass(frozen=True)
class CachedResult(T.Generic[V]):
    """The result of a request (`WorkRequest`), or why it cannot be computed.

    Attributes:
        value: None if the computation failed.
        message: Why the computation failed, empty if it did not.
    """

    value: V | None
    message: str
    computed_at: datetime


@dc.dataclass(frozen=True)
class ResultState:
    """The result of a request (`WorkRequest`), e.g. the intervals of a row: computed or failed
    (`cached`), being computed, or neither."""

    cached: CachedResult | None = None
    is_pending: bool = False


class WorkRequest(T.Protocol[V]):
    """A computation that pages request from the scoring worker (`ScoringWorker.request`), whose
    result is cached in the database, e.g. intervals (`IntervalRequest`).

    Its fields are the fingerprints of what the result depends on, so `key` identifies the
    result in the cache.
    """

    @property
    def key(self) -> str:
        """A SHA-256 of the kind of request and its fields."""
        ...

    @property
    def submission_ids(self) -> tuple[int, ...]:
        """The submissions that the result is of, for log messages."""
        ...

    def compute(self, archive: SubmissionArchive,
                dataset_contexts: T.Mapping[str, DatasetContext],
                settings: ScoringSettings) -> V:
        """Computes the result from the prediction files.

        Raises:
            ValueError: If it cannot be computed, e.g. since the request is out of date. The
                message is cached instead of the result.
        """
        ...

    def value_to_json(self, value: V) -> str: ...

    def value_from_json(self, text: str) -> V: ...


def metric_values_to_json(values: ie.MetricValues) -> str:
    """The JSON of the cache, which keeps NaN and infinite values, unlike standard JSON and
    `evaluation_report.to_json_dict`. Dataclass values become JSON objects."""
    return json.dumps(dc.asdict(values))


def metric_values_from_json(text: str, convert: T.Callable[[T.Any], T.Any] | None = None
                            ) -> ie.MetricValues:
    """The values of `metric_values_to_json`.

    Args:
        convert: Gets each value, e.g. to make a dataclass of a JSON object.
    """
    values = json.loads(text)
    metric_values = ie.MetricValues(averages=values["averages"],
                                    per_attribute=values["per_attribute"])
    return metric_values if convert is None else ie.map_metric_values(convert, metric_values)


def _to_evaluation_set_scores(row: sqlite3.Row) -> EvaluationSetScores:
    return EvaluationSetScores(
        evaluation_set=row["evaluation_set"],
        context_offsets=tuple(json.loads(row["context_offsets"])),
        evaluation_set_fingerprint=row["evaluation_set_fingerprint"],
        attributes=tuple(json.loads(row["attributes"])),
        metrics=metric_values_from_json(row["metrics"]),
        num_invalid=json.loads(row["num_invalid"]),
        missing_attributes=tuple(json.loads(row["missing_attributes"])))


def _to_run_scoring(row: sqlite3.Row,
                    evaluation_set_scores: T.Iterable[EvaluationSetScores]) -> RunScoring:
    return RunScoring(
        submission_id=row["submission_id"], status=row["status"], message=row["message"],
        notes=tuple(json.loads(row["notes"])), settings_fingerprint=row["settings_fingerprint"],
        evaluation_sets_fingerprint=row["evaluation_sets_fingerprint"],
        scored_at=datetime.fromisoformat(row["scored_at"]),
        evaluation_set_scores=tuple(evaluation_set_scores))


#: The columns of `run_scores` but the large report.
_RUN_SCORES_COLUMNS = ("submission_id, evaluation_set, context_offsets, evaluation_set_fingerprint,"
                       " attributes, metrics, num_invalid, missing_attributes")


class ScoreStore:
    """The cached scores and intervals, in the database of the archive."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        with connect(database_path) as connection:
            connection.executescript(_SCHEMA)

    def get_run_scorings(self, submission_ids: T.Iterable[int]) -> dict[int, RunScoring]:
        """Submission id -> its scoring, for those of `submission_ids` that are scored."""
        where = "WHERE submission_id IN (SELECT value FROM json_each(?))"
        parameters = (json.dumps(list(submission_ids)),)
        with connect(self.database_path) as connection:
            submission_id_to_scores: dict[int, list[EvaluationSetScores]] = {}
            for row in connection.execute(
                    f"SELECT {_RUN_SCORES_COLUMNS} FROM run_scores {where}"
                    f" ORDER BY submission_id, evaluation_set DESC", parameters):
                submission_id_to_scores.setdefault(row["submission_id"], []).append(
                    _to_evaluation_set_scores(row))
            return {row["submission_id"]: _to_run_scoring(
                        row, submission_id_to_scores.get(row["submission_id"], ()))
                    for row in connection.execute(f"SELECT * FROM run_scorings {where}",
                                                  parameters)}

    def get_run_scoring(self, submission_id: int) -> RunScoring | None:
        return self.get_run_scorings([submission_id]).get(submission_id)

    def set_run_scoring(self, scoring: RunScoring,
                        reports: T.Mapping[str, T.Mapping[str, T.Any]]) -> None:
        """Stores a scoring, replacing the previous one of the submission.

        Args:
            reports: Evaluation set name -> its `evaluation_report.to_json_dict` document, for
                each of `scoring.evaluation_set_scores`.
        """
        if set(reports) != {s.evaluation_set for s in scoring.evaluation_set_scores}:
            raise ValueError("Each evaluation set of the scores needs a report.")
        with begin_write(self.database_path) as connection:
            connection.execute("DELETE FROM run_scores WHERE submission_id = ?",
                               (scoring.submission_id,))
            connection.execute(
                "INSERT OR REPLACE INTO run_scorings (submission_id, status, message, notes,"
                " settings_fingerprint, evaluation_sets_fingerprint, scored_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (scoring.submission_id, scoring.status, scoring.message,
                 json.dumps(list(scoring.notes)), scoring.settings_fingerprint,
                 scoring.evaluation_sets_fingerprint, scoring.scored_at.isoformat()))
            for scores in scoring.evaluation_set_scores:
                connection.execute(
                    "INSERT INTO run_scores (submission_id, evaluation_set, context_offsets,"
                    " evaluation_set_fingerprint, attributes, metrics, num_invalid,"
                    " missing_attributes, report) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (scoring.submission_id, scores.evaluation_set,
                     json.dumps(list(scores.context_offsets)),
                     scores.evaluation_set_fingerprint, json.dumps(list(scores.attributes)),
                     metric_values_to_json(scores.metrics), json.dumps(scores.num_invalid),
                     json.dumps(list(scores.missing_attributes)),
                     json.dumps(reports[scores.evaluation_set])))

    def get_score_report(self, submission_id: int,
                         evaluation_set: str) -> dict[str, T.Any] | None:
        """The `evaluation_report.to_json_dict` document of a run on an evaluation set, None if
        it is not scored on it."""
        with connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT report FROM run_scores WHERE submission_id = ? AND evaluation_set = ?",
                (submission_id, evaluation_set)).fetchone()
        return None if row is None else json.loads(row["report"])

    def get_result(self, request: WorkRequest[V]) -> CachedResult[V] | None:
        with connect(self.database_path) as connection:
            row = connection.execute("SELECT * FROM result_cache WHERE key = ?",
                                     (request.key,)).fetchone()
        if row is None:
            return None
        return CachedResult(
            value=None if row["value"] is None else request.value_from_json(row["value"]),
            message=row["message"], computed_at=datetime.fromisoformat(row["computed_at"]))

    def set_result(self, request: WorkRequest[V], cached: CachedResult[V]) -> None:
        with begin_write(self.database_path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO result_cache (key, value, message, computed_at)"
                " VALUES (?, ?, ?, ?)",
                (request.key,
                 None if cached.value is None else request.value_to_json(cached.value),
                 cached.message, cached.computed_at.isoformat()))


def is_run_scoring_current(scoring: RunScoring, submission: Submission, context: DatasetContext,
                           settings: ScoringSettings) -> bool:
    """Whether a scoring holds for the current settings and evaluation sets, so that it need not
    be scored again. An 'error' scoring is never current."""
    if scoring.status == "error" or scoring.settings_fingerprint != settings.fingerprint:
        return False
    try:
        sets_fingerprint = get_evaluation_sets_fingerprint(context, submission.split,
                                                           submission.context_offsets)
    except ValueError:  # The split is no longer in the metadata.
        return False
    return scoring.evaluation_sets_fingerprint == sets_fingerprint


def get_scored_evaluation_set(context: DatasetContext, submission: Submission,
                              evaluation_set: str, fingerprint: str) -> ie.EvaluationSet:
    """The evaluation set of a run with this name, which its stored scores are of.

    Raises:
        ValueError: If the run has no such set, or the set has changed since the scoring (its
            fingerprint differs).
    """
    name_to_set = context.get_run_evaluation_sets(submission.split, submission.context_offsets)
    if evaluation_set not in name_to_set:
        raise ValueError(f"Run {submission.run_label!r} has no evaluation set"
                         f" {evaluation_set!r}.")
    if name_to_set[evaluation_set].fingerprint != fingerprint:
        raise ValueError(f"The {evaluation_set} set of {submission.run_label!r} has changed since"
                         f" it was scored.")
    return name_to_set[evaluation_set]


def score_submission(archive: SubmissionArchive, context: DatasetContext, submission: Submission,
                     settings: ScoringSettings) -> tuple[RunScoring, dict[str, dict[str, T.Any]]]:
    """Scores a stored submission on its evaluation sets.

    A file that `irap_evaluation` refuses (a `ValueError`, e.g. one that lacks segments of a new
    metadata build) gives a 'failed' scoring with the message. Other errors are raised.

    Returns:
        The scoring and the reports for `ScoreStore.set_run_scoring`.
    """
    scored_at = get_utc_now()
    try:
        sets_fingerprint = get_evaluation_sets_fingerprint(context, submission.split,
                                                           submission.context_offsets)
    except ValueError as e:  # The split is not in the metadata, so it is never current.
        return make_unscored_run_scoring(submission.id, "failed", str(e), settings), {}
    try:
        predictions = ie.read_predictions(archive.get_predictions_path(submission.id))
        evaluation_sets, notes = context.select_evaluation_sets(predictions)
        results = [evaluate_run(predictions, s, settings) for s in evaluation_sets]
    except ValueError as e:  # Also irap_evaluation.PredictionFormatError.
        return make_unscored_run_scoring(submission.id, "failed", str(e), settings,
                                         sets_fingerprint), {}
    scoring = RunScoring(
        submission_id=submission.id, status="scored", message="", notes=tuple(notes),
        settings_fingerprint=settings.fingerprint, evaluation_sets_fingerprint=sets_fingerprint,
        scored_at=scored_at,
        evaluation_set_scores=tuple(EvaluationSetScores.from_result(r) for r in results))
    return scoring, {r.evaluation_set.name: to_json_dict(r) for r in results}


# Methods ##########################################################################################

@dc.dataclass(frozen=True)
class ScoredRun:
    """A submission with its scores on one evaluation set."""

    submission: Submission
    scores: EvaluationSetScores


def get_run_evaluation_set(context: DatasetContext, scored_run: ScoredRun) -> ie.EvaluationSet:
    """The evaluation set that the scores of a run are of.

    Raises:
        ValueError: See `get_scored_evaluation_set`.
    """
    return get_scored_evaluation_set(context, scored_run.submission,
                                     scored_run.scores.evaluation_set,
                                     scored_run.scores.evaluation_set_fingerprint)


@dc.dataclass(frozen=True)
class MethodScores:
    """The scores of a method on an evaluation set over a subset of the attributes: the means
    over its runs (`irap_evaluation.compute_mean_metrics`).

    Attributes:
        runs: In the order of their submission ids.
        metrics: None if the runs cannot be combined.
        error: Why the runs cannot be combined, e.g. they differ in output kind, or None.
        missing_attributes: The attributes of the subset that some run does not predict,
            in the order of the subset.
    """

    method_name: str
    runs: tuple[ScoredRun, ...]
    metrics: ie.MetricValues[float] | None
    error: str | None
    missing_attributes: tuple[str, ...]

    @property
    def num_runs(self) -> int:
        return len(self.runs)


def select_scored_runs(submissions: T.Iterable[Submission],
                       submission_id_to_scoring: T.Mapping[int, RunScoring],
                       evaluation_set: str) -> list[ScoredRun]:
    """The submissions that are scored on the evaluation set with this name, with those scores.

    Args:
        submission_id_to_scoring: The current scorings (see `is_run_scoring_current`).
    """
    runs = []
    for submission in submissions:
        scoring = submission_id_to_scoring.get(submission.id)
        if scoring is not None and (scores := scoring.get_scores(evaluation_set)) is not None:
            runs.append(ScoredRun(submission, scores))
    return runs


def group_scored_runs_by_method(runs: T.Iterable[ScoredRun]) -> dict[str, list[ScoredRun]]:
    """Method name -> its runs, in the order of their submission ids."""
    method_to_runs: dict[str, list[ScoredRun]] = {}
    for run in sorted(runs, key=lambda r: r.submission.id):
        method_to_runs.setdefault(run.submission.method.name, []).append(run)
    return method_to_runs


def get_combination_error(runs: T.Sequence[ScoredRun]) -> str | None:
    """Why the runs cannot be combined in means (`compute_method_scores`) or intervals
    (`make_interval_request`), or None if they can."""
    if not runs:
        return "At least one run is required."
    if len({r.scores.evaluation_set_fingerprint for r in runs}) > 1:
        offsets = ", ".join(f"{r.submission.run_label}: {r.scores.context_offsets}" for r in runs)
        return (f"The runs are scored on different segments (context offsets {offsets}), so they"
                f" cannot be combined.")
    if len({(r.scores.metrics.names, r.scores.attributes) for r in runs}) > 1:
        return ("The runs differ in metrics or attributes, e.g. hard and probabilistic"
                " predictions, so they cannot be combined.")
    return None


def check_attributes(attributes: T.Collection[str]) -> None:
    """Checks the attributes of means or a request.

    Raises:
        ValueError: If there is none.
    """
    if not attributes:
        raise ValueError("At least one attribute is required.")


def to_run_fingerprints(runs: T.Iterable[ScoredRun]) -> tuple[tuple[int, str], ...]:
    """The sorted (submission id, file SHA-256) of the runs, which identify their predictions in
    a request (see `WorkRequest`)."""
    return tuple(sorted((r.submission.id, r.submission.file_sha256) for r in runs))


def compute_method_scores(method_name: str, runs: T.Sequence[ScoredRun],
                          attributes: T.Sequence[str]) -> MethodScores:
    """The means over the runs of a method of their metrics over `attributes`.

    Errors of combining the runs, and attributes that they are not scored on, are in
    `MethodScores.error`, so that a page can show them next to the other methods.

    Raises:
        ValueError: If `attributes` is empty.
    """
    check_attributes(attributes)
    runs = tuple(sorted(runs, key=lambda r: r.submission.id))
    unpredicted = {a for r in runs for a in r.scores.missing_attributes}
    error = get_combination_error(runs)
    if error is None and (unscored := [a for a in attributes
                                       if a not in runs[0].scores.attributes]):
        error = (f"The runs are not scored on {', '.join(unscored)}, e.g. since their evaluation"
                 f" set has no labels of them.")
    metrics = None if error is not None else ie.compute_mean_metrics(
        [ie.select_metric_attributes(r.scores.metrics, attributes) for r in runs])
    return MethodScores(method_name=method_name, runs=runs, metrics=metrics, error=error,
                        missing_attributes=tuple(a for a in attributes if a in unpredicted))


# Intervals ########################################################################################

@dc.dataclass(frozen=True)
class IntervalRequest:
    """Bootstrap intervals of the metrics of a method (the means over its runs, or a single run)
    on an evaluation set, over a subset of the attributes.

    Its fields are the fingerprints of what the intervals depend on, so `key` identifies them in
    the cache.

    Attributes:
        runs: (submission id, file SHA-256) of each run, sorted.
        evaluation_set: The name of the set, 'reference' or 'model'.
        attributes: Sorted.
    """

    runs: tuple[tuple[int, str], ...]
    evaluation_set: str
    evaluation_set_fingerprint: str
    attributes: tuple[str, ...]
    settings_fingerprint: str

    @functools.cached_property
    def key(self) -> str:
        return compute_json_sha256({"kind": "intervals", **dc.asdict(self)})

    @property
    def submission_ids(self) -> tuple[int, ...]:
        return tuple(submission_id for submission_id, _ in self.runs)

    def compute(self, archive: SubmissionArchive,
                dataset_contexts: T.Mapping[str, DatasetContext],
                settings: ScoringSettings) -> ie.MetricValues[ie.BootstrapInterval]:
        """See `WorkRequest.compute`. The runs are scored again (`evaluate_runs`).

        Raises:
            ValueError: See `evaluate_runs`, and if `irap_evaluation.compute_bootstrap_intervals`
                refuses the runs.
        """
        return ie.compute_bootstrap_intervals(
            self.evaluate_runs(archive, dataset_contexts, settings),
            num_resamples=settings.num_resamples, confidence=settings.confidence,
            seed=settings.bootstrap_seed)

    def evaluate_runs(self, archive: SubmissionArchive,
                      dataset_contexts: T.Mapping[str, DatasetContext],
                      settings: ScoringSettings) -> list[ie.EvaluationResult]:
        """Scores the runs again from their files, restricted to `attributes`, since their
        per-sequence statistics are not cached.

        Raises:
            ValueError: If the request is out of date: other settings, files or evaluation set.
        """
        if self.settings_fingerprint != settings.fingerprint:
            raise ValueError("The request is of other scoring settings.")
        results = []
        for submission_id, file_sha256 in self.runs:
            submission = archive.get_submission(submission_id)
            if submission.file_sha256 != file_sha256:
                raise ValueError(f"Submission #{submission_id} has another file than requested.")
            scored_set = get_scored_evaluation_set(
                dataset_contexts[submission.dataset], submission, self.evaluation_set,
                self.evaluation_set_fingerprint)
            predictions = ie.read_predictions(archive.get_predictions_path(submission_id))
            results.append(ie.select_result_attributes(
                evaluate_run(predictions, scored_set, settings), self.attributes))
        return results

    def value_to_json(self, value: ie.MetricValues[ie.BootstrapInterval]) -> str:
        return metric_values_to_json(value)

    def value_from_json(self, text: str) -> ie.MetricValues[ie.BootstrapInterval]:
        return metric_values_from_json(text, lambda d: ie.BootstrapInterval(**d))


def make_interval_request(runs: T.Sequence[ScoredRun], attributes: T.Collection[str],
                          settings: ScoringSettings) -> IntervalRequest:
    """
    Raises:
        ValueError: If the runs cannot be combined (`get_combination_error`), or `attributes` is
            empty.
    """
    if (error := get_combination_error(runs)) is not None:
        raise ValueError(error)
    check_attributes(attributes)
    return IntervalRequest(
        runs=to_run_fingerprints(runs), evaluation_set=runs[0].scores.evaluation_set,
        evaluation_set_fingerprint=runs[0].scores.evaluation_set_fingerprint,
        attributes=tuple(sorted(attributes)), settings_fingerprint=settings.fingerprint)
