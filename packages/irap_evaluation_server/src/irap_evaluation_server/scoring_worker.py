"""The background thread that scores submissions (see `scoring`).

Submissions are scored one at a time, in the order of queuing. Pages read the scores from the
store, and the resampled values through `ScoringWorker.get_method_resamples`, which keeps them in
memory.
"""

import functools
import logging
import threading
import typing as T

import irap_evaluation as ie
import numpy as np

from .archive import Model, ModelArchive, Submission
from .datasets import DatasetContext
from .scoring import (
    SCORING_SETTINGS,
    MethodScores,
    ScoreStore,
    ScoringSettings,
    SubmissionScoring,
    combine_method_resamples,
    is_scoring_current,
    make_unscored_scoring,
    score_submission,
)

_logger = logging.getLogger(__name__)

#: The number of resampled values of a model on an evaluation set (`ScoreStore.get_resamples`)
#: that the worker keeps in memory, about 1.7 MB each on iRAP-Vietnam.
NUM_CACHED_MODEL_RESAMPLES = 200
#: The number of resampled values of a method over a subset of the attributes
#: (`ScoringWorker.get_method_resamples`) that the worker keeps in memory, so that a page renders
#: again at once, e.g. after sorting. On iRAP-Vietnam, they take about 0.05 MB each for the
#: attribute averages, 0.14 MB for a per-attribute metric and 0.6 MB for the Model page.
NUM_CACHED_METHOD_RESAMPLES = 1000


def _format_unexpected_error(e: Exception) -> str:
    return f"Unexpected error ({type(e).__name__}): {e}"


class ScoringWorker:
    """Scores submissions in a background thread (`start`), or in the calling thread
    (`run_pending`, e.g. in tests).

    Pages read the scorings from the store and the resampled values of methods with
    `get_method_resamples`, and report changed submissions with `update_model_submissions`.
    """

    def __init__(self, archive: ModelArchive, store: ScoreStore,
                 dataset_contexts: T.Mapping[str, DatasetContext],
                 settings: ScoringSettings = SCORING_SETTINGS):
        self.archive = archive
        self.store = store
        self.dataset_contexts = dataset_contexts
        self.settings = settings
        self._condition = threading.Condition()
        #: The ids of the submissions queued for scoring, in the order of queuing (an ordered set).
        self._queued_ids: dict[int, None] = {}
        self._running_ids: set[int] = set()
        self._is_stopping = False
        self._thread: threading.Thread | None = None
        # The resampled values of a submission on a set with a fingerprint do not change while
        # the server runs, since the settings and the code are the same.
        self._get_model_resamples = functools.lru_cache(maxsize=NUM_CACHED_MODEL_RESAMPLES)(
            store.get_resamples)
        self._get_cached_method_resamples = functools.lru_cache(
            maxsize=NUM_CACHED_METHOD_RESAMPLES)(self._compute_method_resamples)

    # Queue ########################################################################################

    def _enqueue(self, submission_id: int) -> None:
        with self._condition:
            self._queued_ids.setdefault(submission_id)
            self._condition.notify_all()

    def _pop_queued_id(self, block: bool) -> int | None:
        """Pops the id of the next submission to score and marks it as running. Returns None if
        there is none (`block` False) or the worker is stopping (`block` True)."""
        with self._condition:
            while not (block and self._is_stopping):
                if self._queued_ids:
                    submission_id = next(iter(self._queued_ids))
                    del self._queued_ids[submission_id]
                    self._running_ids.add(submission_id)
                    return submission_id
                if not block:
                    return None
                self._condition.wait()
            return None

    def _score_queued(self, submission_id: int) -> None:
        try:
            self._score(submission_id)
        except Exception:
            # Scoring records expected errors itself. The thread keeps running.
            _logger.exception("Scoring submission #%d failed, and the failure is not stored.",
                              submission_id)
        finally:
            with self._condition:
                self._running_ids.discard(submission_id)
                self._condition.notify_all()

    def run_pending(self) -> None:
        """Scores the queued submissions in the calling thread."""
        while (submission_id := self._pop_queued_id(block=False)) is not None:
            self._score_queued(submission_id)

    def start(self) -> None:
        """Starts the background thread."""
        if self._thread is not None:
            raise RuntimeError("The worker is already started.")
        self._thread = threading.Thread(target=self._run_thread, name="scoring", daemon=True)
        self._thread.start()

    def _run_thread(self) -> None:
        while (submission_id := self._pop_queued_id(block=True)) is not None:
            self._score_queued(submission_id)

    def stop(self, timeout_s: float | None = None) -> None:
        """Stops the background thread after the current scoring."""
        with self._condition:
            self._is_stopping = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout_s)

    def wait_until_idle(self, timeout_s: float | None = None) -> bool:
        """Waits until no submission is queued or being scored. Returns False on timeout."""
        with self._condition:
            return self._condition.wait_for(
                lambda: not self._queued_ids and not self._running_ids, timeout_s)

    # Scoring ######################################################################################

    def _get_context(self, submission: Submission) -> DatasetContext | None:
        """Returns the dataset of a submission, None if the dataset is no longer configured (its
        submissions are not scored or shown)."""
        return self.dataset_contexts.get(submission.dataset)

    def is_scoring_current(self, submission: Submission,
                           scoring: SubmissionScoring | None) -> bool:
        """Checks whether `scoring`, the stored one of `submission`, is current (see
        `scoring.is_scoring_current`)."""
        context = self._get_context(submission)
        return (scoring is not None and context is not None
                and is_scoring_current(scoring, submission, context, self.settings))

    def select_current_scorings(
            self, submissions: T.Iterable[Submission],
            submission_id_to_scoring: T.Mapping[int, SubmissionScoring]
    ) -> dict[int, SubmissionScoring]:
        """Selects submission id -> its scoring, for those of `submissions` whose scoring is
        current (see `is_scoring_current`).

        Args:
            submission_id_to_scoring: The stored scorings (`ScoreStore.get_scorings`).
        """
        return {s.id: submission_id_to_scoring[s.id] for s in submissions
                if self.is_scoring_current(s, submission_id_to_scoring.get(s.id))}

    def get_current_scorings(self, submissions: T.Collection[Submission]
                             ) -> dict[int, SubmissionScoring]:
        """Loads the stored scorings of `submissions` and selects the current ones
        (`select_current_scorings`)."""
        return self.select_current_scorings(
            submissions, self.store.get_scorings(s.id for s in submissions))

    def load_split_scorings(self, dataset: str, split: str) -> tuple[
            list[Submission], dict[int, SubmissionScoring], dict[int, SubmissionScoring]]:
        """Loads the submissions of a dataset split, submission id -> its stored scoring
        (`ScoreStore.get_scorings`), and submission id -> its current scoring
        (`select_current_scorings`)."""
        submissions = [s for s in self.archive.list_submissions()
                       if (s.dataset, s.split) == (dataset, split)]
        submission_id_to_scoring = self.store.get_scorings(s.id for s in submissions)
        return (submissions, submission_id_to_scoring,
                self.select_current_scorings(submissions, submission_id_to_scoring))

    def is_scoring_pending(self, submission_id: int) -> bool:
        with self._condition:
            return submission_id in self._queued_ids or submission_id in self._running_ids

    def _score(self, submission_id: int) -> None:
        try:
            submission = self.archive.get_submission(submission_id)
        except LookupError:  # Deleted since it was queued.
            return
        context = self._get_context(submission)
        # Current if it was queued again while it was being scored.
        if context is None or self.is_scoring_current(submission,
                                                      self.store.get_scoring(submission_id)):
            return
        try:
            scoring, reports, resamples = score_submission(self.archive, context, submission,
                                                           self.settings)
        except Exception as e:
            _logger.exception("Scoring submission #%d failed.", submission_id)
            scoring, reports, resamples = make_unscored_scoring(
                submission_id, "error", _format_unexpected_error(e), self.settings), {}, {}
        self.store.set_scoring(scoring, reports, resamples)

    def update_submissions(self, submissions: T.Collection[Submission]) -> None:
        """Deletes the scorings of deleted submissions
        (`ScoreStore.delete_scorings_of_deleted_submissions`), and queues the scoring of the
        submissions whose scoring is not current, e.g. after they are added.

        Args:
            submissions: As they are in the archive after the change.
        """
        self.store.delete_scorings_of_deleted_submissions()
        current = self.get_current_scorings(submissions)
        for submission in submissions:
            if self._get_context(submission) is not None and submission.id not in current:
                self._enqueue(submission.id)

    def update_model_submissions(self, model: Model) -> None:
        """Calls `update_submissions` with all submissions of a model after a change, e.g. an
        upload that replaced some of them."""
        self.update_submissions(self.archive.list_model_submissions(model.id))

    def update_all_submissions(self) -> None:
        """Calls `update_submissions` with every submission, e.g. at the start of the
        server, since the metadata, the settings or the scoring code may have changed."""
        self.update_submissions(self.archive.list_submissions())

    # Resampled values #############################################################################

    def get_method_resamples(self, row: MethodScores, attributes: T.Sequence[str],
                             metric_names: T.Collection[str]
                             ) -> ie.MetricValues[np.ndarray] | None:
        """Returns the resampled values of a method over `attributes`
        (`scoring.combine_method_resamples`), or None if the method has an error, or the
        resampled values of a model are no longer stored, e.g. since it was deleted after its
        scores were read. The values are cached in memory, and must not be changed.

        Args:
            row: `scoring.compute_method_scores` over `attributes`.
            metric_names: See `scoring.combine_method_resamples`.
        """
        if row.error is not None:
            return None
        models = tuple((m.submission.id, m.scores.evaluation_set,
                        m.scores.evaluation_set_fingerprint) for m in row.models)
        return self._get_cached_method_resamples(row.method_name, models, tuple(attributes),
                                                 tuple(metric_names))

    def _compute_method_resamples(self, method_name: str, models: tuple[tuple[int, str, str], ...],
                                  attributes: tuple[str, ...], metric_names: tuple[str, ...]
                                  ) -> ie.MetricValues[np.ndarray] | None:
        """See `get_method_resamples`.

        Args:
            models: The arguments of `ScoreStore.get_resamples` of each model.
        """
        try:
            model_resamples = [self._get_model_resamples(*m) for m in models]
        except LookupError:
            return None
        return combine_method_resamples(method_name, model_resamples, attributes, metric_names,
                                        self.settings.bootstrap_seed)
