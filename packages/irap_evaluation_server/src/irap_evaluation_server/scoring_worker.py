"""The background thread that scores submissions and computes intervals (see `scoring`).

Tasks run one at a time, by priority: scoring first, then intervals that a page waits for, then
the intervals of the default views (all scored attributes), which are computed in advance so
that pages show them at once. A task can be queued at several priorities, e.g. default intervals
that a page waits for. It runs once, at the first of them, and withdrawing the request of a page
(`ScoringWorker.set_requested_intervals`) leaves it queued at the others.
"""

import enum
import logging
import threading
import typing as T

from .archive import Submission, SubmissionArchive
from .database import get_utc_now
from .datasets import EVALUATION_SET_NAMES, DatasetContext
from .scoring import (
    SCORING_SETTINGS,
    CachedIntervals,
    IntervalRequest,
    RunScoring,
    ScoreStore,
    ScoringSettings,
    compute_requested_intervals,
    get_combination_error,
    is_run_scoring_current,
    make_interval_request,
    make_unscored_run_scoring,
    score_submission,
    select_scored_runs,
)

_logger = logging.getLogger(__name__)

TaskKey = tuple[T.Any, ...]


class TaskPriority(enum.IntEnum):
    SCORING = 0
    REQUESTED_INTERVALS = 1
    DEFAULT_INTERVALS = 2


def _format_unexpected_error(e: Exception) -> str:
    return f"Unexpected error ({type(e).__name__}): {e}"


class ScoringWorker:
    """Scores submissions and computes intervals in a background thread (`start`), or in the
    calling thread (`run_pending`, e.g. in tests).

    Pages read the results from the store and ask for missing ones with `update_submissions` and
    `set_requested_intervals`.
    """

    def __init__(self, archive: SubmissionArchive, store: ScoreStore,
                 dataset_contexts: T.Mapping[str, DatasetContext],
                 settings: ScoringSettings = SCORING_SETTINGS):
        self.archive = archive
        self.store = store
        self.dataset_contexts = dataset_contexts
        self.settings = settings
        self._condition = threading.Condition()
        #: Priority -> the tasks queued at it, in the order of queuing.
        self._priority_to_tasks: dict[TaskPriority, dict[TaskKey, T.Callable[[], None]]] = {
            priority: {} for priority in TaskPriority}
        self._running_keys: set[TaskKey] = set()
        #: Interval key -> an unexpected error of computing them, which is not cached, so that
        #: the next start computes the intervals again.
        self._interval_key_to_error: dict[str, CachedIntervals] = {}
        #: Owner -> the keys of the intervals that it waits for (see `set_requested_intervals`).
        self._owner_to_interval_keys: dict[T.Hashable, set[str]] = {}
        self._is_stopping = False
        self._thread: threading.Thread | None = None

    # Queue ########################################################################################

    def _enqueue(self, key: TaskKey, priority: TaskPriority,
                 function: T.Callable[[], None]) -> None:
        with self._condition:
            self._priority_to_tasks[priority].setdefault(key, function)
            self._condition.notify_all()

    def _pop_task(self, block: bool) -> tuple[TaskKey, T.Callable[[], None]] | None:
        """The next task, marked as running, or None if there is none (`block` False) or the
        worker is stopping (`block` True)."""
        with self._condition:
            while not (block and self._is_stopping):
                for tasks in self._priority_to_tasks.values():
                    if tasks:
                        key, function = next(iter(tasks.items()))
                        for queued in self._priority_to_tasks.values():
                            queued.pop(key, None)
                        self._running_keys.add(key)
                        return key, function
                if not block:
                    return None
                self._condition.wait()
            return None

    def _run_task(self, key: TaskKey, function: T.Callable[[], None]) -> None:
        try:
            function()
        except Exception:
            # Tasks record expected errors themselves. The thread keeps running.
            _logger.exception("Task %s failed.", key)
        finally:
            with self._condition:
                self._running_keys.discard(key)
                self._condition.notify_all()

    def _is_task_pending(self, key: TaskKey) -> bool:
        with self._condition:
            return (key in self._running_keys
                    or any(key in tasks for tasks in self._priority_to_tasks.values()))

    def run_pending(self) -> None:
        """Runs the queued tasks, also those that they queue, in the calling thread."""
        while (task := self._pop_task(block=False)) is not None:
            self._run_task(*task)

    def start(self) -> None:
        """Starts the background thread."""
        if self._thread is not None:
            raise RuntimeError("The worker is already started.")
        self._thread = threading.Thread(target=self._run_thread, name="scoring", daemon=True)
        self._thread.start()

    def _run_thread(self) -> None:
        while (task := self._pop_task(block=True)) is not None:
            self._run_task(*task)

    def stop(self, timeout_s: float | None = None) -> None:
        """Stops the background thread after its current task."""
        with self._condition:
            self._is_stopping = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout_s)

    def wait_until_idle(self, timeout_s: float | None = None) -> bool:
        """Waits until no task is queued or running. Returns False on timeout."""
        with self._condition:
            return self._condition.wait_for(
                lambda: not any(self._priority_to_tasks.values()) and not self._running_keys,
                timeout_s)

    # Scoring ######################################################################################

    def _get_context(self, submission: Submission) -> DatasetContext | None:
        """None for a dataset that is no longer configured, whose submissions are not shown."""
        return self.dataset_contexts.get(submission.dataset)

    def is_scoring_current(self, submission: Submission, scoring: RunScoring | None) -> bool:
        """Whether `scoring`, the stored one of `submission`, is current (see
        `scoring.is_run_scoring_current`)."""
        context = self._get_context(submission)
        return (scoring is not None and context is not None
                and is_run_scoring_current(scoring, submission, context, self.settings))

    def select_current_scorings(
            self, submissions: T.Iterable[Submission],
            submission_id_to_scoring: T.Mapping[int, RunScoring]) -> dict[int, RunScoring]:
        """Submission id -> its scoring, for those of `submissions` whose scoring is current (see
        `is_scoring_current`).

        Args:
            submission_id_to_scoring: The stored scorings (`ScoreStore.get_run_scorings`).
        """
        return {s.id: submission_id_to_scoring[s.id] for s in submissions
                if self.is_scoring_current(s, submission_id_to_scoring.get(s.id))}

    def get_current_scorings(self, submissions: T.Collection[Submission] | None = None
                             ) -> dict[int, RunScoring]:
        """`select_current_scorings` of `submissions` with their stored scorings.

        Args:
            submissions: By default all submissions, also the deleted ones.
        """
        if submissions is None:
            submissions = self.archive.list_submissions(include_deleted=True)
        return self.select_current_scorings(
            submissions, self.store.get_run_scorings(s.id for s in submissions))

    def is_scoring_pending(self, submission_id: int) -> bool:
        return self._is_task_pending(("score", submission_id))

    def _score(self, submission_id: int) -> None:
        submission = self.archive.get_submission(submission_id)
        context = self._get_context(submission)
        if submission.is_deleted or context is None:
            return
        try:
            scoring, reports = score_submission(self.archive, context, submission, self.settings)
        except Exception as e:
            _logger.exception("Scoring submission #%d failed.", submission_id)
            scoring, reports = make_unscored_run_scoring(
                submission_id, "error", _format_unexpected_error(e), self.settings), {}
        self.store.set_run_scoring(scoring, reports)
        self._enqueue_default_intervals(submission.dataset, submission.split,
                                        submission.method.name)

    def update_submissions(self, submission_ids: T.Iterable[int]) -> None:
        """Queues what changes when submissions are added, deleted or restored: the scoring of
        those that are not current and not deleted, and the default intervals of their methods."""
        id_to_submission = {s.id: s for s in self.archive.list_submissions(include_deleted=True)}
        submissions = [id_to_submission[i] for i in submission_ids]
        current = self.get_current_scorings(submissions)
        for submission in submissions:
            if self._get_context(submission) is None:
                continue
            if not submission.is_deleted and submission.id not in current:
                self._enqueue(("score", submission.id), TaskPriority.SCORING,
                              lambda i=submission.id: self._score(i))
            else:  # Scoring queues them otherwise.
                self._enqueue_default_intervals(submission.dataset, submission.split,
                                                submission.method.name)

    def update_all_submissions(self) -> None:
        """`update_submissions` of every submission, e.g. at the start of the server, after the
        metadata or the settings have changed."""
        self.update_submissions(s.id for s in self.archive.list_submissions())

    # Intervals ####################################################################################

    def get_intervals(self, request: IntervalRequest) -> CachedIntervals | None:
        """The cached intervals, or an unexpected error of computing them, or None if they are
        not computed."""
        return (self._interval_key_to_error.get(request.key)
                or self.store.get_intervals(request.key))

    def is_intervals_pending(self, request: IntervalRequest) -> bool:
        return self._is_task_pending(("intervals", request.key))

    def _compute_intervals(self, request: IntervalRequest) -> None:
        if self.store.get_intervals(request.key) is not None:
            return
        try:
            intervals = compute_requested_intervals(request, self.archive, self.dataset_contexts,
                                                    self.settings)
            cached = CachedIntervals(intervals=intervals, message="", computed_at=get_utc_now())
        except ValueError as e:
            cached = CachedIntervals(intervals=None, message=str(e), computed_at=get_utc_now())
        except Exception as e:
            _logger.exception("Computing intervals of submissions %s failed.",
                              request.submission_ids)
            self._interval_key_to_error[request.key] = CachedIntervals(
                intervals=None, message=_format_unexpected_error(e), computed_at=get_utc_now())
            return
        self.store.set_intervals(request.key, cached)

    def request_intervals(self, request: IntervalRequest,
                          priority: TaskPriority = TaskPriority.REQUESTED_INTERVALS
                          ) -> CachedIntervals | None:
        """The cached intervals (see `get_intervals`), or None after queuing their computation."""
        if (cached := self.get_intervals(request)) is not None:
            return cached
        self._enqueue(("intervals", request.key), priority,
                      lambda: self._compute_intervals(request))
        return None

    def set_requested_intervals(self, owner: T.Hashable,
                                requests: T.Iterable[IntervalRequest]) -> None:
        """Requests the intervals that `owner`, e.g. a page, waits for, instead of those that it
        requested before.

        Earlier requests that no other owner waits for are withdrawn, e.g. those of an attribute
        subset that the user has left: they leave the queue, unless they are also queued
        otherwise, e.g. as default intervals. A running computation finishes.
        """
        key_to_request = {r.key: r for r in requests}
        with self._condition:
            earlier_keys = self._owner_to_interval_keys.pop(owner, set())
            if key_to_request:
                self._owner_to_interval_keys[owner] = set(key_to_request)
            held_keys = set().union(*self._owner_to_interval_keys.values())
            queued = self._priority_to_tasks[TaskPriority.REQUESTED_INTERVALS]
            for key in earlier_keys - held_keys:
                queued.pop(("intervals", key), None)
        for request in key_to_request.values():
            self.request_intervals(request)

    def _enqueue_default_intervals(self, dataset: str, split: str, method_name: str) -> None:
        self._enqueue(("default intervals", dataset, split, method_name),
                      TaskPriority.DEFAULT_INTERVALS,
                      lambda: self._request_default_intervals(dataset, split, method_name))

    def _request_default_intervals(self, dataset: str, split: str, method_name: str) -> None:
        """Queues the intervals over all scored attributes of the method and of each of its runs
        that are not deleted, on each evaluation set."""
        submissions = [s for s in self.archive.list_submissions()
                       if (s.dataset, s.split, s.method.name) == (dataset, split, method_name)]
        current = self.get_current_scorings(submissions)
        for evaluation_set in EVALUATION_SET_NAMES:
            runs = sorted(select_scored_runs(submissions, current, evaluation_set),
                          key=lambda r: r.submission.id)
            if not runs:
                continue
            attributes = runs[0].scores.attributes
            run_groups = [[run] for run in runs]
            if len(runs) > 1 and get_combination_error(runs) is None:
                run_groups.append(runs)
            for run_group in run_groups:
                self.request_intervals(make_interval_request(run_group, attributes, self.settings),
                                       TaskPriority.DEFAULT_INTERVALS)
