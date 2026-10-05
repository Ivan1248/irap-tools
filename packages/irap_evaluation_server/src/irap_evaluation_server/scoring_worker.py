"""The background thread that scores submissions and computes the results of requests, e.g.
intervals (see `scoring`).

Tasks run one at a time, by priority: scoring first, then requests that a page waits for, then
the intervals of the default views (all scored attributes), which are computed in advance so
that pages show them at once. A task can be queued at several priorities, e.g. default intervals
that a page waits for. It runs once, at the first of them, and withdrawing the request of a page
(`ScoringWorker.set_requests`) leaves it queued at the others.
"""

import enum
import logging
import threading
import typing as T

from .archive import Model, ModelArchive, Submission
from .database import get_utc_now
from .datasets import EVALUATION_SET_NAMES, DatasetContext
from .scoring import (
    SCORING_SETTINGS,
    CachedResult,
    ScoreStore,
    ScoringSettings,
    SubmissionScoring,
    WorkRequest,
    get_combination_error,
    is_scoring_current,
    make_interval_request,
    make_unscored_scoring,
    score_submission,
    select_scored_models,
)

_logger = logging.getLogger(__name__)

TaskKey = tuple[T.Any, ...]
V = T.TypeVar("V")


class TaskPriority(enum.IntEnum):
    SCORING = 0
    REQUESTED = 1
    DEFAULT_INTERVALS = 2


def _get_request_task_key(request: WorkRequest) -> TaskKey:
    return ("request", request.key)


def _format_unexpected_error(e: Exception) -> str:
    return f"Unexpected error ({type(e).__name__}): {e}"


class ScoringWorker:
    """Scores submissions and computes the results of requests in a background thread (`start`),
    or in the calling thread (`run_pending`, e.g. in tests).

    Pages read the scorings from the store and the results of requests with `get_result`, ask for
    missing results with `set_requests`, and report changed submissions with
    `update_model_submissions`.
    """

    def __init__(self, archive: ModelArchive, store: ScoreStore,
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
        #: Request key -> an unexpected error of computing its result, which is not cached, so
        #: that the next start computes the result again.
        self._request_key_to_error: dict[str, CachedResult] = {}
        #: Owner -> the task keys of the requests that it waits for (see `set_requests`).
        self._owner_to_task_keys: dict[T.Hashable, set[TaskKey]] = {}
        self._is_stopping = False
        self._thread: threading.Thread | None = None

    # Queue ########################################################################################

    def _enqueue(self, key: TaskKey, priority: TaskPriority,
                 function: T.Callable[[], None]) -> None:
        with self._condition:
            self._priority_to_tasks[priority].setdefault(key, function)
            self._condition.notify_all()

    def _pop_task(self, block: bool) -> tuple[TaskKey, T.Callable[[], None]] | None:
        """Pops the next task and marks it as running. Returns None if there is none (`block`
        False) or the worker is stopping (`block` True)."""
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
        """Loads the active submissions of a dataset split, submission id -> its stored scoring
        (`ScoreStore.get_scorings`), and submission id -> its current scoring
        (`select_current_scorings`)."""
        submissions = [s for s in self.archive.list_submissions()
                       if (s.dataset, s.split) == (dataset, split)]
        submission_id_to_scoring = self.store.get_scorings(s.id for s in submissions)
        return (submissions, submission_id_to_scoring,
                self.select_current_scorings(submissions, submission_id_to_scoring))

    def is_scoring_pending(self, submission_id: int) -> bool:
        return self._is_task_pending(("score", submission_id))

    def _score(self, submission_id: int) -> None:
        try:
            submission = self.archive.get_submission(submission_id)
        except LookupError:  # Deleted since it was queued.
            return
        context = self._get_context(submission)
        # Current if it was queued again while it was being scored.
        if (not submission.is_active or context is None
                or self.is_scoring_current(submission, self.store.get_scoring(submission_id))):
            return
        try:
            scoring, reports = score_submission(self.archive, context, submission, self.settings)
        except Exception as e:
            _logger.exception("Scoring submission #%d failed.", submission_id)
            scoring, reports = make_unscored_scoring(
                submission_id, "error", _format_unexpected_error(e), self.settings), {}
        # Raises `sqlite3.IntegrityError` if the submission was deleted meanwhile.
        self.store.set_scoring(scoring, reports)
        self._enqueue_default_intervals(submission.dataset, submission.split,
                                        submission.model.method_name)

    def update_submissions(self, submissions: T.Collection[Submission]) -> None:
        """Queues the work that follows when submissions become active or replaced, e.g. when
        they are added: the scoring of the active ones whose scoring is not current, and the
        default intervals of their methods.

        Args:
            submissions: As they are in the archive after the change.
        """
        current = self.get_current_scorings([s for s in submissions if s.is_active])
        for submission in submissions:
            if self._get_context(submission) is None:
                continue
            if submission.is_active and submission.id not in current:
                self._enqueue(("score", submission.id), TaskPriority.SCORING,
                              lambda i=submission.id: self._score(i))
            else:  # Scoring queues them otherwise.
                self._enqueue_default_intervals(submission.dataset, submission.split,
                                                submission.model.method_name)

    def update_model_submissions(self, model: Model) -> None:
        """Calls `update_submissions` with all submissions of a model after a change, e.g. an
        upload that replaced some of them, and queues the default intervals of its method on
        every split, also those of deleted submissions or a deleted model."""
        self.update_submissions(self.archive.list_model_submissions(model.id))
        if (context := self.dataset_contexts.get(model.dataset)) is not None:
            for split in context.split_names:
                self._enqueue_default_intervals(model.dataset, split, model.method_name)

    def update_all_submissions(self) -> None:
        """Calls `update_submissions` with every active submission, e.g. at the start of the
        server, since the metadata or the settings may have changed."""
        self.update_submissions(self.archive.list_submissions())

    # Requests #####################################################################################

    def get_result(self, request: WorkRequest[V]) -> CachedResult[V] | None:
        """Returns the cached result of a request, or an unexpected error of computing it, or
        None if it is not computed."""
        return self._request_key_to_error.get(request.key) or self.store.get_result(request)

    def is_pending(self, request: WorkRequest) -> bool:
        return self._is_task_pending(_get_request_task_key(request))

    def _compute(self, request: WorkRequest) -> None:
        if self.store.get_result(request) is not None:
            return
        try:
            value = request.compute(self.archive, self.dataset_contexts, self.settings)
            cached = CachedResult(value=value, message="", computed_at=get_utc_now())
        except ValueError as e:
            cached = CachedResult(value=None, message=str(e), computed_at=get_utc_now())
        except Exception as e:
            _logger.exception("Computing %s of submissions %s failed.", type(request).__name__,
                              request.submission_ids)
            self._request_key_to_error[request.key] = CachedResult(
                value=None, message=_format_unexpected_error(e), computed_at=get_utc_now())
            return
        # Raises `sqlite3.IntegrityError` if a submission was deleted meanwhile.
        self.store.set_result(request, cached)

    def request(self, request: WorkRequest[V],
                priority: TaskPriority = TaskPriority.REQUESTED) -> CachedResult[V] | None:
        """Returns the cached result (see `get_result`), or queues its computation and returns
        None."""
        if (cached := self.get_result(request)) is not None:
            return cached
        self._enqueue(_get_request_task_key(request), priority, lambda: self._compute(request))
        return None

    def set_requests(self, owner: T.Hashable, requests: T.Iterable[WorkRequest]) -> None:
        """Requests the results that `owner`, e.g. a page, waits for, instead of those that it
        requested before.

        Earlier requests that no other owner waits for are withdrawn, e.g. those of an attribute
        subset that the user has left: they leave the queue, unless they are also queued
        otherwise, e.g. as default intervals. A running computation finishes.
        """
        task_key_to_request = {_get_request_task_key(r): r for r in requests}
        with self._condition:
            earlier_keys = self._owner_to_task_keys.pop(owner, set())
            if task_key_to_request:
                self._owner_to_task_keys[owner] = set(task_key_to_request)
            held_keys = set().union(*self._owner_to_task_keys.values())
            queued = self._priority_to_tasks[TaskPriority.REQUESTED]
            for key in earlier_keys - held_keys:
                queued.pop(key, None)
        for request in task_key_to_request.values():
            self.request(request)

    def _enqueue_default_intervals(self, dataset: str, split: str, method_name: str) -> None:
        self._enqueue(("default intervals", dataset, split, method_name),
                      TaskPriority.DEFAULT_INTERVALS,
                      lambda: self._request_default_intervals(dataset, split, method_name))

    def _request_default_intervals(self, dataset: str, split: str, method_name: str) -> None:
        """Queues the intervals over all scored attributes of the method and of each of its
        models, of their active submissions of the split, on each evaluation set."""
        submissions = [s for s in self.archive.list_submissions()
                       if (s.dataset, s.split, s.model.method_name)
                       == (dataset, split, method_name)]
        current = self.get_current_scorings(submissions)
        for evaluation_set in EVALUATION_SET_NAMES:
            models = sorted(select_scored_models(submissions, current, evaluation_set),
                            key=lambda m: m.submission.id)
            if not models:
                continue
            model_groups = [[model] for model in models]
            # Models that can be combined have the same attributes.
            if len(models) > 1 and get_combination_error(models) is None:
                model_groups.append(models)
            for model_group in model_groups:
                self.request(make_interval_request(model_group, model_group[0].scores.attributes,
                                                   self.settings),
                             TaskPriority.DEFAULT_INTERVALS)
