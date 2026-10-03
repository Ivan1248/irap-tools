"""The requests of the intervals that a page shows."""

import asyncio
import typing as T

from nicegui import ui

from ..scoring import IntervalRequest
from ..scoring_worker import ScoringWorker
from .score_tables import IntervalState

K = T.TypeVar("K")


class IntervalRequester:
    """Requests the intervals that a page shows from the scoring worker, and withdraws those that
    it no longer shows, e.g. those of an attribute subset that the user has left, since they can
    take tens of seconds (`ScoringWorker.set_requested_intervals`).

    While the user changes the view, e.g. the subset, the requests wait until `delay_s` after the
    last change (`delay_requests`).
    """

    def __init__(self, worker: ScoringWorker, delay_s: float = 1.0):
        self._worker = worker
        self._delay_s = delay_s
        self._owner = ui.context.client.id
        self._delayed_call: asyncio.TimerHandle | None = None
        #: The requests of the latest `update` that are not computed.
        self._missing_requests: list[IntervalRequest] = []
        ui.context.client.on_delete(self._withdraw_requests)

    def update(self, key_to_request: T.Mapping[K, IntervalRequest]) -> dict[K, IntervalState]:
        """The state of the intervals of each request, and requests those that are not computed.

        Called on each render, with all the requests of the page.
        """
        key_to_state = {}
        self._missing_requests = []
        for key, request in key_to_request.items():
            cached = self._worker.get_intervals(request)
            key_to_state[key] = IntervalState(cached=cached, is_pending=cached is None)
            if cached is None:
                self._missing_requests.append(request)
        if self._delayed_call is None:  # Otherwise the delayed call sends them.
            self._send_requests()
        return key_to_state

    def delay_requests(self) -> None:
        """Delays the requests until `delay_s` after the last call. Call it before the render of
        a change of the view."""
        if self._delayed_call is not None:
            self._delayed_call.cancel()
        self._delayed_call = asyncio.get_running_loop().call_later(self._delay_s,
                                                                   self._send_requests)

    def _send_requests(self) -> None:
        self._delayed_call = None
        self._worker.set_requested_intervals(self._owner, self._missing_requests)

    def _withdraw_requests(self) -> None:
        if self._delayed_call is not None:
            self._delayed_call.cancel()
        self._worker.set_requested_intervals(self._owner, [])

    def get_pending_checks(self) -> list[T.Callable[[], bool]]:
        """For `RefreshTimer.watch`: one per interval of the latest `update` that is not
        computed, true until it is computed. The queue of the worker is checked first, so that
        the store is read only after the computation."""
        return [lambda r=r: (self._worker.is_intervals_pending(r)
                             or self._worker.get_intervals(r) is None)
                for r in self._missing_requests]
