"""The requests of the results that a page shows, e.g. intervals, from the scoring worker."""

import asyncio
import typing as T

from nicegui import ui

from ..scoring import ResultState, WorkRequest
from ..scoring_worker import ScoringWorker

K = T.TypeVar("K")


class WorkRequester:
    """Requests the results that a page shows from the scoring worker, and withdraws those that
    it no longer shows, e.g. those of an attribute subset that the user has left, since they can
    take tens of seconds (`ScoringWorker.set_requests`).

    While the user changes the view, e.g. the subset, the requests wait until `delay_s` after the
    last change (`delay_requests`).
    """

    def __init__(self, worker: ScoringWorker, delay_s: float = 1.0):
        self._worker = worker
        self._delay_s = delay_s
        self._owner = ui.context.client.id
        self._delayed_call: asyncio.TimerHandle | None = None
        #: The requests of the latest `update` that are not computed.
        self._missing_requests: list[WorkRequest] = []
        ui.context.client.on_delete(self._withdraw_requests)

    def update(self, key_to_request: T.Mapping[K, WorkRequest]) -> dict[K, ResultState]:
        """The state of the result of each request, and requests those that are not computed.

        Called on each render, with all the requests of the page.
        """
        key_to_state = {}
        self._missing_requests = []
        for key, request in key_to_request.items():
            cached = self._worker.get_result(request)
            key_to_state[key] = ResultState(cached=cached, is_pending=cached is None)
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
        self._worker.set_requests(self._owner, self._missing_requests)

    def _withdraw_requests(self) -> None:
        if self._delayed_call is not None:
            self._delayed_call.cancel()
        self._worker.set_requests(self._owner, [])

    def get_pending_checks(self) -> list[T.Callable[[], bool]]:
        """For `RefreshTimer.watch`: one per result of the latest `update` that is not computed,
        true until it is computed. The queue of the worker is checked first, so that the store is
        read only after the computation."""
        return [lambda r=r: self._worker.is_pending(r) or self._worker.get_result(r) is None
                for r in self._missing_requests]
