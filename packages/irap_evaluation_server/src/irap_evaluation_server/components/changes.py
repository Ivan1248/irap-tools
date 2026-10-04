"""Running work that a user starts on a page off the event loop, e.g. a change of the archive or
of an account, with refusals shown in a status label."""

import functools
import typing as T

from nicegui import run, ui

from .native_controls import set_status

R = T.TypeVar("R")


def _show_error(status_label: ui.label, text: str) -> None:
    if not status_label.client.is_deleted:
        set_status(status_label, text, is_error=True)


async def run_in_thread_showing_errors(function: T.Callable[[], R],
                                       status_label: ui.label) -> R | None:
    """Runs `function` in a thread, e.g. since it may wait for a database lock or take seconds.
    Shows a refusal or an internal error in `status_label`.

    Args:
        function: Raises `ValueError` or `LookupError` if the work is refused.

    Returns:
        The result of `function`, or None if it was refused or the server is stopping.
    """
    try:
        return await run.io_bound(function)
    except (ValueError, LookupError) as e:
        _show_error(status_label, str(e))
        return None
    except Exception as e:
        _show_error(status_label, f"Internal error: {e!r}. See the server log.")
        raise


async def run_change_in_thread(change: T.Callable[[str], T.Any], get_actor: T.Callable[[], str],
                               status_label: ui.label) -> bool:
    """Runs a change with `run_in_thread_showing_errors`. Returns whether it was done.

    Args:
        change: Gets the actor. It returns something other than None, and raises `ValueError`
            or `LookupError` if the change is refused.
        get_actor: Returns the name of the signed-in account that may make the change, e.g.
            `AccountSessions.require_writer_name`, or raises `ValueError`. It is called on the
            event loop, since the user storage is not available in the thread of the change.
    """
    try:
        actor = get_actor()
    except ValueError as e:
        _show_error(status_label, str(e))
        return False
    return await run_in_thread_showing_errors(functools.partial(change, actor),
                                              status_label) is not None
