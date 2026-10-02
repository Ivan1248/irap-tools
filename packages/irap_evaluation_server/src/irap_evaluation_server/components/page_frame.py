"""The frame of every page: the styles, the navigation and the submitter name."""

import contextlib
import typing as T
from pathlib import Path

from nicegui import app, ui

from .native_controls import create_native_input
from .routes import ACTION_LOG_PATH, SUBMISSIONS_PATH

#: (Title, path) of the pages in the navigation.
NAVIGATION_PAGES = (("Submissions", SUBMISSIONS_PATH), ("Action log", ACTION_LOG_PATH))

_STYLES = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
_SUBMITTER_NAME_KEY = "submitter_name"


def get_submitter_name() -> str:
    """The name that the user entered in the top bar, remembered per browser."""
    return app.storage.user.get(_SUBMITTER_NAME_KEY, "")


def _set_submitter_name(name: str) -> None:
    app.storage.user[_SUBMITTER_NAME_KEY] = name.strip()


@contextlib.contextmanager
def create_page_frame(current_path: str) -> T.Iterator[ui.element]:
    """The top bar and the content container, in which the page creates its content.

    Args:
        current_path: The path of the navigation entry to mark as current.
    """
    ui.add_css(_STYLES)
    ui.query(".nicegui-content").classes("p-0 gap-0")
    with ui.element("header").classes("top-bar w-full"):
        ui.label("iRAP evaluation").classes("title")
        with ui.element("nav"):
            for title, path in NAVIGATION_PAGES:
                ui.link(title, path).classes("current" if path == current_path else "")
        with ui.element("div").classes("submitter"):
            create_native_input("Your name", get_submitter_name(), _set_submitter_name,
                                placeholder="for the action log", size=18)
    with ui.element("main").classes("page-content") as content:
        yield content
