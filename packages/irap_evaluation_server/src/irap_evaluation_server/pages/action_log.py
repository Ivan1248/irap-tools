"""The Action log page: who uploaded, deleted or restored which submission, and when."""

from nicegui import ui

from ..archive import SubmissionArchive
from ..components.formatting import make_action_table_html
from ..components.page_frame import create_page_frame
from ..components.routes import ACTION_LOG_PATH


def register_action_log_page(archive: SubmissionArchive) -> None:
    @ui.page(ACTION_LOG_PATH, title="Action log · iRAP evaluation")
    def action_log_page() -> None:
        with create_page_frame(ACTION_LOG_PATH), ui.element("section").classes("panel"):
            ui.label("Action log").classes("section-title")
            if actions := archive.list_actions():
                ui.html(make_action_table_html(actions), sanitize=False).classes("w-full")
            else:
                ui.label("No actions.").classes("muted")
