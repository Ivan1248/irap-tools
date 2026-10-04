"""The Action log page: all changes of the archive, with who made them and when."""

from nicegui import ui

from ..archive import ModelArchive
from ..components.formatting import make_action_table_html
from ..components.page_frame import create_page_frame
from ..components.routes import ACTION_LOG_PATH


def register_action_log_page(archive: ModelArchive) -> None:
    @ui.page(ACTION_LOG_PATH, title="Action log · iRAP evaluation")
    def action_log_page() -> None:
        with create_page_frame(ACTION_LOG_PATH), ui.element("section").classes("panel"):
            ui.label("Action log").classes("section-title")
            if actions := archive.list_actions():
                models = {m.id: m for m in archive.list_models(include_deleted=True)}
                ui.html(make_action_table_html(actions, models), sanitize=False).classes("w-full")
            else:
                ui.label("No actions.").classes("muted")
