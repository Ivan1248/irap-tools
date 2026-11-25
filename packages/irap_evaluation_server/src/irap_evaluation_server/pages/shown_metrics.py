"""The Shown metrics page, on which admins show or hide attribute aggregates on the pages (see
`display_settings`)."""

import html

import irap_evaluation as ie
from nicegui import ui

from ..accounts import is_admin
from ..components.account_sessions import AccountSessions
from ..components.changes import run_admin_change
from ..components.native_controls import create_native_checkbox
from ..components.page_frame import create_page_frame
from ..components.routes import SHOWN_METRICS_PATH
from ..display_settings import DisplaySettingsStore
from ..method_ranking import HIDEABLE_METRIC_NAMES, RANKING_METRIC_NAMES, is_shown_by_default


def make_metric_label_html(metric_name: str) -> str:
    """Makes the name of an attribute aggregate with the description of its per-attribute
    metric, which the per-attribute view shows too, and whether it is shown by default."""
    parsed = ie.parse_metric_name(metric_name)
    description = ie.describe_metric(
        metric_name if parsed.attribute_aggregation == "tot" else parsed.name)
    if metric_name not in HIDEABLE_METRIC_NAMES:
        description += " The main metric, always shown."
    elif not is_shown_by_default(metric_name):
        description += " Hidden by default."
    return (f'<b>{html.escape(metric_name)}</b>'
            f' <span class="muted">{html.escape(description)}</span>')


def _create_metric_checkboxes(store: DisplaySettingsStore, sessions: AccountSessions) -> None:
    """Creates a checkbox per attribute aggregate, which shows or hides it at once. The store
    checks that the signed-in account is still an admin when a change runs."""
    @ui.refreshable
    def show_checkboxes() -> None:
        shown = store.get_shown_metrics().aggregate_names
        for name in RANKING_METRIC_NAMES:
            checkbox = create_native_checkbox(
                make_metric_label_html(name), name in shown,
                lambda is_shown, n=name: on_changed(n, is_shown), is_label_html=True)
            if name not in HIDEABLE_METRIC_NAMES:
                checkbox.props["disabled"] = True

    async def on_changed(metric_name: str, is_shown: bool) -> None:
        await run_admin_change(
            lambda actor: store.set_metric_shown(metric_name, is_shown, actor), sessions,
            status_label, f"{metric_name} is now {'shown' if is_shown else 'hidden'}. Open pages"
                          f" show the change after a reload.",
            show_checkboxes.refresh)

    with ui.element("div").classes("flex flex-col gap-1"):
        show_checkboxes()
    status_label = ui.label().classes("muted")


def register_shown_metrics_page(store: DisplaySettingsStore, sessions: AccountSessions) -> None:
    @ui.page(SHOWN_METRICS_PATH, title="Shown metrics · iRAP evaluation")
    def shown_metrics_page() -> None:
        account = sessions.get_account()
        with create_page_frame(SHOWN_METRICS_PATH, account), \
                ui.element("section").classes("panel"):
            ui.label("Shown metrics").classes("section-title")
            if not is_admin(account):
                ui.label("Only admins can change the shown metrics.").classes("error")
                return
            ui.label("The attribute averages and totals that the pages show, with their"
                     " per-attribute metrics, e.g. aBrier with Brier. A hidden metric is still"
                     " scored and in the downloaded scores, so showing it needs no"
                     " scoring.").classes("muted")
            _create_metric_checkboxes(store, sessions)
