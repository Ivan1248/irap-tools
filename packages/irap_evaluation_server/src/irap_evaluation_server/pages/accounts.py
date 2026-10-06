"""The Accounts page, on which admins change the permissions of the accounts and remove them (see
`accounts`)."""

import html
import typing as T

from nicegui import ui

from ..accounts import PERMISSION_DESCRIPTIONS, PERMISSIONS, Account, is_admin
from ..components.account_sessions import AccountSessions
from ..components.changes import run_change_in_thread
from ..components.confirmation_dialog import confirm_changes
from ..components.formatting import format_utc_time, make_table_html
from ..components.native_controls import set_status
from ..components.page_frame import create_page_frame
from ..components.routes import ACCOUNTS_PATH


def _make_account_row_html(account: Account) -> str:
    name = html.escape(account.name)
    options = "".join(f'<option value="{p}"{" selected" if p == account.permission else ""}>'
                      f"{p}</option>" for p in PERMISSIONS)
    return (f"<tr><td>{name}</td>"
            f'<td><select class="native-control" data-permission-of="{name}">{options}</select>'
            f"</td><td>{format_utc_time(account.registered_at)}</td>"
            f'<td><button class="native-control" type="button" data-remove="{name}">Remove'
            f"</button></td></tr>")


def _create_accounts_table(sessions: AccountSessions) -> None:
    """Creates the table of the accounts, in which an admin changes them. The store checks that
    the signed-in account is still an admin when a change runs."""
    accounts = sessions.accounts

    @ui.refreshable
    def show_table() -> None:
        # Plain HTML, which a render of the page does not change, so a select keeps the user's
        # choice until the table is refreshed after the change.
        ui.html(make_table_html(["Account", "Permission", "Registered", ""],
                                [_make_account_row_html(a) for a in accounts.list_accounts()]),
                sanitize=False).on(
            "change", lambda e: on_permission_changed(e.args["name"], e.args["permission"]),
            js_handler="(e) => { const select = e.target.closest('[data-permission-of]');"
                       " if (select) emit({name: select.dataset.permissionOf,"
                       " permission: select.value}); }").on(
            "click", lambda e: on_remove_clicked(e.args),
            js_handler="(e) => { const button = e.target.closest('[data-remove]');"
                       " if (button) emit(button.dataset.remove); }")

    async def run_change(change: T.Callable[[str], Account], done_text: str) -> None:
        """Runs a change of an account (`run_change_in_thread`) and shows the accounts as they are
        then, also after a refused change, or reloads the page if the user is no longer an
        admin."""
        is_done = await run_change_in_thread(change, sessions.require_account_name,
                                             status_label)
        if status_label.client.is_deleted:
            return
        if not is_admin(sessions.get_account()):
            ui.navigate.reload()
            return
        if is_done:
            set_status(status_label, done_text)
        show_table.refresh()

    async def on_permission_changed(name: str, permission: str) -> None:
        await run_change(lambda actor: accounts.set_permission(name, permission, actor),
                         f"{name} now has the permission {permission}.")

    async def on_remove_clicked(name: str) -> None:
        change = (f"Removes the account {name}, which signs it out. The action log keeps the"
                  f" name. Anyone can register the name again.")
        if await confirm_changes(f"Remove the account {name}", [change], "Remove"):
            await run_change(lambda actor: accounts.remove_account(name, actor),
                             f"Removed the account {name}.")

    show_table()
    status_label = ui.label().classes("muted")


def register_accounts_page(sessions: AccountSessions) -> None:
    @ui.page(ACCOUNTS_PATH, title="Accounts · iRAP evaluation")
    def accounts_page() -> None:
        account = sessions.get_account()
        with create_page_frame(ACCOUNTS_PATH, account), ui.element("section").classes("panel"):
            ui.label("Accounts").classes("section-title")
            if not is_admin(account):
                ui.label("Only admins can see the accounts.").classes("error")
                return
            with ui.element("div").classes("key-values"):
                for permission, description in PERMISSION_DESCRIPTIONS.items():
                    ui.label(permission).classes("key")
                    ui.label(description)
            ui.label("A change applies at once, but pages that are open show or hide their"
                     " controls and the action log only after a reload.").classes("muted")
            _create_accounts_table(sessions)
