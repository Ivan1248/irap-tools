"""What the pages show, which admins choose: the shown metrics (`method_ranking.ShownMetrics`).

A metric is shown or hidden by default (`method_ranking.is_shown_by_default`). The database stores
the choices of admins, which override the defaults, so that a metric whose default changes keeps
the choice. Hiding a metric changes only the pages, not the scoring or the downloads, so a metric
that is shown again appears at once.

The settings are in `display_settings.sqlite3` in the data directory, apart from the archive and
the accounts.
"""

import logging
import sqlite3
from pathlib import Path

from .accounts import AccountStore, is_admin
from .database import Migration, begin_write, connect, initialize_schema
from .method_ranking import HIDEABLE_METRIC_NAMES, ShownMetrics

_logger = logging.getLogger(__name__)

DATABASE_FILE_NAME = "display_settings.sqlite3"
#: The version of the database schema (`PRAGMA user_version`). A database of an older version is
#: migrated (`_MIGRATIONS`).
SCHEMA_VERSION = 2


def _replace_hidden_metrics(connection: sqlite3.Connection) -> None:
    """Migrates version 1 to 2: replaces the table of the hidden metrics, of which all others were
    shown, by the choices of admins, in which the hidden metrics are hidden."""
    connection.execute("CREATE TABLE metric_choices (metric_name TEXT PRIMARY KEY,"
                       " is_shown INTEGER NOT NULL CHECK (is_shown IN (0, 1)))")
    connection.execute("INSERT INTO metric_choices (metric_name, is_shown)"
                       " SELECT metric_name, 0 FROM hidden_metrics")
    connection.execute("DROP TABLE hidden_metrics")


#: Version v -> the migration from v to v + 1 (`database.initialize_schema`).
_MIGRATIONS: dict[int, Migration] = {1: _replace_hidden_metrics}

_SCHEMA = """
CREATE TABLE metric_choices (
    metric_name TEXT PRIMARY KEY,
    is_shown INTEGER NOT NULL CHECK (is_shown IN (0, 1))
);
"""


def _read_metric_choices(connection: sqlite3.Connection) -> dict[str, bool]:
    return {row["metric_name"]: bool(row["is_shown"]) for row in
            connection.execute("SELECT metric_name, is_shown FROM metric_choices")}


class DisplaySettingsStore:
    """The display settings database of a data directory (see the module docstring), created on
    first use.

    Each operation opens its own database connection, so the store can be used from several
    threads.

    Raises:
        ValueError: If the database cannot be migrated to `SCHEMA_VERSION`, or its tables differ
            from those of its version (see `database.initialize_schema`).
    """

    def __init__(self, data_dir: Path, accounts: AccountStore):
        """Args:
            accounts: Whose admins can change the settings.
        """
        self.database_path = data_dir / DATABASE_FILE_NAME
        self.accounts = accounts
        initialize_schema(self.database_path, _SCHEMA, SCHEMA_VERSION, "display settings",
                          _MIGRATIONS)

    def get_metric_choices(self) -> dict[str, bool]:
        """Returns metric -> whether admins chose to show it, for the metrics that they chose
        to show or hide. Those that are not in `HIDEABLE_METRIC_NAMES`, e.g. stored before a
        library update, are ignored (`ShownMetrics.from_choices`)."""
        with connect(self.database_path) as connection:
            return _read_metric_choices(connection)

    def get_shown_metrics(self) -> ShownMetrics:
        """Returns the metrics that the pages show."""
        return ShownMetrics.from_choices(self.get_metric_choices())

    def set_metric_shown(self, metric_name: str, is_shown: bool, actor: str) -> dict[str, bool]:
        """Shows or hides an attribute aggregate, also if this is its default. Changing one metric
        at a time keeps the changes of other metrics, e.g. by another admin meanwhile.

        Args:
            actor: The name of the admin who changes it.

        Returns:
            The choices of the admins after the change (`get_metric_choices`).

        Raises:
            ValueError: If `actor` is not an admin, e.g. after another admin changed it, or the
                metric is not in `HIDEABLE_METRIC_NAMES`.
        """
        if metric_name not in HIDEABLE_METRIC_NAMES:
            raise ValueError(f"The metric {metric_name!r} cannot be hidden. Metrics:"
                             f" {', '.join(HIDEABLE_METRIC_NAMES)}.")
        with begin_write(self.database_path) as connection:
            # After the lock is taken, so that the check is of the permission at the write.
            if not is_admin(self.accounts.find_account(actor)):
                raise ValueError("Only admins can change the shown metrics.")
            connection.execute("INSERT OR REPLACE INTO metric_choices (metric_name, is_shown)"
                               " VALUES (?, ?)", (metric_name, int(is_shown)))
            choices = _read_metric_choices(connection)
        _logger.info("%s %s the metric %s.", actor, "showed" if is_shown else "hid", metric_name)
        return choices

    def warn_about_unknown_metrics(self) -> None:
        """Logs a warning about stored choices of metrics that are not in
        `HIDEABLE_METRIC_NAMES`, which have no effect, e.g. at the start of the server."""
        if unknown := sorted(set(self.get_metric_choices()) - set(HIDEABLE_METRIC_NAMES)):
            _logger.warning("The display settings %s have choices of the unknown metrics [%s],"
                            " which are ignored.", self.database_path, ", ".join(unknown))
