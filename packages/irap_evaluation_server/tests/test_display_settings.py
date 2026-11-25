import contextlib
import logging
import sqlite3

import pytest

from irap_evaluation_server.accounts import AccountStore
from irap_evaluation_server.display_settings import (
    DATABASE_FILE_NAME,
    SCHEMA_VERSION,
    DisplaySettingsStore,
)
from irap_evaluation_server.method_ranking import (
    DEFAULT_HIDDEN_METRIC_NAMES,
    DEFAULT_SORT_METRIC,
    RANKING_METRIC_NAMES,
)
from irap_evaluation_server.pages.shown_metrics import make_metric_label_html

PASSWORD = "correct horse battery"


@pytest.fixture
def accounts(tmp_path):
    accounts = AccountStore(tmp_path)
    accounts.register("Al", PASSWORD)  # The admin.
    accounts.register("Bo", PASSWORD)
    accounts.set_permission("Bo", "write", "Al")
    return accounts


def test_show_and_hide_metrics(tmp_path, accounts):
    store = DisplaySettingsStore(tmp_path, accounts)
    assert store.get_metric_choices() == {}
    default_names = store.get_shown_metrics().aggregate_names
    assert not set(default_names) & set(DEFAULT_HIDDEN_METRIC_NAMES)
    assert store.set_metric_shown("aBrier", False, "Al") == {"aBrier": False}
    for _ in range(2):  # Showing a shown metric changes nothing.
        assert store.set_metric_shown("totnc_{R>0}", True, "Al") == {"aBrier": False,
                                                                     "totnc_{R>0}": True}
    # A choice equal to the default is kept, so that it holds if the default changes.
    assert store.set_metric_shown("aBrier", True, "Al") == {"aBrier": True, "totnc_{R>0}": True}
    assert store.set_metric_shown("aECE", False, "Al")["aECE"] is False
    # Opened again, e.g. after a restart.
    shown = DisplaySettingsStore(tmp_path, accounts).get_shown_metrics().aggregate_names
    assert shown == (*default_names, "totnc_{R>0}")


@pytest.mark.parametrize("metric_name, actor, message", [
    ("aBrier", "Bo", "Only admins"),
    ("aBrier", "Cid", "Only admins"),  # No such account.
    (DEFAULT_SORT_METRIC, "Al", "cannot be hidden"),
    ("aUnknown", "Al", "cannot be hidden"),
])
def test_set_metric_shown_refuses(tmp_path, accounts, metric_name, actor, message):
    store = DisplaySettingsStore(tmp_path, accounts)
    with pytest.raises(ValueError, match=message):
        store.set_metric_shown(metric_name, False, actor)
    assert store.get_metric_choices() == {}


def test_hidden_metrics_of_schema_version_1_stay_hidden(tmp_path, accounts, caplog):
    # The database of schema version 1, in which the metrics that were not hidden were shown.
    with contextlib.closing(sqlite3.connect(tmp_path / DATABASE_FILE_NAME)) as connection:
        connection.executescript(
            "CREATE TABLE hidden_metrics (metric_name TEXT PRIMARY KEY);"
            " INSERT INTO hidden_metrics VALUES ('amP'), ('aBrier'), ('aUnknown');"
            " PRAGMA user_version = 1;")
    with caplog.at_level(logging.WARNING):
        store = DisplaySettingsStore(tmp_path, accounts)
        store.warn_about_unknown_metrics()
    assert "from schema version 1 to 2" in caplog.text
    assert "unknown metrics [aUnknown]" in caplog.text
    assert store.get_metric_choices() == {"amP": False, "aBrier": False, "aUnknown": False}
    # The other metrics keep their defaults.
    assert store.get_shown_metrics().aggregate_names == tuple(
        n for n in RANKING_METRIC_NAMES
        if n not in ("amP", "aBrier", *DEFAULT_HIDDEN_METRIC_NAMES))
    with contextlib.closing(sqlite3.connect(tmp_path / DATABASE_FILE_NAME)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 2
    assert list(tmp_path.glob("display_settings.v1.*.backup.sqlite3"))


def test_metric_labels_escape_metric_names():
    label_html = make_metric_label_html("totnc_{R>0}")
    assert "totnc_{R&gt;0}" in label_html and "R>0" not in label_html
