"""Connections to the SQLite database of the data directory, shared by the archive and the score
cache. Times are stored in UTC as ISO 8601 (`get_utc_now`)."""

import contextlib
import sqlite3
import typing as T
from datetime import UTC, datetime
from pathlib import Path


def get_utc_now() -> datetime:
    """Returns the current time in UTC, to whole seconds."""
    return datetime.now(UTC).replace(microsecond=0)


@contextlib.contextmanager
def connect(database_path: Path) -> T.Iterator[sqlite3.Connection]:
    """Opens a connection in autocommit mode, so that transactions are explicit (`begin_write`).

    Each operation opens its own connection, so that the database can be used from several
    threads.
    """
    connection = sqlite3.connect(database_path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()


@contextlib.contextmanager
def begin_write(database_path: Path) -> T.Iterator[sqlite3.Connection]:
    """Opens a write transaction, which is committed at the end of the block, or rolled back on
    an exception. It holds the database lock from the start, so that a check and the write that
    depends on it are not interleaved with another write."""
    with connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except BaseException:
            # SQLite has already rolled back after some errors, e.g. a full disk.
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
