"""Connections to the SQLite databases of the data directory: that of the archive, which the score
cache shares, and that of the accounts. Times are stored in UTC as ISO 8601 (`get_utc_now`)."""

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
    """Opens a connection in autocommit mode, so that transactions are explicit (`begin_write`),
    with foreign keys enforced, e.g. so that deleting a submission deletes its cached scores.

    Each operation opens its own connection, so that the database can be used from several
    threads.
    """
    connection = sqlite3.connect(database_path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
    finally:
        connection.close()


def initialize_schema(database_path: Path, schema: str, schema_version: int,
                      database_name: str) -> None:
    """Creates the tables of a new database, with `schema_version` as its `PRAGMA user_version`.

    Args:
        schema: The SQL statements that create the tables.
        database_name: What the database holds, for the error message, e.g. 'archive'.

    Raises:
        ValueError: If the database has another schema version, since there is no migration.
    """
    with connect(database_path) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version == schema_version:
            return
        num_tables = connection.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0]
        if version != 0 or num_tables:
            raise ValueError(f"The database {database_path} is of another version of the"
                             f" {database_name} ({version}, not {schema_version}). Use a new"
                             f" data directory.")
        connection.executescript(f"BEGIN; {schema} PRAGMA user_version = {schema_version};"
                                 f" COMMIT;")


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
