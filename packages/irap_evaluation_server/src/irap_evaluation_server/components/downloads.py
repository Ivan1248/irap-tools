"""Responses of the endpoints that download files."""

import typing as T
from pathlib import Path

from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

PARQUET_MEDIA_TYPE = "application/vnd.apache.parquet"


def make_prepared_download_response(path: Path, write_file: T.Callable[[Path], None], *,
                                    file_name: str, media_type: str) -> FileResponse:
    """Writes a file for a download, and makes the response that sends it and then removes it.

    The file is also removed if writing fails.

    Args:
        path: A new path of the file, e.g. of `ModelArchive.make_download_path`, whose files
            `ModelArchive.remove_uploads` removes if the download is cut off.
        write_file: Writes the file to the path that it gets.
        file_name: The name of the file that the browser saves.
    """
    try:
        write_file(path)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return FileResponse(path, filename=file_name, media_type=media_type,
                        background=BackgroundTask(path.unlink))
