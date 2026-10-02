"""The `irap-eval-server <config.toml>` command, which runs the web app.

`config.load_server_config` describes the configuration file.
"""

import argparse
import os
import secrets
import time
from pathlib import Path

from .archive import SubmissionArchive
from .config import load_server_config
from .datasets import load_dataset_contexts


def _load_storage_secret(data_dir: Path) -> str:
    """The key of NiceGUI's session cookies, created on first use, so that the per-browser
    storage (e.g. the submitter name) survives restarts."""
    path = data_dir / "storage_secret.txt"
    if not path.is_file():
        path.write_text(secrets.token_hex(32), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Runs the iRAP evaluation web app.")
    parser.add_argument("config", type=Path, help="The configuration file (TOML).")
    args = parser.parse_args(argv)
    config = load_server_config(args.config)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    # NiceGUI reads the path of its per-browser storage when it is imported.
    os.environ["NICEGUI_STORAGE_PATH"] = str(config.data_dir / "nicegui_storage")
    from nicegui import app, ui

    from .pages.action_log import register_action_log_page
    from .pages.submissions import register_submission_pages

    start_time_s = time.perf_counter()
    dataset_contexts = load_dataset_contexts(config.datasets)
    print(f"Loaded the metadata of {', '.join(dataset_contexts)} in"
          f" {time.perf_counter() - start_time_s:.1f} s.")
    archive = SubmissionArchive(config.data_dir)
    if num_removed := archive.remove_uploads():
        print(f"Removed {num_removed} unfinished uploads.")

    register_submission_pages(archive, dataset_contexts)
    register_action_log_page(archive)
    # No ripple effects on clicks, and no loading bar on requests.
    app.config.quasar_config["ripple"] = False
    app.config.quasar_config["loadingBar"]["skipHijack"] = True
    ui.run(host=config.host, port=config.port, title="iRAP evaluation", reload=False, show=False,
           storage_secret=_load_storage_secret(config.data_dir))


if __name__ == "__main__":
    main()
