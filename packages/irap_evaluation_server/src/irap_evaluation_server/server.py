"""The `irap-eval-server <config.toml>` command, which runs the web app.

`config.load_server_config` describes the configuration file.
"""

import argparse
import http.client
import os
import secrets
import time
from pathlib import Path

from .accounts import AccountStore
from .archive import ModelArchive
from .config import load_server_config
from .datasets import load_dataset_contexts
from .map_libraries import ensure_map_libraries
from .prediction_analysis import AlignedPredictionsCache
from .scoring import ScoreStore
from .scoring_worker import ScoringWorker

#: The length of the key of the session cookies, in hex digits (256 bits).
_STORAGE_SECRET_LENGTH = 64


def _load_storage_secret(data_dir: Path) -> str:
    """Loads the key of NiceGUI's session cookies, created on first use, so that the
    per-browser storage (e.g. the signed-in account) survives restarts.

    Raises:
        ValueError: If the stored key is shorter than `_STORAGE_SECRET_LENGTH`, e.g. empty after
            a crash, since cookies signed with it could be forged.
    """
    path = data_dir / "storage_secret.txt"
    if not path.is_file():
        # Readable only by the server's user, since the key signs the session cookies.
        file_descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            file.write(secrets.token_hex(_STORAGE_SECRET_LENGTH // 2))
    secret = path.read_text(encoding="utf-8").strip()
    if len(secret) < _STORAGE_SECRET_LENGTH:
        raise ValueError(f"{path} has a key of {len(secret)} characters, fewer than"
                         f" {_STORAGE_SECRET_LENGTH}. Delete it to make a new one, which signs"
                         f" out all users.")
    return secret


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Runs the iRAP evaluation web app.")
    parser.add_argument("config", type=Path, help="The configuration file (TOML).")
    args = parser.parse_args(argv)
    config = load_server_config(args.config)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    # NiceGUI reads the path of its per-browser storage when it is imported.
    os.environ["NICEGUI_STORAGE_PATH"] = str(config.data_dir / "nicegui_storage")
    from nicegui import app, ui

    from .components.account_sessions import AccountSessions
    from .components.request_checks import SameOriginMiddleware
    from .components.routes import REGISTER_PATH, VENDOR_PATH
    from .pages.accounts import register_accounts_page
    from .pages.action_log import register_action_log_page
    from .pages.analysis import register_analysis_page
    from .pages.ensemble import register_ensemble_page
    from .pages.model import register_model_page
    from .pages.models import register_models_pages
    from .pages.scores import register_scores_page
    from .pages.sign_in import register_sign_in_pages

    vendor_dir = config.data_dir / "vendor"
    try:
        ensure_map_libraries(vendor_dir)
        map_error = None
    except (OSError, ValueError, http.client.HTTPException) as e:
        print(f"The map libraries could not be downloaded into {vendor_dir}: {e} The Analysis"
              f" page shows no map.")
        # Without the path and the error, which visitors need not see.
        map_error = "The map libraries could not be downloaded when the server started."
    start_time_s = time.perf_counter()
    dataset_contexts = load_dataset_contexts(config.datasets)
    print(f"Loaded the metadata of {', '.join(dataset_contexts)} in"
          f" {time.perf_counter() - start_time_s:.1f} s.")
    archive = ModelArchive(config.data_dir)
    if num_removed := archive.remove_uploads():
        print(f"Removed {num_removed} unfinished uploads.")
    scoring_worker = ScoringWorker(archive, ScoreStore(archive.database_path), dataset_contexts)
    # Scores what is new or out of date, e.g. after a metadata update, in the background.
    scoring_worker.update_all_submissions()
    scoring_worker.start()
    app.on_shutdown(scoring_worker.stop)

    accounts = AccountStore(config.data_dir)
    if not accounts.has_accounts():
        print(f"There are no accounts yet. The first to register at {REGISTER_PATH} becomes the"
              f" admin.")
    sessions = AccountSessions(accounts)
    # The session cookie is also sent with the requests of pages on other hosts of the domain.
    app.add_middleware(SameOriginMiddleware)
    register_sign_in_pages(sessions)
    register_accounts_page(sessions)
    register_scores_page(dataset_contexts, scoring_worker, sessions)
    register_analysis_page(dataset_contexts, scoring_worker,
                           AlignedPredictionsCache(archive, scoring_worker.settings), map_error,
                           sessions)
    register_models_pages(archive, dataset_contexts, scoring_worker, sessions)
    register_model_page(archive, dataset_contexts, scoring_worker, sessions)
    register_ensemble_page(archive, dataset_contexts, scoring_worker, sessions)
    register_action_log_page(archive, sessions)
    vendor_dir.mkdir(exist_ok=True)
    app.add_static_files(VENDOR_PATH, vendor_dir)
    # No ripple effects on clicks, and no loading bar on requests.
    app.config.quasar_config["ripple"] = False
    app.config.quasar_config["loadingBar"]["skipHijack"] = True
    # No sad face on NiceGUI's error pages (`nicegui/error.py`, whose SVG has the id "Ebene_1").
    ui.add_css("div:has(> svg#Ebene_1) { display: none; }", shared=True)
    # Browsers refuse a cookie with the prefix __Host- from other hosts of the domain, e.g. one
    # that a page on another host sets before a user signs in, so that its browser would share
    # the session (session fixation). The prefix needs HTTPS.
    session_cookie = "__Host-session" if config.is_served_over_https else "session"
    ui.run(host=config.host, port=config.port, title="iRAP evaluation", favicon="🛣️",
           reload=False, show=False, storage_secret=_load_storage_secret(config.data_dir),
           session_middleware_kwargs={"https_only": config.is_served_over_https,
                                      "session_cookie": session_cookie})


if __name__ == "__main__":
    main()
