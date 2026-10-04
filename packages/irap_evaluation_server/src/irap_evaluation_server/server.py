"""The `irap-eval-server <config.toml>` command, which runs the web app.

`config.load_server_config` describes the configuration file.
"""

import argparse
import http.client
import os
import secrets
import time
from pathlib import Path

from .archive import ModelArchive
from .config import load_server_config
from .datasets import load_dataset_contexts
from .map_libraries import ensure_map_libraries
from .prediction_analysis import AlignedPredictionsCache
from .scoring import ScoreStore
from .scoring_worker import ScoringWorker


def _load_storage_secret(data_dir: Path) -> str:
    """Loads the key of NiceGUI's session cookies, created on first use, so that the
    per-browser storage (e.g. the user's name) survives restarts."""
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

    from .components.routes import VENDOR_PATH
    from .pages.action_log import register_action_log_page
    from .pages.analysis import register_analysis_page
    from .pages.ensemble import register_ensemble_page
    from .pages.model import register_model_page
    from .pages.models import register_models_pages
    from .pages.scores import register_scores_page

    vendor_dir = config.data_dir / "vendor"
    try:
        ensure_map_libraries(vendor_dir)
        map_error = None
    except (OSError, ValueError, http.client.HTTPException) as e:
        map_error = f"The map libraries could not be downloaded into {vendor_dir}: {e}"
        print(f"{map_error} The Analysis page shows no map.")
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

    register_scores_page(dataset_contexts, scoring_worker)
    register_analysis_page(dataset_contexts, scoring_worker,
                           AlignedPredictionsCache(archive, scoring_worker.settings), map_error)
    register_models_pages(archive, dataset_contexts, scoring_worker)
    register_model_page(archive, dataset_contexts, scoring_worker)
    register_ensemble_page(archive, dataset_contexts, scoring_worker)
    register_action_log_page(archive)
    vendor_dir.mkdir(exist_ok=True)
    app.add_static_files(VENDOR_PATH, vendor_dir)
    # No ripple effects on clicks, and no loading bar on requests.
    app.config.quasar_config["ripple"] = False
    app.config.quasar_config["loadingBar"]["skipHijack"] = True
    # No sad face on NiceGUI's error pages (`nicegui/error.py`, whose SVG has the id "Ebene_1").
    ui.add_css("div:has(> svg#Ebene_1) { display: none; }", shared=True)
    ui.run(host=config.host, port=config.port, title="iRAP evaluation", favicon="🛣️",
           reload=False, show=False, storage_secret=_load_storage_secret(config.data_dir))


if __name__ == "__main__":
    main()
