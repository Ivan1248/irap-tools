# iRAP evaluation server

An internal web app for saved iRAP model predictions, built with [NiceGUI](https://nicegui.io) on [`irap_evaluation`](../irap_evaluation/):

- an archive of submissions (prediction files) with an action log,
- (planned) scoring and a leaderboard, method comparison, prediction analysis with a map, ensembling and coding-table export.

There are no logins. Anyone who can open the app can upload, delete and restore submissions. Each user enters a name in the top bar, which the action log records. Deleted submissions are kept and can be restored.

## Installation and start

From the irap-tools checkout:

```bash
uv pip install -e packages/irap_data -e packages/irap_evaluation -e packages/irap_evaluation_server
irap-eval-server server.toml
```

`server.toml` (relative paths are relative to its directory):

```toml
data_dir = "data"       # the archive
host = "127.0.0.1"      # optional, "0.0.0.0" to serve the lab network
port = 8600             # optional

[datasets.vietnam]      # a key of irap_data.DATASET_PRESETS
metadata_dir = "/data/IRAP_Vietnam"
images_dir = "/data/IRAP_Vietnam_images"   # optional
analysis_splits = ["train", "val"]          # splits with the detailed analysis (planned)
```

## Submissions

A submission is one prediction file (see the [format](../irap_evaluation/docs/prediction_format.md)), that is one run of a method on one split. Its header gives the dataset, split, method name, seed and context offsets. A file is refused if:

- the file is not a valid prediction file,
- its dataset is not configured, or its attributes or iRAP codes differ from the dataset's,
- it lacks segments of its evaluation sets ([evaluation sets](../irap_evaluation/docs/evaluation.md#evaluation-sets)),
- it is not distinct from the runs of its method on that split that are not deleted (`irap_evaluation.check_runs_distinct`). The Seed field of the upload form replaces the seed in the header, and the action log records the original one.

Files of unlabeled splits are accepted but not scored.

A `.zip` archive uploads the files of one run on several splits at once. Each file in it is checked as above, and in addition:

- every member that is not a directory must be a `.parquet` prediction file (other files are refused, not skipped),
- all files must have the same dataset, method name, seed and context offsets, and distinct splits,
- the Seed field replaces the seed in all of them.

Either all files of an archive are stored, each as its own submission, or none. The action log records the archive name and the name of each file in it.

The data directory holds `archive.sqlite3` (the index and the action log), `submissions/<id>/predictions.parquet` and NiceGUI's per-browser storage.

## Layout

```
irap_evaluation_server/
├─ config.py        ServerConfig, DatasetConfig, load_server_config
├─ datasets.py      DatasetContext: the metadata of a dataset and its evaluation sets
├─ archive.py       SubmissionArchive: files, SQLite index and action log, upload checks
├─ components/      Page frame, native form controls, CSS, HTML fragments
├─ pages/           One module per page or group of pages
└─ server.py        The irap-eval-server command
```

The logic modules (`config`, `datasets`, `archive`) do not import NiceGUI and are tested with pytest:

```bash
python -m pytest packages/irap_evaluation_server/tests
```
