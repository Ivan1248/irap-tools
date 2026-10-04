# iRAP evaluation server

An web app that keeps an archive of model predictions, scores them, and compares, analyses and ensembles them. It is built with [NiceGUI](https://nicegui.io) on [`irap_evaluation`](../irap_evaluation/).

| Page | Path | Content |
|---|---|---|
| Scores | `/scores` (start page) | The methods of a dataset split ranked by an attribute average, or one per-attribute metric, with intervals and differences from a reference method. |
| Models | `/models` | The models of each dataset by method, with the active file of each split, and the upload form. |
| Model | `/models/<id>` | A model's files, also the deleted ones, its scores, the coding-table export and its actions. |
| Analysis | `/analysis` | A model's confusion matrix for one attribute, a map of its outcomes, and segment details, optionally compared with a second model. |
| Ensemble | `/ensemble` | Creates an ensemble of stored models. |
| Action log | `/actions` | All changes, with the name of the user. |
| Accounts | `/accounts` | The accounts, whose permissions admins change. Only for admins. |

Anyone who can open the app can view it and download files. Only signed-in [accounts](#accounts) with write permission can upload, replace and delete files, delete and restore models, edit descriptions and create ensembles. The action log shows the account name.

## Installation and start

From the irap-tools checkout, install the packages and start the server with a [configuration file](#configuration):

```bash
uv pip install -e packages/irap_data -e packages/irap_evaluation -e packages/irap_evaluation_server
irap-eval-server server.toml
```

The map of the Analysis page needs internet access:

- On its first start, the server downloads MapLibre and deck.gl (pinned versions checked by SHA-256) into `data_dir/vendor/`, since MapLibre must be served by the server itself. Without internet access, it starts without the map.
- The browser loads the basemap tiles from Carto.

## Configuration

`server.toml` (relative paths are relative to its directory):

```toml
data_dir = "data"       # the archive
host = "127.0.0.1"      # optional, "0.0.0.0" to serve the lab network
port = 8600             # optional
is_served_over_https = false   # optional, true behind an HTTPS proxy: the session cookie is then sent only over HTTPS

[datasets.vietnam]      # a key of irap_data.DATASET_PRESETS
dataset_dir = "/data/IRAP_Vietnam"   # the images (FRAMES/) and the metadata
analysis_splits = ["train", "val"]   # the splits of the Analysis page, e.g. without test to keep its labels unseen

[datasets.bh]
dataset_dir = "/data/IRAP_BIH"
metadata_dir = "/data/IRAP_BIH_METADATA"   # optional, default dataset_dir
analysis_splits = ["train", "val"]
```

The image paths in the metadata (`segment_id_to_data_paths_rel.json`) are relative to `dataset_dir`. A Vietnam dataset directory, as [`irap_vietnam_data_preparation`](../irap_vietnam_data_preparation/README.md#directory-layout) builds it, also holds the metadata. Without the images, e.g. on a server with only the metadata files, the Analysis page shows no images.

The data directory holds:

- `archive.sqlite3`: the index of models and files, the action log and the score cache,
- `submissions/<id>/predictions.parquet`: the stored files,
- `uploads/`: uploaded files that are not stored yet, removed at start, and when another upload begins if they are older than 24 h,
- `vendor/`: the map libraries,
- `accounts.sqlite3`: the [accounts](#accounts),
- `nicegui_storage/` and `storage_secret.txt`: NiceGUI's per-browser storage, e.g. of the signed-in account, and the key of its session cookie.

The databases have a schema version and no migrations: the server refuses a data directory of another version, so start with a new `data_dir`.

## Accounts

Users register and sign in with the links in the top bar. An account has one of the permissions:

| Permission | Can |
|---|---|
| view | View, like visitors without an account. |
| write | Also upload, replace and delete files, delete and restore models, edit descriptions and create ensembles. |
| admin | Also change the permissions of the accounts and remove them, on the Accounts page. |

The first account to register is an admin. Later accounts can view until an admin gives them more. The app keeps at least one admin: it refuses to remove the last admin or to take away its admin permission.

`accounts.sqlite3` has scrypt hashes of the passwords. Account names are unique regardless of the case of ASCII letters. A permission change applies at once, also to accounts that are signed in. A session lasts 14 days, or until the account is removed. Every change is checked on the server when it is made, also if a page was opened before a sign-out or a permission change.

There is no password change. An admin can remove an account with a lost password, so that the user can register the name again. If no admin can sign in any more, stop the server and delete `accounts.sqlite3`, which removes all accounts, so that the next to register becomes the admin. The action log keeps the names.

The app does not limit registrations or sign-in attempts. Use long passwords. See [Public deployment](#public-deployment) for HTTPS.

## Public deployment

On a server with a public DNS name, the app listens only on `127.0.0.1`, and [Caddy](https://caddyserver.com) serves it over HTTPS. The files are in [`deployment/`](deployment/). The steps are for Arch Linux.

1. Install the packages, create a system user, and install the app in a virtual environment:

   ```bash
   sudo pacman -S --needed caddy git uv
   sudo useradd --system --create-home --home-dir /srv/irap-eval --shell /usr/bin/nologin irap-eval
   sudo -u irap-eval -H bash -c 'cd ~ && mkdir -p data \
       && git clone https://github.com/Ivan1248/irap-tools.git && uv venv venv \
       && VIRTUAL_ENV=~/venv uv pip install -e irap-tools/packages/irap_data \
          -e irap-tools/packages/irap_evaluation -e irap-tools/packages/irap_evaluation_server'
   ```

2. Copy [`deployment/server.toml`](deployment/server.toml) to `/srv/irap-eval/server.toml` and set the dataset directories in it:

   ```bash
   sudo -u irap-eval cp /srv/irap-eval/irap-tools/packages/irap_evaluation_server/deployment/server.toml /srv/irap-eval/
   ```

   The user `irap-eval` must be able to read the dataset directories. The service cannot read `/home` (`ProtectHome`), so the datasets must be elsewhere, e.g. under `/data`.

3. Start the app as a service:

   ```bash
   sudo cp /srv/irap-eval/irap-tools/packages/irap_evaluation_server/deployment/irap-eval-server.service /etc/systemd/system/
   sudo systemctl enable --now irap-eval-server
   journalctl -u irap-eval-server -f   # the log
   ```

4. Register the first account, which becomes the admin, before the app is public. On your computer, forward a port to the app over SSH and open `http://localhost:8600/register`:

   ```bash
   ssh -L 8600:127.0.0.1:8600 <server>
   ```

   With `is_served_over_https = true`, the browser may not keep the sign-in over plain HTTP, but the account is registered.

5. Copy [`deployment/Caddyfile`](deployment/Caddyfile) to `/etc/caddy/Caddyfile`, with the server's name in it, and start Caddy:

   ```bash
   sudo cp /srv/irap-eval/irap-tools/packages/irap_evaluation_server/deployment/Caddyfile /etc/caddy/Caddyfile
   sudo systemctl enable --now caddy
   journalctl -u caddy -f   # shows whether the certificate was issued
   ```

   Caddy gets a Let's Encrypt certificate for the name and renews it. Let's Encrypt must reach the server on port 80 or 443. Caddy also redirects HTTP to HTTPS.

6. Open only SSH, HTTP and HTTPS in the firewall, e.g. with `ufw`:

   ```bash
   sudo pacman -S ufw
   sudo ufw allow ssh && sudo ufw allow http && sudo ufw allow https
   sudo ufw enable && sudo systemctl enable ufw
   ```

   A network firewall of the institution may also block incoming connections.

To update the app, pull the checkout as `irap-eval` and run `sudo systemctl restart irap-eval-server`. Back up `/srv/irap-eval/data`.

## Models and files

A model is, e.g., one trained network, identified by the dataset, method name and seed in the header of its [prediction files](../irap_evaluation/docs/prediction_format.md). It has at most one active file per split. The pages call a stored file "file #<id>", and the code calls it a submission (`archive.Submission`).

### Uploads

A file is refused if:

- it is not a valid prediction file,
- its dataset is not configured, or its splits, attributes or iRAP codes are not those of the dataset,
- it lacks segments of its [evaluation sets](../irap_evaluation/docs/evaluation.md#evaluation-sets),
- its output kind, context offsets, or training or early stopping splits differ from those of the model's other active files,
- the method's models would not have distinct seeds or would differ in training or early stopping splits (`irap_evaluation.check_method_models`), since a method is one training recipe and its models are averaged.

Files of unlabeled splits are accepted but not scored.

- The Seed field replaces the seed in the header, e.g. to add another model of a method. The action log records the original seed.
- A file of a split that the model already has replaces the active file, which is [deleted](#deletion).
- A blank description keeps the model's description.
- The user confirms replaced files, a restored deleted model, and a replaced description before anything is stored. If the model changes in the meantime, e.g. by another user, the upload is refused and can be retried.

A `.zip` archive uploads the files of one model on several splits at once. Every member must be a `.parquet` prediction file (others are refused, not skipped), of the same model, output kind, context offsets, and training and early stopping splits (`archive.MODEL_FILE_FIELDS`), and of distinct splits. Their header `details` may differ. Either all files are stored, each as its own submission, or none. An archive that replaces all active files of a model may change their output kind, context offsets, and training and early stopping splits.

### Deletion

The Model page deletes one active file, after a confirmation. If another user has replaced or deleted that file in the meantime, the deletion is refused. A deleted file is kept and can be downloaded, but it cannot be restored, only uploaded again.

Deleting a model's last active file deletes the model. A model without active files is restored only by an upload, so every model that is not deleted has an active file.

A deleted model is left out of the Scores, Analysis and Ensemble pages, and the Models page shows it only with "Show deleted". Its files are kept unchanged, so it can be restored.

## Scoring

A background thread scores each new active file on its evaluation sets, with the defaults of `irap-eval evaluate` and 1000 bootstrap resamples (`scoring.ScoringSettings`). Attributes that a file does not predict count as invalid cells (`--missing-attribute-policy invalid`), so all models are scored on the same attributes. It then computes the intervals of each model and of the mean over the models of each method.

The database caches the attribute averages and per-attribute values, from which the means over models are computed, also over any subset of attributes. Intervals and comparisons need per-sequence statistics, which are too large to cache, so the thread recomputes them from the files when a page asks for them, and caches the results.

Cached values are out of date when the scoring settings, the evaluation sets (e.g. after a metadata update) or the files of a method's models change. At start, the server rescores the active files whose scores are out of date or failed with an unexpected error. A file that `irap_evaluation` refuses, e.g. one that lacks segments of a new metadata build, is marked as failed with the reason. Deleted files and the files of deleted models are not scored.

## Scores

The Scores page shows the means over each method's models with their bootstrap intervals.

- Methods trained on the shown split are hidden by default, since their scores are not of held-out data. Methods that used the split for early stopping, or whose training splits are unknown, are marked.
- The Method filter is a case-insensitive regular expression that matches any part of the name, e.g. `foo`, or `^(foo|bar)$` for exactly these two.
- On the reference set, "Compare with" adds each method's difference from a reference method, with its interval (`irap_evaluation.compare_methods`, as in `irap-eval compare`). See [comparing methods](../irap_evaluation/docs/evaluation.md#comparing-methods) for how to read them.
- The attribute filter, here and on the Model page, chooses the attributes that the averages cover. Intervals over a subset are computed a second after the last change and take about 0.4 s per model on Vietnam val and 1–2 s on train.

The URL keeps the view, so it can be shared as a link. The Model page shows the scores of a file on each evaluation set and downloads them as the JSON document of `irap-eval evaluate`, without intervals (`/api/submissions/<id>/scores/<set>.json`).

## Analysis

The Analysis page shows the predictions of a model (A) for one attribute on an evaluation set of an analysis split:

- **Confusion matrix:** labels (rows) × predictions of A (columns), with a column of invalid predictions. Scores count an invalid prediction as the first class (`null_policy="first_class"`), i.e. they add the invalid column to the first.
- **Segments of a cell:** a click on a cell lists its segments, most confident first, so that confident errors come first.
- **Map:** segments coloured by outcome, or, with a second model (B), by which of A and B is correct. The segments of the selected cell are highlighted.
- **Segment details:** a click on a segment shows its context images (at A's context offsets, from `dataset_dir`), its label, and the distributions predicted by A and B.

The page keeps the 8 most recently used files in memory. Reading and aligning a file takes about 0.06 s on Vietnam val and 0.3 s on train.

## Ensembles

The Ensemble page creates the weighted mean of the distributions of stored models of a dataset, as `irap-eval ensemble` does (`irap_evaluation.ensemble_predictions`). A hard prediction counts as a one-hot distribution.

- The ensemble is a model with the given method name and seed. It gets a file for each split where every member has an active file, and these replace all files of an existing model of that name, so that all its files are of the same members.
- Its training splits are the union of its members', and its early stopping splits are its members' that are not training splits (`irap_evaluation.make_ensemble_model`).
- Members with different context offsets predict different segments. "Only the segments that every member predicts" keeps their common segments.
- Its files are checked, confirmed and scored like uploads, and the action log records the members and weights. If a member's file changes before the ensemble is stored, the ensemble is refused and can be created again.

## Coding tables

The Model page downloads the active files of the chosen splits as one iRAP coding table (`.xlsx` or `.csv`), as `irap-eval export-coding-table` does, with a coder name (default: the method name) and a coding date. An `.xlsx` table of probabilistic predictions has the probability of each exported code on a second sheet.

## Development

```
irap_evaluation_server/
├─ config.py                ServerConfig, DatasetConfig, load_server_config
├─ datasets.py              DatasetContext: the metadata of a dataset and its evaluation sets
├─ database.py              Connections to the SQLite databases
├─ archive.py               ModelArchive: models, their files, the SQLite index and action log
├─ accounts.py              AccountStore: the accounts, their permissions and password hashes
├─ uploads.py               The checks of uploaded files and .zip archives, and their planned updates
├─ model_listing.py         The models of the Models page, grouped by method
├─ scoring.py               Scoring settings, the score cache, means over models and intervals
├─ scoring_worker.py        The background thread that scores and computes requested results
├─ method_ranking.py        The rows of the method table, their order, and the marks of used splits
├─ method_comparison.py     Comparisons with a reference method
├─ prediction_analysis.py   Outcomes of segments, confusion matrices, segment details
├─ map_libraries.py         The download of the map libraries
├─ ensembles.py             Ensembles of stored models
├─ coding_table_export.py   The coding table of the files of a model
├─ components/              Parts of the pages, e.g. the page frame, form controls, tables and the map
├─ pages/                   One module per page
└─ server.py                The irap-eval-server command
```

The logic modules (all but `components`, `pages` and `server`) do not import NiceGUI and are tested with pytest:

```bash
python -m pytest packages/irap_evaluation_server/tests
```

[`scripts/add_synthetic_models.py`](scripts/add_synthetic_models.py) fills an archive with synthetic Vietnam models and an ensemble, e.g. to try the pages:

```bash
python packages/irap_evaluation_server/scripts/add_synthetic_models.py server.toml
```
