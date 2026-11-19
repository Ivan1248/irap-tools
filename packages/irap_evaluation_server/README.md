# iRAP evaluation server

A web app that stores model predictions, scores them, and compares, analyses and ensembles them. It is built with [NiceGUI](https://nicegui.io) on [`irap_evaluation`](../irap_evaluation/).

| Page | Path | Content |
|---|---|---|
| Scores | `/scores` (start page) | The methods of a dataset split, ranked by an attribute average or by a per-attribute metric, with intervals and differences from a reference method. |
| Models | `/models` | The models of each dataset by method, with their files, a filter, the upload form and the [ensemble](#ensembles) form. |
| Model | `/models/<id>` | A model's files, scores and coding-table export, and its actions for writers. |
| Method | `/methods?dataset=<dataset>&method=<method>` | A method's models, the scores of their mean and of each model, its display name and description, and the actions of its models. Method names in the tables link to it. |
| Analysis | `/analysis` | A model's predictions for one attribute, optionally compared with a second model: confusion matrix, class analysis, review curve, map, segment details, and candidate labeling errors with label reviews. |
| Methodology | `/methodology` | How the files are scored and how to read the score tables. The notes below the tables link to its sections. |
| Action log | `/actions` | All changes, with the name of the user. Writers only. |
| Accounts | `/accounts` | The accounts and their permissions. Admins only, in the Admin menu. |
| Shown metrics | `/shown-metrics` | The attribute averages that the pages show. Admins only, in the Admin menu. |

Anyone who can open the app can view it and download files. Changes need a signed-in [account](#accounts) with write permission.

## Installation and start

From the irap-tools checkout, install the packages and start the server with a [configuration file](#configuration):

```bash
uv pip install -e packages/irap_data -e packages/irap_evaluation -e packages/irap_evaluation_server
irap-eval-server server.toml
```

On its first start, the server downloads pinned libraries, checked by SHA-256, into `data_dir/vendor/`:

- MapLibre and deck.gl for the map of the Analysis page. MapLibre must be served by the server itself. Without them, the server starts without the map. The browser loads the basemap tiles from Carto.
- MathJax 3 (`tex-svg.js`) for the formulas of the Methodology page. Without it, the page shows them as TeX.

## Configuration

`server.toml` (relative paths are relative to its directory):

```toml
data_dir = "data"       # the archive
host = "127.0.0.1"      # optional, "0.0.0.0" to serve the lab network
port = 8600             # optional
is_served_over_https = false   # optional, true behind an HTTPS proxy, so the session cookie is sent only over HTTPS

[datasets.vietnam]      # a key of irap_data.DATASET_PRESETS
dataset_dir = "/data/IRAP_Vietnam"   # the images (FRAMES/) and the metadata
analysis_splits = ["train", "val"]   # the splits of the Analysis page, e.g. without test to keep its labels unseen; the others are protected

[datasets.bh]
dataset_dir = "/data/IRAP_BIH"
metadata_dir = "/data/IRAP_BIH_METADATA"   # optional, default dataset_dir
analysis_splits = ["train", "val"]
```

The image paths in the metadata (`segment_id_to_data_paths_rel.json`) are relative to `dataset_dir`. A Vietnam dataset directory built by [`irap_vietnam_data_preparation`](../irap_vietnam_data_preparation/README.md#directory-layout) also holds the metadata. Without the images, e.g. on a server with only the metadata files, the Analysis page shows no images.

The data directory holds:

- `archive.sqlite3`: the index of models and files, and the action log,
- `submissions/<id>/predictions.parquet`: the stored files, removed when they are [deleted](#deletion) (if a file is open, e.g. on Windows, at a later deletion),
- `scores.sqlite3`: the [scores](#scoring). The server computes them again if the file is missing, so backups can leave it out.
- `uploads/`: uploaded files that are not stored yet, removed at start, and at the start of another upload if they are older than 24 h,
- `vendor/`: the map libraries and MathJax,
- `accounts.sqlite3`: the [accounts](#accounts),
- `display_settings.sqlite3`: the [hidden metrics](#hidden-metrics),
- `label_reviews.sqlite3`: the [label reviews](#label-reviews),
- `nicegui_storage/` and `storage_secret.txt`: NiceGUI's per-browser storage, e.g. of the signed-in account, and the key of its session cookie. Deleting the key signs out all users.

Keep the data directory and its backups private.

Each database has a schema version. At start, the server migrates a database of an older version after copying it to a backup next to it, e.g. `archive.v1.2026-10-12T150000Z.backup.sqlite3`. It refuses a database of a newer version, or one whose tables differ from those of its version. A score cache of another version is created again, and all files are scored again. This takes a few minutes, during which the Scores page lists the files that are being scored.

## Accounts

Users register and sign in with the links in the top bar. Each account has one permission:

| Permission | Can |
|---|---|
| view | View and download, like visitors without an account. |
| write | Also upload, replace and delete files, delete models and methods, edit descriptions and method display names, create ensembles, review labels, and see the action log. |
| admin | Also change the permissions of accounts and remove them on the Accounts page, choose the [shown metrics](#hidden-metrics), and see the per-class scores of [protected splits](#protected-splits). |

The first account to register is an admin, and later accounts get view permission. The app refuses to remove the last admin or to take away its admin permission.

- `accounts.sqlite3` stores scrypt hashes of the passwords.
- Account names are unique regardless of the case of ASCII letters.
- A permission change applies at once, also to signed-in accounts. The server checks every change when it is made, also from a page that was opened before a sign-out or a permission change.
- A session lasts until 14 days after the browser's first visit, also if it signed in later, or until the account is removed.

There is no password change. An admin can remove an account with a lost password, so that the user can register the name again. If no admin can sign in, stop the server and delete `accounts.sqlite3`. This removes all accounts, and the next account to register becomes the admin. The action log keeps the names.

The app does not limit registrations or sign-in attempts, so use long passwords, and serve it over HTTPS (see [Public deployment](#public-deployment)).

### Protected splits

The splits of a dataset that are not in its `analysis_splits`, e.g. test, are protected: their labels are not shown in detail. Their files can be stored, and everyone sees their per-attribute and averaged scores and the number of labeled segments of each attribute (Example count, `metrics.per_attribute.n` in the scores JSON). But they are not on the Analysis page, and only admins see:

- their per-class scores, confusion matrices and review curves, which are left out (`"classes": null`) of the scores JSON that others download,
- the Class count, Minority-class example count and Smallest-class example count columns in the per-attribute view of the Scores page.

This does not keep the labels secret from writers, since the scores of chosen predictions can reveal labels. Give write permission only to trusted users. The action log shows who stored which file.

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

   The user `irap-eval` must be able to read the dataset directories. The service cannot read `/home` (`ProtectHome`), so put the datasets elsewhere, e.g. under `/data`.

3. Start the app as a service:

   ```bash
   sudo cp /srv/irap-eval/irap-tools/packages/irap_evaluation_server/deployment/irap-eval-server.service /etc/systemd/system/
   sudo systemctl enable --now irap-eval-server
   journalctl -u irap-eval-server -f   # the log
   ```

4. Before the app is public, register the first account, which becomes the admin. On your computer, forward a port to the app over SSH and open `http://localhost:8600/register`:

   ```bash
   ssh -L 8600:127.0.0.1:8600 <server>
   ```

   With `is_served_over_https = true`, the browser may not keep the sign-in over plain HTTP, but the account is registered.

5. Copy [`deployment/Caddyfile`](deployment/Caddyfile) to `/etc/caddy/Caddyfile`, set the server's name in it, and start Caddy:

   ```bash
   sudo cp /srv/irap-eval/irap-tools/packages/irap_evaluation_server/deployment/Caddyfile /etc/caddy/Caddyfile
   sudo systemctl enable --now caddy
   journalctl -u caddy -f   # shows whether the certificate was issued
   ```

   Caddy gets a Let's Encrypt certificate for the name, renews it, and redirects HTTP to HTTPS. Let's Encrypt must reach the server on port 80 or 443.

6. Open only SSH, HTTP and HTTPS in the firewall, e.g. with `ufw`:

   ```bash
   sudo pacman -S ufw
   sudo ufw allow ssh && sudo ufw allow http && sudo ufw allow https
   sudo ufw enable && sudo systemctl enable ufw
   ```

   A network firewall of the institution may also block incoming connections.

To update the app and its dependencies, e.g. for security fixes:

```bash
sudo -u irap-eval -H bash -c 'cd ~/irap-tools && git pull && cd ~ \
    && VIRTUAL_ENV=~/venv uv pip install --upgrade -e irap-tools/packages/irap_data \
       -e irap-tools/packages/irap_evaluation -e irap-tools/packages/irap_evaluation_server'
sudo systemctl restart irap-eval-server
```

Back up `/srv/irap-eval/data`, and keep the backups private.

## Models and files

A model, e.g. one trained network, is identified by the dataset, method name and seed in the header of its [prediction files](../irap_evaluation/docs/prediction_format.md). Unlike `irap_evaluation`, the app requires a seed also for the only model of a method, so that a second model can be added without changing the first. A model has at most one file per split. The pages call a stored file "file #<id>", and the code calls it a submission (`archive.Submission`).

A model has a Markdown description, e.g. of how it was trained. The tables show its first line, so that line should summarize the model. The Model page shows and edits the description. The Method page shows the description of the model with the lowest seed and edits it for all models of the method. The browser sanitizes the Markdown (DOMPurify), which removes scripts and event handlers.

A method on a dataset can have a display name, which the pages show instead of the method name, and which the Method filter of the Scores page and the Model filter of the Models page match. Uploads set it, and the Method page edits it. The action log records the edit on the method's first model. Files, URLs, download names and the action log keep the method name.

### Uploads

A file is refused if:

- it is larger than 1 GiB, or the files of a `.zip` archive are larger than 1 GiB together,
- it is not a valid prediction file,
- it has no seed, neither in its header nor in the Seed field,
- its dataset is not configured, or its splits, attributes or iRAP codes are not those of the dataset,
- it has no context offsets and lacks segments of both the reference set and the full set,
- its output kind, context offsets, or training or early stopping splits differ from those of the model's other files,
- the method's models would not have distinct seeds or would differ in training or early stopping splits (`irap_evaluation.check_method_models`), since a method is one training recipe and its models are averaged.

Files of unlabeled splits are stored but not scored.

A file that lacks segments of its model-compatible set, e.g. one made with an older metadata build to which a newer build added segments, is accepted with a warning. Its missing segments are scored as invalid predictions ([missing segments](../irap_evaluation/docs/evaluation.md#missing-segments)), so all models are still scored on the same segments. The Model page lists them under the scores of each evaluation set, and the Scores page notes their number. Predicted segments that are in no evaluation set, e.g. ones that a newer build removed from the split, are noted and not scored.

The fields of the upload form:

- **Seed** replaces the seed in the header, e.g. to add another model of a method, or gives a seed to a file without one. The action log records the original seed.
- **Description**: a blank one keeps the model's description.
- **Method display name** replaces the method's display name, without changing the files. A blank one takes the display name from the files, which must then agree, or keeps the method's if they have none. After a file is chosen, the placeholder shows the display name that a blank field gives.

A file of a split that the model already has replaces its file, which is deleted permanently with its scores. Before anything is stored, the user confirms the deletion of replaced files and the replacement of the description or the method display name. If the model changes in the meantime, e.g. by another user, the upload is refused and can be retried.

A `.zip` archive uploads the files of one model on several splits at once. Every member must be a `.parquet` prediction file (others are refused, not skipped) of a distinct split. All members must have the same model, output kind, context offsets, and training and early stopping splits (`archive.MODEL_FILE_FIELDS`), but their header `details` may differ. Either all files are stored, each as its own submission, or none. An archive that replaces all files of a model may change their output kind, context offsets, and training and early stopping splits.

### Deletion

Deletion is permanent and needs a confirmation. The Model page deletes a file, or a whole model with all its files. The Method page deletes a whole method with all its models. Uploads and ensembles delete the files that they replace. A deleted file is removed with its scores and can only be uploaded again.

- The last file of a model is deleted only with the model, so every model has a file.
- Deleting the last model of a method also removes the method's display name.
- The action log keeps the actions on deleted models and files, and their ids are not reused.

To keep a copy, download the files of a model, e.g. an ensemble, as one `.zip` archive from the Model page (`/api/models/<id>/predictions.zip`). It can be uploaded again.

## Scoring

A background thread scores each new file on its evaluation sets, with the defaults of `irap-eval evaluate` and 1000 bootstrap resamples (`scoring.ScoringSettings`). Attributes and segments that a file does not predict count as invalid predictions (`--missing-attribute-policy invalid`, `--missing-segment-policy invalid`), so all models are scored on the same attributes and segments.

`scores.sqlite3` stores the attribute averages and per-attribute values, and their values in each bootstrap resample of road sequences (`irap_evaluation.resample_model_metrics`), as float32: about 1.9 MB per file and evaluation set on Vietnam. All files on an evaluation set share the resamples. So the pages compute means over models, intervals and paired comparisons of methods from the stored values, also over any subset of attributes, as `irap-eval evaluate` and `irap-eval compare` compute them (see [confidence intervals](../irap_evaluation/docs/evaluation.md#confidence-intervals)).

Resampling takes about 1.1 s per file and evaluation set on Vietnam val and 1.5 s on train. Most of it is for the [manual review proportion](../irap_evaluation/docs/evaluation.md#manual-review-proportion), whose statistics also take about 80 MB of memory while a train file is scored.

Stored scores are out of date when the scoring settings or the evaluation sets change, e.g. after a metadata update. To also catch changes of the scoring code, the settings fingerprint includes the scores and resampled values of random predictions on a small synthetic set (2 attributes, 3 classes, 24 segments), computed at start. A change of `irap_evaluation` or NumPy that changes these values, e.g. of the random generator, makes all scores out of date. A change that affects only cases outside the set, e.g. attributes with more classes, does not. After such a change, increase `scoring.SCORE_CACHE_VERSION`, or delete `scores.sqlite3` while the server is stopped.

At start, the server scores again the files whose scores are out of date or failed with an unexpected error. A file that `irap_evaluation` refuses, e.g. one without context offsets that lacks segments of a new metadata build, is marked as failed, with the reason.

## Scores

The Scores page shows the means over each method's models, with their bootstrap intervals.

- **Highlights:** a value is bold if it is the best of its column (of its row, in the per-attribute view), or not significantly worse than the best: a one-sided paired bootstrap test at 5 %, whose 90 % interval of the difference from the best reaches 0 (`method_ranking.is_not_significantly_worse`). The test needs the methods on the same segments, so on the model-compatible sets, values are bold only if all shown methods are scored on the same set.
- **Model-compatible sets** show each method on its model's own set. If the context offsets of a model select the segments of the reference set or the full set, these are its scores on that set, and its Notes say so.
- **Used splits:** methods trained on the shown split are hidden by default, since their scores are not of held-out data. Methods that used the split for early stopping, or whose training splits are unknown, are marked.
- **`aMRP90`** is the mean over the attributes of the [manual review proportion](../irap_evaluation/docs/evaluation.md#manual-review-proportion) `MRP90`: the fraction of an attribute's segments that a reviewer has to correct, starting with the least confident predictions, until its mF1 reaches 0.9. Like NLL and Brier, it needs probabilistic predictions, so methods of hard predictions have no value.
- **Label counts:** before the methods, the per-attribute view shows counts of each attribute's labels in the evaluation set (`label_counts`): Class count (the classes that occur as a label), Example count (the labeled segments), Minority-class example count (the labeled segments outside the most frequent class) and Smallest-class example count (the labeled segments of the rarest class). They are computed from the labels, since the counts of the scores (`n`, `support`) leave out invalid predictions under `null_policy='exclude'`. On the model-compatible sets, they are shown only if all shown methods are scored on the same set.
- **Sorting:** in the per-attribute view, a click on the header of a method or a label count, e.g. on the arrow below it, sorts the attributes by that column: worst first for a method, smallest first for a label count, and reversed after a second click. A click on Attribute restores the order. Hidden label counts cannot be sorted by, since the order would reveal them.
- **Method filter:** a case-insensitive regular expression that matches any part of the name, e.g. `foo`, or `^(foo|bar)$` for exactly these two.
- **Compare with** adds each method's difference from a reference method, with its interval, as `irap-eval compare` computes it. It is available on the reference set and the full set, which are the same for all files of a split. See [comparing methods](../irap_evaluation/docs/evaluation.md#comparing-methods) for how to read the differences.
- **Attribute filter**, here and on the Method and Model pages, chooses the attributes that the averages cover. The averages of a method over a new subset take about 12 ms on Vietnam (35 attributes). The pages compute them in a thread, so that a large table does not hold up the server, and the worker keeps them in memory (`scoring_worker.NUM_CACHED_METHOD_RESAMPLES`).

The URL keeps the view, so it can be shared as a link.

### Method and Model pages

A method name links to the Method page with the same split and attributes. It shows the attribute averages of the method's mean and of each of its models, which are shaded but not tested against each other, and the per-attribute metrics of the mean.

The Model page shows the scores of a file on each evaluation set. It downloads them as the JSON document of `irap-eval evaluate`, without intervals (`/api/submissions/<id>/scores/<set>.json`).

Their per-attribute tables show mF1, mP, mR, MCC, the accuracy A, `MRP90` and the number of invalid predictions. A is next to `MRP90` because the manual review proportion follows the error rate 1 − A more closely than mF1. The columns sort as in the Scores page: worst first for a metric, largest first for the invalid predictions.

### Hidden metrics

On the Shown metrics page, admins can hide any attribute average except the main metric, amF1. Hiding an average also hides its per-attribute metric, e.g. aBrier hides Brier. Hidden metrics are left out of the columns and the sort and View options of the Scores page, the tables of the Method and Model pages, and the metric list of the Methodology page. A link with a hidden metric (`sort` or `metric`) is refused. Files are scored as before, and the downloaded scores have all metrics, so a metric that is shown again appears at once. Open pages show a change after a reload.

## Analysis

The Analysis page shows the predictions of a model (A) for one attribute on an evaluation set of an analysis split, optionally compared with a second model (B). Tabs switch between the confusion matrix with its class table, the class analysis, the review curve and the labeling errors. The URL keeps the chosen tab.

- **Confusion matrix:** labels (rows) × predictions of A (columns), with a column of invalid predictions. Scores count an invalid prediction as the first class (`null_policy="first_class"`), i.e. they add the invalid column to the first.
- **Class table:** the support, precision, recall and F1 of each class, and how well A tells the class apart from the other classes of the attribute, one vs. rest (`class_analysis.compute_class_discrimination`): the AUROC of its predicted probability and the informedness J = TPR − FPR of its predicted classes. By chance, these are 0.5 and 0 at any class size. F1 is not: a predictor that ignores the image gets between p and 2p / (1 + p) for a class of proportion p. A high AUROC with J near 0 marks a class that A ranks well but rarely predicts.
- **Class analysis:** the classes with labeled segments of all selected attributes that A predicts (`class_analysis`), to find the classes that are hard for A. It shows their support, F1, AUROC and J, the number of their segments that are predicted as another class (Missed), and the class that most of these are predicted as. The classes with the lowest AUROC are first by default, and a click on a column header sorts by that column. A click on a class shows its attribute and selects its missed segments, like a click on its row label in the confusion matrix.
- **Review curve:** mF1 and accuracy of A, and of B, after a reviewer has corrected a proportion of the segments, starting with the least confident predictions. It marks where mF1 reaches 0.9 (the manual review proportion `MRP90`). A tooltip shows the scores at each 5 % of the segments. Models of hard predictions have no curve.
- **Labeling errors:** the labels of all selected attributes that the shown models disagree with (`label_errors`). A model disagrees if its prediction is valid and is another class. With B, only the labels that both models disagree with are listed by default. The least probable labels come first: those with the lowest mean probability over the models with a valid prediction, the self-confidence of confident learning. For hard predictions, this is the proportion of the models that predict the label. Probabilities are not fully comparable between attributes with different numbers of classes. All labels are listed, in pages of 100, and the page follows the selected label. A click on a row shows its attribute and segment. The URL keeps both choices of the list.
- **Segments of a cell:** a click on a cell of the matrix lists its segments, most confident first, so that confident errors come first. A click on a row label of the matrix, or on a class of the class analysis, lists the missed segments of the class with their predicted classes, from the most confident errors to the invalid predictions. In the Labeling errors panel, the labeling errors take the place of this list.
- **Map:** segments coloured by outcome, or, with B, by which of A and B is correct. The listed segments are highlighted.
- **Segment details:** a click on a segment shows its context images (at A's context offsets, from `dataset_dir`), and, below the map, the classes of the attribute with its label and the distributions predicted by A and B. The arrows before the segment ID step to the previous and next segment of its road sequence in the evaluation set. The List buttons step through the labeling errors in their panel, also after the selected label was reviewed and hidden, and through the segments of the selected cell in the other panels, also those that are not listed.

The page keeps the 8 most recently used files in memory. Reading and aligning a file takes about 0.06 s on Vietnam val and 0.3 s on train.

### Label reviews

Writers can correct the label of the selected segment for the shown attribute (`label_reviews`). Each class in the segment details has a button that stores it as the correct class, which can also be the class of the label. A correction can instead say that the class is unclear, and it can have a comment.

- Each label has one correction. A later correction replaces it, and it can be withdrawn.
- A page shows the corrections of other reviewers as they were when the page was loaded, when its Labeling errors panel was shown, or when it stored a correction.
- Reviewed labels are hidden from the labeling errors by default.
- A correction of a label that has changed since, e.g. after a metadata update, is marked as one of the old label and counts as unreviewed.
- Corrections do not change the labels that files are scored with.

The reviewed labels of a split can be downloaded as a hard prediction file of the method `label-review`. It has the segments and attributes with corrections, with the class of the correction, and null for unclear and unreviewed labels. The header's model lists the unclear labels in `details["label_review"]["unclear"]`, and the corrections whose attribute or class is no longer in the vocabulary, which are left out, in `details["label_review"]["skipped"]`.

## Ensembles

The "Create an ensemble" form of the Models page creates the weighted mean of the distributions of stored models of a dataset, as `irap-eval ensemble` does (`irap_evaluation.ensemble_predictions`). A hard prediction counts as a one-hot distribution. The members are the models selected with the checkboxes of the tables, also those that the filter hides. The form lists them with their weights.

- The ensemble is a model with the given method name, display name and seed (0 in a new form). A blank method name is made of the sorted method names of the members with their seeds, e.g. `strong-probs_seeds1-3` (`irap_evaluation.make_ensemble_method_name`). A blank display name keeps the method's.
- It gets a file for each split on which every member has a file. These replace all files of an existing model of that name, so that all its files are of the same members.
- Its training splits are the union of its members', and its early stopping splits are its members' early stopping splits that are not among its training splits (`irap_evaluation.make_ensemble_model_info`).
- Members that predict different segments or attributes are refused by default. The form can instead combine the segments and attributes that every member predicts, or all that any member predicts, where one that a member does not predict counts as invalid (`irap_evaluation.MemberCombination`). Members with different context offsets predict different segments. With all attributes, an attribute that only one member predicts is that member's prediction.
- Its files are checked, confirmed and scored like uploads. The action log records the members, weights and combinations of segments and attributes. If a member's file changes before the ensemble is stored, the ensemble is refused and can be created again.

## Coding tables

The Model page downloads the files of the chosen splits as one iRAP coding table (`.xlsx` or `.csv`), as `irap-eval export-coding-table` does, with a coder name (default: the shown method name) and a coding date. An `.xlsx` table of probabilistic predictions has the probability of each exported code on a second sheet.

Tables are made one at a time in a separate process (NiceGUI's `run.cpu_bound`, started with "spawn"). An `.xlsx` table of large splits takes tens of seconds, which in a thread delays the server's other pages until browsers lose the connection. The file is written to `<data_dir>/uploads` and sent once over HTTP. File names are cut to 160 bytes, since Excel does not open a file whose path is longer than 218 characters.

## Development

```
irap_evaluation_server/
├─ config.py                ServerConfig, DatasetConfig, load_server_config
├─ datasets.py              DatasetContext: the metadata of a dataset and its evaluation sets
├─ database.py              Connections to the SQLite databases and the migrations of their schemas
├─ archive.py               ModelArchive: models, their files, the SQLite index and action log
├─ accounts.py              AccountStore: the accounts, their permissions and password hashes
├─ display_settings.py      The hidden metrics, which admins choose
├─ uploads.py               The checks of uploaded files and .zip archives, and their planned updates
├─ model_listing.py         The models of the Models and Method pages, grouped by method
├─ scoring.py               Scoring settings and the score cache
├─ method_scores.py         Means over the models of a method and their resampled values
├─ scoring_worker.py        The background thread that scores, and the resampled values in memory
├─ method_ranking.py        The rows of the method table, their order, highlights and marks of used splits
├─ method_comparison.py     Comparisons with a reference method
├─ label_counts.py          Counts of the labels of attributes in an evaluation set
├─ methodology.py           The text of the Methodology page, with the values of the settings
├─ prediction_analysis.py   Outcomes of segments, confusion matrices, segment details
├─ class_analysis.py        The classes of a model's attributes with their scores and confusions
├─ label_errors.py          Candidate labeling errors, ranked by the models' mean probability of the label
├─ label_reviews.py         LabelReviewStore: corrections of labels, and their export as a prediction file
├─ vendor_libraries.py      The download of the libraries of the map and of MathJax
├─ ensembles.py             Ensembles of stored models
├─ coding_table_export.py   The coding table of the files of a model
├─ components/              Parts of the pages, e.g. the page frame, form controls, tables and the map
├─ pages/                   One module per page
└─ server.py                The irap-eval-server command
```

The logic modules (all but `components`, `pages` and `server`) do not import NiceGUI and are tested with pytest:

```bash
uv run --project packages/irap_evaluation_server --extra test python -m pytest packages/irap_evaluation_server/tests
```

A data directory has been in production since 2026-10-12, so a change of a database schema needs a new schema version and a migration from the previous one: `SCHEMA_VERSION` and `_MIGRATIONS` in `archive.py`, `accounts.py`, `display_settings.py` and `label_reviews.py` (see `database.initialize_schema`). A change of the score cache tables needs only a new `scoring.SCORE_CACHE_VERSION`. The server refuses tables that differ from those of their version, so a forgotten version change shows at the next start.

[`scripts/add_synthetic_predictions.py`](scripts/add_synthetic_predictions.py) fills an archive with synthetic Vietnam predictions and an ensemble of them, e.g. to try the pages:

```bash
python packages/irap_evaluation_server/scripts/add_synthetic_predictions.py server.toml
```
