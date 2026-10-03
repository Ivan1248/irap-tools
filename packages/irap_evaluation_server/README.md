# iRAP evaluation server

An internal web app for saved iRAP model predictions, built with [NiceGUI](https://nicegui.io) on [`irap_evaluation`](../irap_evaluation/):

- an archive of submissions (prediction files) with an action log,
- scores and the method table, with the differences from a reference method and their intervals,
- the analysis of the predictions of a run: confusion matrices, a map, and the details of segments,
- ensembles of stored runs, and the export of a submission as an iRAP coding table.

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
images_dir = "/data/IRAP_Vietnam_images"   # optional, the segment images of the analysis page
analysis_splits = ["train", "val"]          # the splits of the analysis page
```

On its first start, the server downloads the JavaScript libraries of the map (MapLibre and deck.gl, pinned versions checked by SHA-256) into `data_dir/vendor/`, since MapLibre must be served from the server itself. Without internet access, it starts without the map. The basemap tiles come from Carto, so the map also needs internet access in the browser.

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

The data directory holds `archive.sqlite3` (the index, the action log and the score cache), `submissions/<id>/predictions.parquet` and NiceGUI's per-browser storage.

## Scoring

A background thread scores each submission after its upload, on its evaluation sets, with the defaults of `irap-eval evaluate` and 1000 bootstrap resamples (`scoring.ScoringSettings`). Attributes that a file does not predict count as invalid cells (`--missing-attribute-policy invalid`), so all runs are scored on the same attributes. Then it computes the intervals of each run and of the mean over the runs of each method.

The scores are cached in the database, with the attribute averages and the per-attribute values. The means over runs, also over a subset of the attributes, are computed from them. Intervals need the per-sequence statistics, which are too large to cache, so the thread computes intervals over a subset from the files when a page asks for them, and caches them.

Cached values are out of date when the scoring settings, the evaluation sets (e.g. after a metadata update) or the runs of a method change. At its start, the server scores again the submissions whose scores are out of date or failed with an unexpected error. A file that `irap_evaluation` refuses, e.g. one that lacks segments of a new metadata build, is marked as failed with the reason.

## Scores page

The Scores page (`/scores`, also the start page) ranks the methods of a dataset split by an attribute average, with the means over the runs of each method and their bootstrap intervals. A click on a metric header sorts by it, best first. The View select shows one per-attribute metric instead, with attribute rows and method columns. The cells are shaded by their value relative to the other methods in the column (in the row for a per-attribute metric), darkest for the best. The best value is bold, and so are the values whose interval overlaps its interval, a rough sign that they are not clearly worse. Submissions that are being scored, or whose scoring failed, are listed below the table with the reason. Deleted submissions are left out.

The Methods field filters the methods by a regular expression, case-insensitive and matching any part of the name: `foo` shows the methods whose name contains "foo", and `^(foo|bar)$` exactly these two. The shading and the bold values compare the shown methods.

On the reference set, "Compare with" chooses a reference method, which is shown even if the filter does not match it. The cells of the other shown methods then have a third line: the difference from the reference (Δ), its bootstrap interval, and "better" or "worse" if the interval excludes 0. The differences come from `irap_evaluation.compare_methods`, as `irap-eval compare` computes them. The interval resamples the road sequences once for both methods, so it is usually narrower than the separate intervals suggest, and it includes the variation between the runs of each method. With many methods or attributes, some intervals exclude 0 by chance, so the differences of single attributes are exploratory. Like intervals, a comparison needs the per-sequence statistics, so the background thread computes it from the files (one scoring per run of both methods), row by row, and caches it.

The page of a submission shows its scores on each evaluation set: the attribute averages, and the iRAP metrics and the invalid cells of each attribute, with the metrics shaded relative to the other attributes. A link downloads the `irap-eval evaluate` JSON document over all attributes, without intervals (`/api/submissions/<id>/scores/<set>.json`).

The attribute filter of both pages chooses the attributes that the averages cover. The URL keeps the subset in repeated `attribute` parameters, so a filtered view can be shared as a link. Values change at once. Intervals over a subset are computed from the files a second after the last change of the filter, which takes about 0.4 s per run on Vietnam val and 1–2 s on train, and are cached.

## Analysis page

The Analysis page (`/analysis`) shows the predictions of one run (A) for one attribute on an evaluation set of an analysis split. The test split can be left out of `analysis_splits`, so that its labels are not browsed.

- **Confusion matrix:** the labeled segments by label (rows) and prediction of A (columns), with a column of invalid predictions. Scores count an invalid prediction as the first class (`null_policy="first_class"`), so the matrix of the scores is this one with the invalid column added to the first. The class table shows the precision, recall and F1 of each class as scored.
- **Segments of a cell:** a click on a cell lists its segments, the most confident predictions first, so that a cell of errors starts with the confident errors.
- **Map:** the segments that have a location, coloured by outcome (correct, wrong, invalid, unlabeled). With a second run (B), the colours compare them (both correct, only A correct, only B correct, neither correct). The segments of the selected cell are highlighted.
- **Segment details:** a click on a segment in the list or on the map shows its context images (at the context offsets of A, from `images_dir`), its label, and the distributions predicted by A and B.

The runs are ordered by amF1 over the attribute subset of the filter, which also restricts the attribute choices. The URL keeps the whole view. The page reads the prediction files of A and B, about 0.1 s on Vietnam val and 0.3 s on train, and keeps the 8 most recently used runs in memory. Deleted submissions are not offered. The submission page links to the analysis of a run.

## Ensembles

The Ensemble page (`/ensemble`) creates the ensemble of stored runs with `irap_evaluation.ensemble_predictions`, as `irap-eval ensemble` does: the weighted mean of their distributions, where a hard prediction counts as a one-hot distribution. The members are runs (a method name and seed) of a dataset, each with a weight. A regular expression filters the listed runs by run label, as the Methods field of the Scores page does, and "Select shown" selects the listed ones. Selected runs stay listed, so that every member is visible. One ensemble is stored for each split where every member has a submission, all or none, as a new run with the given method name and seed. Each is checked like an upload, and the action log records its members and weights (action `ensemble`). The ensembles are scored like uploads.

Members with different context offsets predict different segments. The option "Only the segments that every member predicts" keeps their common segments.

## Coding tables

The submission page downloads the submission as an iRAP coding table (`.xlsx` or `.csv`), as `irap-eval export-coding-table` does, with the packaged template, a coder name (default: the method name) and a coding date. A `.xlsx` file of probabilistic predictions has the probability of each exported code on a second sheet.

## Layout

```
irap_evaluation_server/
├─ config.py        ServerConfig, DatasetConfig, load_server_config
├─ datasets.py      DatasetContext: the metadata of a dataset and its evaluation sets
├─ database.py      Connections to the SQLite database
├─ archive.py       SubmissionArchive: files, SQLite index and action log, upload checks
├─ scoring.py       Scoring settings, the score cache, means over runs and intervals
├─ scoring_worker.py  The background thread that scores and computes requested results
├─ method_ranking.py  The rows of the method table and their order
├─ method_comparison.py  Comparisons with a reference method (cached requests of the background thread)
├─ prediction_analysis.py  Outcomes of segments, confusion matrices, segment details
├─ map_libraries.py   The download of the map libraries
├─ ensembles.py     Ensembles of stored runs
├─ coding_table_export.py  The coding table of a submission
├─ components/      Page frame, native form controls, attribute filter, tables, map, page views, CSS
├─ pages/           One module per page
└─ server.py        The irap-eval-server command
```

The logic modules (all but `components`, `pages` and `server`) do not import NiceGUI and are tested with pytest:

```bash
python -m pytest packages/irap_evaluation_server/tests
```
