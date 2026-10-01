# iRAP evaluation

Evaluation of saved iRAP model predictions:

- a prediction file format for full class distributions or hard predictions, with classes identified by iRAP code ([format](docs/prediction_format.md)),
- the iRAP metrics, equal to those of the training-time evaluation in `vidlu_irap_gaim` ([definitions](docs/evaluation.md)),
- confidence intervals and comparisons of methods with a bootstrap over road sequences that includes the spread of a method's runs (e.g. training seeds),
- ensembling,
- CSV and markdown tables,
- export of predictions as iRAP coding tables.

The package is a library with a thin command-line interface, `irap-eval`. It needs numpy, pandas, pyarrow and the torch-free core of [`irap_data`](../irap_data/), and does not need torch. Its layers import only downward:

```
irap_evaluation/
├─ predictions, prediction_io                        The Predictions datatype and the file format
├─ evaluation_sets, metrics, evaluation, bootstrap   Scoring, intervals and comparisons
├─ ensembling                                        Combining predictions
├─ reports/                                          Tables and JSON documents of results
│                                                    (evaluation_report), coding tables
└─ tools/                                            The irap-eval command (cli)
```

## Installation

From the irap-tools checkout:

```bash
uv pip install -e packages/irap_data -e "packages/irap_evaluation[xlsx]"
```

From another project, without a checkout:

```bash
uv pip install "irap-data @ git+https://github.com/Ivan1248/irap-tools#subdirectory=packages/irap_data" \
               "irap-evaluation[xlsx] @ git+https://github.com/Ivan1248/irap-tools#subdirectory=packages/irap_evaluation"
```

The `xlsx` extra (openpyxl) is needed only to write `.xlsx` coding tables.

## Command-line use

```bash
DATA=/data/IRAP_Vietnam   # the metadata directory of the release, e.g. IRAP_BIH_METADATA for BH

# Check files.
irap-eval validate resnet.predictions.parquet vit.predictions.parquet

# Combine two models (the weighted mean of their distributions).
irap-eval ensemble resnet.predictions.parquet vit.predictions.parquet \
    --name resnet+vit -o ensemble.predictions.parquet

# Score files, with 95 % bootstrap intervals from 1000 resamples.
irap-eval evaluate $DATA resnet.predictions.parquet vit.predictions.parquet \
    ensemble.predictions.parquet --out results/ --bootstrap 1000 --table-metrics mF1 MCC

# Compare two methods on the reference set. The files hold their runs (method.name in the header,
# with distinct method.seed values).
irap-eval compare $DATA resnet_seed*.predictions.parquet vit_seed*.predictions.parquet \
    --a resnet --b vit

# Write predictions as an iRAP coding table, with the probability of each code on a second sheet.
irap-eval export-coding-table $DATA ensemble.predictions.parquet -o ensemble.xlsx --confidence-table
```

`evaluate` scores each file on the reference set and on its model set ([evaluation sets](docs/evaluation.md#evaluation-sets)), with nulls in the file scored according to `--null-policy` ([invalid cells](docs/evaluation.md#invalid-cells)). It writes to `--out`:

| File | Content |
|---|---|
| `summary_long.csv` | One row per (run, evaluation set, attribute or `average`, metric), with the method and seed of the run, the null policy, confidence bounds and the number of undefined bootstrap resamples that the bounds leave out (`ci_num_undefined`). |
| `averages_<dataset>_<split>_<set>.csv` | Run rows × attribute-average columns. |
| `method_averages_<dataset>_<split>_<set>.csv` | For methods with several runs, only with `--null-policy first_class`: method rows × the number of runs, the null policy and the means over runs of the attribute averages, with confidence bounds that include the spread of the runs. |
| `per_attribute_<metric>_<dataset>_<split>_<set>.csv` | Attribute rows × run columns, with the average last. |
| `<run>/metrics_<set>.json` | All metrics, intervals, and per attribute the iRAP codes of the classes and, in their order, the per-class precision, recall, F1 and support and the confusion matrix (rows = ground truth). Undefined values are `null`. The directory is the run name with `/` replaced by `_`. |

A run is named `<method>` or `<method>/seed<seed>`. Run `irap-eval <command> --help` for all options.

## Python use

[`docs/prediction_format.md`](docs/prediction_format.md#examples) shows how to write prediction files. Scoring:

```python
from irap_data import load_irap_metadata
import irap_evaluation as ie

metadata = load_irap_metadata("/data/IRAP_Vietnam")
predictions = ie.read_predictions("resnet.predictions.parquet")
reference_set = ie.get_reference_set(metadata, "vietnam", "val")
result = ie.evaluate_predictions(predictions, reference_set)
intervals = ie.compute_bootstrap_intervals([result], num_resamples=1000)
print(result.metrics.averages["amF1"], intervals.averages["amF1"])
print(result.metrics.per_attribute["mF1"]["Lane width"])
class_metrics = ie.compute_class_metrics(result)

# Methods with several runs: means over runs, intervals and differences.
results = [ie.evaluate_predictions(ie.read_predictions(path), reference_set) for path in paths]
runs = ie.group_runs_by_method(results)
print(ie.compute_method_metrics(runs["resnet"]).averages["amF1"],
      ie.compute_bootstrap_intervals(runs["resnet"]).averages["amF1"])
differences = ie.compare_methods(runs["resnet"], runs["vit"], ["amF1"])
print(differences.averages["amF1"])  # MetricDifference(difference, interval)

# Other evaluation sets: the model set of some offsets, and a subset, e.g. one region.
model_set = ie.get_evaluation_set(metadata, "vietnam", "val", (0, -1, -4), name="model")
north_set = ie.select_evaluation_subset(reference_set, is_north, name="north")
```

Tables, JSON documents and the files of `irap-eval evaluate`:

```python
from irap_evaluation.reports import evaluation_report

long_table = evaluation_report.to_long_table([result])
print(evaluation_report.to_average_table(long_table))
evaluation_report.write_evaluation_reports([result], "results/", table_metrics=["mF1"])
```

## Coding tables

`export-coding-table` (`irap_evaluation.reports.coding_tables`) writes one row per predicted segment in the iRAP coding-table layout of `src/irap_evaluation/reports/data/coding_table_template.json`, or of `--template`. The attribute columns hold the iRAP code of the most probable class. The location columns come from the dataset metadata, and the end coordinates from the adjacent next segment. The other columns stay blank, and the `_comment` of the template says why.
