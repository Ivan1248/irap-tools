# iRAP prediction file format

A prediction file is an [Apache Parquet](https://parquet.apache.org/) file holding the predictions of a method for a split of an iRAP dataset. Files are conventionally named `<method>.predictions.parquet`, or `<method>_seed<seed>.predictions.parquet` when distinguishing runs.

`irap_evaluation.prediction_io` reads and writes the format (`read_predictions`, `write_predictions`), represented in memory by `irap_evaluation.predictions.Predictions`.

## Columns

A file holds either predicted class distributions (`probs`) or predicted class indices (`hard`) for each segment.

| Column | Type | Content |
|---|---|---|
| `segment_id` | `string \| large_string` | Segment identifier. |
| `<attribute name>` (one per attribute) | `list<float32> \| int64 \| null` | Probability vector or class index (`null` for invalid prediction – [invalid cells](evaluation.md#invalid-cells)). |

There is one `<attribute name>` column for each attribute, named as in `attribute_metadata.json`.
All attribute columns share the same output kind:

- **`probs`**: a list with one value per class of the attribute. Values are $\ge 0$, and sum to 1 within $10^{-4}$. Writers should use `list<float32>` or `large_list<float32>` (e.g. from Polars). `float64` values are acceptable too.
- **`hard`**: a 0-based class index of the predicted class, represented as `int64` or some other integer type.

An invalid cell is `null`.

## Header

The header is a UTF-8 JSON object (without `NaN` or infinity) stored under the key `irap_predictions` in the file-level key-value metadata of the Parquet file.

In PyArrow, the file-level metadata is `pyarrow.parquet.read_metadata(path).metadata`. The schema metadata of a table read with `pyarrow.parquet.read_table` lacks the key in files written by some libraries, such as Polars.
Pandas cannot write key-value metadata. Pandas, Polars, and DuckDB drop it when reading and rewriting a file.

| Field | Type | Content |
|---|---|---|
| `format_version` | `int` | `1`. |
| `dataset` | `str` | Dataset identifier: `"bh"` or `"vietnam"`. |
| `split` | `str` | Segment split, e.g. `"val"`. |
| `context_offsets` | `list[int] \| None` | Distinct relative positions of sequence segments read by the model, e.g. `[0, -1, -4]`. Along with `dataset` and `split`, defines the model's [evaluation set](evaluation.md#evaluation-sets).Absent or `null` if unknown. |
| `method` | `dict` | Method information (see [Method field](#method-field)). |
| `attribute_to_irap_codes` | `dict[str, list[int]]` | Attribute column name → iRAP codes defining class order for `probs` lists and `hard` class indices. |

Only `context_offsets` is optional.

### Method field

| Field | Type | Content |
|---|---|---|
| `name` | `str` | Short name identifying the method in reports. |
| `seed` | `int \| None` | Run identifier, e.g. training seed. |
| `details` | `dict \| None` | Free-form JSON metadata. |

Only `name` is required.

#### Details field of an ensemble

An ensemble written by `irap_evaluation` is a `probs` file whose `context_offsets` are the union of its members' offsets (or null if any member's offsets are unknown). It records member provenance in `details.ensemble`:

```json
{"members": [{"method": ..., "context_offsets": ..., "weight": ...}, ...]}
```

## Examples

### Writing and reading with `irap_evaluation`

```python
import numpy as np
import irap_evaluation as ie

# Write predictions
header = ie.PredictionHeader(
  dataset="vietnam",
  split="val",
  context_offsets=(0, -1, -4),
  method=ie.MethodInfo(name="resnet-seq", seed=3)
)
predictions = ie.Predictions.from_logits(
  header,
  attribute_to_irap_codes = {"Lane width": (1, 2, 3)},
  segment_ids = ["<segment ID 1>", "<segment ID 2>"],
  logits = {"Lane width": np.array([[2.0, 0.1, -1.0], [0.3, 1.5, 0.2]])}
)
ie.write_predictions("resnet-seq_seed3.predictions.parquet", predictions)

# Read predictions
assert predictions == ie.read_predictions("resnet-seq_seed3.predictions.parquet")
```

For hard predictions (e.g. VLM answers), use `ie.Predictions.from_class_indices` with class indices or `ie.Predictions.from_irap_codes` with iRAP codes, using `irap_data.IGNORE_LABEL_INDEX` or `ie.INVALID_IRAP_CODE` (both −1) for missing answers.

### Writing without `irap_evaluation`

```python
import json
import polars as pl

header = dict(
  format_version=1,
  dataset="vietnam",
  split="val",
  method=dict(name="resnet-seq", seed=3),
  context_offsets=[0, -1, -4],
  attribute_to_irap_codes={"Lane width": [1, 2, 3]}
)

# Write predictions with Polars
predictions = pl.DataFrame(
  {
    "segment_id": ["<segment ID 1>", "<segment ID 2>"],
    "Lane width": [[0.85, 0.12, 0.03], [0.20, 0.70, 0.10]]
  },
  schema={"segment_id": pl.String, "Lane width": pl.List(pl.Float32)}
)
metadata = {"irap_predictions": json.dumps(header)}
predictions.write_parquet("resnet-seq_seed3.predictions.parquet", metadata=metadata)
```

For hard predictions, attribute columns are `pl.Int64` class indices, with `None` for invalid cells.
