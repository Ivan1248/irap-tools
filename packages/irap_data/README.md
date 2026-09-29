# irap_data

Loaders, metadata and tools for the iRAP road-attribute datasets iRAP-BiH and iRAP-Vietnam: a torch-free metadata API, PyTorch datasets, a Streamlit dataset viewer and a statistics report. The code is based on code by Marin Kačan (see [docs/differences-to-the-original-dataset-code.md](docs/differences-to-the-original-dataset-code.md)).

## Installation

```bash
uv pip install -e packages/irap_data           # from an irap-tools checkout
uv pip install "irap-data[torch] @ git+https://github.com/Ivan1248/irap-tools#subdirectory=packages/irap_data"
```

| Extra | Adds |
|---|---|
| none | The metadata API: `irap_data.metadata`, `irap_data.attrs`, `irap_data.dataset_statistics` (numpy only). `import irap_data` does not import torch. |
| `torch` | The datasets (`IRAPDataset`, `make_*_data`, `InferenceImageDataset`), image utilities and `irap_data.jitter` |
| `viewer` | The dataset viewer (includes `torch`) |
| `report` | The plots of the statistics report (matplotlib) |

## Data layout

Set `IRAP_HOME` (or `DATASETS_PATH`) to a directory that contains:

- `IRAP_BIH/` (images) and `IRAP_BIH_METADATA/` (metadata) for iRAP-BiH,
- `IRAP_Vietnam/` (images and metadata) for iRAP-Vietnam.

A metadata directory contains `splits.json`, `segment_id_to_data_paths_rel.json`, `segment_id_to_road_data.json`, `attribute_metadata.json` and `road_id_to_segment_id_sequence.json`.

## Usage

```python
from irap_data import make_bih_data, make_vietnam_data

splits = make_bih_data(context_offsets=(0, -1, -4))  # directories from $IRAP_HOME
train = splits["train"]
print(len(train), train.info.class_counts)

example = train[0]
# {"rgb": (S, 3, H, W) float tensor in [0, 1], "target": (A,) LongTensor (-1 = unlabeled),
#  "segment_id": str, "sequence_id": str}

vn = make_vietnam_data(dataset_dir="/data/IRAP_Vietnam")  # missing attributes become -1
```

Labels and segment selection without images or torch:

```python
from irap_data import load_irap_metadata, select_preset_split_segments

metadata = load_irap_metadata("/data/IRAP_Vietnam")
val = select_preset_split_segments(metadata, "vietnam", "val", context_offsets=(0, -1, -4))
print(len(val.segment_ids), metadata.vocabulary.class_counts)
```

A folder of unlabeled images, ordered by file name:

```python
from irap_data import InferenceImageDataset

ds = InferenceImageDataset.from_folder(
    "/path/to/images", reference_dataset=splits["test"], context_offsets=(0, -1, -4))
```

### Segment selection

`irap_data.metadata.select_split_segments` selects the segments of a split for every loader. It keeps a segment if it has an image, labels, and a complete context window: every context offset falls inside its road sequence and has an image.

- **BiH N-context filter.** `make_bih_data()` also restricts the labeled splits to the precomputed subsets in `seg_to_res/{train,val,test}.pickle`. These subsets hold the segments with at least 10 valid context frames on each side, so the labeled set does not change for any `max(abs(context_offsets)) ≤ 10`. `use_ncontext_filter=False` disables the filter. iRAP-Vietnam has no such filter.
- **Reference set.** `DATASET_PRESETS` holds the loading defaults of each release, including `reference_context_offsets`. The reference set of a split is the segments with a complete context window for these offsets (`select_preset_split_segments` with `context_offsets=None`). Every model whose context offsets lie within the reference window can be scored on the same segments.
- **Unlabeled splits.** `splits.json` can contain `unlabeled_train`, `unlabeled_val` and `unlabeled_test` (unlabeled segments from the same sequences or areas as the labeled split) and `unlabeled_unlocated` (segments with an unknown relationship to the other splits). The loaders return them with all targets `-1`. The context-window check applies to them, the N-context filter does not.

### Conventions

- Targets use `-1` (`irap_data.metadata.IGNORE_LABEL_INDEX`) for a missing label, as `torch.nn.CrossEntropyLoss(ignore_index=-1)` expects.
- Class indices follow the value order of `attribute_value_to_irap_number` in `attribute_metadata.json` (see `ClassVocabulary`).
- `Dataset.info` is a `LazyDict`: `info["class_counts"]` and `info.class_counts` are equivalent, and `Lazy(...)` entries are computed on first access.
- The normalization constants `RGB_MEAN`, `RGB_STD` and `INPUT_DIM` are in `irap_data.irap_dataset`.

## Dataset viewer

```bash
IRAP_HOME=/path/to/IRAP_HOME streamlit run -m irap_data.dataset_viewer  # Streamlit >= 1.41
IRAP_HOME=/path/to/IRAP_HOME streamlit run packages/irap_data/irap_data/dataset_viewer.py
```

The viewer finds the datasets under `$IRAP_HOME`, filters segments by attribute value, and shows the road sequence around each segment.

## Dataset statistics report

`irap-dataset-stats` computes the class frequencies and other statistics of a release from its metadata, without images or torch.

```bash
irap-dataset-stats report vietnam -o reports/vietnam
irap-dataset-stats report bih --metadata-dir /data/IRAP_BIH_METADATA -o reports/bih --context-offsets 0,-1,-4
irap-dataset-stats plot vietnam --split train --sort-attributes imbalance --sort-classes count -o vietnam_skew.pdf
```

`report` writes one report for each selection of segments (see `SELECTION_DESCRIPTIONS` in `irap_data.dataset_statistics`):

| Selection | Segments | Report |
|---|---|---|
| `labeled` | With an image and labels | `<dataset>_labeled_segments` |
| `reference` | The [reference set](#segment-selection) | `<dataset>_labeled_segments_with_context` |
| `model` | With a complete context window for `--context-offsets`; written only if it differs from `reference` | `<dataset>_labeled_segments_with_context_<offsets>` |

Each report is an HTML page (or Markdown with `--format md`) with a class-frequency plot per split (`--plot-format`, `--no-plots`) and a `<report>_class_frequencies.csv` table. It has these sections: overview (split sizes, road sequences and length), label coverage, class balance relative to the base split (the first split of `splits.json` or `--splits`), rare classes (below `--rare-fraction`, default 1 %), missing classes, and per-attribute class frequencies. `<dataset>_statistics.json` holds all statistics.

Segments on the same road are strongly correlated, so the number of road sequences of a class is closer to its number of independent examples than the number of segments.

`plot` draws the class frequencies of one split in one figure, in the format given by the extension of `-o`. A class without segments is drawn as a short red bar.
