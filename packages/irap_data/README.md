# irap_data

Standalone Python package for loading the iRAP road-attribute datasets (iRAP-BiH, iRAP-Vietnam) and their metadata, and an interactive Streamlit-based viewer. This code is based on code by Marin Kačan: see [docs/differences-to-the-original-dataset-code.md](docs/differences-to-the-original-dataset-code.md) for a comparison.


## Installation

From the irap-tools checkout:

```bash
uv pip install -e packages/irap_data           # metadata API only (numpy)
uv pip install -e packages/irap_data[torch]    # adds the dataset classes and image utilities
uv pip install -e packages/irap_data[viewer]   # adds streamlit for the dataset viewer
```

From another project, without a checkout:

```bash
uv pip install "irap-data[torch] @ git+https://github.com/Ivan1248/irap-tools#subdirectory=packages/irap_data"
```

The metadata API – `irap_data.metadata` (class vocabularies, labels, split and context selection) and `irap_data.attrs` – needs only numpy, and `import irap_data` does not import torch. The dataset classes (`IRAPDataset`, `make_*_data`, `InferenceImageDataset`), `irap_data.jitter` and the viewer need the `torch` extra.

## Data layout

Set `IRAP_HOME` (or `DATASETS_PATH`) to a parent directory containing:

- iRAP-BiH: `IRAP_BIH/` (images) and `IRAP_BIH_METADATA/` (metadata JSONs as a sibling)
- iRAP-Vietnam: `IRAP_Vietnam/` (images and metadata together)

Required metadata files (in the metadata dir):

- `splits.json`
- `segment_id_to_data_paths_rel.json`
- `segment_id_to_road_data.json`
- `attribute_metadata.json`
- `road_id_to_segment_id_sequence.json`

### Context-window filtering

`irap_data.metadata.select_split_segments` implements the segment selection of every loader. It drops segments whose context window (per `context_offsets`) would step off the end of its sequence in `road_id_to_segment_id_sequence.json`, or whose context frames have no image on disk. This is the only context filter `make_vietnam_data()` applies.

`make_bih_data()` additionally restricts segments to the precomputed BiH subsets in `seg_to_res/{train,val,test}.pickle`. These pickles retain only segments that have at least 10 valid context frames on each side within their road sequence, so the labeled-set size is stable for any `max(abs(context_offsets)) ≤ 10`. Pass `use_ncontext_filter=False` to skip the pre-computed filter. The iRAP-Vietnam dataset does not provide such a filter.

### Reference context

`DATASET_PRESETS` holds the loading defaults of each release, including `reference_context_offsets`. The reference set of a split is the segments whose context window with these offsets is complete (`select_preset_split_segments(metadata, name, split)`). A model whose context offsets lie within the reference window can predict every reference segment, so all such models can be scored on the same segments.

### Unlabeled splits

`splits.json` may contain extra keys:

- `unlabeled_train`, `unlabeled_val`, `unlabeled_test` – unlabeled segments assigned to a split by geography (unlabeled segments from the same sequences or geographic areas as the labeled subsets).
- `unlabeled_unlocated` – unlabeled segments from folders with unknown geographic relationship to other splits.

When present, `make_vietnam_data` includes these splits in the returned dict. Unlabeled subsets are loaded with `allow_missing_attributes=True`, so every sample's `target` is an all-`-1` tensor compatible with `torch.nn.CrossEntropyLoss(ignore_index=-1)`. The context-window check still applies; the BiH `seg_to_res/*.pickle` allow-list is bypassed because it only enumerates labeled segments.

## Quickstart

```python
from irap_data import make_bih_data, make_vietnam_data

# iRAP-BiH (uses $IRAP_HOME by default)
splits = make_bih_data(context_offsets=(0, -1, -4))
train = splits["train"]
print(len(train), train.info.class_counts)

example = train[0]
# example: {"rgb": (S, 3, H, W) float tensor in [0, 1],
#           "target": (A,) LongTensor, -1 = unlabeled,
#           "segment_id": str, "sequence_id": str}

# iRAP-Vietnam (loose labels – missing attributes become -1)
vn = make_vietnam_data(dataset_dir="/data/IRAP_Vietnam")
```

Labels and segment selection without loading images (no torch needed):

```python
from irap_data import load_irap_metadata, select_preset_split_segments

metadata = load_irap_metadata("/data/IRAP_Vietnam")
val = select_preset_split_segments(metadata, "vietnam", "val", context_offsets=(0, -1, -4))
print(len(val.segment_ids), metadata.vocabulary.class_counts)
```

Inference on a folder of images (no labels required):

```python
from irap_data import InferenceImageDataset

ds = InferenceImageDataset.from_folder(
    "/path/to/images",
    reference_dataset=splits["test"],   # borrows attribute metadata
    context_offsets=(0, -1, -4),
)
```

## Dataset viewer

```bash
IRAP_HOME=/path/to/IRAP_HOME streamlit run packages/irap_data/irap_data/dataset_viewer.py
```

Or, once `irap_data` is installed:

```bash
python -m streamlit run -m irap_data.dataset_viewer
```

The viewer auto-detects which datasets exist under `$IRAP_HOME`, exposes per-attribute filtering, navigates by index or random sample, and previews the surrounding road sequence for each segment.

## Package contents

```
irap_data/
├── __init__.py                # Public API (torch-based names are imported on first access)
├── metadata.py                # ClassVocabulary, IRAPMetadata, labels, split and context selection, DATASET_PRESETS
├── dataset.py                 # Dataset base class + transformations (map/filter/zip/...)
├── irap_dataset.py            # IRAPDataset, make_bih_data, make_vietnam_data
├── inference_dataset.py       # InferenceImageDataset (label-free folder loader)
├── attrs.py                   # IRAP_BH_ATTRS_TO_INCLUDE + name-to-index helpers
├── attribute_frequencies.py   # AttributeFrequencyStats, compute_attribute_frequency_stats
├── image_utils.py             # load_image_cv2, center_crop, resize_to_cover, hwc_to_chw_float_tensor
├── lazy_dict.py               # LazyDict, Lazy
├── vis_utils.py               # AttributeMetadataDecoder + color/composite helpers
├── jitter.py                  # make_sequence_color_jitter, JITTER_STANDARD, JITTER_STRONG
└── dataset_viewer.py          # Streamlit viewer (entry point)
```

## Conventions

- The normalization constants (`RGB_MEAN`, `RGB_STD`, `INPUT_DIM`) live in `irap_data.irap_dataset`, and the ignore sentinel (`IGNORE_LABEL_INDEX = -1`) in `irap_data.metadata`.
- Class indices follow the value order of `attribute_value_to_irap_number` in `attribute_metadata.json` (see `ClassVocabulary`). IRAP codes are parsed to ints, whether the metadata stores them as ints (Vietnam) or strings (BiH).
- `Dataset.info` is a `LazyDict` with attribute access – both `info["class_counts"]` and `info.class_counts` work. Entries wrapped in `Lazy(...)` are computed on first access.
- Targets use `-1` as the ignore index (matches `torch.nn.CrossEntropyLoss(ignore_index=-1)`).
