# Equivalence with `irap_gaim/dataset_wrapper.py`

Date: 2026-05-24, updated 2026-11-23

This document compares `irap_data/src/irap_data/irap_dataset.py` (`IRAPDataset` / `make_bh_data`) against the original [`irap_gaim/dataset_wrapper.py`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py) (`DatasetWrapper`), [`image_sequence_dataset.py`](https://github.com/mkacan/irap_gaim/blob/main/image_sequence_dataset.py) (`ImageSequenceDataset`), and the [`train_local_rec.py`](https://github.com/mkacan/irap_gaim/blob/main/train_local_rec.py) driver. The comparison is restricted to the **standard supervised iRAP-BH configuration** defined by the original [`irap_gaim/config.json`](https://github.com/mkacan/irap_gaim/blob/main/config.json):

```jsonc
// excerpt
"context_sequence":          [0, -1, -4],
"data_types":                ["rgb"],
"input_dim": { "rgb":        [384, 288, 3] },
"attribute_value_mapping_path": "",        // -> identity mapping branch
"normalization_statistics":  { "mean": [...BH...], "std": [...BH...] }
```

`make_bh_data(use_ncontext_filter=True)` with the other defaults is the new-code counterpart of that config (`context_offsets = (0, -1, -4)`, `allow_missing_attributes = False`, `mean/std = RGB_MEAN/STD`, `input_dim_rgb = (384, 288, 3)`, `transforms = None`).

Segment counts are given as train / val / test. They were measured on 2026-11-20 and checked independently on 2026-11-23 on `IRAP_BIH_METADATA`, the default metadata of `make_bh_data`, whose `splits.json` has 215,481 / 5,898 / 6,614 segments. Without the N-context filter, the selection rules of the original give exactly the key sets of `seg_to_res/<split>.pickle`. The iRAP-BH release on Zenodo (`IRAP_BH`) contains only the segments of `make_bh_data(use_ncontext_filter=True)`.

## Equivalent behavior

| Step | Original | New |
|---|---|---|
| Drop split entries with no data path | [`dataset_wrapper.py:13-22`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L13-L22) | `metadata.compute_split_labels` |
| `seg_to_res/{split}.pickle` N-context filter, `max_N = 10`, per-split, union over splits | [`train_local_rec.py:196-229`](https://github.com/mkacan/irap_gaim/blob/main/train_local_rec.py#L196-L229) → [`n_context_dataset.py:4-35`](https://github.com/mkacan/irap_gaim/blob/main/n_context_dataset.py#L4-L35) | `metadata.load_ncontext_segment_ids` |
| Label construction (identity remap) | [`dataset_wrapper.py:189-272`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L189-L272) (`_load_default_attribute_metadata` + `_map_attribute_values` + `_get_segment_id_to_attribute_value`) | `metadata.ClassVocabulary` and `metadata.compute_segment_labels` |
| Per-attribute class count | `len(attribute_value_to_irap_number[attr])` ([`dataset_wrapper.py:289-294`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L289-L294)) | the same count (`ClassVocabulary.class_counts`) |
| Attribute ordering | sorted by `attribute_to_idx` ([`dataset_wrapper.py:355`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L355)) | sorted by `attribute_to_idx` (`ClassVocabulary.from_attribute_metadata`) |
| Drop segments with unknown iRAP codes (663 / 60 / 35 of the segments with an image) | `KeyError` in [`dataset_wrapper.py:256-272`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L256-L272) | `metadata.compute_segment_labels` with `allow_missing_attributes=False` |
| Apply `ncontext_segment_id_subset` | [`dataset_wrapper.py:57-62`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L57-L62) | `metadata.select_split_segments` |
| Context-frame offsets `(0, -1, -4)` ordering | iterated in `context_sequence` order (`config.json` key) | iterated in `context_offsets` order |

The filter by a valid context window is not in this table, because the two versions find the context frames differently ([§ 1](#1-context-id-resolution-mechanism)). With the N-context filter, the selected segments are still the same: 209,459 / 5,509 / 6,341 in both.

[`_remove_filtered_out_segments`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L370-L381) is a no-op under identity remap, so the new version's omission of it is equivalent.

The following differences also have no effect under the standard iRAP-BH configuration:

- **Filtering order.** The original ([`dataset_wrapper.py:13-91`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L13-L91)) filters in the order data paths → context → `ncontext_subset` → label mapping → two more `_filter_segments` passes. The new code filters in the order data paths → label mapping → context → `ncontext_subset`. The final intersection is the same.
- **Missing-attribute handling.** The original raises a `KeyError` in [`_create_segment_id_to_labels`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L339-L362) when `required_attributes` is incomplete. The new code drops the segment instead (or labels it `-1` when `allow_missing_attributes=True`). iRAP-BH has complete attribute coverage, so neither branch fires.
- **Sample-dict shape.** The original ([`image_sequence_dataset.py:52-72`](https://github.com/mkacan/irap_gaim/blob/main/image_sequence_dataset.py#L52-L72)) emits `{rgb, target, segment_id}` plus optional `depth`. The new code emits `{rgb, target, segment_id, sequence_id}` and never `depth`. With `data_types: ["rgb"]` the original also never produces `depth`, and `sequence_id` (road id) is purely additive.

## Behavioral differences

### 1. Context-id resolution mechanism

| | Original | New |
|---|---|---|
| Method | integer arithmetic on segment IDs: `str(int(sid) + offset)` | road-sequence lookup via `road_id_to_segment_id_sequence.json` |
| Validity domain for a context frame | `int(seg_id) ∈ set(map(int, splits['all']))` (i.e. labelled `train ∪ val ∪ test` after the data-path filter) | `cid ∈ seg_to_paths` (every segment with an image on disk) |

The original code filters each split by this context window in [`dataset_wrapper.py:38-55`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L38-L55), the new code in `metadata.select_split_segments`. The road-based lookup cannot pull a frame from a different road, which allows the same code to handle iRAP-Vietnam.

The two methods give the same context frames only if (a) segment IDs are consecutive along each road, (b) IDs of different roads are not adjacent, and (c) every `segment_id_to_data_paths_rel.json` entry is in a labeled split. On iRAP-BH, (c) holds, and (a) and (b) do not:

- **(a)** Neighbors in a road sequence have non-consecutive IDs at 1,827 places. At 1,816 of them, the next segment is a later frame of the same video. The road sequences are thus in recording order, and ID arithmetic selects unrelated frames at these places.
- **(b)** For 99 of the 190 non-empty road sequences, the segment with the ID before that of the first segment has an image. For 80 of these 99, it is on another road.

Effects under the standard configuration:

| | `use_ncontext_filter=False` | `use_ncontext_filter=True` |
|---|---|---|
| Segments selected by the original | 214,073 / 5,813 / 6,563 | 209,459 / 5,509 / 6,341 |
| Segments selected by the new code | 213,755 / 5,775 / 6,555 | 209,459 / 5,509 / 6,341 |
| Selected only by the original | 380 / 38 / 8 | 0 / 0 / 0 |
| Selected only by the new code | 62 / 0 / 0 | 0 / 0 / 0 |
| Selected by both, with different context frames | 2,239 / 12 / 8 | 2,131 / 12 / 8 |

- **Selected only by the original.** They are among the first 4 segments of their road, so they have no complete context window on it. The segments with the IDs `id - 1` and `id - 4` have images, but for 350 of the 426 segments at least one of them is on another road.
- **Selected only by the new code.** At least one of the segments with the IDs `id - 1` and `id - 4` has no image, while the segments 1 and 4 positions earlier on the road have images.
- **N-context filter.** It keeps only the segments of a split's `seg_to_res` file that are at least 10 positions from both ends of their road sequence and whose 10 neighbors on each side are in the same file. This removes every segment that only one version selects: the segments selected only by the original are among the first 4 of their road, and the segments selected only by the new code are in no `seg_to_res` file.
- **Different context frames.** At least one of the frames that the original reads, at `id - 1` and `id - 4`, is not 1 or 4 positions earlier on the road. This affects about 1 % of the training segments, with or without the N-context filter.

### 2. Image resize / crop

| | Original ([`dataset_wrapper.py:141-144`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L141-L144)) | New (`image_utils.py`) |
|---|---|---|
| Resize | none | to the target width, preserving the aspect ratio (Lanczos), by default (`fit_mode="fit_width"`) |
| Crop | `CenterCrop((H, W))` only when `input_dim['rgb'] != [384, 284, 3]` | the height is center-cropped or padded with the mean color to the target height, by default |

The `[384, 284, 3]` literal looks like a typo for `288`: `input_dim.rgb` is `[384, 288, 3]`, so the `!=` branch fires and the original does crop.

For iRAP-BH, source images are at exactly `384 × 288`, so the default fit leaves pixels unchanged, as the original does (the only remaining difference is that the original emits a standardized tensor rather than a `[0, 1]` `float32` – see §3). `fit_mode="center_crop"` reproduces the crop of the original for any source size. For 1920×1080 iRAP-Vietnam images, that crop keeps only about 5% of the image, which is why the default fit resizes.

### 3. Augmentation and normalization moved out of the dataset

The original applies, inside `__getitem__`:

- [`ColorJitter(0.6, 0.3, 0.2, 0.02)`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L139-L140) on the `train` split, and
- [`Normalize(mean, std)`](https://github.com/mkacan/irap_gaim/blob/main/dataset_wrapper.py#L145) on every split, with iRAP-BH mean/std from [`config.json:11-15`](https://github.com/mkacan/irap_gaim/blob/main/config.json#L11-L15).

`IRAPDataset` does neither: `rgb` is deterministic, unnormalized, `float32` in `[0, 1]`. The iRAP-BH mean/std are exposed as `RGB_MEAN` / `RGB_STD` and via `info.pixel_stats`, so the caller owns augmentation and standardization. To match the original numerically, the caller must apply that same `ColorJitter` to the "train" split and standardize every split with `info.pixel_stats` – outside `irap_data`'s scope.

## Bottom line

Under the standard iRAP-BH configuration with `use_ncontext_filter=True`, `IRAPDataset` matches the original in **segment selection, per-attribute labels, class counts, and the order of the 3 frames**. Without the N-context filter, the selected segments differ at the starts of road sequences ([§ 1](#1-context-id-resolution-mechanism)).

The context frames differ for 2,131 / 12 / 8 of the selected segments ([§ 1](#1-context-id-resolution-mechanism)), so their sample tensors cannot be bit-identical.

For the other segments, bit-identical sample tensors additionally require that the caller applies `ColorJitter(0.6, 0.3, 0.2, 0.02)` to the train split and standardizes with `info.pixel_stats` ([§ 3](#3-augmentation-and-normalization-moved-out-of-the-dataset)), which is outside the scope of `irap_data`.
