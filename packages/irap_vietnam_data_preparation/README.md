# iRAP-Vietnam data preparation

Scripts that turn the Vietnam panoramic images and Excel coding tables into a dataset compatible with the iRAP-BH format. [`vietnam_data_preparation.md`](vietnam_data_preparation.md) gives the reasons for the design decisions.

## Requirements

- `unrar` (preferred) or `7z` on `PATH`.
- Python packages: `pandas openpyxl xlrd pyarrow tqdm numpy`. The split editor also needs `streamlit>=1.35 plotly`, and the cross-annotator evaluation needs `scikit-learn`.
- A dataset root directory, `<data_dir>`. Put `coding-tables.zip` (or an unzipped `coding-tables/` directory) and `attribute_metadata.json` from the [releases](https://github.com/Ivan1248/irap-tools/releases) in `<data_dir>/_raw/`.

## Directory layout

```
<data_dir>/
  _raw/                                  # inputs
    coding-tables.zip
    attribute_metadata.json
    image_rars/                          # step 1
  _work/                                 # intermediate files and reports, deletable
    rows.parquet, parse_report.json      # step 3
    cross_annotator_report.json          # step 3, optional
    build_report.json                    # step 4
    FRAMES_duplicates/                   # step 2, with --ignore-duplicates
  FRAMES/<video_dir>/<video_dir>_seg<N>.png   # step 2
  segment_id_to_data_paths_rel.json      # step 4
  segment_id_to_road_data.json
  road_id_to_segment_id_sequence.json
  attribute_metadata.json
  unlabeled_segment_ids.json
  unlabeled_sequence_id_to_data.json
  unlabeled_unlocated_segment_ids.json
  splits.json                            # step 5
```

## Steps

Run the scripts from this directory, with `DATA_DIR` set to the dataset root.

### 1. Download the image archives

```bash
python download_images.py $DATA_DIR
```

Downloads about 110 GB of `split*.rar` archives from Seafile. An interrupted download resumes, and a file whose size already matches is skipped.

### 2. Extract the images

```bash
python extract_images.py $DATA_DIR --ignore-duplicates
```

- Writes the images to `FRAMES/<video_dir>/`, without the `splitN/` directory that wraps them in the archives. If `FRAMES/` is not empty, it asks for confirmation before it deletes it (`--yes` skips the question).
- Stops if two archives contain the same path. With `--ignore-duplicates`, the first archive wins and all copies go to `_work/FRAMES_duplicates/` for review.
- Applies the `missing_segments*.rar` archives last. Their images replace or add files in `FRAMES/`.

### 3. Parse the coding tables

```bash
python parse_coding_tables.py $DATA_DIR
```

A coding table is an Excel sheet with one row per 20 m segment, for example:

| Section | Distance | Length | Latitude start | Longitude start | Image Reference FPZ | Comments | Carriageway | Number of lanes | … |
|---|---|---|---|---|---|---|---|---|---|
| VID_20241211_153327_00_024 | 0.00 | 0.02 | 10.372859 | 105.449173 | [seg. no. 1809249](…) | | 3 | 1 | … |
| VID_20241211_153327_00_024 | 0.02 | 0.02 | 10.372734 | 105.449042 | [seg. no. 1809251](…) | | 3 | 1 | … |

- Distance and length are in km.
- The `seg_id` in `Image Reference FPZ` identifies the image, e.g. `FRAMES/VID_20241211_153327_00_024/VID_20241211_153327_00_024_seg1809249.png`.
- The images between two rows, e.g. `…_seg1809250.png`, have no row and become unlabeled segments.
- The attribute columns, from `Carriageway` to just before `Additional comments`, hold iRAP codes. `attribute_metadata.json` maps the codes to values, e.g. `3` is an undivided road. `incompatible_attributes.json` maps each iRAP-BH attribute name to its header where the two differ, e.g. `Carriageway label` to `Carriageway`.

Writes one row per segment to `_work/rows.parquet` and a summary to `_work/parse_report.json`. The docstring of `parse_coding_tables.py` lists the rules for dropped rows. A duplicate `seg_id` keeps the row with the most coded attributes. A blank attribute cell becomes `-1`, which is an _ignore_ label. `--drop-rows-missing-attributes` drops such rows instead.

Optional: `python cross_annotator_eval.py $DATA_DIR` measures the agreement between the coding tables that code the same segments. It writes `_work/cross_annotator_report.json`.

### 4. Build the metadata

```bash
python build_metadata.py $DATA_DIR
```

- **Matching:** matches each row to an image by `seg_id` and drops rows without an image. Images without a corresponding coding table row become unlabeled segments.
- **Road sequences:** builds one road sequence per section. Adjacent labeled segments must be continuous: each `seg_id` step is 10 m of distance (± 2.5 m). A section with a gap is split into `<section>__part<N>` road sequences, so that context windows never span a gap.
- **Unlabeled images:** an unlabeled image is placed into a road sequence if it lies between two adjacent labeled segments of the same recording. A recording is an image folder, or a run of consecutive `seg_id`s in a folder that holds several recordings (such as `FRAMES/nan/`). The unlabeled images of a recording without labeled segments have no map position and are marked as unlocated.

To check what a code change does to the metadata, copy the generated files to another directory before the rebuild, and compare the two builds:

```bash
python compare_metadata.py <old_dir> $DATA_DIR
```

### 5. Assign the splits

```bash
streamlit run split_editor.py -- $DATA_DIR
```

The editor shows the road sequences on a map, and the unlabeled recordings in muted colors. Pick a split in the sidebar and draw a rectangle: all sequences whose centroid is inside go to that split. **Save** writes `splits.json`. An existing `splits.json` is the starting assignment, so the editor can be opened again after a rebuild.

The sidebar shows the classes that each split misses, where a class is an (attribute, iRAP code) pair. Support is counted in road sequences, as the segments of one road are near-duplicates. *Fixable* classes occur in enough sequences to be in every split.

#### `splits.json` format

`{<split>: [seg_id, ...]}`, with seg_ids as strings:

- `train`, `val`, `test`: labeled segments.
- `unlabeled_train`, `unlabeled_val`, `unlabeled_test`: unlabeled recordings, assigned in the editor.
- `unlabeled_unlocated`: unlabeled recordings without a labeled segment, so without a map position. They are added automatically.

### 6. Clean up

After the dataset is verified, `_raw/` and `_work/` can be deleted.

Re-running the pipeline needs no cleanup: each step overwrites its outputs, and the split editor starts from the existing `splits.json`.
