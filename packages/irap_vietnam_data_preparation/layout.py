"""Standard subdirectory layout under an ``IRAP_Vietnam/`` dataset root.

All Vietnam pipeline scripts take a single positional ``<data_dir>`` argument (the
dataset root) and derive their inputs/outputs from these constants.
"""

import re
from pathlib import Path

# The file name of a segment image, ``<video_dir>_seg<seg_id>.png``, with the groups
# video_dir and seg_id.
SEG_IMAGE_NAME_RE = re.compile(r"^(?P<video_dir>.*)_seg(?P<seg_id>\d+)\.png$", re.IGNORECASE)

RAW_SUBDIR = "_raw"
WORK_SUBDIR = "_work"
IMAGES_SUBDIR = "FRAMES"
IMAGES_DUPLICATES_SUBDIR = "FRAMES_duplicates"
PATCH_ARCHIVES_TMP_SUBDIR = "patch_archives_tmp"

RARS_SUBDIR_UNDER_RAW = "image_rars"
ATTR_META_FILENAME = "attribute_metadata.json"
CODING_TABLES_SUBDIR = "coding-tables"
ROWS_PARQUET = "rows.parquet"
PARSE_REPORT = "parse_report.json"
CROSS_ANNOTATOR_REPORT = "cross_annotator_report.json"
BUILD_REPORT = "build_report.json"
REPEATED_IMAGES_REPORT = "repeated_images_report.json"
UNLABELED_SEGMENT_IDS_FILENAME = "unlabeled_segment_ids.json"
UNLABELED_SEQUENCE_ID_TO_DATA_FILENAME = "unlabeled_sequence_id_to_data.json"
UNLABELED_UNLOCATED_SEGMENT_IDS_FILENAME = "unlabeled_unlocated_segment_ids.json"
SEGMENT_ID_TO_DATA_PATHS_REL_FILENAME = "segment_id_to_data_paths_rel.json"
SEGMENT_ID_TO_ROAD_DATA_FILENAME = "segment_id_to_road_data.json"
ROAD_ID_TO_SEGMENT_ID_SEQUENCE_FILENAME = "road_id_to_segment_id_sequence.json"
SPLITS_FILENAME = "splits.json"

# The keys of splits.json: the names of the labeled splits, the names of the unlabeled splits,
# which are the names of the labeled splits with the prefix, and UNLOCATED_SPLIT for the
# unlabeled segments without a location.
LABELED_SPLIT_NAMES = ("train", "val", "test")
UNLABELED_SPLIT_PREFIX = "unlabeled_"
UNLOCATED_SPLIT = "unlabeled_unlocated"


def is_unlabeled_split(split: str) -> bool:
    return split.startswith(UNLABELED_SPLIT_PREFIX)


def seg_id_from_image_name(file_name: str) -> str | None:
    """Returns the seg_id of a segment image file name without leading zeros, or None."""
    m = SEG_IMAGE_NAME_RE.match(file_name)
    return str(int(m["seg_id"])) if m else None


def video_dir_from_image_name(file_name: str) -> str | None:
    """Returns the video_dir of a segment image file name, or None.

    The video folder an image belongs in is recoverable from its name alone,
    independent of how deeply the image was wrapped inside its source archive.
    """
    m = SEG_IMAGE_NAME_RE.match(file_name)
    return m["video_dir"] if m else None


def index_images_by_seg_id(images_dir: Path) -> dict[str, list[Path]]:
    """Indexes the segment images under *images_dir*, at any depth, by seg_id."""
    index: dict[str, list[Path]] = {}
    for path in sorted(images_dir.rglob("*.png")):
        seg_id = seg_id_from_image_name(path.name)
        if seg_id is not None:
            index.setdefault(seg_id, []).append(path)
    return index


def raw_dir(data_dir: Path) -> Path:
    return data_dir / RAW_SUBDIR


def work_dir(data_dir: Path) -> Path:
    return data_dir / WORK_SUBDIR


def images_dir(data_dir: Path) -> Path:
    return data_dir / IMAGES_SUBDIR


def images_duplicates_dir(data_dir: Path) -> Path:
    return work_dir(data_dir) / IMAGES_DUPLICATES_SUBDIR


def patch_archives_tmp_dir(data_dir: Path) -> Path:
    return work_dir(data_dir) / PATCH_ARCHIVES_TMP_SUBDIR


def metadata_dir(data_dir: Path) -> Path:
    return data_dir


def rars_dir(data_dir: Path) -> Path:
    return raw_dir(data_dir) / RARS_SUBDIR_UNDER_RAW


def attr_meta_path(data_dir: Path) -> Path:
    return raw_dir(data_dir) / ATTR_META_FILENAME


def coding_tables_dir(data_dir: Path) -> Path:
    return raw_dir(data_dir) / CODING_TABLES_SUBDIR


def rows_path(data_dir: Path) -> Path:
    return work_dir(data_dir) / ROWS_PARQUET


def parse_report_path(data_dir: Path) -> Path:
    return work_dir(data_dir) / PARSE_REPORT


def cross_annotator_report_path(data_dir: Path) -> Path:
    return work_dir(data_dir) / CROSS_ANNOTATOR_REPORT


def build_report_path(data_dir: Path) -> Path:
    return work_dir(data_dir) / BUILD_REPORT


def repeated_images_report_path(data_dir: Path) -> Path:
    return work_dir(data_dir) / REPEATED_IMAGES_REPORT


def unlabeled_segment_ids_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / UNLABELED_SEGMENT_IDS_FILENAME


def unlabeled_sequence_id_to_data_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / UNLABELED_SEQUENCE_ID_TO_DATA_FILENAME


def unlabeled_unlocated_segment_ids_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / UNLABELED_UNLOCATED_SEGMENT_IDS_FILENAME


def output_attr_meta_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / ATTR_META_FILENAME


def segment_id_to_data_paths_rel_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / SEGMENT_ID_TO_DATA_PATHS_REL_FILENAME


def segment_id_to_road_data_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / SEGMENT_ID_TO_ROAD_DATA_FILENAME


def road_id_to_segment_id_sequence_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / ROAD_ID_TO_SEGMENT_ID_SEQUENCE_FILENAME


def splits_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / SPLITS_FILENAME
