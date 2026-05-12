"""Standard subdirectory layout under an ``IRAP_Vietnam/`` dataset root.

All Vietnam pipeline scripts take a single positional ``<data_dir>`` argument (the
dataset root) and derive their inputs/outputs from these constants.
"""

from pathlib import Path

RAW_SUBDIR = "_raw"
WORK_SUBDIR = "_work"
IMAGES_SUBDIR = "images"

RARS_SUBDIR_UNDER_RAW = "image_rars"
ATTR_META_FILENAME = "attribute_metadata.json"
CODING_TABLES_ZIP = "coding-tables.zip"
ROWS_PARQUET = "rows.parquet"
PARSE_REPORT = "parse_report.json"
BUILD_REPORT = "build_report.json"
UNLABELED_SEG_IDS_FILENAME = "unlabeled_seg_ids.json"


def raw_dir(data_dir: Path) -> Path:
    return data_dir / RAW_SUBDIR


def work_dir(data_dir: Path) -> Path:
    return data_dir / WORK_SUBDIR


def images_dir(data_dir: Path) -> Path:
    return data_dir / IMAGES_SUBDIR


def metadata_dir(data_dir: Path) -> Path:
    return data_dir


def rars_dir(data_dir: Path) -> Path:
    return raw_dir(data_dir) / RARS_SUBDIR_UNDER_RAW


def attr_meta_path(data_dir: Path) -> Path:
    return raw_dir(data_dir) / ATTR_META_FILENAME


def rows_path(data_dir: Path) -> Path:
    return work_dir(data_dir) / ROWS_PARQUET


def parse_report_path(data_dir: Path) -> Path:
    return work_dir(data_dir) / PARSE_REPORT


def build_report_path(data_dir: Path) -> Path:
    return work_dir(data_dir) / BUILD_REPORT


def unlabeled_seg_ids_path(data_dir: Path) -> Path:
    return metadata_dir(data_dir) / UNLABELED_SEG_IDS_FILENAME
