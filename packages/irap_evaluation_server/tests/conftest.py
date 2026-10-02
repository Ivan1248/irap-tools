import sys
from pathlib import Path

import pytest

# The synthetic metadata and predictions of the irap_evaluation tests.
sys.path.insert(0, str(Path(__file__).parents[2] / "irap_evaluation" / "tests"))

from synthetic_vietnam import SPLIT, write_metadata_dir

from irap_evaluation_server.archive import SubmissionArchive
from irap_evaluation_server.config import DatasetConfig
from irap_evaluation_server.datasets import load_dataset_contexts


@pytest.fixture
def metadata_dir(tmp_path: Path) -> Path:
    return write_metadata_dir(tmp_path / "IRAP_Vietnam")


@pytest.fixture
def dataset_contexts(metadata_dir):
    return load_dataset_contexts({"vietnam": DatasetConfig(
        metadata_dir=metadata_dir, images_dir=None, analysis_splits=(SPLIT,))})


@pytest.fixture
def archive(tmp_path: Path) -> SubmissionArchive:
    return SubmissionArchive(tmp_path / "data")
