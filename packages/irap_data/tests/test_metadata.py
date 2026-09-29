"""Tests of the torch-free metadata API in `irap_data.metadata`."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from irap_data.metadata import (
    DATASET_PRESETS,
    IGNORE_LABEL_INDEX,
    ClassVocabulary,
    MetaFiles,
    compute_segment_context_ids,
    compute_segment_labels,
    get_segment_location,
    load_irap_metadata,
    select_preset_split_segments,
    select_split_segments,
    to_irap_code,
)

ATTRS = ("A", "B")
# BiH stores codes as strings, and class order is the value order, not the code order.
ATTRIBUTE_METADATA = {
    "attribute_to_idx": {"B": 1, "A": 0},
    "attribute_value_to_irap_number": {"A": {"a20": "20", "a10": "10"},
                                       "B": {"b1": 1, "b2": 2, "b3": 3}},
}


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def metadata_dir(tmp_path: Path) -> Path:
    """Two roads: r0 = S0..S5 (S3 has no image), r1 = T0..T1. S4 misses the code of B."""
    road_to_seq = {"r0": [f"S{i}" for i in range(6)], "r1": ["T0", "T1"]}
    all_segments = [s for seq in road_to_seq.values() for s in seq]
    codes = {sid: {"A": 10, "B": 2} for sid in all_segments}
    codes["S1"] = {"A": "20", "B": "3"}
    codes["S4"] = {"A": 10, "B": -1}
    _write_json(tmp_path / MetaFiles.ATTRIBUTE_METADATA, ATTRIBUTE_METADATA)
    _write_json(tmp_path / MetaFiles.SEGMENT_ID_TO_DATA_PATHS,
                {sid: {"rgb": f"{sid}.png"} for sid in all_segments if sid != "S3"})
    _write_json(tmp_path / MetaFiles.SEGMENT_ID_TO_ROAD_DATA, {
        sid: {"section": "r0", "distance_km": 0.02 * i, "length_km": 0.02, "lat": 1.0,
              "lon": 2.0, "required_attributes": codes[sid]}
        for i, sid in enumerate(all_segments)})
    _write_json(tmp_path / MetaFiles.ROAD_ID_TO_SEGMENT_ID_SEQUENCE, road_to_seq)
    _write_json(tmp_path / MetaFiles.SPLITS, {"val": all_segments, "unlabeled_val": []})
    return tmp_path


def test_to_irap_code():
    assert [to_irap_code(v) for v in (3, "3", " 7 ", -1, "-1", "None", None)] == \
           [3, 3, 7, None, None, None, None]
    # A value no writer should produce is an error rather than a silently dropped label.
    for value in (True, 2.0, "2.0", "N/A", ""):
        with pytest.raises(ValueError, match="Not an IRAP code"):
            to_irap_code(value)


def test_vocabulary_equality_depends_on_order():
    vocabulary = ClassVocabulary({"a": {"x": 10, "y": 20}, "b": {"z": 1}})
    assert vocabulary == ClassVocabulary({"a": {"x": 10, "y": 20}, "b": {"z": 1}})
    assert vocabulary != ClassVocabulary({"a": {"y": 20, "x": 10}, "b": {"z": 1}})
    assert vocabulary != ClassVocabulary({"b": {"z": 1}, "a": {"x": 10, "y": 20}})
    with pytest.raises(TypeError):
        hash(vocabulary)


def test_metadata_caches_the_sequence_index(metadata_dir):
    metadata = load_irap_metadata(metadata_dir)
    assert metadata.sequence_index["S2"] == ("r0", 2)
    assert metadata.sequence_index is metadata.sequence_index


def test_torch_names_are_in_all():
    import irap_data

    assert {"IRAPDataset", "make_bih_data", "ClassVocabulary"} <= set(irap_data.__all__)
    assert not {"importlib", "T", "metadata"} & set(irap_data.__all__)


def test_vocabulary_follows_attribute_index_and_value_order():
    vocabulary = ClassVocabulary.from_attribute_metadata(ATTRIBUTE_METADATA)
    assert vocabulary.attribute_names == ATTRS
    assert vocabulary.get_irap_codes("A") == (20, 10)
    assert vocabulary.get_values("A") == ("a20", "a10")
    assert vocabulary.class_counts == (2, 3)
    assert vocabulary.attr_to_value_to_class_idx["A"] == {"a20": 0, "a10": 1}
    assert vocabulary.restrict_to_attributes(["B"]).attribute_names == ("B",)
    with pytest.raises(KeyError):
        vocabulary.restrict_to_attributes(["C"])


def test_vocabulary_rejects_duplicate_codes():
    with pytest.raises(ValueError, match="duplicate"):
        ClassVocabulary({"A": {"x": 1, "y": 1}})


def test_compute_segment_labels_maps_codes_and_handles_missing(metadata_dir):
    metadata = load_irap_metadata(metadata_dir)
    segment_ids = ["S0", "S1", "S4", "unknown"]
    loose = compute_segment_labels(metadata.vocabulary, metadata.segment_id_to_road_data,
                                   segment_ids, allow_missing_attributes=True)
    assert loose == {"S0": [1, 1], "S1": [0, 2], "S4": [1, IGNORE_LABEL_INDEX],
                     "unknown": [IGNORE_LABEL_INDEX, IGNORE_LABEL_INDEX]}
    strict = compute_segment_labels(metadata.vocabulary, metadata.segment_id_to_road_data,
                                    segment_ids, allow_missing_attributes=False)
    assert list(strict) == ["S0", "S1"]


def test_context_stays_on_the_road_and_needs_images(metadata_dir):
    metadata = load_irap_metadata(metadata_dir)
    context = compute_segment_context_ids(
        metadata.road_id_to_segment_id_sequence, ["S0", "S1", "S2", "S4", "S5", "T0", "T1"],
        context_offsets=(0, -1), available_segment_ids=metadata.segment_id_to_data_paths_rel)
    # S0 and T0 have no predecessor, and S4 needs S3, which has no image.
    assert context == {"S1": ("S1", "S0"), "S2": ("S2", "S1"), "S5": ("S5", "S4"),
                       "T1": ("T1", "T0")}


def test_select_split_segments(metadata_dir):
    metadata = load_irap_metadata(metadata_dir)
    selection = select_split_segments(metadata, "val", context_offsets=(0, -1),
                                      allow_missing_attributes=True)
    assert selection.segment_ids == ("S1", "S2", "S5", "T1")
    assert selection.attr_to_num_labeled == {"A": 4, "B": 4}
    subset = select_split_segments(metadata, "val", context_offsets=(0, -1),
                                   allow_missing_attributes=True,
                                   ncontext_segment_id_subset={"S2", "T1"})
    assert subset.segment_ids == ("S2", "T1")
    with pytest.raises(ValueError, match="Invalid split"):
        select_split_segments(metadata, "test", context_offsets=(0,),
                              allow_missing_attributes=True)


def test_select_preset_split_segments_uses_reference_offsets(metadata_dir):
    metadata = load_irap_metadata(metadata_dir)
    # The -4..0 Vietnam reference window of S4 and S5 includes S3, which has no image, and r1 is
    # too short.
    assert select_preset_split_segments(metadata, "vietnam", "val").segment_ids == ()
    model = select_preset_split_segments(metadata, "vietnam", "val", context_offsets=(0,))
    assert model.segment_ids == ("S0", "S1", "S2", "S4", "S5", "T0", "T1")
    with pytest.raises(ValueError, match="Unknown IRAP dataset"):
        select_preset_split_segments(metadata, "atlantis", "val")


def test_select_split_segments_matches_irap_dataset(metadata_dir):
    pytest.importorskip("torch")
    from irap_data import IRAPDataset

    metadata = load_irap_metadata(metadata_dir)
    for offsets in [(0,), (0, -1), (1, 0)]:
        dataset = IRAPDataset(metadata_dir / "images", "val", metadata_dir=metadata_dir,
                              context_offsets=offsets, allow_missing_attributes=True)
        selection = select_split_segments(metadata, "val", context_offsets=offsets,
                                          allow_missing_attributes=True)
        assert tuple(dataset.segment_ids) == selection.segment_ids
        assert dataset.info.attr_to_num_labeled == selection.attr_to_num_labeled


def test_dataset_presets_cover_the_dataset_factories():
    pytest.importorskip("torch")
    from irap_data import IRAP_DATASET_FACTORIES

    assert set(DATASET_PRESETS) == set(IRAP_DATASET_FACTORIES)


def test_get_segment_location_reads_both_layouts():
    vietnam = get_segment_location({"section": "s", "distance_km": 0.04, "length_km": 0.02,
                                    "lat": 1.5, "lon": 2.5})
    bih = get_segment_location({"metadata": {"Section": "s", "Distance": "0.04",
                                             "Length": "0.02", "Latitude": "1.5",
                                             "Longitude": "2.5"}})
    assert vietnam == bih
    assert vietnam.distance_m == pytest.approx(40.0)
    with pytest.raises(ValueError, match="location layout"):
        get_segment_location({"required_attributes": {}})


def test_metadata_api_does_not_import_torch():
    code = ("import sys, irap_data, irap_data.metadata; "
            "sys.exit(int('torch' in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0
