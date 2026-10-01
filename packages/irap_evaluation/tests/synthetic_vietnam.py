"""Builders of a small synthetic IRAP-Vietnam metadata directory and of predictions for it."""

import json
from pathlib import Path

import numpy as np

from irap_data.metadata import ClassVocabulary, MetaFiles
from irap_evaluation.predictions import MethodInfo, PredictionHeader, Predictions

# Three attributes of the canonical subset and one outside it. The codes of "Lane width" are
# not in ascending order, so class indices differ from code order.
ATTRIBUTE_CODES = {"Carriageway label": [1, 2, 3], "Number of lanes": [1, 2, 3, 4],
                   "Lane width": [3, 2, 1], "Curvature": [1, 2]}
NUM_ROADS, ROAD_LENGTH = 4, 15
SPLIT = "val"


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj), encoding="utf-8")


def write_metadata_dir(path: Path) -> Path:
    """Writes metadata of NUM_ROADS roads of ROAD_LENGTH segments, all in the 'val' split.

    'Curvature' is missing (-1) in every fifth segment.
    """
    rng = np.random.default_rng(0)
    path.mkdir(parents=True, exist_ok=True)
    road_to_seq = {f"road{r}": [f"{r}{i:03d}" for i in range(ROAD_LENGTH)]
                   for r in range(NUM_ROADS)}
    segments = [sid for seq in road_to_seq.values() for sid in seq]
    _write_json(path / MetaFiles.ATTRIBUTE_METADATA, {
        "attribute_to_idx": {a: i for i, a in enumerate(ATTRIBUTE_CODES)},
        "attribute_value_to_irap_number": {a: {f"{a} {c}": c for c in codes}
                                           for a, codes in ATTRIBUTE_CODES.items()}})
    _write_json(path / MetaFiles.SEGMENT_ID_TO_DATA_PATHS,
                {sid: {"rgb": f"FRAMES/road/{sid}.png"} for sid in segments})

    def road_data(road: int, position: int):
        codes = {a: int(rng.choice(c)) for a, c in ATTRIBUTE_CODES.items()}
        if position % 5 == 0:
            codes["Curvature"] = -1
        return {"section": f"section{road}", "distance_km": 0.02 * position, "length_km": 0.02,
                "lat": 45.0 + 0.001 * position, "lon": 16.0 + road, "comments": "",
                "required_attributes": codes}

    _write_json(path / MetaFiles.SEGMENT_ID_TO_ROAD_DATA,
                {sid: road_data(r, i) for r, seq in enumerate(road_to_seq.values())
                 for i, sid in enumerate(seq)})
    _write_json(path / MetaFiles.ROAD_ID_TO_SEGMENT_ID_SEQUENCE, road_to_seq)
    _write_json(path / MetaFiles.SPLITS, {SPLIT: segments})
    return path


def make_predictions(vocabulary: ClassVocabulary, segment_ids, *, name: str, seed: int,
                     method_seed: int | None = None, context_offsets=(0, -1),
                     reverse_classes: bool = False):
    """Random predictions of every attribute, optionally with the class order reversed.

    Args:
        seed: Seed of the random predictions.
        method_seed: `MethodInfo.seed`, which identifies the run.
    """
    rng = np.random.default_rng(seed)
    attribute_to_irap_codes = vocabulary.attribute_to_irap_codes
    if reverse_classes:
        attribute_to_irap_codes = {a: codes[::-1] for a, codes in attribute_to_irap_codes.items()}
    header = PredictionHeader(dataset="vietnam", split=SPLIT,
                              method=MethodInfo(name=name, seed=method_seed),
                              context_offsets=context_offsets)
    logits = {a: 3 * rng.normal(size=(len(segment_ids), len(codes)))
              for a, codes in attribute_to_irap_codes.items()}
    return Predictions.from_logits(header, attribute_to_irap_codes, segment_ids, logits)
