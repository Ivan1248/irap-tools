"""Adds synthetic models to the archive of a server configuration, through the same checks
as an upload, and an ensemble of them.

It adds 3 methods with 3 seeds each on val and test, 'strong-probs' seed 1 also on the unlabeled
splits, and an ensemble of the 'strong-probs' models. Running it again replaces the files of these
models. A running server scores the new files after a restart.

The fake models are those of `prototypes/analysis_ui/make_data.py` (branch prototype/analysis-ui):
their logits favour the label of a segment with a strength that varies per road sequence, and the
neighbouring classes get part of it. Where a segment has no label, the logits follow the class
frequencies of 'train'.

Usage:
    python add_synthetic_models.py <server.toml>
"""

import dataclasses as dc
import sys
from pathlib import Path

import irap_evaluation as ie
import numpy as np
from irap_data.metadata import IGNORE_LABEL_INDEX, IRAPMetadata, compute_segment_labels

from irap_evaluation_server.archive import ModelArchive
from irap_evaluation_server.config import load_server_config
from irap_evaluation_server.datasets import load_dataset_contexts
from irap_evaluation_server.ensembles import EnsembleMember, plan_ensemble, prepare_ensemble
from irap_evaluation_server.uploads import apply_upload, plan_upload

DATASET = "vietnam"
SUBMITTER = "synthetic data script"
CONTEXT_OFFSETS = (0, -1, -4)
SEEDS = (1, 2, 3)
LABELED_SPLITS = ("val", "test")


@dc.dataclass(frozen=True)
class FakeModel:
    name: str
    #: Mean logit bonus of the label class. Larger means more accurate.
    label_strength: float
    #: Std. dev. of the per-sequence log-multiplier of `label_strength`.
    sequence_difficulty_std_dev: float
    #: Probability of an invalid cell (hard models only).
    invalid_rate: float = 0.0
    output_kind: ie.OutputKind = "probs"


FAKE_MODELS = (
    FakeModel("strong-probs", label_strength=3.0, sequence_difficulty_std_dev=0.4),
    FakeModel("weak-probs", label_strength=1.6, sequence_difficulty_std_dev=0.6),
    FakeModel("vlm-hard", label_strength=2.2, sequence_difficulty_std_dev=0.6, invalid_rate=0.05,
              output_kind="hard"),
)


def compute_train_log_priors(metadata: IRAPMetadata) -> dict[str, np.ndarray]:
    """Computes the (K,) log class frequencies of each attribute over 'train', with add-one
    smoothing."""
    vocabulary = metadata.vocabulary
    labels = compute_segment_labels(vocabulary, metadata.segment_id_to_road_data,
                                    metadata.splits["train"], allow_missing_attributes=True)
    label_matrix = np.array(list(labels.values()), dtype=np.int64)
    log_priors = {}
    for i, (attr, num_classes) in enumerate(zip(vocabulary.attribute_names,
                                                vocabulary.class_counts)):
        column = label_matrix[:, i]
        counts = np.bincount(column[column != IGNORE_LABEL_INDEX], minlength=num_classes) + 1
        log_priors[attr] = np.log(counts / counts.sum())
    return log_priors


def compute_fake_logits(model: FakeModel, labels: np.ndarray, log_prior: np.ndarray,
                        sequence_multipliers: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Computes the (N, K) logits of one attribute for segments with (N,) class labels,
    `IGNORE_LABEL_INDEX` where unlabeled."""
    num_segments, num_classes = len(labels), len(log_prior)
    logits = log_prior + rng.normal(0.0, 1.0, (num_segments, num_classes))
    rows = np.flatnonzero(labels != IGNORE_LABEL_INDEX)
    strength = model.label_strength * sequence_multipliers[rows]
    logits[rows, labels[rows]] += strength
    for neighbor_offset in (-1, 1):
        neighbors = labels[rows] + neighbor_offset
        is_inside = (neighbors >= 0) & (neighbors < num_classes)
        logits[rows[is_inside], neighbors[is_inside]] += 0.4 * strength[is_inside]
    return logits


def make_fake_predictions(model: FakeModel, metadata: IRAPMetadata, split: str,
                          log_priors: dict[str, np.ndarray], model_seed: int,
                          random_seed: int) -> ie.Predictions:
    rng = np.random.default_rng(random_seed)
    vocabulary = metadata.vocabulary
    segment_ids = list(metadata.splits[split])
    labels = compute_segment_labels(vocabulary, metadata.segment_id_to_road_data, segment_ids,
                                    allow_missing_attributes=True)
    label_matrix = np.array([labels[sid] for sid in segment_ids], dtype=np.int64)
    sequence_index = metadata.sequence_index
    road_ids = [sequence_index[sid][0] if sid in sequence_index else sid for sid in segment_ids]
    unique_road_ids, road_indices = np.unique(road_ids, return_inverse=True)
    road_multipliers = np.exp(rng.normal(0.0, model.sequence_difficulty_std_dev,
                                         len(unique_road_ids)))
    sequence_multipliers = road_multipliers[road_indices]
    header = ie.PredictionHeader(
        dataset=DATASET, split=split, context_offsets=CONTEXT_OFFSETS,
        model=ie.ModelInfo(model.name, training_splits=("train",), early_stopping_splits=(),
                           seed=model_seed, details={"synthetic": True}))
    codes = vocabulary.attribute_to_irap_codes
    logits = {attr: compute_fake_logits(model, label_matrix[:, i], log_priors[attr],
                                        sequence_multipliers, rng)
              for i, attr in enumerate(vocabulary.attribute_names)}
    if model.output_kind == "probs":
        return ie.Predictions.from_logits(header, codes, segment_ids, logits)
    class_indices = {attr: np.where(rng.random(len(segment_ids)) < model.invalid_rate,
                                    IGNORE_LABEL_INDEX, v.argmax(1))
                     for attr, v in logits.items()}
    return ie.Predictions.from_class_indices(header, codes, segment_ids, class_indices)


def main():
    config = load_server_config(Path(sys.argv[1]))
    dataset_contexts = load_dataset_contexts(config.datasets)
    context = dataset_contexts[DATASET]
    metadata = context.metadata
    archive = ModelArchive(config.data_dir)
    log_priors = compute_train_log_priors(metadata)
    file_specs = [
        *((m, split, seed) for m in FAKE_MODELS for split in LABELED_SPLITS for seed in SEEDS),
        *((FAKE_MODELS[0], split, 1) for split in ("unlabeled_val", "unlabeled_unlocated")
          if split in metadata.splits)]
    for random_seed, (model, split, model_seed) in enumerate(file_specs):
        predictions = make_fake_predictions(model, metadata, split, log_priors, model_seed,
                                            random_seed)
        file_name = f"{model.name}_seed{model_seed}.{split}.predictions.parquet"
        path = archive.make_upload_path(file_name)
        ie.write_predictions(path, predictions)
        planned = plan_upload(archive, dataset_contexts, path, file_name=file_name,
                              description=f"Synthetic model {model.name!r}, seed {model_seed}"
                                          f" (add_synthetic_models.py).")
        [submission] = apply_upload(archive, planned, submitter=SUBMITTER)
        print(f"#{submission.id} {file_name}"
              + "".join(f"\n    {note}" for note in planned.notes))

    submissions = archive.list_submissions()
    members = [EnsembleMember(s.model, 1.0) for s in submissions
               if s.model.method_name == "strong-probs" and s.split == "val"]
    plan = plan_ensemble(submissions, DATASET, members, LABELED_SPLITS)
    update = prepare_ensemble(archive, context, plan, method_name="ensemble1", seed=None,
                              intersect_segments=False,
                              description="Ensemble of strong-probs seeds 1-3.")
    created = archive.apply_model_update(update, submitter=SUBMITTER)
    print("Ensemble: " + ", ".join(f"#{s.id} {s.label} {s.split}" for s in created))


if __name__ == "__main__":
    main()
