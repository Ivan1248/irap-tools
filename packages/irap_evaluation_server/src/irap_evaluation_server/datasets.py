"""The configured datasets: their metadata and the evaluation sets of their splits."""

import dataclasses as dc
import threading
import typing as T

import irap_evaluation as ie
from irap_data.metadata import IRAPMetadata, load_irap_metadata

from .config import DatasetConfig


@dc.dataclass(eq=False)
class DatasetContext:
    """A configured dataset with its metadata and cached evaluation sets.

    The cache is shared by the threads that check uploads.

    Attributes:
        name: The dataset name, a key of `irap_data.DATASET_PRESETS`.
    """

    name: str
    config: DatasetConfig
    metadata: IRAPMetadata
    _split_and_offsets_to_evaluation_set: dict[tuple[str, tuple[int, ...] | None],
                                               ie.EvaluationSet] = dc.field(
        default_factory=dict, init=False, repr=False)
    #: Held while a set is computed, so that concurrent first uses compute it once.
    _cache_lock: threading.Lock = dc.field(default_factory=threading.Lock, init=False,
                                           repr=False)

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(self.metadata.splits)

    def _get_evaluation_set(self, split: str,
                            context_offsets: tuple[int, ...] | None) -> ie.EvaluationSet:
        """The reference set if `context_offsets` is None, otherwise the model set."""
        key = (split, context_offsets)
        with self._cache_lock:
            if key not in self._split_and_offsets_to_evaluation_set:
                if split not in self.metadata.splits:
                    raise ValueError(f"The dataset {self.name!r} has no split {split!r}. Its"
                                     f" splits: {', '.join(self.split_names)}.")
                self._split_and_offsets_to_evaluation_set[key] = (
                    ie.get_reference_set(self.metadata, self.name, split)
                    if context_offsets is None
                    else ie.get_evaluation_set(self.metadata, self.name, split, context_offsets,
                                               name="model"))
            return self._split_and_offsets_to_evaluation_set[key]

    def get_reference_set(self, split: str) -> ie.EvaluationSet:
        """`irap_evaluation.get_reference_set` of a split, computed on first use.

        Raises:
            ValueError: If the dataset has no such split.
        """
        return self._get_evaluation_set(split, None)

    def get_model_set(self, split: str, context_offsets: T.Sequence[int]) -> ie.EvaluationSet:
        """The evaluation set named 'model' of a model's context offsets, computed on first use.

        Raises:
            ValueError: If the dataset has no such split.
        """
        return self._get_evaluation_set(split, tuple(context_offsets))

    def select_evaluation_sets(
            self, predictions: ie.Predictions) -> tuple[list[ie.EvaluationSet], list[str]]:
        """`irap_evaluation.select_evaluation_sets` of a run, with the model set of the context
        offsets in its header.

        Raises:
            ValueError: If the predictions are of another dataset or of an unknown split, or
                `irap_evaluation.select_evaluation_sets` refuses them.
        """
        header = predictions.header
        if header.dataset != self.name:
            raise ValueError(f"The predictions are of the dataset {header.dataset!r},"
                             f" not {self.name!r}.")
        offsets = header.context_offsets
        return ie.select_evaluation_sets(
            predictions.segment_ids, self.get_reference_set(header.split),
            None if offsets is None else self.get_model_set(header.split, offsets))

    def check_predicted_classes(self, predictions: ie.Predictions) -> None:
        """Checks that the predicted attributes and their iRAP codes are those of the dataset.

        Raises:
            irap_evaluation.PredictionFormatError: If an attribute is unknown or its codes differ.
        """
        attribute_to_irap_codes = self.metadata.vocabulary.attribute_to_irap_codes
        if unknown := [a for a in predictions.attributes if a not in attribute_to_irap_codes]:
            raise ie.PredictionFormatError(f"Attributes unknown in the dataset {self.name!r}:"
                                           f" {unknown}.")
        ie.check_class_codes(predictions,
                             {a: attribute_to_irap_codes[a] for a in predictions.attributes})


def load_dataset_contexts(
        datasets: T.Mapping[str, DatasetConfig]) -> dict[str, DatasetContext]:
    """Loads the metadata of each dataset.

    Raises:
        ValueError: If an analysis split is not in the metadata, or an images directory does not
            exist.
    """
    contexts = {}
    for name, config in datasets.items():
        metadata = load_irap_metadata(config.metadata_dir)
        if unknown := [s for s in config.analysis_splits if s not in metadata.splits]:
            raise ValueError(f"Dataset {name!r}: the analysis splits {unknown} are not in"
                             f" {config.metadata_dir}. Its splits: {', '.join(metadata.splits)}.")
        if config.images_dir is not None and not config.images_dir.is_dir():
            raise ValueError(f"Dataset {name!r}: the images directory {config.images_dir} does"
                             f" not exist.")
        contexts[name] = DatasetContext(name=name, config=config, metadata=metadata)
    return contexts
