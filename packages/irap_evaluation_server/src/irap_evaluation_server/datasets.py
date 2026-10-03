"""The configured datasets: their metadata and the evaluation sets of their splits."""

import dataclasses as dc
import threading
import typing as T

import irap_evaluation as ie
from irap_data.metadata import IRAPMetadata, load_irap_metadata

from .config import DatasetConfig

#: The names of the evaluation sets that a run can be scored on.
EVALUATION_SET_NAMES = (ie.REFERENCE_SET_NAME, ie.MODEL_COMPATIBLE_SET_NAME)


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
    #: (split, context offsets or None for the reference set) -> the set.
    _evaluation_set_cache: dict[tuple[str, tuple[int, ...] | None], ie.EvaluationSet] = dc.field(
        default_factory=dict, init=False, repr=False)
    #: Held while a set is computed, so that concurrent first uses compute it once. Cached sets
    #: are read without it, so that a page does not wait for the computation of another set.
    _cache_lock: threading.Lock = dc.field(default_factory=threading.Lock, init=False,
                                           repr=False)

    @property
    def split_names(self) -> tuple[str, ...]:
        return tuple(self.metadata.splits)

    def _get_cached_evaluation_set(self, split: str,
                                   context_offsets: T.Sequence[int] | None) -> ie.EvaluationSet:
        """The reference set if `context_offsets` is None, otherwise the model-compatible set,
        computed on first use.

        Raises:
            ValueError: If the dataset has no such split.
        """
        key = (split, None if context_offsets is None else tuple(context_offsets))
        if (evaluation_set := self._evaluation_set_cache.get(key)) is not None:
            return evaluation_set
        with self._cache_lock:
            if key not in self._evaluation_set_cache:
                if split not in self.metadata.splits:
                    raise ValueError(f"The dataset {self.name!r} has no split {split!r}. Its"
                                     f" splits: {', '.join(self.split_names)}.")
                self._evaluation_set_cache[key] = (
                    ie.get_reference_set(self.metadata, self.name, split)
                    if context_offsets is None
                    else ie.get_model_compatible_set(self.metadata, self.name, split,
                                                     context_offsets))
            return self._evaluation_set_cache[key]

    def get_reference_set(self, split: str) -> ie.EvaluationSet:
        """`irap_evaluation.get_reference_set` of a split (see `_get_cached_evaluation_set`)."""
        return self._get_cached_evaluation_set(split, None)

    def get_model_compatible_set(self, split: str,
                                 context_offsets: T.Sequence[int]) -> ie.EvaluationSet:
        """`irap_evaluation.get_model_compatible_set` of a split (see
        `_get_cached_evaluation_set`)."""
        return self._get_cached_evaluation_set(split, context_offsets)

    def get_run_evaluation_sets(
            self, split: str, context_offsets: T.Sequence[int] | None
    ) -> dict[str, ie.EvaluationSet]:
        """Name -> set, for the evaluation sets that a run of a split can be scored on: the
        reference set, and the model-compatible set if the context offsets are known.

        Raises:
            ValueError: If the dataset has no such split.
        """
        name_to_set = {ie.REFERENCE_SET_NAME: self.get_reference_set(split)}
        if context_offsets is not None:
            name_to_set[ie.MODEL_COMPATIBLE_SET_NAME] = self.get_model_compatible_set(
                split, context_offsets)
        return name_to_set

    def select_evaluation_sets(
            self, predictions: ie.Predictions) -> tuple[list[ie.EvaluationSet], list[str]]:
        """`irap_evaluation.select_evaluation_sets` of a run, with the model-compatible set of
        the context offsets in its header.

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
            None if offsets is None else self.get_model_compatible_set(header.split, offsets))

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
