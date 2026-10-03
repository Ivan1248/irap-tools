"""The view of the Comparison page: two methods of a dataset split, as in its URL."""

import dataclasses as dc
import typing as T

from ..datasets import DatasetContext
from .routes import ATTRIBUTE_PARAMETER, COMPARISON_PATH, make_query_path
from .view_queries import parse_dataset_and_split, parse_per_attribute_metric


@dc.dataclass(frozen=True)
class ComparisonView:
    """What the Comparison page shows, as in its URL. '' means the default: the two best methods
    on the Scores page, and the attribute averages.

    Attributes:
        metric_name: The per-attribute metric of the per-attribute table, or '' for the
            attribute averages.
    """

    dataset: str
    split: str
    method_a: str = ""
    method_b: str = ""
    metric_name: str = ""

    def to_path(self, attributes: T.Sequence[str] = ()) -> str:
        """The URL path, without the default values.

        Args:
            attributes: The attribute subset (`AttributeSubset.to_query`).
        """
        return make_query_path(COMPARISON_PATH, {
            "dataset": self.dataset, "split": self.split, "a": self.method_a,
            "b": self.method_b, "metric": self.metric_name, ATTRIBUTE_PARAMETER: attributes})

    @classmethod
    def from_query(cls, query: T.Mapping[str, str],
                   dataset_contexts: T.Mapping[str, DatasetContext]) -> T.Self:
        """The view of the query parameters of `to_path`, with the dataset and split filled in.
        The method names are checked by the page, against the scored runs.

        Raises:
            ValueError: See `parse_dataset_and_split` and `parse_per_attribute_metric`.
        """
        dataset, split = parse_dataset_and_split(query, dataset_contexts)
        return cls(dataset=dataset, split=split, method_a=query.get("a", ""),
                   method_b=query.get("b", ""), metric_name=parse_per_attribute_metric(query))
