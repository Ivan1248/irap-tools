"""The reviews of labels: corrections by reviewers of the label of a segment for one attribute,
e.g. of the candidate labeling errors (`label_errors`), and their export as a prediction file.

The reviews are in `label_reviews.sqlite3` in the data directory, apart from the archive, with
one correction per label. A new correction replaces it, and a withdrawn one is deleted. They do
not change the labels that submissions are scored with.
"""

import dataclasses as dc
import sqlite3
import typing as T
from datetime import datetime
from pathlib import Path

import irap_evaluation as ie
import numpy as np
from irap_data.metadata import IGNORE_LABEL_INDEX, ClassVocabulary

from .database import Migration, begin_write, connect, get_utc_now, initialize_schema

DATABASE_FILE_NAME = "label_reviews.sqlite3"
#: The version of the database schema (`PRAGMA user_version`). A database of an older version is
#: migrated (`_MIGRATIONS`).
SCHEMA_VERSION = 1
#: Version v -> the migration from v to v + 1 (`database.initialize_schema`).
_MIGRATIONS: dict[int, Migration] = {}

#: See `LabelReview.correction`.
LabelCorrection = T.Literal["class", "unclear"]
LABEL_CORRECTIONS: tuple[LabelCorrection, ...] = T.get_args(LabelCorrection)
#: The method name in the header of an exported prediction file (`make_review_predictions`).
LABEL_REVIEW_METHOD_NAME = "label-review"

_SCHEMA = f"""
CREATE TABLE label_reviews (
    dataset TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    attribute TEXT NOT NULL,
    correction TEXT NOT NULL CHECK (correction IN {LABEL_CORRECTIONS}),
    irap_code INTEGER CHECK ((irap_code IS NOT NULL) = (correction = 'class')),
    label_irap_code INTEGER,
    comment TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (dataset, segment_id, attribute)
);
"""

#: (segment id, attribute), which identifies a label of a dataset.
LabelKey = tuple[str, str]


@dc.dataclass(frozen=True)
class LabelReview:
    """A correction by a reviewer of the label of a segment for one attribute.

    Attributes:
        correction: 'class' if the class of the segment is `irap_code`, 'unclear' if the class
            cannot be told, e.g. from the images.
        irap_code: The class of a 'class' correction, else None.
        label_irap_code: The label when the correction was given, None if the segment had none.
        reviewer: The name of the reviewer's account.
        created_at: In UTC.
    """

    dataset: str
    segment_id: str
    attribute: str
    correction: LabelCorrection
    irap_code: int | None
    label_irap_code: int | None
    comment: str
    reviewer: str
    created_at: datetime

    @property
    def key(self) -> LabelKey:
        return self.segment_id, self.attribute

    def is_outdated(self, label_irap_code: int | None) -> bool:
        """Returns whether the label has changed since the correction, e.g. by a metadata update,
        given the current label."""
        return label_irap_code != self.label_irap_code


def _to_label_review(row: sqlite3.Row) -> LabelReview:
    return LabelReview(**{k: row[k] for k in row.keys() if k != "created_at"},
                       created_at=datetime.fromisoformat(row["created_at"]))


class LabelReviewStore:
    """The label review database of a data directory (see the module docstring), created on
    first use.

    Each operation opens its own database connection, so the store can be used from several
    threads.

    Raises:
        ValueError: If the database cannot be migrated to `SCHEMA_VERSION`, or its tables differ
            from those of its version (see `database.initialize_schema`).
    """

    def __init__(self, data_dir: Path):
        self.database_path = data_dir / DATABASE_FILE_NAME
        initialize_schema(self.database_path, _SCHEMA, SCHEMA_VERSION, "label review",
                          _MIGRATIONS)

    def set_review(self, dataset: str, segment_id: str, attribute: str, correction: LabelCorrection,
                   irap_code: int | None, label_irap_code: int | None, comment: str,
                   reviewer: str) -> LabelReview:
        """Stores a correction of a label, which replaces its correction, if any. The caller checks
        that the segment, the attribute and the codes are of the dataset (`review_label`).

        Raises:
            ValueError: For an unknown correction, or an iRAP code that is given for 'unclear' or
                missing for 'class'.
        """
        if correction not in LABEL_CORRECTIONS:
            raise ValueError(f"Unknown correction {correction!r}. Corrections:"
                             f" {', '.join(LABEL_CORRECTIONS)}.")
        if (irap_code is None) == (correction == "class"):
            raise ValueError("A 'class' correction needs a class, and an 'unclear' one has none.")
        review = LabelReview(dataset=dataset, segment_id=segment_id, attribute=attribute,
                             correction=correction, irap_code=irap_code,
                             label_irap_code=label_irap_code, comment=comment.strip(),
                             reviewer=reviewer, created_at=get_utc_now())
        row = {**dc.asdict(review), "created_at": review.created_at.isoformat()}
        with begin_write(self.database_path) as connection:
            connection.execute(
                f"INSERT OR REPLACE INTO label_reviews ({', '.join(row)})"
                f" VALUES ({', '.join(f':{column}' for column in row)})", row)
        return review

    def remove_review(self, dataset: str, segment_id: str, attribute: str) -> None:
        """Withdraws the correction of a label, if it has one."""
        with begin_write(self.database_path) as connection:
            connection.execute(
                "DELETE FROM label_reviews WHERE dataset = ? AND segment_id = ? AND attribute = ?",
                (dataset, segment_id, attribute))

    def get_reviews(self, dataset: str) -> dict[LabelKey, LabelReview]:
        """Returns the corrections of the labels of a dataset."""
        with connect(self.database_path) as connection:
            rows = connection.execute("SELECT * FROM label_reviews WHERE dataset = ?",
                                      (dataset,)).fetchall()
        return {(review := _to_label_review(row)).key: review for row in rows}


def get_label_irap_code(evaluation_set: ie.EvaluationSet, attribute: str,
                        segment_index: int) -> int | None:
    """Returns the label of a segment of a set for an attribute, None if it has none."""
    label_index = int(evaluation_set.class_indices[attribute][segment_index])
    if label_index == IGNORE_LABEL_INDEX:
        return None
    return evaluation_set.vocabulary.get_irap_codes(attribute)[label_index]


def review_label(store: LabelReviewStore, evaluation_set: ie.EvaluationSet, segment_index: int,
                 attribute: str, correction: LabelCorrection, irap_code: int | None, comment: str,
                 reviewer: str) -> LabelReview:
    """Stores a correction of the label of a segment of a set for an attribute
    (`LabelReviewStore.set_review`), as one of the label that the segment has in the set.

    Args:
        segment_index: Into the segments of the set.

    Raises:
        ValueError: If the set has no such attribute, the class of a 'class' correction is not a
            class of the attribute, or the correction is refused by `set_review`.
    """
    if attribute not in evaluation_set.class_indices:
        raise ValueError(f"{attribute!r} is not an attribute of the evaluation set.")
    if (irap_code is not None
            and irap_code not in evaluation_set.vocabulary.get_irap_codes(attribute)):
        raise ValueError(f"{irap_code} is not a class of {attribute!r}.")
    return store.set_review(
        evaluation_set.dataset, evaluation_set.segment_ids[segment_index], attribute, correction,
        irap_code, get_label_irap_code(evaluation_set, attribute, segment_index), comment, reviewer)


def list_reviewed_labels(reviews: T.Mapping[LabelKey, LabelReview],
                         evaluation_set: ie.EvaluationSet) -> list[tuple[int, str]]:
    """Lists the labels of a set whose correction is on the current label
    (`LabelReview.is_outdated`), as (segment index, attribute).

    Args:
        reviews: The corrections of the dataset (`LabelReviewStore.get_reviews`).
    """
    segment_to_index = {s: i for i, s in enumerate(evaluation_set.segment_ids)}
    return [(i, review.attribute) for review in reviews.values()
            if (i := segment_to_index.get(review.segment_id)) is not None
            and review.attribute in evaluation_set.class_indices
            and not review.is_outdated(get_label_irap_code(evaluation_set, review.attribute, i))]


def make_review_predictions(reviews: T.Mapping[LabelKey, LabelReview],
                            evaluation_set: ie.EvaluationSet,
                            exported_at: datetime) -> ie.Predictions:
    """Makes hard predictions of the corrections of the labels of the segments of a set, e.g. the
    full set of a split, to export them as a prediction file.

    The predictions have the segments and the attributes with corrections, in the order of the set
    and its vocabulary. A 'class' correction is its class, also if the label has changed since.
    Other labels are invalid predictions (null), and the unclear ones are listed in
    `details["label_review"]["unclear"]` (attribute -> segment ids) of the header's model.

    A correction whose attribute, or class, is not in the vocabulary of the set, e.g. after a
    metadata update that removed it, is left out and listed in
    `details["label_review"]["skipped"]` (attribute -> segment ids).

    Args:
        reviews: The corrections of the dataset (`LabelReviewStore.get_reviews`).

    Raises:
        LookupError: If no segment of the set has a correction that is not left out.
    """
    vocabulary = evaluation_set.vocabulary
    segment_to_index = {s: i for i, s in enumerate(evaluation_set.segment_ids)}
    set_reviews = sorted((r for r in reviews.values() if r.segment_id in segment_to_index),
                         key=lambda r: segment_to_index[r.segment_id])
    exported, skipped = [], []
    for review in set_reviews:
        (exported if _fits_vocabulary(review, vocabulary) else skipped).append(review)
    if not exported:
        skipped_note = (f", apart from {len(skipped)} of attributes or classes that are not in"
                        f" its vocabulary" if skipped else "")
        raise LookupError(f"No segment of {evaluation_set.dataset}/{evaluation_set.split} has a"
                          f" label review{skipped_note}.")
    segment_ids = list(dict.fromkeys(r.segment_id for r in exported))
    segment_to_position = {s: i for i, s in enumerate(segment_ids)}
    exported_attributes = {r.attribute for r in exported}
    attributes = [a for a in vocabulary.attribute_names if a in exported_attributes]
    irap_codes = {a: np.full(len(segment_ids), ie.INVALID_IRAP_CODE, dtype=np.int64)
                  for a in attributes}
    for review in exported:
        if review.correction == "class":
            irap_codes[review.attribute][segment_to_position[review.segment_id]] = (
                review.irap_code)
    model = ie.ModelInfo(
        method_name=LABEL_REVIEW_METHOD_NAME, training_splits=None, early_stopping_splits=(),
        details={"label_review": {
            "exported_at": exported_at.isoformat(),
            "unclear": _group_segment_ids(r for r in exported if r.correction != "class"),
            "skipped": _group_segment_ids(skipped)}})
    header = ie.PredictionHeader(dataset=evaluation_set.dataset, split=evaluation_set.split,
                                 model=model)
    return ie.Predictions.from_irap_codes(
        header, {a: vocabulary.get_irap_codes(a) for a in attributes}, segment_ids, irap_codes)


def _fits_vocabulary(review: LabelReview, vocabulary: ClassVocabulary) -> bool:
    """Returns whether the attribute of a correction, and its class if it has one, are in the
    vocabulary."""
    return (review.attribute in vocabulary.attribute_names
            and (review.correction != "class"
                 or review.irap_code in vocabulary.get_irap_codes(review.attribute)))


def _group_segment_ids(reviews: T.Iterable[LabelReview]) -> dict[str, list[str]]:
    """Groups the segments of corrections by attribute, in the order of the corrections."""
    attribute_to_segment_ids: dict[str, list[str]] = {}
    for review in reviews:
        attribute_to_segment_ids.setdefault(review.attribute, []).append(review.segment_id)
    return attribute_to_segment_ids
