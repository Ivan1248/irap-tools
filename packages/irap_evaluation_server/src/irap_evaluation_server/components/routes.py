"""The URL paths of the pages and endpoints. `{submission_id}` and `{evaluation_set}` are path
parameters."""

import typing as T
import urllib.parse

SCORES_PATH = "/scores"
SUBMISSIONS_PATH = "/submissions"
SUBMISSION_PATH = "/submissions/{submission_id}"
ACTION_LOG_PATH = "/actions"
UPLOAD_PATH = "/api/uploads"
PREDICTIONS_DOWNLOAD_PATH = "/api/submissions/{submission_id}/predictions.parquet"
SCORES_DOWNLOAD_PATH = "/api/submissions/{submission_id}/scores/{evaluation_set}.json"

#: The query parameter of an attribute subset, repeated for each attribute.
ATTRIBUTE_PARAMETER = "attribute"


def make_query_path(path: str, query: T.Mapping[str, str | T.Sequence[str]]) -> str:
    """`path` with the query parameters that are not empty, a sequence as repeated parameters."""
    query = {k: v for k, v in query.items() if v}
    return f"{path}?{urllib.parse.urlencode(query, doseq=True)}" if query else path


def get_submission_path(submission_id: int, attributes: T.Sequence[str] = ()) -> str:
    """The submission page, with an attribute subset (`AttributeSubset.to_query`)."""
    return make_query_path(SUBMISSION_PATH.format(submission_id=submission_id),
                           {ATTRIBUTE_PARAMETER: attributes})


def get_download_path(submission_id: int) -> str:
    return PREDICTIONS_DOWNLOAD_PATH.format(submission_id=submission_id)


def get_scores_download_path(submission_id: int, evaluation_set: str) -> str:
    return SCORES_DOWNLOAD_PATH.format(submission_id=submission_id, evaluation_set=evaluation_set)
