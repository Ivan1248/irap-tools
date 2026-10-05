"""The URL paths of the pages and endpoints. The names in braces are path parameters."""

import typing as T
import urllib.parse

SCORES_PATH = "/scores"
#: Its query parameters are those of `analysis_view.AnalysisView.to_path`.
ANALYSIS_PATH = "/analysis"
#: With the query parameter `dataset`.
ENSEMBLE_PATH = "/ensemble"
#: With the query parameter `dataset`.
MODELS_PATH = "/models"
#: With the query parameters `split` (of the scores) and an attribute subset
#: (`ATTRIBUTE_PARAMETER`).
MODEL_PATH = "/models/{model_id}"
ACTION_LOG_PATH = "/actions"
#: With the query parameters `next` (the path to return to) and `failed` ('1' after a wrong name
#: or password).
SIGN_IN_PATH = "/sign-in"
#: With the query parameter `next`.
REGISTER_PATH = "/register"
ACCOUNTS_PATH = "/accounts"
#: Endpoints of plain forms, with the form field `next`.
SIGN_IN_ENDPOINT_PATH = "/api/sign-in"
REGISTER_ENDPOINT_PATH = "/api/register"
SIGN_OUT_ENDPOINT_PATH = "/api/sign-out"
UPLOAD_PATH = "/api/uploads"
PREDICTIONS_DOWNLOAD_PATH = "/api/submissions/{submission_id}/predictions.parquet"
#: The active files of a model (`ModelArchive.write_model_predictions_zip`).
MODEL_PREDICTIONS_DOWNLOAD_PATH = "/api/models/{model_id}/predictions.zip"
SCORES_DOWNLOAD_PATH = "/api/submissions/{submission_id}/scores/{evaluation_set}.json"
SEGMENT_IMAGE_PATH = "/api/datasets/{dataset}/images/{segment_id}"
#: The static files of the map libraries (`map_libraries`).
VENDOR_PATH = "/vendor"

#: The query parameter of an attribute subset, repeated for each attribute.
ATTRIBUTE_PARAMETER = "attribute"


def make_query_path(path: str, query: T.Mapping[str, str | T.Sequence[str]]) -> str:
    """Adds the query parameters that are not empty to `path`, a sequence as repeated
    parameters."""
    query = {k: v for k, v in query.items() if v}
    return f"{path}?{urllib.parse.urlencode(query, doseq=True)}" if query else path


def to_local_path(url: str, default: str) -> str:
    """Returns `url` if it is a path on this server, e.g. a `next` parameter, otherwise
    `default`, so that a link cannot redirect to another site."""
    parts = urllib.parse.urlsplit(url)
    # Browsers read '/\' as '//', the start of another host, and remove tabs and line breaks.
    is_local = (not parts.scheme and not parts.netloc and url.startswith("/")
                and not url.startswith(("//", "/\\")) and url.isprintable())
    return url if is_local else default


def get_sign_in_path(next_path: str) -> str:
    """Makes the path of the Sign-in page, which returns to `next_path`."""
    return make_query_path(SIGN_IN_PATH, {"next": next_path})


def get_register_path(next_path: str) -> str:
    """Makes the path of the Register page, which returns to `next_path`."""
    return make_query_path(REGISTER_PATH, {"next": next_path})


def get_model_path(model_id: int, split: str = "", attributes: T.Sequence[str] = ()) -> str:
    """Makes the path of the Model page, with the split of its scores and an attribute subset
    (`AttributeSubset.to_query`)."""
    return make_query_path(MODEL_PATH.format(model_id=model_id),
                           {"split": split, ATTRIBUTE_PARAMETER: attributes})


def get_segment_image_path(dataset: str, segment_id: str) -> str:
    return SEGMENT_IMAGE_PATH.format(dataset=urllib.parse.quote(dataset, safe=""),
                                     segment_id=urllib.parse.quote(segment_id, safe=""))


def get_download_path(submission_id: int) -> str:
    return PREDICTIONS_DOWNLOAD_PATH.format(submission_id=submission_id)


def get_model_download_path(model_id: int) -> str:
    return MODEL_PREDICTIONS_DOWNLOAD_PATH.format(model_id=model_id)


def get_scores_download_path(submission_id: int, evaluation_set: str) -> str:
    return SCORES_DOWNLOAD_PATH.format(submission_id=submission_id, evaluation_set=evaluation_set)
