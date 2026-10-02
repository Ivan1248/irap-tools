"""The URL paths of the pages and endpoints. `{submission_id}` is a path parameter."""

SUBMISSIONS_PATH = "/submissions"
SUBMISSION_PATH = "/submissions/{submission_id}"
ACTION_LOG_PATH = "/actions"
UPLOAD_PATH = "/api/uploads"
PREDICTIONS_DOWNLOAD_PATH = "/api/submissions/{submission_id}/predictions.parquet"


def get_submission_path(submission_id: int) -> str:
    return SUBMISSION_PATH.format(submission_id=submission_id)


def get_download_path(submission_id: int) -> str:
    return PREDICTIONS_DOWNLOAD_PATH.format(submission_id=submission_id)
