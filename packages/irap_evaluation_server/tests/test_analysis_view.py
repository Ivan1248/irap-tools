import dataclasses as dc
import functools
import urllib.parse

import irap_evaluation as ie
import pytest
from run_helpers import add_run, make_run, score_and_get_runs
from synthetic_vietnam import SPLIT

from irap_evaluation_server.components.analysis_view import AnalysisView, resolve_view
from irap_evaluation_server.components.attribute_filter import AttributeSubset
from irap_evaluation_server.scoring import get_run_evaluation_set


@pytest.fixture
def runs(archive, dataset_contexts, worker):
    """Two runs scored on the reference set."""
    added = [add_run(archive, dataset_contexts, make_run(dataset_contexts, name))
             for name in ("a", "b")]
    return score_and_get_runs(worker, [s.id for s in added])


def _resolve(view, runs, dataset_contexts, attributes=()):
    subset = AttributeSubset.from_query(attributes, runs[0].scores.attributes)
    return resolve_view(view, runs, subset,
                        functools.partial(get_run_evaluation_set, dataset_contexts["vietnam"]))


def test_path_and_query_round_trip(dataset_contexts):
    view = AnalysisView("vietnam", SPLIT, ie.MODEL_COMPATIBLE_SET_NAME, run_id=3, compare_id=4,
                        attribute_name="Lane width", cell=(1, 3), segment_id="0004")
    path = view.to_path(["Lane width", "Curvature"])
    query = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
    assert query["attribute"] == ["Lane width", "Curvature"]
    assert AnalysisView.from_query({k: v[0] for k, v in query.items()}, dataset_contexts) == view
    assert AnalysisView("vietnam", SPLIT).to_path() == f"/analysis?dataset=vietnam&split={SPLIT}"
    assert AnalysisView.from_query({}, dataset_contexts) == AnalysisView("vietnam", SPLIT)
    for query, message in [({"split": "test"}, "not an analysis split"),
                           ({"set": "x"}, "Unknown evaluation set"),
                           ({"run": "x"}, "submission number"), ({"cell": "1"}, "two indices")]:
        with pytest.raises(ValueError, match=message):
            AnalysisView.from_query(query, dataset_contexts)


def test_resolve_view_chooses_the_defaults(runs, dataset_contexts):
    resolved = _resolve(AnalysisView("vietnam", SPLIT), runs, dataset_contexts)
    assert resolved.run_a is runs[0] and resolved.run_b is None
    assert resolved.view.run_id == runs[0].submission.id
    assert resolved.view.attribute_name == runs[0].scores.attributes[0]
    assert resolved.evaluation_set.fingerprint == runs[0].scores.evaluation_set_fingerprint
    resolved = _resolve(AnalysisView("vietnam", SPLIT), runs, dataset_contexts,
                        attributes=["Lane width"])
    assert resolved.view.attribute_name == "Lane width"


def test_resolve_view_refuses_what_is_not_in_the_runs(runs, dataset_contexts):
    a, b = runs[0].submission.id, runs[1].submission.id
    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    resolved = _resolve(AnalysisView("vietnam", SPLIT, run_id=a, compare_id=b, cell=(2, 3),
                                     attribute_name="Lane width",
                                     segment_id=reference_set.segment_ids[0]),
                        runs, dataset_contexts)
    assert resolved.run_b is runs[1]
    for changes, message in [
            (dict(run_id=999), "not a scored run"),
            (dict(compare_id=a), "cannot be compared"),
            (dict(attribute_name="Nope"), "not scored on the attribute"),
            (dict(cell=(3, 0)), "outside the matrix"),  # 'Lane width' has 3 classes.
            (dict(cell=(0, 4)), "outside the matrix"),
            (dict(segment_id="x"), "not in the evaluation set")]:
        with pytest.raises(ValueError, match=message):
            _resolve(dc.replace(resolved.view, **changes), runs, dataset_contexts)
    with pytest.raises(ValueError, match="not in the attribute subset"):
        _resolve(resolved.view, runs, dataset_contexts, attributes=["Curvature"])
