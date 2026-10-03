import irap_evaluation as ie
import pytest
from run_helpers import add_run, assert_close, evaluate_submission, make_run, score_and_get_runs
from synthetic_vietnam import SPLIT

from irap_evaluation_server.method_comparison import (
    COMPARED_METRIC_NAMES,
    make_comparison_request,
)
from irap_evaluation_server.scoring import group_scored_runs_by_method, make_interval_request


@pytest.fixture
def method_to_runs(archive, dataset_contexts, worker):
    """Method 'a' with two runs, 'b' with one, and 'hard' with one hard-prediction run."""
    submissions = [
        add_run(archive, dataset_contexts, make_run(dataset_contexts, "a", 1, random_seed=1)),
        add_run(archive, dataset_contexts, make_run(dataset_contexts, "a", 2, random_seed=2)),
        add_run(archive, dataset_contexts, make_run(dataset_contexts, "b", random_seed=3)),
        add_run(archive, dataset_contexts,
                make_run(dataset_contexts, "hard", random_seed=4, is_hard=True))]
    return group_scored_runs_by_method(score_and_get_runs(worker, [s.id for s in submissions]))


def test_comparison_equals_compare_methods(dataset_contexts, worker, method_to_runs):
    runs_a, runs_b = method_to_runs["a"], method_to_runs["b"]
    subset = runs_a[0].scores.attributes[1:]
    request = make_comparison_request(runs_a, runs_b, subset, worker.settings)
    assert worker.request(request) is None and worker.is_pending(request)
    worker.run_pending()
    cached = worker.get_result(request)
    assert cached.message == ""

    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    results_a, results_b = (
        [ie.select_result_attributes(evaluate_submission(worker, r.submission, reference_set),
                                     subset) for r in runs]
        for runs in (runs_a, runs_b))
    expected = ie.compare_methods(results_a, results_b, COMPARED_METRIC_NAMES,
                                  num_resamples=worker.settings.num_resamples, seed=0)
    assert_close(cached.value, expected)
    # The subset order does not matter, but the side does.
    assert make_comparison_request(runs_a, runs_b, subset[::-1], worker.settings).key \
           == request.key
    assert make_comparison_request(runs_b, runs_a, subset, worker.settings).key != request.key


def test_hard_predictions_are_compared_on_the_common_metrics(worker, method_to_runs):
    runs = method_to_runs["a"][0].scores.attributes
    request = make_comparison_request(method_to_runs["a"], method_to_runs["hard"], runs,
                                      worker.settings)
    worker.request(request)
    worker.run_pending()
    differences = worker.get_result(request).value
    assert "amF1" in differences.averages and "aNLL" not in differences.averages
    assert "mF1" in differences.per_attribute


def test_comparison_requests_are_checked(worker, method_to_runs):
    attributes = method_to_runs["a"][0].scores.attributes
    with pytest.raises(ValueError, match="Method B: .*differ in metrics"):
        make_comparison_request(method_to_runs["a"],
                                [*method_to_runs["b"], *method_to_runs["hard"]], attributes,
                                worker.settings)
    with pytest.raises(ValueError, match="At least one attribute"):
        make_comparison_request(method_to_runs["a"], method_to_runs["b"], (), worker.settings)


def test_set_requests_withdraws_requests_of_both_kinds(worker, method_to_runs):
    runs_a, runs_b = method_to_runs["a"], method_to_runs["b"]
    subset = runs_a[0].scores.attributes[:2]
    intervals = make_interval_request(runs_a, subset, worker.settings)
    comparison = make_comparison_request(runs_a, runs_b, subset, worker.settings)
    worker.set_requests("page", [intervals, comparison])
    assert worker.is_pending(intervals) and worker.is_pending(comparison)
    worker.set_requests("page", [comparison])
    assert not worker.is_pending(intervals) and worker.is_pending(comparison)
    worker.run_pending()
    assert worker.get_result(comparison).value is not None
    assert worker.get_result(intervals) is None
