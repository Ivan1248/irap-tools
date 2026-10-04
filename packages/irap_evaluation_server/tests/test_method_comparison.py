import math

import irap_evaluation as ie
import pytest
from run_helpers import (
    add_model,
    assert_close,
    make_model,
    score_and_get_models,
    score_submission_again,
)
from synthetic_vietnam import SPLIT

from irap_evaluation_server.method_comparison import (
    COMPARED_METRIC_NAMES,
    get_better_method,
    make_comparison_request,
)
from irap_evaluation_server.scoring import group_scored_models_by_method


@pytest.fixture
def method_to_models(archive, dataset_contexts, worker):
    """Method 'a' with two models, 'b' with one, and 'hard' with one of hard predictions."""
    submissions = [
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "a", 1, random_seed=1)),
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "a", 2, random_seed=2)),
        add_model(archive, dataset_contexts, make_model(dataset_contexts, "b", random_seed=3)),
        add_model(archive, dataset_contexts,
                  make_model(dataset_contexts, "hard", random_seed=4, is_hard=True))]
    return group_scored_models_by_method(
        score_and_get_models(worker, [s.id for s in submissions]))


def test_comparison_equals_compare_methods(dataset_contexts, worker, method_to_models):
    models_a, models_b = method_to_models["a"], method_to_models["b"]
    subset = models_a[0].scores.attributes[1:]
    request = make_comparison_request(models_a, models_b, subset, worker.settings)
    assert worker.request(request) is None and worker.is_pending(request)
    worker.run_pending()
    cached = worker.get_result(request)
    assert cached.message == ""

    reference_set = dataset_contexts["vietnam"].get_reference_set(SPLIT)
    results_a, results_b = (
        [ie.select_result_attributes(score_submission_again(worker, m.submission, reference_set),
                                     subset) for m in models]
        for models in (models_a, models_b))
    expected = ie.compare_methods(results_a, results_b, COMPARED_METRIC_NAMES,
                                  num_resamples=worker.settings.num_resamples, seed=0)
    assert_close(cached.value, expected)
    # The subset order does not matter, but the side does.
    assert make_comparison_request(models_a, models_b, subset[::-1], worker.settings).key \
           == request.key
    assert make_comparison_request(models_b, models_a, subset, worker.settings).key \
           != request.key


def test_hard_predictions_are_compared_on_the_common_metrics(worker, method_to_models):
    attributes = method_to_models["a"][0].scores.attributes
    request = make_comparison_request(method_to_models["a"], method_to_models["hard"],
                                      attributes, worker.settings)
    worker.request(request)
    worker.run_pending()
    differences = worker.get_result(request).value
    assert "amF1" in differences.averages and "aNLL" not in differences.averages
    assert "mF1" in differences.per_attribute


def test_comparison_requests_are_checked(worker, method_to_models):
    attributes = method_to_models["a"][0].scores.attributes
    with pytest.raises(ValueError, match="Method B: .*differ in metrics"):
        make_comparison_request(method_to_models["a"],
                                [*method_to_models["b"], *method_to_models["hard"]], attributes,
                                worker.settings)
    with pytest.raises(ValueError, match="At least one attribute"):
        make_comparison_request(method_to_models["a"], method_to_models["b"], (),
                                worker.settings)


def test_get_better_method():
    def interval(low, high):
        return ie.BootstrapInterval(low=low, high=high, num_undefined=0)

    assert get_better_method(interval(0.05, 0.15), "amF1") == "A"
    assert get_better_method(interval(-0.15, -0.05), "amF1") == "B"
    # Lower is better.
    assert get_better_method(interval(0.05, 0.15), "aNLL") == "B"
    assert get_better_method(interval(-0.15, -0.05), "aNLL") == "A"
    # The interval includes 0, is undefined, or there is none.
    for undecided in (interval(-0.1, 0.1), interval(0.0, 0.1), interval(math.nan, math.nan),
                      None):
        assert get_better_method(undecided, "amF1") is None
