import dataclasses as dc

import numpy as np
import pytest

from irap_data.metadata import IGNORE_LABEL_INDEX
from irap_evaluation.metrics import (
    ClassificationStatistics,
    add_per_attribute_metric_names,
    compute_classification_metrics,
    compute_classification_statistics,
    compute_multi_attribute_metrics,
    describe_metric,
    get_irap_metric_names,
    is_lower_better,
    parse_metric_name,
    select_metric_attributes,
    sum_statistics,
)


def test_confusion_matrix_metrics_by_hand():
    # Class 2 is neither in the ground truth nor predicted, so the means leave it out.
    confusion_matrix = np.array([[3, 1, 0], [2, 4, 0], [0, 0, 0]])
    metrics = compute_classification_metrics(
        ClassificationStatistics(confusion_matrix), ["P", "R", "mF1", "mP", "A", "n", "MCC"])
    np.testing.assert_allclose(metrics["P"], [3 / 5, 4 / 5, 0])
    np.testing.assert_allclose(metrics["R"], [3 / 4, 4 / 6, 0])
    f1 = [2 * 0.6 * 0.75 / 1.35, 2 * 0.8 * (4 / 6) / (0.8 + 4 / 6)]
    assert metrics["mF1"] == pytest.approx(np.mean(f1))
    assert metrics["mP"] == pytest.approx(0.7)
    assert metrics["A"] == pytest.approx(0.7)
    assert metrics["n"] == 10
    # MCC = (c n - sum_k p_k t_k) / sqrt((n^2 - sum p^2) (n^2 - sum t^2))
    assert metrics["MCC"] == pytest.approx((7 * 10 - (5 * 4 + 5 * 6))
                                           / np.sqrt((100 - 50) * (100 - 52)))
    all_classes = compute_classification_metrics(ClassificationStatistics(confusion_matrix),
                                                 ["mP"], ignore_missing_classes=False)
    assert all_classes["mP"] == pytest.approx(1.4 / 3)


def test_support_restricted_metrics():
    confusion_matrix = np.array([[6, 0, 0], [0, 4, 1], [0, 0, 1]])
    metrics = compute_classification_metrics(ClassificationStatistics(confusion_matrix),
                                             ["F1", "mF1_supp5", "nc_supp5", "mF1_supp9"])
    assert metrics["nc_supp5"] == 2
    assert metrics["mF1_supp5"] == pytest.approx(metrics["F1"][:2].mean())
    assert np.isnan(metrics["mF1_supp9"])


def test_describe_metric_depends_on_ignore_missing_classes_only_where_it_applies():
    def is_affected(name: str) -> bool:
        return (describe_metric(name, ignore_missing_classes=True)
                != describe_metric(name, ignore_missing_classes=False))

    assert is_affected("mF1") and is_affected("amIoU")
    assert not any(map(is_affected, ["mF1_supp10", "nc_supp5", "mNLL", "F1", "MCC"]))


def test_statistics_ignore_unlabeled_examples_and_sum_over_groups():
    rng = np.random.default_rng(0)
    targets = rng.integers(-1, 3, size=200)
    probs = rng.dirichlet(np.ones(3), size=200).astype(np.float32)
    groups = rng.integers(0, 7, size=200)
    grouped = compute_classification_statistics(targets, probs.argmax(1), 3, probs=probs,
                                                group_indices=groups, num_groups=7)
    whole = compute_classification_statistics(targets, probs.argmax(1), 3, probs=probs)
    summed = sum_statistics(grouped)
    np.testing.assert_array_equal(summed.confusion_matrix, whole.confusion_matrix)
    np.testing.assert_allclose(summed.nll_sums, whole.nll_sums)
    assert whole.confusion_matrix.sum() == (targets != IGNORE_LABEL_INDEX).sum()
    # Weights of 2 for every group double every count.
    doubled = sum_statistics(grouped, np.full((1, 7), 2))
    np.testing.assert_array_equal(doubled.confusion_matrix[0], 2 * whole.confusion_matrix)


def test_weighted_sums_leave_out_infinite_sums_of_undrawn_groups():
    # Group 1 gives its label probability 0, so its NLL sum is infinite.
    targets, groups = np.array([0, 1, 0]), np.array([0, 1, 2])
    probs = np.array([[0.5, 0.5], [1.0, 0.0], [0.25, 0.75]], dtype=np.float32)
    grouped = compute_classification_statistics(targets, probs.argmax(1), 2, probs=probs,
                                                group_indices=groups, num_groups=3)
    weighted = sum_statistics(grouped, np.array([[1, 0, 2], [0, 3, 0]]))
    np.testing.assert_allclose(weighted.nll_sums[0], [np.log(2) - 2 * np.log(0.25), 0])
    np.testing.assert_equal(weighted.nll_sums[1], [0, np.inf])


def test_probabilistic_metrics():
    targets = np.array([0, 0, 1])
    probs = np.array([[0.5, 0.5], [1.0, 0.0], [0.25, 0.75]], dtype=np.float32)
    metrics = compute_classification_metrics(
        compute_classification_statistics(targets, probs.argmax(1), 2, probs=probs),
        ["NLL", "Brier", "mNLL"])
    nll = [np.log(2), 0, -np.log(0.75)]
    assert metrics["NLL"] == pytest.approx(np.mean(nll))
    assert metrics["mNLL"] == pytest.approx(np.mean([np.mean(nll[:2]), nll[2]]))
    assert metrics["Brier"] == pytest.approx(np.mean([0.5, 0, 2 * 0.25 ** 2]))


def test_hard_statistics_refuse_probabilistic_metrics():
    stats = compute_classification_statistics(np.array([0, 1]), np.array([0, 0]), 2)
    with pytest.raises(ValueError, match="hard"):
        compute_classification_metrics(stats, ["NLL"])


def test_attribute_average_leaves_out_undefined_values():
    defined = ClassificationStatistics(np.array([[3, 1], [1, 3]]))
    # A single ground-truth and predicted class makes MCC undefined (0/0).
    undefined = ClassificationStatistics(np.array([[4, 0], [0, 0]]))
    metrics = compute_multi_attribute_metrics({"x": defined, "y": undefined}, ["aMCC", "MCC"])
    assert np.isnan(metrics.per_attribute["MCC"]["y"])
    assert metrics.averages["aMCC"] == pytest.approx(metrics.per_attribute["MCC"]["x"])
    with pytest.raises(ValueError, match="per-class"):
        compute_multi_attribute_metrics({"x": defined}, ["aF1"])


def test_add_per_attribute_metric_names():
    assert add_per_attribute_metric_names(["amF1", "aNLL", "mF1", "amF1_supp5"]) == (
        "amF1", "aNLL", "mF1", "amF1_supp5", "NLL", "mF1_supp5")


def _to_float_values(metrics):
    return type(metrics)(
        averages={n: float(v) for n, v in metrics.averages.items()},
        per_attribute={n: {a: float(v) for a, v in values.items()}
                       for n, values in metrics.per_attribute.items()})


def test_select_metric_attributes_equals_computing_on_the_subset():
    # x has an undefined MCC (one class), y an infinite NLL (label probability 0).
    targets = {"x": np.array([0, 0, 0]), "y": np.array([0, 1, 1]), "z": np.array([1, 0, 1])}
    probs = {"x": np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]),
             "y": np.array([[1.0, 0.0], [1.0, 0.0], [0.4, 0.6]]),
             "z": np.array([[0.3, 0.7], [0.6, 0.4], [0.5, 0.5]])}
    statistics = {a: compute_classification_statistics(
                      targets[a], probs[a].argmax(1), 2, probs=probs[a].astype(np.float32))
                  for a in targets}
    names = add_per_attribute_metric_names(["amF1", "aMCC", "aNLL"])
    metrics = _to_float_values(compute_multi_attribute_metrics(statistics, names))
    for subset in (["x", "z"], ["y", "z"], ["x"], ["z", "x", "y"]):
        expected = _to_float_values(compute_multi_attribute_metrics(
            {a: s for a, s in statistics.items() if a in subset}, names))
        # Equal values, NaN where undefined.
        np.testing.assert_equal(dc.asdict(select_metric_attributes(metrics, subset)),
                                dc.asdict(expected))
    assert np.isnan(select_metric_attributes(metrics, ["x"]).averages["aMCC"])
    assert select_metric_attributes(metrics, ["y", "z"]).averages["aNLL"] == np.inf

    with pytest.raises(ValueError, match="At least one"):
        select_metric_attributes(metrics, [])
    with pytest.raises(ValueError, match=r"\['w'\]"):
        select_metric_attributes(metrics, ["x", "w"])
    averages_only = _to_float_values(compute_multi_attribute_metrics(statistics, ["amF1"]))
    with pytest.raises(ValueError, match="add_per_attribute_metric_names"):
        select_metric_attributes(averages_only, ["x"])


def test_batched_metrics_equal_separate_computations():
    rng = np.random.default_rng(1)
    matrices = rng.integers(0, 5, size=(4, 3, 3))
    batched = compute_classification_metrics(ClassificationStatistics(matrices), ["mF1", "MCC"])
    for i, matrix in enumerate(matrices):
        single = compute_classification_metrics(ClassificationStatistics(matrix), ["mF1", "MCC"])
        assert batched["mF1"][i] == pytest.approx(single["mF1"])
        assert batched["MCC"][i] == pytest.approx(single["MCC"], nan_ok=True)


def test_parse_metric_name():
    assert parse_metric_name("amF1_supp10") == (True, "mF1_supp10", "mF1", 10)
    assert parse_metric_name("A") == (False, "A", "A", None)
    assert parse_metric_name("aA").is_averaged
    for name in ["nc", "A_supp5", "mF1_suppx", "xyz"]:
        with pytest.raises(ValueError):
            parse_metric_name(name)


def test_is_lower_better():
    assert [n for n in get_irap_metric_names() if is_lower_better(n)] == [
        "aNLL", "aBrier", "amNLL", "amNLL_supp5", "amNLL_supp10", "NLL", "Brier", "mNLL"]


def test_get_irap_metric_names_puts_averages_first():
    names = get_irap_metric_names()
    assert names[:5] == ("amF1", "amP", "amR", "aMCC", "aA")
    assert "aNLL" not in get_irap_metric_names(output_kind="hard")
