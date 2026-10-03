import dataclasses as dc
import json

import numpy as np
import pytest

from irap_data.metadata import IGNORE_LABEL_INDEX
from irap_evaluation.bootstrap import (
    BootstrapInterval,
    MetricDifference,
    _compute_interval,
    _resample_method_metrics,
    compare_methods,
    compute_bootstrap_intervals,
    compute_mean_metrics,
    compute_method_metrics,
    draw_group_weights,
)
from irap_evaluation.evaluation import (
    align_to_evaluation_set,
    compute_class_metrics,
    evaluate_predictions,
    group_runs_by_method,
    select_evaluated_attributes,
    select_result_attributes,
)
from irap_evaluation.evaluation_sets import (
    get_evaluation_set,
    get_model_compatible_set,
    select_evaluation_sets,
    select_evaluation_subset,
)
from irap_evaluation.metrics import (
    MetricValues,
    compute_classification_metrics,
    compute_classification_statistics,
    compute_grouped_metrics,
    compute_multi_attribute_metrics,
    select_metric_attributes,
    sum_statistics,
)
from irap_evaluation.predictions import MethodInfo, PredictionFormatError, select_segments
from irap_evaluation.reports.evaluation_report import (
    get_partially_undefined_intervals,
    to_class_table,
    to_json_dict,
    to_long_table,
    to_per_attribute_table,
    write_evaluation_reports,
)

from synthetic_vietnam import NUM_ROADS, ROAD_LENGTH, SPLIT, make_predictions


def _invalidate(predictions, attr, is_valid):
    probs = np.where(is_valid[:, None], predictions.probs[attr], np.nan).astype(np.float32)
    return dc.replace(predictions, probs={**predictions.probs, attr: probs})


def test_reference_set_is_a_subset_of_each_model_compatible_set(metadata, reference_set):
    model_compatible_set = get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -1))
    assert (reference_set.name, model_compatible_set.name) == ("reference", "model")
    # The -4..0 window leaves out the first 4 segments of each road, (0, -1) only the first.
    assert reference_set.num_segments == NUM_ROADS * (ROAD_LENGTH - 4)
    assert model_compatible_set.num_segments == NUM_ROADS * (ROAD_LENGTH - 1)
    assert set(reference_set.segment_ids) <= set(model_compatible_set.segment_ids)
    assert reference_set.sequence_ids == tuple(f"road{r}" for r in range(NUM_ROADS))
    np.testing.assert_array_equal(
        np.array(reference_set.sequence_ids)[reference_set.segment_sequence_indices],
        reference_set.segment_sequence_ids)


def test_evaluation_set_fingerprint(metadata, reference_set):
    fingerprint = reference_set.fingerprint
    assert get_evaluation_set(metadata, "vietnam", SPLIT, reference_set.context_offsets,
                              reference_set.name).fingerprint == fingerprint
    attr = next(iter(reference_set.class_indices))
    labels = reference_set.class_indices[attr].copy()
    labels[0] = (labels[0] + 1) % 2
    changed = dc.replace(reference_set,
                         class_indices={**reference_set.class_indices, attr: labels})
    assert changed.fingerprint != fingerprint
    assert get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -1)).fingerprint != fingerprint


def test_select_evaluation_sets(metadata, reference_set):
    model_compatible_set = get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -1))
    # The window of (0, -5) reaches beyond the -4..0 reference window, so that the fifth
    # segment of each road is in the reference set but not in this model-compatible set.
    wide_model_compatible_set = get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -5))
    all_segments = model_compatible_set.segment_ids
    without_a_reference_segment = [s for s in all_segments if s != reference_set.segment_ids[0]]

    def select(segment_ids, model_compatible_set):
        sets, notes = select_evaluation_sets(segment_ids, reference_set, model_compatible_set)
        return [s.name for s in sets], notes

    assert select(all_segments, model_compatible_set) == (["reference", "model"], [])
    names, notes = select(all_segments, None)
    assert names == ["reference"] and "not scored on its model-compatible set" in notes[0]
    names, notes = select(wide_model_compatible_set.segment_ids, wide_model_compatible_set)
    assert names == ["model"] and "not scored on the reference set" in notes[0]
    names, notes = select(all_segments, dc.replace(reference_set, name="model"))
    assert names == ["reference"] and "model-compatible set is left out" in notes[0]
    with pytest.raises(ValueError, match="cannot be scored"):
        select(without_a_reference_segment, None)
    # Incomplete predictions, also when the model-compatible set has the segments of the
    # reference set.
    for incomplete_set in (model_compatible_set, dc.replace(reference_set, name="model")):
        with pytest.raises(ValueError, match="of the model-compatible set .* incomplete"):
            select(without_a_reference_segment, incomplete_set)


def test_sets_without_labels_are_left_out(metadata, reference_set):
    def without_labels(evaluation_set):
        return dc.replace(evaluation_set, class_indices={
            a: np.full_like(v, IGNORE_LABEL_INDEX) for a, v in evaluation_set.class_indices.items()})

    model_compatible_set = get_model_compatible_set(metadata, "vietnam", SPLIT, (0, -1))
    segment_ids = model_compatible_set.segment_ids
    sets, notes = select_evaluation_sets(segment_ids, without_labels(reference_set),
                                         model_compatible_set)
    assert [s.name for s in sets] == ["model"]
    assert notes == ["The reference set has no labels, so the model is not scored on it."]
    sets, notes = select_evaluation_sets(segment_ids, without_labels(reference_set),
                                         without_labels(model_compatible_set))
    assert sets == [] and len(notes) == 2
    with pytest.raises(ValueError, match="of the model-compatible set .* incomplete"):
        select_evaluation_sets(model_compatible_set.segment_ids[1:], without_labels(reference_set),
                               without_labels(model_compatible_set))


def test_subsets_keep_their_road_sequences(reference_set, predictions):
    is_selected = np.isin(reference_set.segment_sequence_ids, ["road1", "road3"])
    subset = select_evaluation_subset(reference_set, is_selected, "roads 1 and 3")
    assert subset.name == "roads 1 and 3"
    assert subset.context_offsets == reference_set.context_offsets
    assert subset.sequence_ids == ("road1", "road3")
    np.testing.assert_array_equal(subset.class_indices["Curvature"],
                                  reference_set.class_indices["Curvature"][is_selected])
    result = evaluate_predictions(predictions, subset)
    assert result.metrics.per_attribute["n"]["Lane width"] == int(is_selected.sum())
    with pytest.raises(ValueError, match="bool array"):
        select_evaluation_subset(reference_set, np.flatnonzero(is_selected), "x")


def test_evaluated_attributes_are_the_labeled_canonical_ones(reference_set, predictions):
    assert select_evaluated_attributes(reference_set) == ("Number of lanes", "Lane width",
                                                          "Curvature")
    with pytest.raises(ValueError, match=r"not in the dataset: \['Lane widht'\]"):
        select_evaluated_attributes(reference_set, ["Lane width", "Lane widht"])
    with pytest.raises(ValueError, match="not in the dataset"):
        evaluate_predictions(predictions, reference_set, attributes=["Lane width", "Speed limit"])


def test_evaluation_matches_direct_metric_computation(metadata, reference_set):
    # In reverse segment order, so that the rows have to be aligned with the evaluation set.
    predictions = make_predictions(metadata.vocabulary, list(reversed(metadata.splits[SPLIT])),
                                   name="m", seed=0)
    result = evaluate_predictions(predictions, reference_set)
    selected = select_segments(predictions, reference_set.segment_ids)
    expected = compute_multi_attribute_metrics(
        {a: compute_classification_statistics(
            reference_set.class_indices[a], selected.probs[a].argmax(1),
            selected.probs[a].shape[1], probs=selected.probs[a])
         for a in result.attributes}, result.metrics.names)
    for name in ["amF1", "aMCC", "aNLL", "amF1_supp5"]:
        assert result.metrics.averages[name] == pytest.approx(float(expected.averages[name]),
                                                              nan_ok=True)
    assert result.metrics.per_attribute["n"]["Curvature"] == int(
        (reference_set.class_indices["Curvature"] != IGNORE_LABEL_INDEX).sum())


def test_evaluation_is_independent_of_the_predicted_class_order(metadata, reference_set,
                                                                predictions):
    reversed_ = make_predictions(metadata.vocabulary, list(metadata.splits[SPLIT]), name="m",
                                 seed=0, reverse_classes=True)
    # Same random logits, so reversing the class order changes which class gets which logit.
    # Re-align the reversed file's probabilities to check that alignment is by code.
    realigned = dc.replace(reversed_,
                           probs={a: p[:, ::-1].copy() for a, p in reversed_.probs.items()})
    np.testing.assert_equal(dc.asdict(evaluate_predictions(predictions, reference_set).metrics),
                            dc.asdict(evaluate_predictions(realigned, reference_set).metrics))


def test_align_to_evaluation_set(metadata, reference_set):
    segments = list(metadata.splits[SPLIT])
    reversed_ = make_predictions(metadata.vocabulary.restrict_to_attributes(["Curvature"]),
                                 segments[::-1], name="m", seed=0, reverse_classes=True)
    aligned = align_to_evaluation_set(reversed_, reference_set, ["Curvature", "Lane width"],
                                      missing_attribute_policy="invalid")
    assert aligned.segment_ids == reference_set.segment_ids
    assert aligned.attribute_to_irap_codes == {
        a: metadata.vocabulary.get_irap_codes(a) for a in ["Curvature", "Lane width"]}
    rows = [segments[::-1].index(s) for s in reference_set.segment_ids]
    np.testing.assert_array_equal(aligned.probs["Curvature"],
                                  reversed_.probs["Curvature"][rows][:, ::-1])
    assert not aligned.is_valid["Lane width"].any()
    with pytest.raises(PredictionFormatError, match="not predicted"):
        align_to_evaluation_set(reversed_, reference_set, ["Lane width"])


def test_null_policies(reference_set, predictions):
    with_invalid = _invalidate(predictions, "Lane width",
                               np.arange(predictions.num_segments) % 2 == 0)
    first_class = evaluate_predictions(with_invalid, reference_set, null_policy="first_class")
    excluded = evaluate_predictions(with_invalid, reference_set, null_policy="exclude")
    num_labeled = reference_set.attr_to_num_labeled["Lane width"]
    assert first_class.metrics.per_attribute["n"]["Lane width"] == num_labeled
    assert excluded.metrics.per_attribute["n"]["Lane width"] == \
           num_labeled - excluded.num_invalid["Lane width"]
    assert excluded.num_invalid["Lane width"] > 0
    all_invalid = _invalidate(predictions, "Lane width", np.zeros(predictions.num_segments, bool))
    with pytest.raises(ValueError, match=r"\['Lane width'\] is invalid"):
        evaluate_predictions(all_invalid, reference_set, null_policy="exclude")


def test_first_class_scores_invalid_cells_as_uniform_distributions(metadata, reference_set,
                                                                   predictions):
    invalid = _invalidate(predictions, "Lane width", np.zeros(predictions.num_segments, bool))
    metrics = evaluate_predictions(invalid, reference_set,
                                   null_policy="first_class").metrics.per_attribute
    num_classes = len(metadata.vocabulary.get_irap_codes("Lane width"))
    # Finite, unlike the NLL of a one-hot distribution of class 0.
    assert metrics["NLL"]["Lane width"] == pytest.approx(np.log(num_classes))
    assert metrics["Brier"]["Lane width"] == pytest.approx(1 - 1 / num_classes)
    # Every invalid cell counts as a prediction of class 0.
    labels = reference_set.class_indices["Lane width"]
    assert metrics["A"]["Lane width"] == pytest.approx(
        np.mean(labels[labels != IGNORE_LABEL_INDEX] == 0))


def test_class_metrics(metadata, reference_set, predictions):
    result = evaluate_predictions(predictions, reference_set)
    lane_width = compute_class_metrics(result)["Lane width"]
    assert lane_width.irap_codes == metadata.vocabulary.get_irap_codes("Lane width")
    assert list(lane_width.metrics) == ["P", "R", "F1", "IoU", "cNLL", "cBrier"]
    statistics = sum_statistics(result.statistics["Lane width"])
    np.testing.assert_array_equal(lane_width.support, statistics.support)
    np.testing.assert_allclose(lane_width.metrics["F1"],
                               compute_classification_metrics(statistics, ["F1"])["F1"])
    table = to_class_table(lane_width)
    assert list(table.columns) == ["value", "support", "P", "R", "F1", "IoU", "cNLL", "cBrier"]
    assert list(table.index) == list(lane_width.irap_codes)
    with pytest.raises(ValueError, match="Not per-class"):
        compute_class_metrics(result, ["mF1"])


def test_results_do_not_keep_per_class_metrics(reference_set, predictions):
    with pytest.raises(ValueError, match="compute_class_metrics"):
        evaluate_predictions(predictions, reference_set, metric_names=["mF1", "F1"])


def test_evaluation_requires_every_segment_and_attribute(metadata, reference_set):
    segments = list(reference_set.segment_ids)
    with pytest.raises(PredictionFormatError, match="no prediction"):
        evaluate_predictions(make_predictions(metadata.vocabulary, segments[1:], name="m", seed=0),
                             reference_set)
    partial = make_predictions(metadata.vocabulary.restrict_to_attributes(["Curvature"]),
                               segments, name="m", seed=0)
    with pytest.raises(PredictionFormatError, match="not predicted"):
        evaluate_predictions(partial, reference_set)
    assert evaluate_predictions(partial, reference_set, attributes=["Curvature"]).attributes == \
           ("Curvature",)


def test_missing_attributes_score_as_invalid_cells(metadata, reference_set, predictions):
    without = dc.replace(predictions, attribute_to_irap_codes={
        a: c for a, c in predictions.attribute_to_irap_codes.items() if a != "Lane width"},
        probs={a: p for a, p in predictions.probs.items() if a != "Lane width"})
    result = evaluate_predictions(without, reference_set, missing_attribute_policy="invalid")
    all_invalid = evaluate_predictions(
        _invalidate(predictions, "Lane width", np.zeros(predictions.num_segments, bool)),
        reference_set)
    np.testing.assert_equal(dc.asdict(result.metrics), dc.asdict(all_invalid.metrics))
    assert result.num_invalid == all_invalid.num_invalid
    assert result.missing_attributes == ("Lane width",)
    assert all_invalid.missing_attributes == ()
    assert to_json_dict(result)["missing_attributes"] == ["Lane width"]
    with pytest.raises(ValueError, match="exclude"):
        evaluate_predictions(without, reference_set, missing_attribute_policy="invalid",
                             null_policy="exclude")
    # Without missing attributes, 'exclude' is allowed.
    evaluate_predictions(predictions, reference_set, missing_attribute_policy="invalid",
                         null_policy="exclude")
    with pytest.raises(ValueError, match="missing_attribute_policy"):
        evaluate_predictions(predictions, reference_set, missing_attribute_policy="skip")


def test_select_result_attributes_equals_scoring_the_subset(reference_set, predictions):
    result = evaluate_predictions(predictions, reference_set)
    subset = ["Curvature", "Number of lanes"]  # Not in the order of the result.
    assert result.attributes == ("Number of lanes", "Lane width", "Curvature")
    selected = select_result_attributes(result, subset)
    expected = evaluate_predictions(predictions, reference_set,
                                    attributes=[a for a in result.attributes if a in subset])
    assert selected.attributes == expected.attributes
    np.testing.assert_equal(dc.asdict(selected.metrics), dc.asdict(expected.metrics))
    np.testing.assert_equal(dc.asdict(select_metric_attributes(result.metrics, subset)),
                            dc.asdict(expected.metrics))
    assert selected.num_invalid == expected.num_invalid
    assert selected.intervals is None
    np.testing.assert_equal(dc.asdict(compute_bootstrap_intervals([selected], num_resamples=20)),
                            dc.asdict(compute_bootstrap_intervals([expected], num_resamples=20)))
    with pytest.raises(ValueError, match="not scored"):
        select_result_attributes(result, ["Lane width", "Speed limit"])
    with pytest.raises(ValueError, match="At least one"):
        select_result_attributes(result, [])


def test_compute_mean_metrics_equals_the_method_metrics(metadata, reference_set):
    runs = _evaluate_runs(metadata, reference_set, "m", [0, 1, 2])
    np.testing.assert_equal(dc.asdict(compute_mean_metrics([r.metrics for r in runs])),
                            dc.asdict(compute_method_metrics(runs)))
    subset = ["Number of lanes", "Lane width"]
    np.testing.assert_equal(
        dc.asdict(compute_mean_metrics([select_metric_attributes(r.metrics, subset)
                                        for r in runs])),
        dc.asdict(compute_method_metrics([select_result_attributes(r, subset) for r in runs])))
    with pytest.raises(ValueError, match="differ"):
        compute_mean_metrics([runs[0].metrics, select_metric_attributes(runs[1].metrics, subset)])
    with pytest.raises(ValueError, match="At least one"):
        compute_mean_metrics([])


def test_bootstrap_intervals_contain_the_point_estimate(metadata, predictions):
    evaluation_set = get_evaluation_set(metadata, "vietnam", SPLIT, (0,), "all")
    result = evaluate_predictions(predictions, evaluation_set)
    intervals = compute_bootstrap_intervals([result], num_resamples=200, seed=1)
    assert intervals == compute_bootstrap_intervals([result], num_resamples=200, seed=1)
    for name in ["amF1", "aNLL"]:
        interval = intervals.averages[name]
        assert interval.low <= result.metrics.averages[name] <= interval.high
    assert set(intervals.per_attribute["mF1"]) == set(result.attributes)


def test_bootstrap_argument_errors(metadata, reference_set, predictions):
    result = evaluate_predictions(predictions, reference_set)
    with pytest.raises(ValueError, match="At least one resample"):
        compute_bootstrap_intervals([result], num_resamples=0)
    with pytest.raises(ValueError, match="At least one resample"):
        compare_methods([result], [result], num_resamples=0)
    with pytest.raises(ValueError, match="confidence"):
        compute_bootstrap_intervals([result], num_resamples=10, confidence=95)
    with pytest.raises(ValueError, match="at least one run"):
        compute_bootstrap_intervals([], num_resamples=10)
    # No segment has a context window of 100 segments.
    empty_set = get_evaluation_set(metadata, "vietnam", SPLIT, (0, -100), "empty")
    empty_result = evaluate_predictions(predictions, empty_set, attributes=["Lane width"])
    with pytest.raises(ValueError, match="no road sequences"):
        compute_bootstrap_intervals([empty_result], num_resamples=10)


def test_compare_methods(metadata, reference_set, predictions):
    predictions_b = make_predictions(metadata.vocabulary, list(metadata.splits[SPLIT]),
                                     name="b", seed=1)
    result_a, result_b = (evaluate_predictions(p, reference_set)
                          for p in (predictions, predictions_b))
    self_difference = compare_methods([result_a], [result_a], ["amF1"],
                                      num_resamples=50).averages["amF1"]
    assert self_difference == MetricDifference(0.0, BootstrapInterval(0.0, 0.0, 0))
    per_attribute = compare_methods([result_a], [result_b], ["mF1"], num_resamples=20)
    assert set(per_attribute.per_attribute["mF1"]) == set(result_a.attributes)
    assert compare_methods([result_a], [result_b], num_resamples=20).names == \
           tuple(result_a.metrics.averages)

    with pytest.raises(ValueError, match="not in both methods"):
        compare_methods([result_a], [result_b], ["amIoU"], num_resamples=20)
    other_set = evaluate_predictions(
        predictions_b, get_evaluation_set(metadata, "vietnam", SPLIT, (0,), "all"))
    with pytest.raises(ValueError, match="different segments"):
        compare_methods([result_a], [other_set])
    with pytest.raises(ValueError, match="null policies"):
        compare_methods([result_a], [evaluate_predictions(predictions_b, reference_set,
                                                          null_policy="exclude")])
    half_invalid = _invalidate(predictions, "Lane width",
                               np.arange(predictions.num_segments) % 2 == 0)
    with pytest.raises(ValueError, match="exclude"):
        compare_methods(
            [evaluate_predictions(half_invalid, reference_set, null_policy="exclude")],
            [evaluate_predictions(predictions_b, reference_set, null_policy="exclude")])


def _evaluate_runs(metadata, evaluation_set, name, prediction_seeds):
    """Results of runs with seeds 0, 1, ... of random predictions with `prediction_seeds`."""
    segments = list(metadata.splits[SPLIT])
    return [evaluate_predictions(make_predictions(metadata.vocabulary, segments, name=name,
                                                  seed=s, method_seed=i), evaluation_set)
            for i, s in enumerate(prediction_seeds)]


def _to_bounds(intervals):
    """(num_intervals, 2) the bounds of `MetricValues` of intervals."""
    per_attribute = [i for attr_to_interval in intervals.per_attribute.values()
                     for i in attr_to_interval.values()]
    return np.array([(i.low, i.high) for i in [*intervals.averages.values(), *per_attribute]])


def test_identical_runs_add_no_uncertainty(metadata, reference_set):
    [run] = _evaluate_runs(metadata, reference_set, "m", [0])
    copies = [dc.replace(run, method=MethodInfo("m", seed=i)) for i in range(3)]
    assert compute_method_metrics(copies) == run.metrics
    # The mean of the resampled values can differ from the value by rounding.
    np.testing.assert_allclose(
        _to_bounds(compute_bootstrap_intervals(copies, num_resamples=50, seed=4)),
        _to_bounds(compute_bootstrap_intervals([run], num_resamples=50, seed=4)), rtol=1e-12)


def test_resampled_method_values_add_the_run_term(metadata, reference_set):
    runs = _evaluate_runs(metadata, reference_set, "m", [0, 1, 2])
    rng = np.random.default_rng(0)
    weights = draw_group_weights(len(reference_set.sequence_ids), 20, rng)
    t_values = rng.standard_t(2, size=20)
    resampled = _resample_method_metrics(runs, ["amF1", "mF1"], weights, t_values)
    run_values = np.stack([compute_grouped_metrics(r.statistics, ["amF1"], weights)
                           .averages["amF1"] for r in runs])
    np.testing.assert_allclose(
        resampled.averages["amF1"],
        run_values.mean(0) + run_values.std(0, ddof=1) / np.sqrt(3) * t_values)
    assert set(resampled.per_attribute["mF1"]) == set(runs[0].attributes)

    point = compute_method_metrics(runs)
    assert point.averages["amF1"] == pytest.approx(
        np.mean([r.metrics.averages["amF1"] for r in runs]))
    interval = compute_bootstrap_intervals(runs, num_resamples=200).averages["amF1"]
    assert interval.low <= point.averages["amF1"] <= interval.high


def test_an_infinite_run_mean_stays_infinite(metadata, reference_set, predictions):
    # Probability 0 for every class but the first gives an infinite NLL.
    probs = np.zeros_like(predictions.probs["Lane width"])
    probs[:, 0] = 1
    certain = dc.replace(predictions, probs={**predictions.probs, "Lane width": probs},
                         header=dc.replace(predictions.header, method=MethodInfo("m", seed=1)))
    runs = [evaluate_predictions(dc.replace(predictions, header=dc.replace(
                predictions.header, method=MethodInfo("m", seed=0))), reference_set),
            evaluate_predictions(certain, reference_set)]
    assert compute_method_metrics(runs).averages["aNLL"] == np.inf
    interval = compute_bootstrap_intervals(runs, num_resamples=50).averages["aNLL"]
    assert (interval.low, interval.high) == (np.inf, np.inf)


def test_runs_must_be_distinct_comparable_runs_of_one_method(metadata, reference_set):
    run_0, run_1 = _evaluate_runs(metadata, reference_set, "m", [0, 1])
    without_seed = dc.replace(run_1, method=MethodInfo("m"))
    for runs, message in [
        ([run_0, dc.replace(run_1, method=MethodInfo("m", seed=0))], "same seed"),
        ([run_0, without_seed], "each needs a seed"),
        ([run_0, dc.replace(run_1, method=MethodInfo("other", seed=1))], "different methods"),
        ([run_0, dc.replace(run_1, null_policy="exclude")], "null policies"),
        ([run_0, evaluate_predictions(
            make_predictions(metadata.vocabulary, list(metadata.splits[SPLIT]), name="m",
                             seed=1, method_seed=1),
            reference_set, metric_names=["amF1"])], "different metrics"),
    ]:
        with pytest.raises(ValueError, match=message):
            compute_bootstrap_intervals(runs, num_resamples=10)
    with pytest.raises(ValueError, match="same seed"):
        group_runs_by_method([run_0, run_0])
    assert group_runs_by_method([run_0, run_1]) == {"m": (run_0, run_1)}


@pytest.mark.parametrize("values, expected", [
    ([1.0] * 3 + [np.inf] * 97, (np.inf, np.inf)),
    ([1.0] * 50 + [np.inf] * 50, (1.0, np.inf)),
    ([np.inf] * 10, (np.inf, np.inf)),
    ([-np.inf] * 50 + [0.0] * 50, (-np.inf, 0.0)),
    ([np.nan] * 5, (np.nan, np.nan)),
])
def test_intervals_of_infinite_values(values, expected):
    interval = _compute_interval(np.array(values), 0.95)
    np.testing.assert_equal((interval.low, interval.high), expected)
    assert interval.num_undefined == np.isnan(values).sum()


def test_intervals_of_finite_values_are_numpy_percentiles():
    values = np.random.default_rng(0).normal(size=101)
    interval = _compute_interval(np.concatenate([values, [np.nan]]), 0.9)
    np.testing.assert_allclose((interval.low, interval.high), np.quantile(values, [0.05, 0.95]))
    assert interval.num_undefined == 1


def test_only_partially_undefined_intervals_are_reported():
    intervals = MetricValues(
        averages={"amF1": BootstrapInterval(0.1, 0.2, 0), "aNLL": BootstrapInterval(-1, 1, 3)},
        per_attribute={"MCC": {"Lane width": BootstrapInterval(np.nan, np.nan, 10)}})
    assert get_partially_undefined_intervals(intervals) == {"aNLL": 3}


def test_json_report_has_no_nan(reference_set, predictions):
    # Predicting one class everywhere makes MCC undefined (NaN).
    probs = np.zeros_like(predictions.probs["Lane width"])
    probs[:, 0] = 1
    constant = dc.replace(predictions, probs={**predictions.probs, "Lane width": probs})
    report = to_json_dict(evaluate_predictions(constant, reference_set))
    assert report["metrics"]["per_attribute"]["MCC"]["Lane width"] is None
    json.dumps(report, allow_nan=False)


def test_per_attribute_tables_need_per_attribute_metrics(reference_set, predictions):
    long_table = to_long_table([evaluate_predictions(predictions, reference_set)])
    with pytest.raises(ValueError, match="no per-attribute values of 'amF1'"):
        to_per_attribute_table(long_table, "amF1")


def test_reports_refuse_evaluation_sets_that_map_to_one_file(reference_set, predictions,
                                                             tmp_path):
    results = [evaluate_predictions(predictions, dc.replace(reference_set, name=name))
               for name in ["a b", "a_b"]]
    with pytest.raises(ValueError, match="map to the files"):
        write_evaluation_reports(results, tmp_path / "out", table_metrics=["mF1"])
    assert not (tmp_path / "out").exists()
