import dataclasses as dc
import math
from datetime import datetime, timezone

import irap_evaluation as ie
import pytest
from irap_evaluation.metrics import IRAP_ATTRIBUTE_METRIC_NAMES

from irap_evaluation_server.archive import Model, Submission
from irap_evaluation_server.components.attribute_filter import AttributeSubset
from irap_evaluation_server.components.score_tables import (
    make_method_table_html,
    make_per_attribute_table_html,
)
from irap_evaluation_server.components.view_queries import select_matching_names
from irap_evaluation_server.method_ranking import (
    CellHighlight,
    compute_highlights,
    get_unscored_reason,
    rank_methods,
    select_model_metrics,
    select_shown_methods,
    sort_method_scores,
    sort_scored_models,
)
from irap_evaluation_server.model_listing import summarize_models_by_method
from irap_evaluation_server.scoring import (
    CachedResult,
    EvaluationSetScores,
    MethodScores,
    ResultState,
    ScoredModel,
    SubmissionScoring,
)

_TIME = datetime(2026, 10, 3, tzinfo=timezone.utc)


def make_model_row(model_id, method_name="m", seed=None, method_display_name=None):
    return Model(id=model_id, dataset="vietnam", method_name=method_name,
                 method_display_name=method_display_name, seed=seed, description="",
                 created_at=_TIME)


def make_submission(submission_id, method_name="m", seed=None, replaced_at=None,
                    training_splits=(), model=None, split="val", num_attributes=2):
    return Submission(
        id=submission_id, model=model or make_model_row(submission_id, method_name, seed),
        split=split, file_sha256="0" * 64, output_kind="probs", context_offsets=(0,),
        training_splits=training_splits, early_stopping_splits=(), num_segments=10,
        num_attributes=num_attributes, submitter="Bo", uploaded_at=_TIME,
        replaced_at=replaced_at)


def make_scored_model(submission_id, attribute_to_value, method_name="m",
                      missing_attributes=(), training_splits=()):
    """Makes a scored model whose iRAP metrics of each attribute equal the given value."""
    metrics = ie.MetricValues(
        averages={f"a{n}": math.nan for n in IRAP_ATTRIBUTE_METRIC_NAMES},
        per_attribute={n: dict(attribute_to_value) for n in IRAP_ATTRIBUTE_METRIC_NAMES})
    scores = EvaluationSetScores(
        evaluation_set="reference", context_offsets=(0,), evaluation_set_fingerprint="f",
        attributes=tuple(attribute_to_value),
        metrics=ie.select_metric_attributes(metrics, list(attribute_to_value)),
        num_invalid={a: 3 for a in attribute_to_value}, missing_attributes=missing_attributes)
    return ScoredModel(make_submission(submission_id, method_name, seed=submission_id,
                                       training_splits=training_splits), scores)


def make_method_scores(method_name, averages, error=None, training_splits=()):
    return MethodScores(
        method_name=method_name,
        models=(make_scored_model(1, {"x": 0.5}, method_name, training_splits=training_splits),),
        metrics=None if error else ie.MetricValues(averages=averages, per_attribute={}),
        error=error, missing_attributes=())


def test_sort_method_scores():
    rows = [make_method_scores("a", {"amF1": 0.5, "aNLL": 1.0}),
            make_method_scores("b", {"amF1": 0.7, "aNLL": 2.0}),
            make_method_scores("c", {"amF1": math.nan, "aNLL": math.inf}),
            make_method_scores("d", {}, error="differ"),
            make_method_scores("e", {"amF1": 0.6})]  # Hard predictions, without aNLL.
    assert [r.method_name for r in sort_method_scores(rows, "amF1")] == ["b", "e", "a", "c", "d"]
    # Lower is better, infinite is worst.
    assert [r.method_name for r in sort_method_scores(rows, "aNLL")] == ["a", "b", "c", "d", "e"]


def test_ties_are_sorted_by_shown_method_name():
    def make_row(method_name, method_display_name):
        row = make_method_scores(method_name, {"amF1": 0.5})
        [model] = row.models
        submission = dc.replace(model.submission, model=make_model_row(
            1, method_name, method_display_name=method_display_name))
        return dc.replace(row, models=(dc.replace(model, submission=submission),))

    rows = [make_row("a", "z"), make_row("b", None)]
    assert [r.shown_method_name for r in rows] == ["z", "b"]
    assert [r.method_name for r in sort_method_scores(rows, "amF1")] == ["b", "a"]


def test_rank_methods_means_over_models():
    models = [make_scored_model(1, {"x": 0.2, "y": 0.4}),
              make_scored_model(2, {"x": 0.4, "y": 0.6}),
              make_scored_model(3, {"x": 0.5, "y": 0.1}, method_name="n")]
    rows = rank_methods(models, ["x"], "amF1")
    assert [(r.method_name, r.num_models) for r in rows] == [("n", 1), ("m", 2)]
    assert rows[1].metrics.averages["amF1"] == pytest.approx(0.3)
    [row] = rank_methods(models[:1], ["x", "z"], "amF1")
    assert "not scored on z" in row.error


def test_sort_scored_models_by_the_average_over_a_subset():
    models = [make_scored_model(1, {"x": 0.2, "y": 0.9}),
              make_scored_model(2, {"x": 0.4, "y": 0.1}), make_scored_model(3, {"x": 0.3})]
    assert select_model_metrics(models[0], ["x", "y"]).averages["amF1"] == pytest.approx(0.55)
    assert select_model_metrics(models[2], ["x", "y"]) is None  # It is not scored on y.
    assert select_model_metrics(models[0], []) is None
    assert [m.submission.id for m in sort_scored_models(models, "amF1", ["x"])] == [2, 3, 1]
    assert [m.submission.id for m in sort_scored_models(models, "amF1", ["x", "y"])] == [
        1, 2, 3]


def test_select_shown_methods():
    rows = [make_method_scores("a-1", {"amF1": 0.6}),
            make_method_scores("x", {}, error="differ"), make_method_scores("B-2", {"amF1": 0.5}),
            make_method_scores("t", {"amF1": 0.9}, training_splits=("val",))]

    def select(text, reference_method="", hidden_training_split=None):
        shown, num_hidden = select_shown_methods(
            rows, select_matching_names(text, [r.method_name for r in rows]), reference_method,
            hidden_training_split)
        return [r.method_name for r in shown], num_hidden

    assert select("") == (["a-1", "x", "B-2", "t"], 0)
    assert select("b") == (["B-2"], 0)  # A part of the name, case-insensitive.
    assert select(r"^(a|b)-\d$") == (["a-1", "B-2"], 0)
    # The reference, unmatched.
    assert select("^x$", reference_method="B-2") == (["x", "B-2"], 0)
    assert select("nothing") == ([], 0)
    # The methods trained on the split are hidden, except the reference.
    assert select("", hidden_training_split="val") == (["a-1", "x", "B-2"], 1)
    assert select("", hidden_training_split="test") == (["a-1", "x", "B-2", "t"], 0)
    assert select("", "t", hidden_training_split="val") == (["a-1", "x", "B-2", "t"], 0)
    assert select("^a", hidden_training_split="val") == (["a-1"], 0)  # Only matching rows.


def test_select_matching_names_refuses_invalid_and_slow_patterns():
    with pytest.raises(ValueError, match="Invalid regular expression"):
        select_matching_names("a(", ["a"])
    # Exponential backtracking, which `regex` does not avoid, unlike for e.g. '(a+)+#'.
    with pytest.raises(ValueError, match="too long"):
        select_matching_names(r"^(\w|\w\w)*$", ["a" * 40 + "!"])


def test_summarize_models_by_method():
    models = [make_model_row(1, "a", seed=1), make_model_row(2, "a", seed=0),
              make_model_row(3, "b"), make_model_row(4, "c")]
    submissions = [make_submission(10, model=models[0], split="val", num_attributes=3),
                   make_submission(11, model=models[0], split="test", num_attributes=5),
                   make_submission(12, model=models[0], split="val", replaced_at=_TIME),
                   make_submission(13, model=models[1]),
                   make_submission(14, model=models[2], training_splits=None)]
    later = dc.replace(submissions[4], uploaded_at=_TIME.replace(year=2027))
    groups = summarize_models_by_method(models, [*submissions[:4], later])
    assert [g.method_name for g in groups] == ["b", "a", "c"]  # The latest updated first.
    a = groups[1]
    assert [m.model.seed for m in a.models] == [0, 1]
    summary = a.models[1]
    assert {s: x.id for s, x in summary.split_to_submission.items()} == {"val": 10, "test": 11}
    assert (summary.num_segments, summary.attribute_range, summary.output_kind) == (
        20, (3, 5), "probs")
    assert groups[0].model_info.training_splits is None
    c = groups[2].models[0]
    assert (c.split_to_submission, c.output_kind, c.attribute_range, c.model_info) == (
        {}, None, None, None)


def test_get_unscored_reason():
    scoring = SubmissionScoring(submission_id=1, status="scored", message="", notes=(),
                                settings_fingerprint="s", evaluation_sets_fingerprint="e",
                                scored_at=_TIME)
    assert get_unscored_reason(scoring, is_current=True, is_pending=False) is None
    assert get_unscored_reason(scoring, True, is_pending=True) == "Scoring…"
    assert "current settings" in get_unscored_reason(None, False, False)
    failed = dc.replace(scoring, status="failed", message="No predictions for 3 segments.")
    assert get_unscored_reason(failed, True, False) == failed.message
    error = dc.replace(scoring, status="error", message="Unexpected error.")
    assert "scored again" in get_unscored_reason(error, False, False)


def test_attribute_subset():
    options = ("a", "b", "c")
    assert AttributeSubset.from_query([], options).is_all
    subset = AttributeSubset.from_query(["c", "x", "b"], options)
    assert (subset.selected, subset.ignored) == (("b", "c"), ("x",))
    assert subset.to_query() == ("b", "c")
    assert subset.label == "2 of 3 attributes"
    # Unknown values alone select all, and are reported.
    unknown = AttributeSubset.from_query(["x"], options)
    assert unknown.is_all and unknown.ignored == ("x",)
    assert unknown.to_query() == () and unknown.label == "all 3 attributes"
    assert subset.select(["c", "a"]).selected == ("a", "c")
    assert subset.select([]).selected == ()


def test_compute_highlights_strength():
    values = [0.2, None, 0.6, math.nan, 0.4, math.inf]
    strengths = [h and h.strength for h in compute_highlights(values, None, False)]
    assert strengths == [0.0, None, 1.0, None, pytest.approx(0.5), None]
    lower_better = compute_highlights(values, None, True)
    assert [h and h.strength for h in lower_better] == [1.0, None, 0.0, None, pytest.approx(0.5),
                                                       None]
    assert not any(h and h.is_best_or_tied for h in lower_better)  # Without intervals.
    assert compute_highlights([0.5, 0.5], None, False) == [CellHighlight(1.0, False)] * 2
    # Nothing to compare with.
    assert compute_highlights([0.5, None, math.nan], None, False) == [None] * 3
    with pytest.raises(ValueError):
        compute_highlights([0.5, 0.6], [None], False)


def test_compute_highlights_ties():
    def interval(low, high):
        return ie.BootstrapInterval(low=low, high=high, num_undefined=0)

    values = [0.5, 0.8, 0.7, 0.75, 0.6]
    intervals = [interval(0.45, 0.55), interval(0.75, 0.85), interval(0.6, 0.75), None,
                 interval(math.nan, math.nan)]
    ties = [h.is_best_or_tied for h in compute_highlights(values, intervals, False)]
    # The best, and the overlapping one. Values without a usable interval are not tied.
    assert ties == [False, True, True, False, False]
    # Without a usable interval of the best, only the best.
    best_unknown = [None, None, *intervals[2:]]
    ties = [h.is_best_or_tied for h in compute_highlights(values, best_unknown, False)]
    assert ties == [False, True, False, False, False]


def test_method_tables_escape_names():
    models = [make_scored_model(1, {"x": 0.2, "y": 0.4}, method_name="<m>",
                                missing_attributes=("y",))]
    rows = [*rank_methods(models, ["x", "y"], "amF1"),
            make_method_scores("<n>", {}, error="Models <differ>.")]
    interval = ie.BootstrapInterval(low=0.1, high=0.3, num_undefined=2)
    cached = CachedResult(
        value=ie.MetricValues(averages={"amF1": interval}, per_attribute={}),
        message="", computed_at=_TIME)
    table = make_method_table_html(rows, {"<m>": ResultState(cached=cached)}, "amF1", "val",
                                   ["x"], is_model_compatible_view=False)
    assert "&lt;m&gt;" in table and "<m>" not in table
    assert "Models &lt;differ&gt;." in table

    per_attribute = make_per_attribute_table_html(rows, {}, "mF1", ["x", "y"], "val")
    assert "<m>" not in per_attribute and "<n>" not in per_attribute
