import dataclasses as dc
import math
from datetime import datetime, timezone

import irap_evaluation as ie
import pytest
from irap_evaluation.metrics import IRAP_ATTRIBUTE_METRIC_NAMES

from irap_evaluation_server.archive import Submission
from irap_evaluation_server.components.attribute_filter import AttributeSubset
from irap_evaluation_server.components.score_tables import (
    ReferenceComparisons,
    make_method_table_html,
    make_per_attribute_table_html,
    make_run_scores_html,
)
from irap_evaluation_server.components.view_queries import compile_name_pattern
from irap_evaluation_server.method_comparison import list_comparable_methods
from irap_evaluation_server.method_ranking import (
    CellHighlight,
    compute_highlights,
    get_unscored_reason,
    rank_methods,
    select_run_metrics,
    select_shown_methods,
    sort_method_scores,
    sort_scored_runs,
)
from irap_evaluation_server.scoring import (
    CachedResult,
    EvaluationSetScores,
    MethodScores,
    ResultState,
    RunScoring,
    ScoredRun,
)

_TIME = datetime(2026, 10, 3, tzinfo=timezone.utc)


def make_submission(submission_id, method_name="m", seed=None, deleted_at=None):
    return Submission(id=submission_id, file_sha256="0" * 64, dataset="vietnam", split="val",
                      method=ie.MethodInfo(method_name, seed), context_offsets=(0,),
                      output_kind="probs", num_segments=10, num_attributes=2, submitter="Bo",
                      description="", uploaded_at=_TIME, deleted_at=deleted_at)


def make_run(submission_id, attribute_to_value, method_name="m", missing_attributes=()):
    """A run whose iRAP metrics of each attribute equal the given value."""
    metrics = ie.MetricValues(
        averages={f"a{n}": math.nan for n in IRAP_ATTRIBUTE_METRIC_NAMES},
        per_attribute={n: dict(attribute_to_value) for n in IRAP_ATTRIBUTE_METRIC_NAMES})
    scores = EvaluationSetScores(
        evaluation_set="reference", context_offsets=(0,), evaluation_set_fingerprint="f",
        attributes=tuple(attribute_to_value),
        metrics=ie.select_metric_attributes(metrics, list(attribute_to_value)),
        num_invalid={a: 3 for a in attribute_to_value}, missing_attributes=missing_attributes)
    return ScoredRun(make_submission(submission_id, method_name, seed=submission_id), scores)


def make_method_scores(method_name, averages, error=None):
    return MethodScores(
        method_name=method_name, runs=(),
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


def test_rank_methods_means_over_runs():
    runs = [make_run(1, {"x": 0.2, "y": 0.4}), make_run(2, {"x": 0.4, "y": 0.6}),
            make_run(3, {"x": 0.5, "y": 0.1}, method_name="n")]
    rows = rank_methods(runs, ["x"], "amF1")
    assert [(r.method_name, r.num_runs) for r in rows] == [("n", 1), ("m", 2)]
    assert rows[1].metrics.averages["amF1"] == pytest.approx(0.3)
    [row] = rank_methods(runs[:1], ["x", "z"], "amF1")
    assert "not scored on z" in row.error


def test_sort_scored_runs_by_the_average_over_a_subset():
    runs = [make_run(1, {"x": 0.2, "y": 0.9}), make_run(2, {"x": 0.4, "y": 0.1}),
            make_run(3, {"x": 0.3})]
    assert select_run_metrics(runs[0], ["x", "y"]).averages["amF1"] == pytest.approx(0.55)
    assert select_run_metrics(runs[2], ["x", "y"]) is None  # It is not scored on y.
    assert select_run_metrics(runs[0], []) is None
    assert [r.submission.id for r in sort_scored_runs(runs, "amF1", ["x"])] == [2, 3, 1]
    assert [r.submission.id for r in sort_scored_runs(runs, "amF1", ["x", "y"])] == [1, 2, 3]


def test_method_table_differences_from_a_reference():
    a = make_method_scores("a", {"amF1": 0.6, "aNLL": 1.0, "amP": 0.5})
    b = make_method_scores("b", {"amF1": 0.5, "aNLL": 0.8, "amP": 0.5})
    c = make_method_scores("c", {"amF1": 0.4, "aNLL": 0.9, "amP": 0.4})

    def difference(value, low, high, num_undefined=0):
        return ie.MetricDifference(value, ie.BootstrapInterval(low, high, num_undefined))

    differences = ie.MetricValues(averages={
        "amF1": difference(0.1, 0.05, 0.15), "aNLL": difference(0.2, 0.1, 0.3),
        "amP": difference(0.0, -0.1, 0.1, num_undefined=3)}, per_attribute={})
    state = ResultState(cached=CachedResult(value=differences, message="", computed_at=_TIME))
    comparisons = ReferenceComparisons(b, {"a": state, "c": ResultState(is_pending=True)})
    table = make_method_table_html([a, b, c], {}, "amF1", [], is_model_compatible_view=False,
                                   comparisons=comparisons)
    row_a, row_b, row_c = table.split("<tr>")[2:]
    cells = row_a.split("<td")
    amf1, amp, anll = (next(c for c in cells if f">{v}" in c) for v in (".6000", ".5000",
                                                                         "1.0000"))
    assert "Δ +.1000 (+.050–+.150) <b>better</b>" in amf1  # Higher amF1 is better.
    assert "Δ .0000 (-.100–+.100*)</div>" in amp  # The interval includes 0.
    assert "Δ +.2000 (+.100–+.300) <b>worse</b>" in anll  # Higher aNLL is worse.
    assert "b (reference)" in row_b and "Δ" not in row_b
    assert "Δ -.1000 (computing…)</div>" in row_c


def test_list_comparable_methods():
    rows = [make_method_scores("a", {"amF1": 0.6}), make_method_scores("x", {}, error="differ"),
            make_method_scores("b", {"amF1": 0.5})]
    assert [r.method_name for r in list_comparable_methods(rows)] == ["a", "b"]


def test_select_shown_methods():
    rows = [make_method_scores("a-1", {"amF1": 0.6}),
            make_method_scores("x", {}, error="differ"), make_method_scores("B-2", {"amF1": 0.5})]

    def select(text, reference_method=""):
        shown = select_shown_methods(rows, compile_name_pattern(text), reference_method)
        return [r.method_name for r in shown]

    assert select("") == ["a-1", "x", "B-2"]
    assert select("b") == ["B-2"]  # A part of the name, case-insensitive.
    assert select(r"^(a|b)-\d$") == ["a-1", "B-2"]
    assert select("^x$", reference_method="B-2") == ["x", "B-2"]  # The reference, unmatched.
    assert select("nothing") == []
    with pytest.raises(ValueError, match="Invalid regular expression"):
        compile_name_pattern("a(")


def test_get_unscored_reason():
    submission = make_submission(1)
    scoring = RunScoring(submission_id=1, status="scored", message="", notes=(),
                         settings_fingerprint="s", evaluation_sets_fingerprint="e",
                         scored_at=_TIME)
    assert get_unscored_reason(submission, scoring, is_current=True, is_pending=False) is None
    assert get_unscored_reason(submission, scoring, True, is_pending=True) == "Scoring…"
    assert "current settings" in get_unscored_reason(submission, None, False, False)
    deleted = make_submission(1, deleted_at=_TIME)
    assert get_unscored_reason(deleted, None, False, False).startswith("Deleted")
    failed = dc.replace(scoring, status="failed", message="No predictions for 3 segments.")
    assert get_unscored_reason(submission, failed, True, False) == failed.message
    error = dc.replace(scoring, status="error", message="Unexpected error.")
    assert "scored again" in get_unscored_reason(submission, error, False, False)


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


def test_method_table_html():
    runs = [make_run(1, {"x": 0.2, "y": 0.4}, method_name="<m>", missing_attributes=("y",))]
    rows = [*rank_methods(runs, ["x", "y"], "amF1"),
            make_method_scores("<n>", {}, error="Runs <differ>.")]
    interval = ie.BootstrapInterval(low=0.1, high=0.3, num_undefined=2)
    cached = CachedResult(
        value=ie.MetricValues(averages={"amF1": interval}, per_attribute={}),
        message="", computed_at=_TIME)
    table = make_method_table_html(rows, {"<m>": ResultState(cached=cached)}, "amF1",
                                   ["x"], is_model_compatible_view=False)
    assert "&lt;m&gt;" in table and "<m>" not in table
    assert '<td colspan="12" class="error">Runs &lt;differ&gt;.</td>' in table
    assert '<th class="sortable sorted" data-sort="amF1"' in table
    assert '<a href="/submissions/1?attribute=x">#1</a>' in table
    assert "1 not predicted" in table and "single run*" in table

    per_attribute = make_per_attribute_table_html(rows, {}, "mF1", ["x", "y"])
    assert "not predicted" in per_attribute and "&lt;n&gt;" not in per_attribute


def test_run_scores_html():
    run = make_run(1, {"x": 0.2, "y": math.nan}, missing_attributes=("y",))
    [scores] = rank_methods([run], ["x", "y"], "amF1")
    html = make_run_scores_html(scores, ResultState(is_pending=True))
    assert "computing…" in html and "NaN" in html and "3 (not predicted)" in html
