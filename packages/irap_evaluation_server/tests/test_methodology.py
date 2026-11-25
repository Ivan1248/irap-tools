import dataclasses as dc

import pytest

from irap_evaluation_server.methodology import (
    COMPARISONS_SECTION,
    EVALUATION_SETS_SECTION,
    INTERVALS_SECTION,
    METRICS_SECTION,
    check_described_settings,
    compute_t_quantile,
    make_methodology_sections,
)
from irap_evaluation_server.method_ranking import ShownMetrics
from irap_evaluation_server.scoring import SCORING_SETTINGS


def test_t_quantiles_of_one_and_two_degrees_of_freedom():
    # Tables of Student's t-distribution, e.g. scipy.stats.t.ppf(0.975, df).
    assert compute_t_quantile(0.975, 1) == pytest.approx(12.7062, abs=1e-4)
    assert compute_t_quantile(0.975, 2) == pytest.approx(4.3027, abs=1e-4)
    assert compute_t_quantile(0.05, 2) == pytest.approx(-2.9200, abs=1e-4)


def test_linked_sections_exist():
    ids = [s.id for s in make_methodology_sections(SCORING_SETTINGS, ["vietnam", "bh"],
                                                   ShownMetrics.from_choices({}))]
    assert len(set(ids)) == len(ids)
    assert {EVALUATION_SETS_SECTION, INTERVALS_SECTION, COMPARISONS_SECTION} <= set(ids)


def test_reference_windows_state_the_in_split_offsets():
    [section] = [s for s in make_methodology_sections(SCORING_SETTINGS, ["vietnam", "bh"],
                                                      ShownMetrics.from_choices({}))
                 if s.id == EVALUATION_SETS_SECTION]
    assert "(vietnam: −4 to 0; bh: −14 to 10, with −10 to 10 in the split)" in section.markdown


def _get_metrics_markdown(shown_metrics):
    [section] = [s for s in make_methodology_sections(SCORING_SETTINGS, ["vietnam"], shown_metrics)
                 if s.id == METRICS_SECTION]
    return section.markdown


def test_metrics_text_describes_the_shown_metrics():
    default = _get_metrics_markdown(ShownMetrics.from_choices({}))
    assert "attribute total" not in default and "`_{…}`" not in default
    assert ("NLL, Brier, mNLL, mNLL_supp5, mNLL_supp10 and MRP90 need probabilistic predictions."
            in default)
    shown = _get_metrics_markdown(ShownMetrics.from_choices({"totnc_{R>0}": True, "aECE": True,
                                                             "aMRP90": False}))
    assert "An **attribute total**, with the prefix `tot`" in shown and "`_{…}`" in shown
    assert "mNLL_supp10 and ECE need probabilistic predictions." in shown
    assert "nc_{R>0}: the number of classes" in shown


def test_settings_that_the_text_does_not_describe_are_refused():
    with pytest.raises(ValueError, match="null, missing attribute and missing segment policies"):
        check_described_settings(dc.replace(SCORING_SETTINGS, null_policy="exclude"))
