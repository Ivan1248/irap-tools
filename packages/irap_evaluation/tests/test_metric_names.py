import pytest

from irap_evaluation.metric_names import (
    ClassCondition,
    add_per_attribute_metric_names,
    describe_metric,
    get_attribute_average_name,
    get_irap_metric_names,
    is_lower_better,
    parse_metric_name,
)


def test_describe_metric_depends_on_ignore_missing_classes_only_where_it_applies():
    def is_affected(name: str) -> bool:
        return (describe_metric(name, ignore_missing_classes=True)
                != describe_metric(name, ignore_missing_classes=False))

    assert is_affected("mF1") and is_affected("amIoU")
    assert not any(map(is_affected, ["mF1_supp10", "nc_supp5", "mNLL", "mF1_{cNLL>1}", "F1",
                                     "MCC"]))


def test_add_per_attribute_metric_names():
    assert add_per_attribute_metric_names(["amF1", "aNLL", "mF1", "amF1_supp5"]) == (
        "amF1", "aNLL", "mF1", "amF1_supp5", "NLL", "mF1_supp5")


def test_parse_metric_name():
    assert parse_metric_name("amF1_supp10") == (True, "mF1_supp10", "mF1", 10, None)
    assert parse_metric_name("A") == (False, "A", "A", None, None)
    assert parse_metric_name("aA").is_averaged
    for name in ["A_supp5", "mF1_suppx", "mF1_supp0", "xyz"]:
        with pytest.raises(ValueError):
            parse_metric_name(name)


def test_parse_class_condition():
    assert parse_metric_name("amF1_supp10_{P<=0.5}") == (
        True, "mF1_supp10_{P<=0.5}", "mF1", 10, ClassCondition("P", "<=", 0.5))
    assert parse_metric_name("nc_{supp<10}").class_condition == ClassCondition("supp", "<", 10)
    assert parse_metric_name("mF1_{cNLL>1}").required_base_metrics == ("mF1", "cNLL")
    for name in ["nc_{R>0,P>0}", "A_{R>0}", "nc_{A>0}", "nc_{R>x}", "nc_R>0}", "mF1_{R>0"]:
        with pytest.raises(ValueError):
            parse_metric_name(name)


def test_describe_class_condition():
    assert describe_metric("nc_{R>0}") == (
        "nc_{R>0}: the number of classes that are labeled or predicted, where recall > 0.")
    assert describe_metric("mF1_{cNLL>=1.5}") == (
        "Macro F1 (mF1_{cNLL>=1.5}): the mean of F1 over labeled classes, where class NLL ≥ 1.5."
        " Higher is better.")


def test_class_extremes():
    assert parse_metric_name("aminR_supp1") == (True, "minR_supp1", "minR", 1, None)
    assert is_lower_better("maxNLL") and not is_lower_better("minF1")
    assert describe_metric("minR_supp5") == (
        "Minimum recall over classes (minR_supp5): the lowest recall among classes with at least 5"
        " labeled examples. Higher is better.")
    with pytest.raises(ValueError):
        parse_metric_name("minA")


def test_get_attribute_average_name():
    assert get_attribute_average_name("mF1_supp10") == "amF1_supp10"
    for name in ["amF1", "xyz"]:
        with pytest.raises(ValueError):
            get_attribute_average_name(name)


def test_is_lower_better():
    assert [n for n in get_irap_metric_names() if is_lower_better(n)] == [
        "aNLL", "aBrier", "amNLL", "amNLL_supp5", "amNLL_supp10", "NLL", "Brier", "mNLL"]
    assert is_lower_better("aMRP90") and is_lower_better("ECE")


def test_manual_review_proportion_metric_is_named_after_its_target():
    assert parse_metric_name("aMRP90") == (True, "MRP90", "MRP90", None, None)
    assert "mF1 reaches 0.9" in describe_metric("MRP90")


def test_get_irap_metric_names_puts_averages_first():
    names = get_irap_metric_names()
    assert names[:5] == ("amF1", "amP", "amR", "aMCC", "aA")
    assert "aNLL" not in get_irap_metric_names(output_kind="hard")
