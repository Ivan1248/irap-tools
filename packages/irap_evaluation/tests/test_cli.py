"""End-to-end runs of the `irap-eval` commands on the synthetic dataset."""

import dataclasses as dc
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from irap_data.metadata import MetaFiles, get_dataset_preset
from irap_evaluation.prediction_io import read_predictions, write_predictions
from irap_evaluation.tools.cli import main

from synthetic_vietnam import SPLIT, make_predictions


def _write_model_files(metadata, tmp_path):
    segments = list(metadata.splits[SPLIT])
    paths = []
    for name, seed, offsets in [("a", 0, (0, -1)), ("b", 1, None)]:
        path = tmp_path / f"{name}.predictions.parquet"
        write_predictions(path, make_predictions(metadata.vocabulary, segments, name=name,
                                                 seed=seed, context_offsets=offsets))
        paths.append(str(path))
    return paths


def test_evaluate_refuses_runs_without_labels(metadata_dir, metadata, tmp_path, capsys):
    paths = _write_model_files(metadata, tmp_path)
    # Without road data, no segment has labels, as in an unlabeled split.
    (metadata_dir / MetaFiles.SEGMENT_ID_TO_ROAD_DATA).write_text("{}", encoding="utf-8")
    out = tmp_path / "results"
    assert main(["evaluate", str(metadata_dir), *paths, "--out", str(out)]) == 1
    assert "nothing is scored" in capsys.readouterr().err
    assert not out.exists()


def test_evaluate_ensemble_compare_export(metadata_dir, metadata, tmp_path, capsys):
    paths = _write_model_files(metadata, tmp_path)
    assert main(["validate", *paths]) == 0

    ensemble_path = str(tmp_path / "ens.predictions.parquet")
    assert main(["ensemble", *paths, "--name", "ens", "-o", ensemble_path]) == 0
    assert read_predictions(ensemble_path).header.context_offsets is None

    out = tmp_path / "results"
    assert main(["evaluate", str(metadata_dir), *paths, ensemble_path, "--out", str(out),
                 "--context-offsets", "b=0", "--bootstrap", "50"]) == 0
    long_table = pd.read_csv(out / "summary_long.csv")
    assert set(long_table["run"]) == {"a", "b", "ens"}
    assert (long_table["ci_num_undefined"] >= 0).all()
    # a and b are also scored on their own sets, ens has no offsets.
    assert set(long_table.groupby("run")["evaluation_set"].unique().map(tuple)) == \
           {("reference", "model"), ("reference",)}
    per_attribute = pd.read_csv(out / "per_attribute_mF1_vietnam_val_reference.csv",
                                index_col=0)
    assert list(per_attribute.columns) == ["a", "b", "ens"]
    assert per_attribute.index[-1] == "average"
    report = json.loads((out / "a" / "metrics_model.json").read_text(encoding="utf-8"))
    assert report["context_offsets"] == [0, -1]
    assert report["intervals"]["averages"]["amF1"]["low"] <= report["metrics"]["averages"]["amF1"]
    assert "Attribute averages per run, vietnam/val, reference set" in capsys.readouterr().out

    assert main(["compare", str(metadata_dir), *paths, "--a", "a", "--b", "b", "--bootstrap",
                 "50"]) == 0
    assert "a - b" in capsys.readouterr().out

    table_path = tmp_path / "coding.csv"
    assert main(["export-coding-table", str(metadata_dir), paths[0], "-o", str(table_path),
                 "--confidence-table"]) == 0
    assert (tmp_path / "coding.confidence.csv").exists()


def _write_run_files(metadata, tmp_path, method_to_num_runs):
    segments = list(metadata.splits[SPLIT])
    paths = []
    for name, num_runs in method_to_num_runs.items():
        for method_seed in range(num_runs):
            path = tmp_path / f"{name}_{method_seed}.predictions.parquet"
            write_predictions(path, make_predictions(
                metadata.vocabulary, segments, name=name, seed=len(paths),
                method_seed=method_seed, context_offsets=None))
            paths.append(str(path))
    return paths


def test_methods_with_several_runs(metadata_dir, metadata, tmp_path, capsys):
    paths = _write_run_files(metadata, tmp_path, {"a": 3, "b": 2, "c": 1})
    out = tmp_path / "results"
    assert main(["evaluate", str(metadata_dir), *paths, "--out", str(out),
                 "--bootstrap", "50"]) == 0
    assert "means over runs" in capsys.readouterr().out
    assert (out / "a_seed0" / "metrics_reference.json").exists()
    long_table = pd.read_csv(out / "summary_long.csv")
    assert set(long_table["method"]) == {"a", "b", "c"}
    methods = pd.read_csv(out / "method_averages_vietnam_val_reference.csv", index_col=0)
    assert methods["runs"].to_dict() == {"a": 3, "b": 2}  # c has a single run
    a_runs = long_table[(long_table["method"] == "a") & (long_table["attribute"] == "average")
                        & (long_table["metric"] == "mF1")]
    assert methods.loc["a", "mF1"] == pytest.approx(a_runs["value"].mean())
    assert methods.loc["a", "mF1 low"] <= methods.loc["a", "mF1"] <= methods.loc["a", "mF1 high"]

    assert main(["compare", str(metadata_dir), *paths[:5], "--a", "a", "--b", "b",
                 "--metrics", "amF1", "--bootstrap", "50"]) == 0
    assert "a = a (3 runs), b = b (2 runs)" in capsys.readouterr().out
    assert main(["compare", str(metadata_dir), *paths, "--a", "a", "--b", "b"]) == 1
    assert "other methods than a and b: ['c']" in capsys.readouterr().err
    assert main(["compare", str(metadata_dir), *paths[:3], "--a", "a", "--b", "b"]) == 1
    assert "No files of the methods ['b']" in capsys.readouterr().err


def test_means_over_runs_need_the_first_class_null_policy(metadata_dir, metadata, tmp_path,
                                                          capsys):
    segments = list(metadata.splits[SPLIT])
    paths = []
    for method_seed in range(2):
        predictions = make_predictions(metadata.vocabulary, segments, name="vlm", seed=method_seed,
                                       method_seed=method_seed, context_offsets=None)
        # The runs have invalid cells in different segments.
        is_valid = np.arange(len(segments)) % 2 == method_seed
        probs = np.where(is_valid[:, None], predictions.probs["Lane width"], np.nan)
        paths.append(str(tmp_path / f"vlm_{method_seed}.predictions.parquet"))
        write_predictions(paths[-1], dc.replace(predictions, probs={
            **predictions.probs, "Lane width": probs.astype(np.float32)}))

    def evaluate(null_policy):
        out = tmp_path / null_policy
        assert main(["evaluate", str(metadata_dir), *paths, "--out", str(out),
                     "--null-policy", null_policy]) == 0
        return out, pd.read_csv(out / "summary_long.csv")

    excluded_out, excluded_runs = evaluate("exclude")
    assert "No means over the runs of a method" in capsys.readouterr().out
    assert not list(excluded_out.glob("method_averages_*.csv"))
    first_class_out, first_class_runs = evaluate("first_class")
    assert "means over runs" in capsys.readouterr().out
    methods = pd.read_csv(first_class_out / "method_averages_vietnam_val_reference.csv",
                          index_col=0)
    assert methods.loc["vlm", "null_policy"] == "first_class"
    assert set(excluded_runs["null_policy"]) == {"exclude"}

    def get_lane_width_n(runs):
        return runs[(runs["attribute"] == "Lane width") & (runs["metric"] == "n")]["value"]

    assert (get_lane_width_n(excluded_runs).to_numpy()
            < get_lane_width_n(first_class_runs).to_numpy()).all()
    assert main(["compare", str(metadata_dir), *paths[:1], *_write_run_files(
        metadata, tmp_path, {"other": 1}), "--a", "vlm", "--b", "other",
        "--bootstrap", "20"]) == 0


def test_runs_of_a_method_need_distinct_seeds(metadata_dir, metadata, tmp_path, capsys):
    segments = list(metadata.splits[SPLIT])
    paths = [str(tmp_path / f"{i}.predictions.parquet") for i in range(2)]
    for i, path in enumerate(paths):
        write_predictions(path, make_predictions(metadata.vocabulary, segments, name="m", seed=i,
                                                 method_seed=7))
    assert main(["evaluate", str(metadata_dir), *paths, "--out", str(tmp_path / "r")]) == 1
    assert "same seed" in capsys.readouterr().err


@pytest.mark.parametrize("names, message", [
    (["m", "m"], "each needs a seed"),
    (["conv seq", "conv_seq"], "output directories"),
])
def test_clashing_method_names_are_an_error(metadata_dir, metadata, tmp_path, capsys, names,
                                            message):
    segments = list(metadata.splits[SPLIT])
    paths = []
    for name in names:
        paths.append(str(tmp_path / f"{len(paths)}.predictions.parquet"))
        write_predictions(paths[-1], make_predictions(metadata.vocabulary, segments, name=name,
                                                      seed=0))
    assert main(["evaluate", str(metadata_dir), *paths, "--out", str(tmp_path / "r")]) == 1
    assert message in capsys.readouterr().err  # without a traceback


def test_models_that_miss_reference_segments_are_scored_on_their_own_set(
        metadata_dir, metadata, tmp_path, capsys):
    # Offset -6 reaches beyond the -4..0 reference window, so the first segments of each road that
    # are in the reference set have no prediction.
    segments = [sid for sid in metadata.splits[SPLIT] if int(sid[1:]) >= 6]
    path = tmp_path / "far.predictions.parquet"
    write_predictions(path, make_predictions(metadata.vocabulary, segments, name="far", seed=0,
                                             context_offsets=(0, -6)))
    out = tmp_path / "results"
    assert main(["evaluate", str(metadata_dir), str(path), "--out", str(out)]) == 0
    assert "not scored on the reference set" in capsys.readouterr().out
    long_table = pd.read_csv(out / "summary_long.csv")
    assert set(long_table["evaluation_set"]) == {"model"}


def test_a_model_that_cannot_be_scored_on_any_set_is_an_error(metadata_dir, metadata,
                                                              reference_set, tmp_path, capsys):
    complete, incomplete = tmp_path / "a.predictions.parquet", tmp_path / "b.predictions.parquet"
    segments = list(metadata.splits[SPLIT])
    reference_segment = reference_set.segment_ids[0]
    write_predictions(complete, make_predictions(metadata.vocabulary, segments, name="a", seed=0))
    write_predictions(incomplete, make_predictions(
        metadata.vocabulary, [sid for sid in segments if sid != reference_segment], name="b",
        seed=0, context_offsets=None))
    assert main(["evaluate", str(metadata_dir), str(complete), str(incomplete), "--out",
                 str(tmp_path / "r")]) == 1
    assert "Run 'b': No predictions for 1 of" in capsys.readouterr().err


def test_a_model_with_the_reference_offsets_is_scored_only_on_the_reference_set(
        metadata_dir, metadata, tmp_path, capsys):
    offsets = get_dataset_preset("vietnam").reference_context_offsets
    path = tmp_path / "ref.predictions.parquet"
    write_predictions(path, make_predictions(metadata.vocabulary, list(metadata.splits[SPLIT]),
                                             name="ref", seed=0, context_offsets=offsets))
    out = tmp_path / "results"
    assert main(["evaluate", str(metadata_dir), str(path), "--out", str(out)]) == 0
    assert "model set is left out" in capsys.readouterr().out
    assert [p.name for p in (out / "ref").iterdir()] == ["metrics_reference.json"]


def test_the_library_does_not_import_the_reports_or_the_tools():
    code = ("import sys, irap_evaluation;"
            " print(sorted(m for m in sys.modules if m.startswith('irap_evaluation.')))")
    modules = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             check=True).stdout
    assert "reports" not in modules and "tools" not in modules
