"""Tests of the `irap-dataset-stats report` command and of `irap_data.report_document`."""

import csv
from pathlib import Path

import pytest

from irap_data import report_document as rd
from irap_data.dataset_report import main
from test_metadata import metadata_dir  # noqa: F401 – a fixture


def test_html_escapes_text_and_converts_inline_markup():
    blocks = [rd.Heading("Title of `x`", 1),
              rd.Paragraph("a <b> & *em* and **0**"),
              rd.make_table(["class", "n"], [["x|y", "**0**"]], "lr"),
              rd.Figure("plot.pdf", "Plot"), rd.Figure("plot.svg", "Plot")]
    page = rd.to_html(blocks)
    assert "<title>Title of x</title>" in page
    assert '<h1 id="title-of-x">Title of <code>x</code></h1>' in page
    assert "<p>a &lt;b&gt; &amp; <em>em</em> and <strong>0</strong></p>" in page
    assert '<td class="l">x|y</td><td class="r"><strong>0</strong></td>' in page
    assert '<a href="plot.pdf">Plot (PDF)</a>' in page and '<img src="plot.svg"' in page
    assert "aria-sort" in page  # The sorting script
    markdown = rd.to_markdown(blocks)
    assert "| x\\|y | **0** |" in markdown
    assert "[Plot (PDF)](plot.pdf)" in markdown and "![Plot](plot.svg)" in markdown


@pytest.mark.parametrize("report_format", ["html", "md"])
def test_report_writes_the_format(metadata_dir, tmp_path: Path, report_format):  # noqa: F811
    out_dir = tmp_path / "report"
    assert main(["report", "vietnam", "--metadata-dir", str(metadata_dir), "-o", str(out_dir),
                 "--format", report_format]) == 0
    report = (out_dir / f"vietnam_labeled_segments.{report_format}").read_text(encoding="utf-8")
    assert ("<h2 id=\"class-frequencies\">" if report_format == "html"
            else "## Class frequencies") in report
    plot_format = "svg" if report_format == "html" else "pdf"
    assert (out_dir / f"vietnam_labeled_segments_val_class_frequencies.{plot_format}").is_file()
    assert (out_dir / "vietnam_statistics.json").is_file()
    # The fixture has no segment with the complete reference context.
    for stem, selections in [("vietnam_labeled_segments", {"labeled"}),
                             ("vietnam_labeled_segments_with_context", set())]:
        with open(out_dir / f"{stem}_class_frequencies.csv", encoding="utf-8") as f:
            assert {row["selection"] for row in csv.DictReader(f)} == selections
