"""HTML of the Analysis page: the confusion matrix, the class table, the segments of a matrix
cell, and the details of a segment (see `prediction_analysis`).

The matrix and the segment list are single HTML elements with `data-cell` and `data-segment`
attributes for one delegated click handler each, since hundreds of NiceGUI elements would be slow.
"""

import html
import typing as T

import irap_evaluation as ie
import numpy as np

from ..prediction_analysis import SegmentDetail
from .deck_map import OUTCOME_COLORS
from .formatting import make_table_html, make_truncated_name_html
from .score_tables import format_optional_metric_value

#: The most classes whose matrix cells show their counts. Larger matrices show them only as
#: tooltips.
_MAX_CLASSES_WITH_COUNTS = 8


def _format_class(irap_code: int, class_name: str) -> str:
    return f"{irap_code} · {class_name}"


def _get_cell_color(label_index: int, predicted_index: int, num_classes: int) -> tuple[int, ...]:
    if predicted_index == num_classes:
        return OUTCOME_COLORS["invalid"]
    return OUTCOME_COLORS["correct" if label_index == predicted_index else "wrong"]


def make_confusion_matrix_html(confusion_matrix: np.ndarray, class_names: T.Sequence[str],
                               irap_codes: T.Sequence[int],
                               selected_cell: tuple[int, int] | None) -> str:
    """Makes the matrix as a grid of cells, shaded by their share of the row.

    Args:
        confusion_matrix: (K, K + 1) counts with the invalid predictions in the last column (see
            `prediction_analysis.AttributeOutcomes.confusion_matrix`).
    """
    num_classes = len(class_names)
    is_count_shown = num_classes <= _MAX_CLASSES_WITH_COUNTS
    parts = ['<div class="header">label ↓ predicted →</div>']
    parts += [f'<div class="header center" title="{html.escape(n)}">{c}</div>'
              for c, n in zip(irap_codes, class_names, strict=True)]
    parts.append('<div class="header center" title="Invalid predictions">inv.</div>')
    for i, (code, name) in enumerate(zip(irap_codes, class_names, strict=True)):
        label = html.escape(_format_class(code, name))
        parts.append(f'<div class="header" title="{label}">{label}</div>')
        row_sum = int(confusion_matrix[i].sum())
        for j in range(num_classes + 1):
            count = int(confusion_matrix[i, j])
            predicted = "invalid" if j == num_classes else _format_class(irap_codes[j],
                                                                          class_names[j])
            share = count / row_sum if row_sum else 0
            classes = "cell" + (" selected" if selected_cell == (i, j) else "")
            attributes = (f' data-cell="{i},{j}"' if count else " disabled")
            color = ", ".join(map(str, _get_cell_color(i, j, num_classes)))
            parts.append(
                f'<div class="{classes}"{attributes}'
                f' style="background: color-mix(in srgb, rgb({color}) {round(share * 100)}%,'
                f' white)" title="{label} → {html.escape(predicted)}: {count}">'
                f'{count if is_count_shown else ""}</div>')
    columns = f"minmax(70px, 9em) repeat({num_classes + 1}, minmax(14px, 1fr))"
    return (f'<div class="confusion-matrix" style="grid-template-columns: {columns}">'
            f'{"".join(parts)}</div>')


def make_class_table_html(class_report: T.Mapping[str, T.Any],
                          confusion_matrix: np.ndarray) -> str:
    """Makes a table of the support, precision, recall and F1 of each class, and its invalid
    predictions.

    Args:
        class_report: The attribute's entry of `classes` in an
            `evaluation_report.to_json_dict` document, whose undefined values are None.
        confusion_matrix: See `make_confusion_matrix_html`.
    """
    rows = []
    for k, (code, name) in enumerate(zip(class_report["irap_codes"], class_report["values"],
                                         strict=True)):
        cells = "".join(f'<td class="number">'
                        f"{format_optional_metric_value(class_report[key][k], 3)}</td>"
                        for key in ("precision", "recall", "f1"))
        rows.append(f"<tr><td>{html.escape(_format_class(code, name))}</td>"
                    f'<td class="number">{class_report["support"][k]}</td>{cells}'
                    f'<td class="number">{int(confusion_matrix[k, -1])}</td></tr>')
    metric_names = ("P", "R", "F1")
    number_headers = ["Support", *metric_names, "Invalid"]
    header_titles = {"Support": "The number of segments labeled with the class.",
                     **{n: ie.describe_metric(n) for n in metric_names}}
    return make_table_html(["Class", *number_headers], rows, number_headers, header_titles)


def make_segment_list_html(segment_indices: T.Sequence[int], segment_ids: T.Sequence[str],
                           selected_index: int | None) -> str:
    """Makes a button with the id of each of the given segments of the set.

    Args:
        segment_indices: Into the segments of the set, in the `data-segment` attribute.
        segment_ids: The ids of all segments of the set.
    """
    return "".join(
        f'<button type="button" data-segment="{i}"'
        f'{" class=selected" if i == selected_index else ""}>'
        f"{html.escape(segment_ids[i])}</button>"
        for i in segment_indices)


def _make_probability_table_html(detail: SegmentDetail) -> str:
    header = "".join(f"<th>{make_truncated_name_html(p.label)}</th>"
                     for p in detail.model_predictions)
    rows = []
    for k, (code, name) in enumerate(zip(detail.irap_codes, detail.class_names, strict=True)):
        cells = []
        for prediction in detail.model_predictions:
            if prediction.probs is None:
                cells.append(f'<td class="muted">{"invalid" if k == 0 else ""}</td>')
                continue
            probability = float(prediction.probs[k])
            bar_class = "bar predicted" if prediction.predicted_index == k else "bar"
            cells.append(f'<td class="bar-cell"><div class="{bar_class}"'
                         f' style="width: {probability * 100:.1f}%"></div>'
                         f"<span>{probability:.2f}</span></td>")
        label_class = ' class="label"' if k == detail.label_index else ""
        rows.append(f"<tr><td{label_class}>{html.escape(_format_class(code, name))}</td>"
                    f'{"".join(cells)}</tr>')
    return (f'<table class="probabilities"><thead><tr><th>Class</th>{header}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def make_segment_detail_html(detail: SegmentDetail,
                             get_image_url: T.Callable[[str], str]) -> str:
    """Makes the details of a segment: its context images, road position, location, label and
    predicted distributions.

    Args:
        get_image_url: Segment id -> the URL of its image.
    """
    figures = []
    for offset, segment_id in detail.context_segment_ids:
        image = (f'<img src="{html.escape(get_image_url(segment_id))}" alt="No image">'
                 if segment_id is not None
                 else '<div class="muted">Outside the road sequence</div>')
        figures.append(f"<figure>{image}<figcaption>offset {offset}</figcaption></figure>")
    images = f'<div class="context-images">{"".join(figures)}</div>'
    road = ("no road sequence" if detail.road_id is None
            else f"road {detail.road_id}, position {detail.position}")
    location = ("no location" if detail.coordinates is None
                else f"{detail.coordinates[0]:.5f}, {detail.coordinates[1]:.5f}")
    label = ("no label" if detail.label_index is None
             else "label " + _format_class(detail.irap_codes[detail.label_index],
                                           detail.class_names[detail.label_index]))
    title = html.escape(f"{detail.segment_id} · {road} · {location} · {label}")
    return (f'<div class="segment-detail">{images}<div class="segment-facts">'
            f'<div class="subsection-title">{title}</div>'
            f"{_make_probability_table_html(detail)}</div></div>")
