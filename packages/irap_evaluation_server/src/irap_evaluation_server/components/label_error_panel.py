"""The list of the candidate labeling errors of the Analysis page (`label_errors`): the choices
of the listed candidates, and the candidates in pages."""

import typing as T

import numpy as np
from nicegui import ui

from ..label_errors import LabelErrorRanking, find_listed_position
from ..label_reviews import LabelKey, LabelReview
from .analysis_html import make_label_error_table_html
from .formatting import make_data_click_js_handler
from .native_controls import (
    create_native_checkbox,
    create_native_step_buttons,
    set_native_button_enabled,
)

#: The number of candidates of a page of the list.
LABEL_ERROR_PAGE_SIZE = 100


class LabelErrorPanel:
    """The choices of the listed candidate labeling errors, and a table of a page of them
    (`analysis_html.make_label_error_table_html`) with buttons that change the page.

    The table has one page of the candidates, of which there can be tens of thousands, since it
    is sent again whenever another row is marked. The page is that of the selected candidate if
    it is listed.
    """

    def __init__(self, *, is_all_models_disagreement_required: bool,
                 is_reviewed_label_hidden: bool,
                 on_all_models_disagreement_changed: T.Callable[[bool], T.Any],
                 on_reviewed_label_hiding_changed: T.Callable[[bool], T.Any],
                 on_candidate_clicked: T.Callable[[int], T.Any]):
        """
        Args:
            is_all_models_disagreement_required, is_reviewed_label_hidden: The choices at first
                (see `analysis_view.AnalysisView`).
            on_all_models_disagreement_changed, on_reviewed_label_hiding_changed: Called with a
                choice that the user has changed, to show the candidates of it (`show`).
            on_candidate_clicked: Called with the rank of a clicked candidate.
        """
        self._page = 0
        #: The ranking, the listed ranks, the selected rank and the reviews of the last `show`.
        self._shown: tuple[LabelErrorRanking, np.ndarray, int | None,
                           T.Mapping[LabelKey, LabelReview]] | None = None
        ui.label("Labels that a model predicts as another class, the least probable first, by"
                 " the mean probability of the label over the models.").classes("muted")
        with ui.element("div").classes("form-row"):
            with ui.element("div") as self._all_models_choice:
                create_native_checkbox("Only where both models disagree with the label",
                                       is_all_models_disagreement_required,
                                       on_all_models_disagreement_changed)
            create_native_checkbox("Hide reviewed labels", is_reviewed_label_hidden,
                                   on_reviewed_label_hiding_changed)
        with ui.element("div").classes("form-row items-center"):
            self._page_buttons = create_native_step_buttons(
                self._change_page, ("Previous page", "Next page"))
            self._status = ui.label().classes("muted")
        self._table_html = ui.html("", sanitize=False).classes("selectable-rows").on(
            "click", lambda e: on_candidate_clicked(int(e.args)),
            js_handler=make_data_click_js_handler("data-label-error-row"))

    def show(self, ranking: LabelErrorRanking, listed_ranks: np.ndarray,
             selected_rank: int | None, reviews: T.Mapping[LabelKey, LabelReview]) -> None:
        """Shows the page of the selected candidate, or the page that was shown if it is not
        listed. The choice between the disagreement of all models and of any is shown if the
        ranking is of several models.

        Args:
            listed_ranks: (M,) ascending ranks of the listed candidates of the ranking, e.g. of
                those whose labels are not reviewed.
            selected_rank: The rank of the candidate whose row is marked, or None.
            reviews: The corrections of the dataset
                (`label_reviews.LabelReviewStore.get_reviews`).
        """
        self._all_models_choice.set_visibility(len(ranking.labeled_aligned_seq) > 1)
        self._shown = (ranking, listed_ranks, selected_rank, reviews)
        if (position := find_listed_position(listed_ranks, selected_rank)) is not None:
            self._page = position // LABEL_ERROR_PAGE_SIZE
        self._page = min(self._page, max(self._get_num_pages() - 1, 0))
        self._show_page()

    def _get_num_pages(self) -> int:
        _, listed_ranks, _, _ = self._shown
        return -(-len(listed_ranks) // LABEL_ERROR_PAGE_SIZE)

    def _show_page(self) -> None:
        ranking, listed_ranks, selected_rank, reviews = self._shown
        start = self._page * LABEL_ERROR_PAGE_SIZE
        page_ranks = listed_ranks[start:start + LABEL_ERROR_PAGE_SIZE]
        text = (f"Labels {start + 1}–{start + len(page_ranks)} of {len(listed_ranks)}."
                if len(page_ranks) else "Number of labels: 0.")
        if num_hidden := len(ranking) - len(listed_ranks):
            text += f" Number of reviewed labels (hidden): {num_hidden}."
        self._status.set_text(text)
        for is_forward, button in self._page_buttons.items():
            set_native_button_enabled(button, self._find_adjacent_page(is_forward) is not None)
        evaluation_set = ranking.evaluation_set
        self._table_html.set_content(make_label_error_table_html(
            ranking.get_candidates(page_ranks), evaluation_set.segment_ids,
            evaluation_set.vocabulary, reviews, selected_rank))

    def _find_adjacent_page(self, is_forward: bool) -> int | None:
        """Returns the next or the previous page, None if there is none."""
        page = self._page + (1 if is_forward else -1)
        return page if self._shown is not None and 0 <= page < self._get_num_pages() else None

    def _change_page(self, is_forward: bool) -> None:
        if (page := self._find_adjacent_page(is_forward)) is not None:
            self._page = page
            self._show_page()
