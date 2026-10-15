"""The Methodology page: how the server scores files and how to read the score tables
(`methodology`), as one document whose formulas MathJax typesets."""

import html
import typing as T

from nicegui import ui

from ..components.account_sessions import AccountSessions
from ..components.page_frame import create_page_frame
from ..components.routes import METHODOLOGY_PATH, get_library_path
from ..methodology import MARKDOWN_EXTRAS, MethodologySection, make_methodology_sections
from ..scoring import ScoringSettings
from ..vendor_libraries import MATHJAX_SCRIPT

#: Inline formulas between `$`, as in the text (`methodology`). MathJax typesets them only when
#: asked to (`methodology_page`), since NiceGUI creates the page's content after MathJax starts.
_MATHJAX_HEAD_HTML = f"""
<script>
window.MathJax = {{tex: {{inlineMath: [['$', '$']]}}, options: {{enableMenu: false}},
                  startup: {{typeset: false}}}};
</script>
<script defer src="{get_library_path(MATHJAX_SCRIPT)}"></script>
"""


def _make_document_markdown(sections: T.Sequence[MethodologySection]) -> str:
    """Makes the Markdown of the page: the title, the contents and the sections, whose headings
    have the ids of the sections."""
    contents = "\n".join(f"{i}. [{s.title}](#{s.id})" for i, s in enumerate(sections, start=1))
    body = "\n\n".join(f'<h2 id="{s.id}">{html.escape(s.title)}</h2>\n\n{s.markdown}'
                       for s in sections)
    return (f"# Methodology\n\nHow the server scores the prediction files, and how to read the"
            f" score tables.\n\n{contents}\n\n{body}")


def register_methodology_page(settings: ScoringSettings, dataset_names: T.Sequence[str],
                              math_error: str | None, sessions: AccountSessions) -> None:
    """
    Args:
        math_error: Why MathJax is missing (`vendor_libraries.ensure_library_files`), or None.
            The formulas are then shown as TeX.

    Raises:
        ValueError: See `make_methodology_sections`.
    """
    document_markdown = _make_document_markdown(make_methodology_sections(settings,
                                                                          dataset_names))

    @ui.page(METHODOLOGY_PATH, title="Methodology · iRAP evaluation")
    async def methodology_page() -> None:
        with create_page_frame(METHODOLOGY_PATH, sessions.get_account()):
            if math_error is None:
                ui.add_head_html(_MATHJAX_HEAD_HTML)
            with ui.element("article").classes("panel methodology"):
                if math_error is not None:
                    ui.label(f"{math_error} The formulas are shown as TeX.").classes("error")
                # Not sanitized, since the text is the server's own, and DOMPurify would remove
                # ids that are also DOM property names, e.g. 'attributes'.
                ui.markdown(document_markdown, extras=MARKDOWN_EXTRAS, sanitize=False)
        if math_error is None:
            await ui.context.client.connected()
            # Scrolls to the section of the link again, since the formulas change the heights of
            # the text above it.
            ui.run_javascript("MathJax.startup.promise.then(() => MathJax.typesetPromise())"
                              ".then(() => document.getElementById(location.hash.slice(1))"
                              "?.scrollIntoView())")
