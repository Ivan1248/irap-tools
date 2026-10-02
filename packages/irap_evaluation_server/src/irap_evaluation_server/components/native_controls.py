"""The browser's own form controls instead of Quasar's, for a compact, desktop-like look.

Each `create_*` function creates the control in the current NiceGUI container and returns its
element. The class `native-control` (`styles.css`) restores the browser's style, which
Tailwind's reset removes.
"""

import json
import typing as T

from nicegui import ui
from nicegui.elements.mixins.text_element import TextElement


def _create_field(label: str) -> ui.element:
    with ui.element("label").classes("field") as field:
        ui.label(label)
    return field


def create_native_select(label: str, options: T.Mapping[str, str], value: str,
                         on_change: T.Callable[[str], None]) -> ui.element:
    """A labeled `<select>`.

    Args:
        options: Value -> displayed text.
    """
    with _create_field(label):
        select = ui.element("select").classes("native-control")
        select.props["value"] = value
        with select:
            for option_value, text in options.items():
                TextElement(tag="option", text=text).props["value"] = option_value
    select.on("change", lambda e: on_change(e.args), js_handler="(e) => emit(e.target.value)")
    return select


def create_native_input(label: str, value: str, on_change: T.Callable[[str], None], *,
                        input_type: str = "text", placeholder: str = "",
                        size: int | None = None) -> ui.element:
    """A labeled `<input>`. `on_change` gets the value when the input loses focus or Enter is
    pressed, which is before a click elsewhere is handled.

    Args:
        input_type: The `type` attribute, e.g. 'number'.
        size: The width in characters, or None for the browser default.
    """
    with _create_field(label):
        element = ui.element("input").classes("native-control")
    element.props.update(type=input_type, value=value, placeholder=placeholder)
    if size is not None:
        element.props["size"] = size
    element.on("change", lambda e: on_change(e.args), js_handler="(e) => emit(e.target.value)")
    return element


def create_native_checkbox(label: str, value: bool,
                           on_change: T.Callable[[bool], None]) -> ui.element:
    with ui.element("label").classes("flex items-center gap-1"):
        element = ui.element("input").classes("native-control")
        element.props.update(type="checkbox", checked=value)
        ui.label(label)
    element.on("change", lambda e: on_change(bool(e.args)),
               js_handler="(e) => emit(e.target.checked)")
    return element


def create_native_button(text: str, on_click: T.Callable[[], T.Any]) -> ui.element:
    button = TextElement(tag="button", text=text).classes("native-control")
    button.props["type"] = "button"
    button.on("click", lambda _: on_click())
    return button


def create_file_upload_input(label: str, upload_url: str, accept: str,
                             on_status: T.Callable[[dict], None]) -> ui.element:
    """A labeled file `<input>` that posts the chosen file to `upload_url` as the form field
    'file'.

    Args:
        upload_url: An endpoint that stores the file and returns a JSON object.
        accept: The `accept` attribute, e.g. '.parquet'.
        on_status: Gets `{'status': 'uploading', 'file_name': ...}` when the upload starts,
            then the JSON object of the endpoint, or `{'status': 'error', 'message': ...}`.
            Each also has an 'upload_id', which increases with each upload, so that the reply
            to a replaced upload can be recognized.
    """
    with _create_field(label):
        element = ui.element("input").classes("native-control")
    element.props.update(type="file", accept=accept)
    url = json.dumps(upload_url)
    element.on("change", lambda e: on_status(e.args), js_handler=f"""(e) => {{
        const file = e.target.files[0];
        if (!file) return;
        const uploadId = e.target.uploadId = (e.target.uploadId || 0) + 1;
        const body = new FormData();
        body.append("file", file);
        emit({{status: "uploading", upload_id: uploadId, file_name: file.name}});
        fetch({url}, {{method: "POST", body}})
            .then(async (r) => r.ok ? await r.json()
                                    : {{status: "error", message: await r.text()}})
            .catch((error) => ({{status: "error", message: String(error)}}))
            .then((status) => emit({{...status, upload_id: uploadId}}));
    }}""")
    return element
