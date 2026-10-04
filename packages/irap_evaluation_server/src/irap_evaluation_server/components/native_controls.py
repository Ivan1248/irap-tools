"""The browser's own form controls instead of Quasar's, for a compact, desktop-like look.

Each `create_*` function creates the control in the current NiceGUI container and returns its
element. The class `native-control` (`styles.css`) restores the browser's style, which
Tailwind's reset removes.

NiceGUI renders native tags in the render function of the page's root, so any update of the
page, e.g. a refreshed table, renders them again. Vue then sets a DOM property only if its prop
has changed since the last render, except `value`, which it sets on each render. A change by the
user is therefore stored in the element's props: for a select, so that a render does not restore
the initial `value`, and for a checkbox, so that `checked` matches the DOM and
`set_native_checkbox_checked` can set it back to an earlier state. A text input has no `value`
prop (see `create_native_input`), so that text that is being typed is not reset either.
"""

import json
import typing as T

from nicegui import ui
from nicegui.elements.mixins.text_element import TextElement


def _on_user_change(element: ui.element, prop: str,
                    on_change: T.Callable[[T.Any], T.Any]) -> T.Callable[[T.Any], T.Any]:
    """Makes a handler of a change event that stores the new value in `element.props[prop]` (see
    the module docstring) before it calls `on_change`. The handler returns the result of
    `on_change`, so that NiceGUI awaits that of an async `on_change`."""
    def handle(event: T.Any) -> T.Any:
        element.props[prop] = event.args
        element.update()
        return on_change(event.args)

    return handle


def _create_field(label: str) -> ui.element:
    with ui.element("label").classes("field") as field:
        ui.label(label)
    return field


def _create_options(options: T.Mapping[str, str]) -> None:
    for option_value, text in options.items():
        TextElement(tag="option", text=text).props["value"] = option_value


def _create_select(label: str, value: str, on_change: T.Callable[[str], T.Any],
                   create_options: T.Callable[[], None]) -> ui.element:
    with _create_field(label):
        select = ui.element("select").classes("native-control")
        select.props["value"] = value
        with select:
            create_options()
    select.on("change", _on_user_change(select, "value", on_change),
              js_handler="(e) => emit(e.target.value)")
    return select


def create_native_select(label: str, options: T.Mapping[str, str], value: str,
                         on_change: T.Callable[[str], T.Any]) -> ui.element:
    """Creates a labeled `<select>`.

    Args:
        options: Value -> displayed text.
        on_change: Gets the new value. It can be async.
    """
    return _create_select(label, value, on_change, lambda: _create_options(options))


def create_native_grouped_select(label: str, groups: T.Mapping[str, T.Mapping[str, str]],
                                 value: str, on_change: T.Callable[[str], T.Any],
                                 ungrouped: T.Mapping[str, str] | None = None) -> ui.element:
    """Creates a labeled `<select>` with groups of options (`<optgroup>`).

    Args:
        groups: Group label -> option value -> displayed text.
        on_change: Gets the new value. It can be async.
        ungrouped: Options before the groups, e.g. one for none.
    """
    def create_options() -> None:
        _create_options(ungrouped or {})
        for group_label, options in groups.items():
            with ui.element("optgroup") as group:
                _create_options(options)
            group.props["label"] = group_label

    return _create_select(label, value, on_change, create_options)


def create_native_input(label: str, value: str, on_change: T.Callable[[str], None], *,
                        input_type: str = "text", placeholder: str = "",
                        size: int | None = None) -> ui.element:
    """Creates a labeled `<input>`. `on_change` gets the value when the input loses focus or
    Enter is pressed, which is before a click elsewhere is handled.

    The initial value is the input's `defaultValue`, not its `value`, so that a render of the page
    does not reset text that the user is typing (see the module docstring). The input cannot be set
    from code.

    Args:
        value: The initial value.
        on_change: Gets the new value. It can be async.
        input_type: The `type` attribute, e.g. 'number'.
        size: The width in characters, or None for the browser default.
    """
    with _create_field(label):
        element = ui.element("input").classes("native-control")
    element.props.update(type=input_type, defaultValue=value, placeholder=placeholder)
    if size is not None:
        element.props["size"] = size
    # The result is returned, so that NiceGUI awaits that of an async `on_change`.
    element.on("change", lambda event: on_change(event.args),
               js_handler="(e) => emit(e.target.value)")
    return element


def create_native_checkbox(label: str, value: bool,
                           on_change: T.Callable[[bool], None]) -> ui.element:
    with ui.element("label").classes("flex items-center gap-1"):
        element = ui.element("input").classes("native-control")
        element.props.update(type="checkbox", checked=value)
        ui.label(label)
    element.on("change", _on_user_change(element, "checked", lambda v: on_change(bool(v))),
               js_handler="(e) => emit(e.target.checked)")
    return element


def set_native_checkbox_checked(checkbox: ui.element, is_checked: bool) -> None:
    """Checks or unchecks a checkbox of `create_native_checkbox` from code, without its
    `on_change`."""
    if checkbox.props["checked"] != is_checked:
        checkbox.props["checked"] = is_checked
        checkbox.update()


def create_native_button(text: str, on_click: T.Callable[[], T.Any]) -> ui.element:
    button = TextElement(tag="button", text=text).classes("native-control")
    button.props["type"] = "button"
    button.on("click", lambda _: on_click())
    return button


def set_status(label: ui.label, text: str, is_error: bool = False) -> None:
    """Shows the status of a form in `label`, e.g. the result of a click."""
    label.set_text(text)
    label.classes(replace="error" if is_error else "muted")


def create_file_upload_input(label: str, upload_url: str, accept: str,
                             on_status: T.Callable[[dict], None]) -> ui.element:
    """Creates a labeled file `<input>` that posts the chosen file to `upload_url` as the form field
    'file'.

    Args:
        upload_url: An endpoint that stores the file and returns a JSON object.
        accept: The `accept` attribute, e.g. '.parquet'.
        on_status: Gets `{'status': 'uploading', 'file_name': ...}` when the upload starts,
            then the JSON object of the endpoint, or `{'status': 'error', 'message': ...}`.
            Each also has an 'upload_id', which increases with each upload, so that the reply
            to an earlier upload can be recognized.
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
