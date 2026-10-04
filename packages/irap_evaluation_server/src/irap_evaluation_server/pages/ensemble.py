"""The Ensemble page: a form that creates an ensemble of stored models (see `ensembles`)."""

import typing as T

from fastapi import Request
from nicegui import run, ui

from ..accounts import can_write
from ..archive import ModelArchive, Submission, discard_model_update
from ..components.account_sessions import AccountSessions
from ..components.confirmation_dialog import confirm_changes
from ..components.formatting import format_splits, make_model_link_html
from ..components.native_controls import (
    create_native_button,
    create_native_checkbox,
    create_native_input,
    create_native_select,
    set_native_checkbox_checked,
    set_status,
)
from ..components.page_frame import create_page_frame, create_write_permission_panel
from ..components.routes import ENSEMBLE_PATH, make_query_path
from ..components.view_queries import compile_name_pattern, parse_dataset
from ..datasets import DatasetContext
from ..ensembles import (
    EnsembleMember,
    EnsemblePlan,
    MemberModel,
    list_member_models,
    plan_ensemble,
    prepare_ensemble,
)
from ..scoring_worker import ScoringWorker
from ..uploads import parse_description, parse_seed

_FORM_TITLE = "Create an ensemble"


def register_ensemble_page(archive: ModelArchive,
                           dataset_contexts: T.Mapping[str, DatasetContext],
                           worker: ScoringWorker, sessions: AccountSessions) -> None:
    @ui.page(ENSEMBLE_PATH, title="Ensemble · iRAP evaluation")
    def ensemble_page(request: Request) -> None:
        account = sessions.get_account()
        with create_page_frame(ENSEMBLE_PATH, account):
            if not can_write(account):
                create_write_permission_panel(account, _FORM_TITLE, "create ensembles")
                return
            try:
                dataset = parse_dataset(request.query_params, dataset_contexts)
            except ValueError as e:
                ui.label(str(e)).classes("error")
                return
            _create_ensemble_form(archive, dataset_contexts, worker, sessions.require_writer_name,
                                  dataset)


def _parse_members(member_models: T.Sequence[MemberModel], is_selected: T.Mapping[int, bool],
                   weight_texts: T.Mapping[int, str]) -> list[EnsembleMember]:
    """Parses the selected models with their weights. A blank weight is 1.

    Args:
        is_selected, weight_texts: By model id.

    Raises:
        ValueError: If a weight is not a number.
    """
    members = []
    for member_model in member_models:
        model = member_model.model
        if not is_selected.get(model.id, False):
            continue
        text = weight_texts.get(model.id, "1").strip() or "1"
        try:
            weight = float(text)
        except ValueError:
            raise ValueError(f"The weight of {model.label} must be a number, got {text!r}."
                             ) from None
        members.append(EnsembleMember(model, weight))
    return members


def _describe_plan(plan: EnsemblePlan) -> str:
    text = f"Creates the ensemble on the splits {', '.join(plan.split_to_submissions)}."
    for split, labels in plan.split_to_missing.items():
        text += f" Skips {split}, which has no file of {', '.join(labels)}."
    model = plan.make_model_info("")
    if model.training_splits is None:
        text += " Its training splits are unknown, since a member's are."
    else:
        text += (f" It is trained on {format_splits(model.training_splits)}, the union of those"
                 f" of the members.")
    if model.early_stopping_splits:
        text += f" Its early stopping splits are {format_splits(model.early_stopping_splits)}."
    return text


def _create_ensemble_form(archive: ModelArchive,
                          dataset_contexts: T.Mapping[str, DatasetContext],
                          worker: ScoringWorker, get_actor: T.Callable[[], str],
                          dataset: str) -> None:
    """
    Args:
        get_actor: Returns the name of the signed-in account that can write, or raises
            `ValueError`.
    """
    context = dataset_contexts[dataset]
    # The plan shown while the user chooses members uses this list. The Create button plans with
    # a new list.
    submissions = archive.list_submissions()
    member_models = list_member_models(submissions, dataset, context.split_names)
    # By model id.
    is_selected: dict[int, bool] = {}
    weight_texts: dict[int, str] = {}
    id_to_checkbox: dict[int, ui.element] = {}
    id_to_container: dict[int, ui.element] = {}
    id_to_label = {m.model.id: m.model.label for m in member_models}
    # The member filter: by model label.
    member_pattern = compile_name_pattern("")
    form = {"name": "", "seed": "", "description": "", "intersect_segments": False}

    def get_plan(submissions: T.Sequence[Submission]) -> EnsemblePlan:
        """
        Raises:
            ValueError: See `_parse_members` and `ensembles.plan_ensemble`.
        """
        return plan_ensemble(submissions, dataset,
                             _parse_members(member_models, is_selected, weight_texts),
                             context.split_names)

    def update_plan() -> None:
        try:
            set_status(plan_label, _describe_plan(get_plan(submissions)))
        except ValueError as e:
            set_status(plan_label, str(e), is_error=True)

    def set_member(model_id: int, is_checked: bool) -> None:
        is_selected[model_id] = is_checked
        update_plan()

    def show_members() -> None:
        """Shows the members that match the filter, and the selected ones, so that every member
        of the plan is visible."""
        for model_id, container in id_to_container.items():
            container.set_visibility(is_selected.get(model_id, False)
                                     or bool(member_pattern.search(id_to_label[model_id])))

    def on_filter_changed(text: str) -> None:
        nonlocal member_pattern
        try:
            member_pattern = compile_name_pattern(text)
        except ValueError as e:
            set_status(filter_status, str(e), is_error=True)
            return
        filter_status.set_text("")
        show_members()

    def select_shown_members(is_checked: bool) -> None:
        for model_id, container in id_to_container.items():
            if container.visible:
                is_selected[model_id] = is_checked
                set_native_checkbox_checked(id_to_checkbox[model_id], is_checked)
        show_members()
        update_plan()

    def set_weight(model_id: int, text: str) -> None:
        weight_texts[model_id] = text
        update_plan()

    def set_creating(is_creating: bool) -> None:
        create_button.props["disabled"] = is_creating
        create_button.update()

    async def on_create_clicked() -> None:
        try:
            plan = get_plan(archive.list_submissions())
            name = form["name"].strip()
            if not name:
                raise ValueError("Enter the method name of the ensemble.")
            actor = get_actor()
            seed = parse_seed(form["seed"])
        except ValueError as e:
            set_status(status_label, str(e), is_error=True)
            return
        client = ui.context.client
        set_status(status_label, "Creating the ensemble…")
        set_creating(True)
        try:
            created = await _create(archive, context, plan, method_name=name, seed=seed,
                                    intersect_segments=form["intersect_segments"],
                                    description=form["description"], actor=actor)
        except ValueError as e:  # Also irap_evaluation.PredictionFormatError.
            if not client.is_deleted:
                set_status(status_label, str(e), is_error=True)
            return
        except Exception as e:
            if not client.is_deleted:
                set_status(status_label, f"Internal error: {e!r}. See the server log.",
                           is_error=True)
            raise
        finally:
            if not client.is_deleted:
                set_creating(False)
        if created:
            await run.io_bound(worker.update_model_submissions, created[0].model.id)
        if client.is_deleted:
            return
        if not created:  # Cancelled, or the server is stopping.
            set_status(status_label, "The ensemble is not stored.")
            return
        status_label.set_text("")
        model = created[0].model
        result_html.set_content(
            f"Created {make_model_link_html(model.id, model.label)} on the splits"
            f" {', '.join(s.split for s in created)}. It is scored in the background.")

    with ui.element("section").classes("panel"):
        ui.label(_FORM_TITLE).classes("section-title")
        ui.label("The ensemble predicts the weighted mean of the members' distributions. A hard"
                 " prediction counts as one-hot, so an ensemble of hard predictions is a weighted"
                 " vote. A cell is invalid only if it is invalid in every member. The ensemble"
                 " gets a file for each split that every member has. If its model exists, these"
                 " files replace all of the model's files. It counts as trained on the training"
                 " splits of all members.").classes("muted")
        with ui.element("div").classes("form-row"):
            create_native_select(
                "Dataset", {n: n for n in dataset_contexts}, dataset,
                lambda v: ui.navigate.to(make_query_path(ENSEMBLE_PATH, {"dataset": v})))
        if not member_models:
            ui.label("The dataset has no models.").classes("muted")
            return
        ui.label("Members and weights").classes("subsection-title")
        with ui.element("div").classes("form-row items-center"):
            create_native_input("Filter (regex)", "", on_filter_changed,
                                placeholder="all", size=24)
            create_native_button("Select shown", lambda: select_shown_members(True))
            create_native_button("Unselect shown", lambda: select_shown_members(False))
            filter_status = ui.label().classes("muted")
        with ui.element("div").classes("ensemble-members"):
            for member_model in sorted(member_models,
                                       key=lambda m: (m.model.method_name, m.model.label)):
                model_id = member_model.model.id
                with ui.element("div").classes("ensemble-member") as container:
                    id_to_checkbox[model_id] = create_native_checkbox(
                        member_model.model.label, False,
                        lambda is_checked, i=model_id: set_member(i, is_checked))
                    create_native_input("", "1", lambda text, i=model_id: set_weight(i, text),
                                        input_type="number", size=6)
                    ui.label(", ".join(member_model.splits)).classes("muted")
                id_to_container[model_id] = container
        plan_label = ui.label("Select at least 2 members.").classes("muted")
        with ui.element("div").classes("form-row"):
            create_native_input("Method name", "", lambda v: form.update(name=v), size=24)
            create_native_input("Seed (optional)", "", lambda v: form.update(seed=v),
                                input_type="number", size=10)
            create_native_input("Description", "", lambda v: form.update(description=v),
                                placeholder="blank: keep the model's", size=40)
        create_native_checkbox(
            "Only the segments that every member predicts (needed for members with different"
            " context offsets)", False,
            lambda v: form.update(intersect_segments=v))
        with ui.element("div").classes("form-row items-center"):
            create_button = create_native_button("Create", on_create_clicked)
            status_label = ui.label().classes("muted")
        result_html = ui.html("", sanitize=False)


async def _create(archive: ModelArchive, context: DatasetContext, plan: EnsemblePlan, *,
                  method_name: str, seed: int | None, intersect_segments: bool,
                  description: str, actor: str) -> list[Submission]:
    """Makes the ensemble files (`ensembles.prepare_ensemble`), asks the user to confirm the
    changes, if any, and stores them.

    Args:
        actor: The name of the signed-in account.

    Returns:
        The new submissions, or [] if the user cancelled, closed the page, or the server is
        stopping.

    Raises:
        ValueError: See `ensembles.prepare_ensemble` and
            `archive.ModelArchive.apply_model_update`.
    """
    client = ui.context.client
    update = await run.io_bound(prepare_ensemble, archive, context, plan,
                                method_name=method_name, seed=seed,
                                intersect_segments=intersect_segments,
                                description=parse_description(description))
    if update is None:  # The server is stopping.
        return []
    try:
        if client.is_deleted:
            return []
        if update.confirmations and not await confirm_changes(
                f"Ensemble {update.label} on {update.dataset}", update.confirmations, "Store"):
            return []
        return await run.io_bound(archive.apply_model_update, update, submitter=actor) or []
    finally:
        discard_model_update(update)  # The files that are not moved into the archive.
