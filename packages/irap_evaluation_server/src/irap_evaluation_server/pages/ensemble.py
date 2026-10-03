"""The Ensemble page: a form that creates the ensemble of stored runs (see `ensembles`)."""

import html
import typing as T

import irap_evaluation as ie
from fastapi import Request
from nicegui import run, ui

from ..archive import Submission, SubmissionArchive, parse_seed
from ..components.formatting import make_submission_link_html
from ..components.native_controls import (
    create_native_button,
    create_native_checkbox,
    create_native_input,
    create_native_select,
    set_status,
)
from ..components.page_frame import create_page_frame, get_submitter_name
from ..components.routes import ENSEMBLE_PATH, make_query_path
from ..components.view_queries import parse_dataset
from ..datasets import DatasetContext
from ..ensembles import (
    EnsembleMember,
    EnsemblePlan,
    MemberRun,
    create_ensemble,
    list_member_runs,
    plan_ensemble,
)
from ..scoring_worker import ScoringWorker


def register_ensemble_page(archive: SubmissionArchive,
                           dataset_contexts: T.Mapping[str, DatasetContext],
                           worker: ScoringWorker) -> None:
    @ui.page(ENSEMBLE_PATH, title="Ensemble · iRAP evaluation")
    def ensemble_page(request: Request) -> None:
        with create_page_frame(ENSEMBLE_PATH):
            try:
                dataset = parse_dataset(request.query_params, dataset_contexts)
            except ValueError as e:
                ui.label(str(e)).classes("error")
                return
            _create_ensemble_form(archive, dataset_contexts, worker, dataset)


def _parse_members(runs: T.Sequence[MemberRun], is_selected: T.Mapping[str, bool],
                   weight_texts: T.Mapping[str, str]) -> list[EnsembleMember]:
    """The selected runs with their weights, by run label.

    Raises:
        ValueError: If a weight is not a number.
    """
    members = []
    for member_run in runs:
        label = member_run.method.run_label
        if not is_selected.get(label, False):
            continue
        text = weight_texts.get(label, "1").strip() or "1"
        try:
            weight = float(text)
        except ValueError:
            raise ValueError(f"The weight of {label} must be a number, got {text!r}.") from None
        members.append(EnsembleMember(member_run.method, weight))
    return members


def _describe_plan(plan: EnsemblePlan) -> str:
    text = f"Creates the ensemble of the splits {', '.join(plan.split_to_submissions)}."
    for split, labels in plan.split_to_missing.items():
        text += f" Skips {split}, which has no run of {', '.join(labels)}."
    return text


def _create_ensemble_form(archive: SubmissionArchive,
                          dataset_contexts: T.Mapping[str, DatasetContext],
                          worker: ScoringWorker, dataset: str) -> None:
    context = dataset_contexts[dataset]
    # The plan shown while the user chooses is of this list. Create plans with a new one.
    submissions = archive.list_submissions()
    runs = list_member_runs(submissions, dataset, context.split_names)
    is_selected: dict[str, bool] = {}
    weight_texts: dict[str, str] = {}
    form = {"name": "", "seed": "", "description": "", "intersect_segments": False}

    def get_plan(submissions: T.Sequence[Submission]) -> EnsemblePlan:
        """
        Raises:
            ValueError: See `_parse_members` and `ensembles.plan_ensemble`.
        """
        return plan_ensemble(submissions, dataset,
                             _parse_members(runs, is_selected, weight_texts),
                             context.split_names)

    def update_plan() -> None:
        try:
            set_status(plan_label, _describe_plan(get_plan(submissions)))
        except ValueError as e:
            set_status(plan_label, str(e), is_error=True)

    def set_member(label: str, is_checked: bool) -> None:
        is_selected[label] = is_checked
        update_plan()

    def set_weight(label: str, text: str) -> None:
        weight_texts[label] = text
        update_plan()

    async def on_create_clicked() -> None:
        try:
            plan = get_plan(archive.list_submissions())
            name = form["name"].strip()
            if not name:
                raise ValueError("Enter the method name of the ensemble.")
            seed = parse_seed(form["seed"])
        except ValueError as e:
            set_status(status_label, str(e), is_error=True)
            return
        set_status(status_label, "Creating the ensemble…")
        try:
            created = await run.io_bound(
                create_ensemble, archive, context, plan,
                method=ie.MethodInfo(name, seed),
                intersect_segments=form["intersect_segments"], submitter=get_submitter_name(),
                description=form["description"])
        except ValueError as e:  # Also irap_evaluation.PredictionFormatError.
            set_status(status_label, str(e), is_error=True)
            return
        except Exception as e:
            set_status(status_label, f"Internal error: {e!r}. See the server log.",
                       is_error=True)
            raise
        if created is None:  # The server is stopping.
            return
        await run.io_bound(worker.update_submissions, [s.id for s in created])
        links = ", ".join(f"{make_submission_link_html(s.id)} ({html.escape(s.split)})"
                          for s in created)
        status_label.set_text("")
        result_html.set_content(f"Created {html.escape(created[0].run_label)}: {links}."
                                f" It is scored in the background.")

    with ui.element("section").classes("panel"):
        ui.label("Create an ensemble").classes("section-title")
        ui.label("The ensemble of runs predicts the weighted mean of their distributions. A hard"
                 " prediction counts as a one-hot distribution, so the ensemble of hard"
                 " predictions is a weighted vote. A cell is invalid only if it is invalid in"
                 " every member. One ensemble is stored for each split where every member has a"
                 " run.").classes("muted")
        with ui.element("div").classes("form-row"):
            create_native_select(
                "Dataset", {n: n for n in dataset_contexts}, dataset,
                lambda v: ui.navigate.to(make_query_path(ENSEMBLE_PATH, {"dataset": v})))
        if not runs:
            ui.label("The dataset has no submissions.").classes("muted")
            return
        ui.label("Members").classes("subsection-title")
        with ui.element("div").classes("ensemble-members"):
            for member_run in runs:
                label = member_run.method.run_label
                create_native_checkbox(label, False,
                                       lambda is_checked, lb=label: set_member(lb, is_checked))
                create_native_input("", "1", lambda text, lb=label: set_weight(lb, text),
                                    input_type="number", size=6)
                ui.label(", ".join(member_run.splits)).classes("muted")
        plan_label = ui.label("Select at least 2 members.").classes("muted")
        with ui.element("div").classes("form-row"):
            create_native_input("Method name", "", lambda v: form.update(name=v), size=24)
            create_native_input("Seed (optional)", "", lambda v: form.update(seed=v),
                                input_type="number", size=10)
            create_native_input("Description", "", lambda v: form.update(description=v),
                                size=40)
        create_native_checkbox(
            "Only the segments that every member predicts (needed when the members have"
            " different context offsets)", False,
            lambda v: form.update(intersect_segments=v))
        with ui.element("div").classes("form-row items-center"):
            create_native_button("Create", on_create_clicked)
            status_label = ui.label().classes("muted")
        result_html = ui.html("", sanitize=False)
