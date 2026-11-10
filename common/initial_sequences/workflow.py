from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

from common.evaluation.feasibility import evaluate_generation
from common.execution.io import load_execution_snapshot, save_execution_snapshot
from common.execution.replay import PlanReplay
from common.execution.selection import SnapshotSelection, select_snapshot
from common.initial_sequences.generator import GenerationResult, GeneratorConfig, generate_initial_plan
from common.initial_sequences.inputs import _number, _plain
from common.initial_sequences.io import load_planning_input, save_planning_input, _invalid_constant, _unique_pairs
from common.initial_sequences.plan import InitialPlan


INITIAL_BACKGROUND_FILES = ("planning_input.json", "proposal.json", "snapshot.json", "summary.json")


def resolve_method_config(spec="greedydp_waiting", *, default_seed=0) -> GeneratorConfig:


    from common.initial_sequences.adapters import foodmatch, greedydp, ocd

    methods = {
        "greedydp_waiting": greedydp.default_config,
        "foodmatch_batch": foodmatch.default_config,
        "ocd_alns_tabu": ocd.default_config,
    }
    aliases = {"greedydp": "greedydp_waiting", "foodmatch": "foodmatch_batch", "ocd": "ocd_alns_tabu"}
    complete = isinstance(spec, GeneratorConfig)
    if isinstance(spec, str):
        values = {"name": spec}
    elif complete:
        values = _plain(spec)
    elif isinstance(spec, Mapping):
        values = dict(spec)
        complete = "objective" in values
        if complete:
            spec = GeneratorConfig(**values)
    else:
        raise ValueError("initial method must be a name, recipe or GeneratorConfig")
    if not complete and set(values) - {"name", "seed", "information_mode", "parameters"}:
        raise ValueError("initial method recipe has unknown fields")
    name = values.get("name")
    if not isinstance(name, str):
        raise ValueError("initial method requires a name")
    canonical = aliases.get(name, name)
    if canonical not in methods:
        raise ValueError(f"unknown initial method {name!r}")
    parameters = values.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ValueError("initial method parameters must be a mapping")
    config = methods[canonical](information_mode=values.get("information_mode", "batch"),
                                seed=values.get("seed", default_seed), **dict(parameters))
    if complete:
        if (spec.name, spec.objective) != (config.name, config.objective):
            raise ValueError("initial method name or objective does not match the supported adapter")


        if spec.parameters != config.parameters:
            raise ValueError("complete initial method config requires the complete normalized parameter set")
        return spec
    return config


def initial_generator(config: GeneratorConfig, *, progress=None):

    from common.initial_sequences.adapters.foodmatch import FoodMatchAdapter
    from common.initial_sequences.adapters.greedydp import GreedyDPAdapter
    from common.initial_sequences.adapters.ocd import OCDAdapter
    config = resolve_method_config(config)
    if config.name == "greedydp_waiting":
        return GreedyDPAdapter(progress=progress)
    return {"foodmatch_batch": FoodMatchAdapter, "ocd_alns_tabu": OCDAdapter}[config.name]()


@dataclass(frozen=True)
class InitialBackground:
    plan: InitialPlan
    replay: PlanReplay
    selection: SnapshotSelection

    @property
    def snapshot(self):
        return self.selection.snapshot

    def summary(self):
        return {
            "network_id": self.plan.network_id, "snapshot_id": self.plan.snapshot_id,
            "method_config": _plain(self.plan.generation_result.method_config),
            "selection": self.selection.to_dict(),
        }


def build_initial_background(planning_input, routing, method="greedydp_waiting", *,
                             default_seed=0, time_s=None, remaining_tasks=None,
                             progress=None) -> InitialBackground:


    if (time_s is None) == (remaining_tasks is None):
        raise ValueError("specify exactly one of time_s and remaining_tasks")
    if time_s is not None:
        time_s = _number(time_s, "snapshot.time_s")
        if time_s < planning_input.planning_time_s:
            raise ValueError("snapshot time must not precede planning time")
    elif type(remaining_tasks) is not int or not 0 <= remaining_tasks <= len(planning_input.tasks):
        raise ValueError("remaining_tasks must be an integer between zero and the input task count")
    config = resolve_method_config(method, default_seed=default_seed)
    adapter = initial_generator(config, progress=progress)
    plan = generate_initial_plan(adapter, planning_input, routing, config)
    replay = PlanReplay(planning_input, plan, routing)
    return InitialBackground(plan, replay, select_snapshot(replay, time_s=time_s, remaining_tasks=remaining_tasks))


def save_initial_background(background: InitialBackground, output_dir) -> None:


    directory = Path(output_dir)
    if directory.exists() and (not directory.is_dir() or any(
            (directory / name).exists() for name in INITIAL_BACKGROUND_FILES)):
        raise ValueError("initial background output must be empty of initial-plan files")
    directory.mkdir(parents=True, exist_ok=True)
    save_planning_input(background.replay.planning_input, directory / "planning_input.json")
    proposal = background.plan.generation_result.to_dict()
    (directory / "proposal.json").write_text(json.dumps(proposal, ensure_ascii=True, allow_nan=False,
        sort_keys=True, separators=(",", ":")) + "\n")
    save_execution_snapshot(background.snapshot, directory / "snapshot.json")
    (directory / "summary.json").write_text(json.dumps(background.summary(), ensure_ascii=True, allow_nan=False,
        sort_keys=True, separators=(",", ":")) + "\n")


def load_initial_background(output_dir, routing) -> InitialBackground:

    directory = Path(output_dir)
    def read(name):
        return json.loads((directory / name).read_text(), parse_constant=_invalid_constant,
                          object_pairs_hook=_unique_pairs)
    summary = read("summary.json")
    planning_input = load_planning_input(directory / "planning_input.json", routing.roads)
    result = GenerationResult.from_dict(read("proposal.json"))
    resolve_method_config(result.method_config)
    plan = evaluate_generation(planning_input, result, routing)
    replay = PlanReplay(planning_input, plan, routing)
    snapshot = load_execution_snapshot(directory / "snapshot.json", replay)
    selection = summary.get("selection", {})
    if selection.get("mode") == "time":
        selected = select_snapshot(replay, time_s=snapshot.time_s)
    elif selection.get("mode") == "remaining_tasks":
        selected = select_snapshot(replay, remaining_tasks=selection.get("target_remaining_tasks"))
    else:
        raise ValueError("stored initial background has an invalid snapshot selection mode")
    if selected.snapshot != snapshot:
        raise ValueError("snapshot selection does not match saved execution state")
    background = InitialBackground(plan, replay, selected)
    return background
