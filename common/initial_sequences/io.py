import json
from pathlib import Path

from common.data.roads.schema import RoadNetwork
from common.initial_sequences.inputs import PlanningInput


def save_planning_input(planning_input: PlanningInput, path: Path) -> None:

    if not isinstance(planning_input, PlanningInput):
        raise TypeError("planning_input must be a PlanningInput")
    record = {"planning_input": planning_input.to_dict()}

    Path(path).write_text(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                    sort_keys=True, separators=(",", ":")) + "\n",
                          encoding="utf-8", errors="backslashreplace")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"invalid JSON number {value}")


def load_planning_input(path: Path, roads: RoadNetwork | None = None) -> PlanningInput:

    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"),
                            parse_constant=_invalid_constant, object_pairs_hook=_unique_pairs)
        if not isinstance(record, dict) or "planning_input" not in record:
            raise ValueError("malformed planning input file")
        result = PlanningInput.from_dict(record["planning_input"])
        if roads is not None:
            result.validate_roads(roads)
        return result
    except ValueError as error:
        raise ValueError(f"planning input file {path}: {error}") from error


def save_initial_plan(plan: "InitialPlan", path: Path) -> None:

    from common.initial_sequences.plan import InitialPlan

    if not isinstance(plan, InitialPlan):
        raise TypeError("plan must be an InitialPlan")
    record = {"initial_plan": plan.to_dict()}
    Path(path).write_text(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                    sort_keys=True, separators=(",", ":")) + "\n",
                          encoding="utf-8", errors="backslashreplace")


def load_initial_plan(path: Path, planning_input: PlanningInput, routing: "Routing") -> "InitialPlan":

    from common.evaluation.feasibility import PlanValidationError, evaluate_generation
    from common.initial_sequences.plan import InitialPlan

    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"),
                            parse_constant=_invalid_constant, object_pairs_hook=_unique_pairs)
        if not isinstance(record, dict) or "initial_plan" not in record:
            raise ValueError("malformed initial plan file")
        stored = InitialPlan.from_dict(record["initial_plan"])
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        raise ValueError(f"initial plan file {path}: {error}") from error
    if not isinstance(planning_input, PlanningInput):
        raise PlanValidationError("input_type", "expected PlanningInput", result=stored.generation_result)
    verified = evaluate_generation(planning_input, stored.generation_result, routing)
    return verified
