from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Any, Literal, Protocol, TYPE_CHECKING

from common.initial_sequences.inputs import PlanningInput, PlanningVisit, _freeze, _identifier, _ids, _mapping, _nonnegative, _plain, _routes, _sequence

if TYPE_CHECKING:
    from common.data.roads.routing import Routing
    from common.initial_sequences.plan import InitialPlan


@dataclass(frozen=True, kw_only=True)
class GeneratorConfig:
    name: str
    information_mode: Literal["batch", "online"]
    objective: str
    references: tuple[str, ...] = ()
    seed: int | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("name", "objective"):
            _identifier(getattr(self, name), f"method.{name}")
        if self.information_mode not in ("batch", "online"):
            raise ValueError("method.information_mode: expected batch or online")
        object.__setattr__(self, "references", _ids(self.references, "method.references"))
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("method.seed: expected an integer or None")
        object.__setattr__(self, "parameters", _freeze(_mapping(self.parameters, "method.parameters")))


@dataclass(frozen=True, kw_only=True)
class GenerationResult:


    input_fingerprint: str
    method_config: GeneratorConfig
    status: Literal["proposed", "partial", "no_plan_found", "unsupported"]
    sequences: Mapping[str, tuple[PlanningVisit, ...]]
    accepted_task_ids: tuple[str, ...] = ()
    rejected_task_ids: tuple[str, ...] = ()
    deferred_task_ids: tuple[str, ...] = ()
    uncovered_task_ids: tuple[str, ...] = ()
    runtime_s: float = 0
    termination_reason: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _identifier(self.input_fingerprint, "generation.input_fingerprint")
        if not isinstance(self.method_config, GeneratorConfig):
            raise ValueError("generation.method_config: expected GeneratorConfig")
        if self.status not in ("proposed", "partial", "no_plan_found", "unsupported"):
            raise ValueError("generation.status: unknown generation status")
        object.__setattr__(self, "sequences", _routes(self.sequences, "generation.sequences", PlanningVisit))
        decisions = set()
        for name in ("accepted_task_ids", "rejected_task_ids", "deferred_task_ids", "uncovered_task_ids"):
            values = _ids(getattr(self, name), f"generation.{name}")
            object.__setattr__(self, name, values)
            if name != "uncovered_task_ids":
                if decisions.intersection(values):
                    raise ValueError("generation: accepted, rejected and deferred task IDs must be disjoint")
                decisions.update(values)
        object.__setattr__(self, "runtime_s", _nonnegative(self.runtime_s, "generation.runtime_s"))
        if not isinstance(self.termination_reason, str):
            raise ValueError("generation.termination_reason: expected a string")
        object.__setattr__(self, "diagnostics", _freeze(_mapping(self.diagnostics, "generation.diagnostics")))

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "GenerationResult":
        try:
            values = _mapping(record, "generation")
            values["method_config"] = GeneratorConfig(**_mapping(values["method_config"], "method_config"))
            values["sequences"] = {
                worker_id: tuple(PlanningVisit(**visit) for visit in _sequence(visits, "visits", Mapping))
                for worker_id, visits in _mapping(values["sequences"], "sequences").items()
            }
            return cls(**values)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid generation result: {error}") from error


class InitialSequenceGenerator(Protocol):
    def generate(self, planning_input: PlanningInput, routing: Routing,
                 method_config: GeneratorConfig) -> GenerationResult:

        ...


def generate_initial_plan(generator: InitialSequenceGenerator, planning_input: PlanningInput,
                          routing: Routing, method_config: GeneratorConfig) -> InitialPlan:


    from common.evaluation.feasibility import PlanValidationError, evaluate_generation, validate_generation_input

    validate_generation_input(planning_input, routing, method_config)
    if not callable(getattr(generator, "generate", None)):
        raise PlanValidationError("generator_contract", "generator must provide a callable generate method")
    started = perf_counter()
    result = generator.generate(planning_input, routing, method_config)
    runtime_s = perf_counter() - started
    if not isinstance(result, GenerationResult):
        raise PlanValidationError("generator_contract", "generate must return GenerationResult")
    result = replace(result, runtime_s=runtime_s)
    if result.method_config != method_config:
        raise PlanValidationError("generator_contract", "returned method_config differs from the invocation", result=result)
    plan = evaluate_generation(planning_input, result, routing)
    return plan
