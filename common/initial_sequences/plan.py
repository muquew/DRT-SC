from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any

from common.data.roads.schema import MovementProfile, RoadLeg, RoadPath, RoadPosition
from common.initial_sequences.generator import GenerationResult
from common.initial_sequences.inputs import PlanningVisit, _identifier, _ids, _mapping, _nonnegative, _number, _plain, _routes, _sequence


def _total(values, name):
    try:
        return _nonnegative(math.fsum(values), name)
    except OverflowError as error:
        raise ValueError(f"{name}: total exceeds the finite numeric range") from error


@dataclass(frozen=True, kw_only=True)
class ScheduledVisit:
    visit: PlanningVisit
    incoming_path: RoadPath
    arrival_time_s: float
    start_time_s: float
    load_after: float
    carried_task_ids: tuple[str, ...]
    service_time_s: float = field(default=0., metadata={"omit_default": True})

    def __post_init__(self):
        if not isinstance(self.visit, PlanningVisit):
            raise ValueError("scheduled.visit: expected PlanningVisit")
        if not isinstance(self.incoming_path, RoadPath):
            raise ValueError("scheduled.incoming_path: expected RoadPath")
        for name in ("arrival_time_s", "start_time_s"):
            object.__setattr__(self, name, _number(getattr(self, name), f"scheduled.{name}"))
        _nonnegative(self.waiting_time_s, "scheduled.waiting_time_s")
        object.__setattr__(self, "service_time_s", _nonnegative(self.service_time_s, "scheduled.service_time_s"))
        _number(self.completion_time_s, "scheduled.completion_time_s")
        object.__setattr__(self, "load_after", _nonnegative(self.load_after, "scheduled.load_after"))
        object.__setattr__(self, "carried_task_ids", _ids(self.carried_task_ids, "scheduled.carried_task_ids"))

    @property
    def completion_time_s(self) -> float:
        completion = self.start_time_s + self.service_time_s if self.service_time_s else self.start_time_s
        if self.service_time_s > 0 and completion <= self.start_time_s:
            raise ValueError("service duration is below the time base's numeric resolution")
        return completion

    @property
    def waiting_time_s(self) -> float:
        return self.start_time_s - self.arrival_time_s

    @property
    def position(self) -> RoadPosition:
        return self.incoming_path.destination


@dataclass(frozen=True, kw_only=True)
class InitialPlan:


    input_fingerprint: str
    network_id: str
    snapshot_id: str
    generation_result: GenerationResult
    routes: Mapping[str, tuple[ScheduledVisit, ...]]
    tardiness_s: Mapping[str, float]
    _validated_for: tuple[str, str] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ("input_fingerprint", "network_id", "snapshot_id"):
            _identifier(getattr(self, name), f"plan.{name}")
        result = self.generation_result
        if not isinstance(result, GenerationResult):
            raise ValueError("plan.generation_result: expected GenerationResult")
        if result.input_fingerprint != self.input_fingerprint:
            raise ValueError("plan: generation input fingerprint does not match")
        if result.status != "proposed" or result.uncovered_task_ids:
            raise ValueError("plan: requires a proposed result with no uncovered tasks")
        object.__setattr__(self, "routes", _routes(self.routes, "plan.routes", ScheduledVisit))
        if self.routes.keys() != result.sequences.keys():
            raise ValueError("plan: route workers do not match generation sequences")
        deliveries = set()
        for worker_id, route in self.routes.items():
            if tuple(scheduled.visit for scheduled in route) != result.sequences[worker_id]:
                raise ValueError(f"plan: route visits for {worker_id} do not match generation sequences")
            for scheduled in route:
                path = scheduled.incoming_path
                if path.network_id != self.network_id or path.snapshot_id != self.snapshot_id:
                    raise ValueError("plan: incoming path does not match the road snapshot")
                if scheduled.visit.operation == "delivery":
                    task_id = scheduled.visit.task_id
                    if task_id in deliveries:
                        raise ValueError(f"plan: duplicate delivery for {task_id}")
                    deliveries.add(task_id)
        tardiness = _mapping(self.tardiness_s, "plan.tardiness_s")
        for task_id, value in tardiness.items():
            _identifier(task_id, "plan.tardiness_s task ID")
            tardiness[task_id] = _nonnegative(value, f"plan.tardiness_s[{task_id}]")
        if tardiness.keys() != deliveries:
            raise ValueError("plan: tardiness task IDs must match scheduled deliveries")
        object.__setattr__(self, "tardiness_s", MappingProxyType(tardiness))
        for name in ("total_tardiness_s", "total_travel_time_s", "total_wait_time_s", "total_distance_m"):
            getattr(self, name)

    @property
    def completion_times_s(self) -> Mapping[str, float]:
        return MappingProxyType({scheduled.visit.task_id: scheduled.completion_time_s
                                 for route in self.routes.values() for scheduled in route
                                 if scheduled.visit.operation == "delivery"})

    @property
    def total_tardiness_s(self) -> float:
        return _total(self.tardiness_s.values(), "plan.total_tardiness_s")

    @property
    def late_task_count(self) -> int:
        return sum(value > 0 for value in self.tardiness_s.values())

    @property
    def total_travel_time_s(self) -> float:
        return _total((scheduled.incoming_path.duration_s for route in self.routes.values()
                       for scheduled in route), "plan.total_travel_time_s")

    @property
    def total_wait_time_s(self) -> float:
        return _total((scheduled.waiting_time_s for route in self.routes.values()
                       for scheduled in route), "plan.total_wait_time_s")

    @property
    def total_distance_m(self) -> float:
        return _total((scheduled.incoming_path.length_m for route in self.routes.values()
                       for scheduled in route), "plan.total_distance_m")

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @cached_property
    def fingerprint(self) -> str:
        record = self.to_dict()
        for name in ("runtime_s", "diagnostics"):
            record["generation_result"].pop(name)
        canonical = json.dumps(record, sort_keys=True, ensure_ascii=True,
                               allow_nan=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "InitialPlan":
        try:
            values = _mapping(record, "plan")
            values["generation_result"] = GenerationResult.from_dict(values["generation_result"])
            routes = {}
            for worker_id, visits in _mapping(values["routes"], "routes").items():
                route = []
                for scheduled in _sequence(visits, "scheduled visits", Mapping):
                    scheduled = dict(scheduled)
                    scheduled["visit"] = PlanningVisit(**_mapping(scheduled["visit"], "visit"))
                    path = _mapping(scheduled["incoming_path"], "incoming_path")
                    for name in ("origin", "destination"):
                        path[name] = RoadPosition(**_mapping(path[name], name))
                    path["profile"] = MovementProfile(**_mapping(path["profile"], "profile"))
                    path["legs"] = tuple(RoadLeg(**leg) for leg in _sequence(path["legs"], "legs", Mapping))
                    scheduled["incoming_path"] = RoadPath(**path)
                    route.append(ScheduledVisit(**scheduled))
                routes[worker_id] = tuple(route)
            values["routes"] = routes
            return cls(**values)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid initial plan: {error}") from error
