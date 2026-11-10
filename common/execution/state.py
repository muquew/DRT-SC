from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import json
import math
from types import MappingProxyType
from typing import Literal

from common.data.roads.schema import MovementProfile, RoadLeg, RoadPath, RoadPosition
from common.initial_sequences.inputs import PlanningTask, PlanningVisit, _exceeds_capacity, _identifier, _ids, _mapping, _nonnegative, _number, _plain, _sequence


@dataclass(frozen=True, kw_only=True)
class TravelProgress:


    path: RoadPath
    departure_time_s: float
    time_s: float

    def __post_init__(self):
        if not isinstance(self.path, RoadPath) or self.path.duration_s <= 0:
            raise ValueError("travel.path: expected a RoadPath with positive duration")
        object.__setattr__(self, "departure_time_s", _number(self.departure_time_s, "travel.departure_time_s"))
        object.__setattr__(self, "time_s", _number(self.time_s, "travel.time_s"))
        arrival = _number(self.departure_time_s + self.path.duration_s, "travel.arrival_time_s")
        if not self.departure_time_s <= self.time_s <= arrival:
            raise ValueError("travel.time_s: outside the departure-to-arrival interval")
        _nonnegative(self.remaining_time_s, "travel.remaining_time_s")

    @property
    def elapsed_time_s(self) -> float:
        return min(self.path.duration_s, max(0.0, self.time_s - self.departure_time_s))

    @property
    def remaining_time_s(self) -> float:


        return max(0.0, (self.departure_time_s + self.path.duration_s) - self.time_s)


@dataclass(frozen=True, kw_only=True)
class ServiceProgress:


    visit: PlanningVisit
    start_time_s: float
    completion_time_s: float
    time_s: float

    def __post_init__(self):
        if not isinstance(self.visit, PlanningVisit):
            raise ValueError("service.visit: expected PlanningVisit")
        for name in ("start_time_s", "completion_time_s", "time_s"):
            object.__setattr__(self, name, _number(getattr(self, name), "service." + name))
        if not self.start_time_s <= self.time_s < self.completion_time_s:
            raise ValueError("service.time_s: outside the active service interval")

    @property
    def remaining_time_s(self):
        return self.completion_time_s - self.time_s


@dataclass(frozen=True, kw_only=True)
class WorkerExecutionState:
    worker_id: str
    location: RoadPosition
    capacity: float
    profile: MovementProfile
    available_time_s: float
    phase: Literal["not_started", "travelling", "waiting", "servicing", "idle"]
    carried_task_ids: tuple[str, ...]
    load: float
    completed_visit_count: int
    remaining_visits: tuple[PlanningVisit, ...]
    fixed_prefix_length: int
    travel: TravelProgress | None = None
    service: ServiceProgress | None = field(default=None, metadata={"omit_default": True})

    def __post_init__(self):
        _identifier(self.worker_id, "worker_state.worker_id")
        if not isinstance(self.location, RoadPosition) or not isinstance(self.profile, MovementProfile):
            raise ValueError("worker_state: expected RoadPosition and MovementProfile")
        object.__setattr__(self, "capacity", _number(self.capacity, "worker_state.capacity", positive=True))
        object.__setattr__(self, "available_time_s", _number(self.available_time_s, "worker_state.available_time_s"))
        object.__setattr__(self, "load", _nonnegative(self.load, "worker_state.load"))
        object.__setattr__(self, "carried_task_ids", _ids(self.carried_task_ids, "worker_state.carried_task_ids"))
        object.__setattr__(self, "remaining_visits", _sequence(self.remaining_visits, "worker_state.remaining_visits", PlanningVisit))
        if type(self.completed_visit_count) is not int or self.completed_visit_count < 0:
            raise ValueError("worker_state.completed_visit_count: expected a nonnegative integer")
        if (type(self.fixed_prefix_length) is not int
                or not 0 <= self.fixed_prefix_length <= len(self.remaining_visits)):
            raise ValueError("worker_state.fixed_prefix_length: outside remaining visits")
        if self.phase not in ("not_started", "travelling", "waiting", "servicing", "idle"):
            raise ValueError("worker_state.phase: unknown execution phase")
        if self.phase == "travelling":
            if not isinstance(self.travel, TravelProgress):
                raise ValueError("worker_state.travel: required while travelling")
            if self.travel.path.profile != self.profile or self.travel.path.network_id != self.location.network_id:
                raise ValueError("worker_state.travel: path profile or network differs from worker")
        elif self.travel is not None:
            raise ValueError("worker_state.travel: only used while travelling")
        if self.phase == "servicing":
            if (not isinstance(self.service, ServiceProgress) or not self.remaining_visits
                    or self.service.visit != self.remaining_visits[0]):
                raise ValueError("worker_state.service: active service must be the first remaining visit")
        elif self.service is not None:
            raise ValueError("worker_state.service: only used while servicing")
        if self.phase in ("travelling", "waiting", "servicing") and not self.remaining_visits:
            raise ValueError("worker_state: an active phase requires a remaining visit")
        if self.phase == "idle" and (self.remaining_visits or self.carried_task_ids):
            raise ValueError("worker_state: idle worker has remaining service or cargo")
        if self.phase == "not_started" and self.completed_visit_count:
            raise ValueError("worker_state: not_started worker cannot have completed visits")
        if bool(self.carried_task_ids) != (self.load > 0) or _exceeds_capacity(self.load, self.capacity):
            raise ValueError("worker_state.load: inconsistent with cargo or capacity")


@dataclass(frozen=True, kw_only=True)
class TaskExecutionState:
    task: PlanningTask
    status: Literal["awaiting_pickup", "carried", "delivered"]
    holder_id: str | None = None
    pickup_time_s: float | None = None
    delivery_time_s: float | None = None

    def __post_init__(self):
        if not isinstance(self.task, PlanningTask):
            raise ValueError("task_state.task: expected PlanningTask")
        for name in ("pickup_time_s", "delivery_time_s"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _number(getattr(self, name), f"task_state.{name}"))
        if self.status == "carried":
            _identifier(self.holder_id, "task_state.holder_id")
            if self.delivery_time_s is not None:
                raise ValueError("task_state: carried task cannot have a delivery time")
        elif self.status == "delivered":
            if self.holder_id is not None or self.delivery_time_s is None:
                raise ValueError("task_state: delivered task needs a delivery time and no current holder")
        elif self.status == "awaiting_pickup":
            if any(value is not None for value in (self.holder_id, self.pickup_time_s, self.delivery_time_s)):
                raise ValueError("task_state: awaiting pickup cannot have executed service or a holder")
            if self.task.pickup is None or self.task.pickup_ready_time_s is None:
                raise ValueError("task_state: awaiting pickup requires a pickup position and readiness")
        else:
            raise ValueError("task_state.status: unknown execution status")
        if (self.pickup_time_s is not None and self.delivery_time_s is not None
                and self.pickup_time_s > self.delivery_time_s):
            raise ValueError("task_state: pickup follows delivery")


@dataclass(frozen=True, kw_only=True)
class ExecutionSnapshot:


    instance_id: str
    input_fingerprint: str
    plan_fingerprint: str
    network_id: str
    snapshot_id: str
    time_s: float
    time_origin_s: float
    load_unit: str
    tasks: tuple[TaskExecutionState, ...]
    workers: tuple[WorkerExecutionState, ...]
    task_map: Mapping[str, TaskExecutionState] = field(init=False, repr=False, compare=False)
    worker_map: Mapping[str, WorkerExecutionState] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ("instance_id", "input_fingerprint", "plan_fingerprint", "network_id", "snapshot_id", "load_unit"):
            _identifier(getattr(self, name), f"execution.{name}")
        for name in ("time_s", "time_origin_s"):
            object.__setattr__(self, name, _number(getattr(self, name), f"execution.{name}"))
        for name, kind, key in (("tasks", TaskExecutionState, lambda s: s.task.id),
                                ("workers", WorkerExecutionState, lambda s: s.worker_id)):
            records = _sequence(getattr(self, name), f"execution.{name}", kind)
            index = {key(record): record for record in records}
            if len(index) != len(records):
                raise ValueError(f"execution.{name}: duplicate IDs")
            object.__setattr__(self, name, records)
            object.__setattr__(self, "task_map" if name == "tasks" else "worker_map", MappingProxyType(index))
        holders, active_services = {}, set()
        for worker in self.workers:
            if worker.location.network_id != self.network_id:
                raise ValueError("execution: worker location belongs to another network")
            if worker.phase == "not_started":
                if worker.available_time_s <= self.time_s:
                    raise ValueError("execution: not_started worker must become available after the snapshot")
            elif worker.available_time_s != self.time_s:
                raise ValueError("execution: active or idle worker availability must equal snapshot time")
            if worker.travel is not None:
                travel = worker.travel
                if travel.path.snapshot_id != self.snapshot_id or travel.time_s != self.time_s:
                    raise ValueError("execution: travel snapshot or observation time is inconsistent")
            if worker.service is not None:
                if worker.service.time_s != self.time_s:
                    raise ValueError("execution: active service observation time differs from snapshot")
                service = worker.service
                state = self.task_map[service.visit.task_id]
                if service.visit.task_id in active_services:
                    raise ValueError("execution: a task has multiple active services")
                active_services.add(service.visit.task_id)
                expected = service.start_time_s + state.task.service_duration_s(service.visit.operation)
                if service.completion_time_s != expected:
                    raise ValueError("execution: active service differs from its declared duration")
                if ((service.visit.operation == 'pickup' and state.status != 'awaiting_pickup') or
                        (service.visit.operation == 'delivery' and
                         (state.status != 'carried' or state.holder_id != worker.worker_id))):
                    raise ValueError("execution: active service and task custody differ")
            for identifier in worker.carried_task_ids:
                if identifier in holders or identifier not in self.task_map:
                    raise ValueError("execution: cargo has duplicate holders or an unknown task")
                holders[identifier] = worker.worker_id
            try:
                expected_load = math.fsum(self.task_map[i].task.load for i in worker.carried_task_ids)
            except OverflowError as error:
                raise ValueError("execution: carried load exceeds the finite numeric range") from error
            if worker.load != expected_load:
                raise ValueError("execution: worker load differs from its carried tasks")
            for visit in worker.remaining_visits:
                state = self.task_map.get(visit.task_id)
                if state is None or state.status == "delivered":
                    raise ValueError("execution: remaining visit references an unknown or delivered task")
                if state.status == "carried" and (state.holder_id != worker.worker_id or visit.operation == "pickup"):
                    raise ValueError("execution: remaining visit contradicts current custody")
        for state in self.tasks:
            if state.task.delivery.network_id != self.network_id or (
                    state.status == "awaiting_pickup" and state.task.pickup.network_id != self.network_id):
                raise ValueError("execution: remaining task position belongs to another network")
            if (state.status == "carried") != (state.task.id in holders) or state.holder_id != holders.get(state.task.id):
                raise ValueError("execution: task status or holder differs from worker custody")
            if any(t is not None and t > self.time_s for t in (state.pickup_time_s, state.delivery_time_s)):
                raise ValueError("execution: recorded service time follows the snapshot")

    @property
    def unfinished_task_ids(self) -> tuple[str, ...]:
        return tuple(state.task.id for state in self.tasks if state.status != "delivered")

    @property
    def completed_task_ids(self) -> tuple[str, ...]:
        return tuple(state.task.id for state in self.tasks if state.status == "delivered")

    def to_dict(self):
        return _plain(self)

    @cached_property
    def fingerprint(self) -> str:
        content = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, record):
        try:
            values = _mapping(record, "execution snapshot")
            tasks, workers = [], []
            for state in _sequence(values["tasks"], "tasks", Mapping):
                state = dict(state)
                task = _mapping(state["task"], "task")
                task["pickup"] = None if task["pickup"] is None else RoadPosition(**task["pickup"])
                task["delivery"] = RoadPosition(**task["delivery"])
                state["task"] = PlanningTask(**task)
                tasks.append(TaskExecutionState(**state))
            for state in _sequence(values["workers"], "workers", Mapping):
                state = dict(state)
                state["location"] = RoadPosition(**state["location"])
                state["profile"] = MovementProfile(**state["profile"])
                state["remaining_visits"] = tuple(PlanningVisit(**v) for v in _sequence(state["remaining_visits"], "remaining_visits", Mapping))
                if state.get("travel") is not None:
                    travel = _mapping(state["travel"], "travel")
                    path = _mapping(travel["path"], "travel.path")
                    for name in ("origin", "destination"):
                        path[name] = RoadPosition(**path[name])
                    path["profile"] = MovementProfile(**path["profile"])
                    path["legs"] = tuple(RoadLeg(**leg) for leg in _sequence(path["legs"], "path.legs", Mapping))
                    travel["path"] = RoadPath(**path)
                    state["travel"] = TravelProgress(**travel)
                if state.get("service") is not None:
                    service = dict(state["service"])
                    service["visit"] = PlanningVisit(**service["visit"])
                    state["service"] = ServiceProgress(**service)
                workers.append(WorkerExecutionState(**state))
            return cls(**{**values, "tasks": tuple(tasks), "workers": tuple(workers)})
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid execution snapshot: {error}") from error
