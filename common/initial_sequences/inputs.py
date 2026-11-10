from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import cached_property
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Literal

from common.data.roads.schema import MovementProfile, RoadNetwork, RoadPosition


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected a nonempty string")


def _number(value, name, *, positive=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name}: expected a finite number")
    if positive and value <= 0:
        raise ValueError(f"{name}: expected > 0")
    return float(value)


def _exceeds_capacity(load, capacity):

    return load - capacity > math.ulp(max(load, capacity))


def _freeze(value):
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("provenance: expected string keys")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        if type(value) in (int, float):
            _number(value, "provenance number")
        return value
    raise ValueError("provenance: expected JSON-like values")


def _provenance(record):
    if not isinstance(record.provenance, Mapping):
        raise ValueError("provenance: expected a mapping")
    object.__setattr__(record, "provenance", _freeze(record.provenance))


def _sequence(value, name, record_type):
    if not isinstance(value, (tuple, list)) or not all(
            isinstance(item, record_type) for item in value):
        raise ValueError(f"{name}: expected {record_type.__name__} records")
    return tuple(value)


def _ids(value, name):
    result = _sequence(value, name, str)
    for item in result:
        _identifier(item, name)
    if len(set(result)) != len(result):
        raise ValueError(f"{name}: duplicate task IDs")
    return result


def _plain(value):


    if value is None or type(value) in (str, int, float, bool):
        return value
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name))
                for item in fields(value) if item.init
                and not (item.metadata.get("omit_default") and getattr(value, item.name) == item.default)}
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}: expected a mapping")
    return dict(value)


def _nonnegative(value, name):
    value = _number(value, name)
    if value < 0:
        raise ValueError(f"{name}: expected >= 0")
    return value


def _routes(value, name, record_type):
    result = _mapping(value, name)
    for worker_id, visits in result.items():
        _identifier(worker_id, f"{name} worker ID")
        result[worker_id] = _sequence(visits, f"{name}[{worker_id}]", record_type)
    return MappingProxyType(result)


@dataclass(frozen=True, kw_only=True)
class PlanningVisit:
    task_id: str
    operation: Literal["pickup", "delivery"]

    def __post_init__(self):
        _identifier(self.task_id, "visit.task_id")
        if self.operation not in ("pickup", "delivery"):
            raise ValueError("visit.operation: expected pickup or delivery")


@dataclass(frozen=True, kw_only=True)
class PlanningTask:


    id: str
    pickup: RoadPosition | None
    delivery: RoadPosition
    load: float
    pickup_ready_time_s: float | None
    delivery_ready_time_s: float
    promised_delivery_time_s: float
    release_time_s: float | None = None
    known_at_s: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    pickup_service_time_s: float = field(default=0., metadata={"omit_default": True})
    delivery_service_time_s: float = field(default=0., metadata={"omit_default": True})

    def __post_init__(self):
        _identifier(self.id, "task.id")
        if self.pickup is not None and not isinstance(self.pickup, RoadPosition):
            raise ValueError("task.pickup: expected RoadPosition or None")
        if not isinstance(self.delivery, RoadPosition):
            raise ValueError("task.delivery: expected RoadPosition")
        object.__setattr__(self, "load", _number(self.load, "task.load", positive=True))
        for name in ("delivery_ready_time_s", "promised_delivery_time_s"):
            object.__setattr__(self, name, _number(getattr(self, name), f"task.{name}"))
        for name in ("pickup_ready_time_s", "release_time_s", "known_at_s"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _number(getattr(self, name), f"task.{name}"))
        for name in ("pickup_service_time_s", "delivery_service_time_s"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), f"task.{name}"))
        _provenance(self)

    def service_duration_s(self, operation):
        if operation not in ("pickup", "delivery"):
            raise ValueError("task service operation must be pickup or delivery")
        return getattr(self, operation + "_service_time_s")


@dataclass(frozen=True, kw_only=True)
class PlanningWorker:
    id: str
    location: RoadPosition
    capacity: float
    available_time_s: float
    profile: MovementProfile = field(default_factory=MovementProfile)
    carried_task_ids: tuple[str, ...] = ()
    remaining_visits: tuple[PlanningVisit, ...] = ()
    fixed_prefix_length: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _identifier(self.id, "worker.id")
        if not isinstance(self.location, RoadPosition):
            raise ValueError("worker.location: expected RoadPosition")
        if not isinstance(self.profile, MovementProfile):
            raise ValueError("worker.profile: expected MovementProfile")
        object.__setattr__(self, "capacity",
                           _number(self.capacity, "worker.capacity", positive=True))
        object.__setattr__(self, "available_time_s",
                           _number(self.available_time_s, "worker.available_time_s"))
        object.__setattr__(self, "carried_task_ids",
                           _ids(self.carried_task_ids, "worker.carried_task_ids"))
        object.__setattr__(self, "remaining_visits",
                           _sequence(self.remaining_visits, "worker.remaining_visits", PlanningVisit))
        if (type(self.fixed_prefix_length) is not int
                or not 0 <= self.fixed_prefix_length <= len(self.remaining_visits)):
            raise ValueError("worker.fixed_prefix_length: expected an integer within remaining_visits")
        _provenance(self)


@dataclass(frozen=True, kw_only=True)
class PlanningInput:
    instance_id: str
    planning_time_s: float
    time_origin_s: float
    network_id: str
    snapshot_id: str
    load_unit: str
    tasks: tuple[PlanningTask, ...]
    workers: tuple[PlanningWorker, ...]
    required_task_ids: tuple[str, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    task_map: Mapping[str, PlanningTask] = field(init=False, repr=False, compare=False)
    worker_map: Mapping[str, PlanningWorker] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ("instance_id", "network_id", "snapshot_id", "load_unit"):
            _identifier(getattr(self, name), f"input.{name}")
        for name in ("planning_time_s", "time_origin_s"):
            object.__setattr__(self, name, _number(getattr(self, name), f"input.{name}"))
        for name, record_type, map_name in (("tasks", PlanningTask, "task_map"),
                                            ("workers", PlanningWorker, "worker_map")):
            values = _sequence(getattr(self, name), f"input.{name}", record_type)
            records = {item.id: item for item in values}
            if len(records) != len(values):
                raise ValueError(f"input.{name}: duplicate IDs")
            object.__setattr__(self, name, values)
            object.__setattr__(self, map_name, MappingProxyType(records))
        required = _ids(self.required_task_ids, "input.required_task_ids")
        if any(task_id not in self.task_map for task_id in required):
            raise ValueError("input.required_task_ids: unknown task ID")
        object.__setattr__(self, "required_task_ids", required)
        _provenance(self)

        required = set(required)
        holders = {}
        for worker in self.workers:
            if worker.available_time_s < self.planning_time_s:
                raise ValueError(f"worker {worker.id}: available_time_s precedes planning_time_s")
            for task_id in worker.carried_task_ids:
                if task_id not in required:
                    raise ValueError(f"worker {worker.id}: carried task {task_id} must be required")
                if task_id in holders:
                    raise ValueError(f"task {task_id}: carried by multiple workers")
                release = self.task_map[task_id].release_time_s
                if release is not None and release > self.planning_time_s:
                    raise ValueError(f"task {task_id}: carried before release_time_s")
                holders[task_id] = worker.id
            try:
                load = math.fsum(self.task_map[task_id].load for task_id in worker.carried_task_ids)
            except OverflowError as error:
                raise ValueError(f"worker {worker.id}: initial carried load exceeds finite range") from error
            if _exceeds_capacity(load, worker.capacity):
                raise ValueError(f"worker {worker.id}: initial carried load exceeds capacity")
        for task in self.tasks:
            if task.id in holders:
                continue
            if task.pickup is None or task.pickup_ready_time_s is None:
                raise ValueError(f"task {task.id}: uncollected task requires pickup and pickup_ready_time_s")
            if task.release_time_s is not None and task.pickup_ready_time_s < task.release_time_s:
                raise ValueError(f"task {task.id}: pickup_ready_time_s precedes release_time_s")
        for position in self._positions():
            if position.network_id != self.network_id:
                raise ValueError("input: positions must belong to network_id")
        self._validate_remaining_visits(required, holders)

    def _positions(self):
        carried = {task_id for worker in self.workers for task_id in worker.carried_task_ids}
        for task in self.tasks:
            if task.id not in carried:
                yield task.pickup
            yield task.delivery
        for worker in self.workers:
            yield worker.location

    def _validate_remaining_visits(self, required, holders):
        owners = dict(holders)
        for worker in self.workers:
            onboard = set(worker.carried_task_ids)
            visited = set()
            for visit in worker.remaining_visits:
                task_id = visit.task_id
                if task_id not in required:
                    raise ValueError(f"worker {worker.id}: remaining visit task {task_id} must be required")
                if owners.setdefault(task_id, worker.id) != worker.id:
                    raise ValueError(f"task {task_id}: remaining visits belong to multiple workers")
                event = (task_id, visit.operation)
                if event in visited:
                    raise ValueError(f"task {task_id}: duplicate remaining {visit.operation}")
                visited.add(event)
                if visit.operation == "pickup":
                    if task_id in onboard or task_id in holders:
                        raise ValueError(f"task {task_id}: already carried before remaining pickup")
                    onboard.add(task_id)
                elif task_id not in onboard:
                    raise ValueError(f"task {task_id}: remaining delivery has no preceding pickup or holder")
                else:
                    onboard.remove(task_id)
            if worker.remaining_visits and onboard:
                raise ValueError(f"worker {worker.id}: remaining visits omit a delivery")

    @property
    def candidate_task_ids(self) -> tuple[str, ...]:
        required = set(self.required_task_ids)
        return tuple(task.id for task in self.tasks if task.id not in required)

    def validate_roads(self, roads: RoadNetwork) -> None:


        if not isinstance(roads, RoadNetwork):
            raise TypeError("roads must be a RoadNetwork")
        if roads.network_id != self.network_id:
            raise ValueError("planning input network_id does not match roads")
        if roads.snapshot_id != self.snapshot_id:
            raise ValueError("planning input snapshot_id does not match roads")
        for position in self._positions():
            roads.normalize_position(position)

    def to_dict(self) -> dict[str, Any]:

        return _plain(self)

    @cached_property
    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=True,
                               allow_nan=False, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "PlanningInput":

        if not isinstance(record, Mapping):
            raise ValueError("planning input: expected a JSON object")
        try:
            values = dict(record)
            tasks = []
            for task in _sequence(values["tasks"], "tasks", Mapping):
                task = dict(task)
                task["pickup"] = None if task["pickup"] is None else RoadPosition(**task["pickup"])
                task["delivery"] = RoadPosition(**task["delivery"])
                tasks.append(PlanningTask(**task))
            workers = []
            for worker in _sequence(values["workers"], "workers", Mapping):
                worker = dict(worker)
                worker["location"] = RoadPosition(**worker["location"])
                if "profile" in worker:
                    worker["profile"] = MovementProfile(**worker["profile"])
                if "remaining_visits" in worker:
                    worker["remaining_visits"] = tuple(
                        PlanningVisit(**visit) for visit in _sequence(
                            worker["remaining_visits"], "remaining_visits", Mapping))
                workers.append(PlanningWorker(**worker))
            values.update(tasks=tuple(tasks), workers=tuple(workers))
            return cls(**values)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid planning input: {error}") from error
