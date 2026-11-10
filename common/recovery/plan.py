from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Any, Literal

from common.data.roads.schema import MovementProfile, RoadLeg, RoadPath, RoadPosition
from common.initial_sequences.inputs import _identifier, _ids, _mapping, _nonnegative, _number, _plain, _routes, _sequence
from common.initial_sequences.plan import _total


@dataclass(frozen=True, kw_only=True)
class RecoveryVisit:
    operation: Literal["pickup", "delivery", "transfer"]
    task_id: str | None = None
    transfer_id: str | None = None

    def __post_init__(self):
        if self.operation not in ("pickup", "delivery", "transfer"):
            raise ValueError("recovery_visit.operation: expected pickup, delivery, or transfer")
        if self.operation == "transfer":
            _identifier(self.transfer_id, "recovery_visit.transfer_id")
            if self.task_id is not None:
                raise ValueError("recovery_visit: transfer visits use only transfer_id")
        else:
            _identifier(self.task_id, "recovery_visit.task_id")
            if self.transfer_id is not None:
                raise ValueError("recovery_visit: pickup and delivery visits use only task_id")


@dataclass(frozen=True, kw_only=True)
class WaypointVisit(RecoveryVisit):


    operation: Literal["waypoint"] = "waypoint"
    location: RoadPosition
    not_before_s: float | None = None
    for_task_id: str | None = None

    def __post_init__(self):
        if self.operation != "waypoint" or self.task_id is not None or self.transfer_id is not None:
            raise ValueError("waypoint: only location and optional not_before_s are allowed")
        if not isinstance(self.location, RoadPosition):
            raise ValueError("waypoint.location: expected RoadPosition")
        if self.not_before_s is not None:
            object.__setattr__(self, "not_before_s", _number(self.not_before_s, "waypoint.not_before_s"))
        if self.for_task_id is not None:
            _identifier(self.for_task_id, "waypoint.for_task_id")


def _read_visit(record):
    values = dict(_mapping(record, "recovery visit"))
    if values.get("operation") == "waypoint":
        values["location"] = RoadPosition(**_mapping(values["location"], "waypoint.location"))
        return WaypointVisit(**values)
    return RecoveryVisit(**values)


@dataclass(frozen=True, kw_only=True)
class TransferEvent:
    id: str
    worker_i: str
    worker_j: str
    location: RoadPosition
    tasks_i_to_j: tuple[str, ...] = ()
    tasks_j_to_i: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("id", "worker_i", "worker_j"):
            _identifier(getattr(self, name), f"transfer.{name}")
        if self.worker_i == self.worker_j:
            raise ValueError("transfer: participants must be distinct workers")
        if not isinstance(self.location, RoadPosition):
            raise ValueError("transfer.location: expected RoadPosition")
        for name in ("tasks_i_to_j", "tasks_j_to_i"):
            object.__setattr__(self, name, _ids(getattr(self, name), f"transfer.{name}"))
        if not self.tasks_i_to_j and not self.tasks_j_to_i:
            raise ValueError("transfer: at least one task must be handed over")
        if set(self.tasks_i_to_j).intersection(self.tasks_j_to_i):
            raise ValueError("transfer: task directions must be disjoint")


@dataclass(frozen=True, kw_only=True)
class RecoveryPlan:


    input_id: str
    network_id: str
    snapshot_id: str
    sequences: Mapping[str, tuple[RecoveryVisit, ...]]
    transfers: tuple[TransferEvent, ...] = ()
    transfer_map: Mapping[str, TransferEvent] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        for name in ("input_id", "network_id", "snapshot_id"):
            _identifier(getattr(self, name), f"recovery_plan.{name}")
        object.__setattr__(self, "sequences", _routes(
            self.sequences, "recovery_plan.sequences", RecoveryVisit))
        transfers = _sequence(self.transfers, "recovery_plan.transfers", TransferEvent)
        index = {event.id: event for event in transfers}
        if len(index) != len(transfers):
            raise ValueError("recovery_plan.transfers: duplicate IDs")
        object.__setattr__(self, "transfers", transfers)
        object.__setattr__(self, "transfer_map", MappingProxyType(index))

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "RecoveryPlan":
        try:
            values = _mapping(record, "recovery plan")
            values["sequences"] = {
                worker_id: tuple(_read_visit(visit) for visit in
                    _sequence(visits, "recovery visits", Mapping))
                for worker_id, visits in _mapping(values["sequences"], "sequences").items()
            }
            transfers = []
            for record in _sequence(values.get("transfers", ()), "transfers", Mapping):
                event = dict(record)
                event["location"] = RoadPosition(**_mapping(event["location"], "location"))
                transfers.append(TransferEvent(**event))
            values["transfers"] = tuple(transfers)
            return cls(**values)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid recovery plan: {error}") from error


@dataclass(frozen=True, kw_only=True)
class ScheduledRecoveryVisit:
    visit: RecoveryVisit
    incoming_path: RoadPath
    arrival_time_s: float
    start_time_s: float
    completion_time_s: float
    load_after: float
    carried_task_ids: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.visit, RecoveryVisit):
            raise ValueError("scheduled.visit: expected RecoveryVisit")
        if not isinstance(self.incoming_path, RoadPath):
            raise ValueError("scheduled.incoming_path: expected RoadPath")
        for name in ("arrival_time_s", "start_time_s", "completion_time_s"):
            object.__setattr__(self, name, _number(getattr(self, name), f"scheduled.{name}"))
        _nonnegative(self.waiting_time_s, "scheduled.waiting_time_s")
        _nonnegative(self.service_time_s, "scheduled.service_time_s")
        object.__setattr__(self, "load_after", _nonnegative(self.load_after, "scheduled.load_after"))
        object.__setattr__(self, "carried_task_ids", _ids(self.carried_task_ids, "scheduled.carried_task_ids"))

    @property
    def position(self) -> RoadPosition:
        return self.incoming_path.destination

    @property
    def waiting_time_s(self) -> float:
        return self.start_time_s - self.arrival_time_s

    @property
    def service_time_s(self) -> float:
        return self.completion_time_s - self.start_time_s


@dataclass(frozen=True, kw_only=True)
class EvaluatedRecoveryPlan:


    plan: RecoveryPlan
    routes: Mapping[str, tuple[ScheduledRecoveryVisit, ...]]
    tardiness_s: Mapping[str, float]

    def __post_init__(self):
        if not isinstance(self.plan, RecoveryPlan):
            raise ValueError("evaluated.plan: expected RecoveryPlan")
        object.__setattr__(self, "routes", _routes(
            self.routes, "evaluated.routes", ScheduledRecoveryVisit))
        if self.routes.keys() != self.plan.sequences.keys():
            raise ValueError("evaluated: route workers do not match plan sequences")
        deliveries = set()
        transfers = {event.id: [] for event in self.plan.transfers}
        for worker_id, route in self.routes.items():
            if tuple(scheduled.visit for scheduled in route) != self.plan.sequences[worker_id]:
                raise ValueError(f"evaluated: route visits for {worker_id} do not match plan sequences")
            for scheduled in route:
                path = scheduled.incoming_path
                if path.network_id != self.plan.network_id or path.snapshot_id != self.plan.snapshot_id:
                    raise ValueError("evaluated: incoming path does not match the road snapshot")
                visit = scheduled.visit
                if visit.operation == "delivery":
                    if visit.task_id in deliveries:
                        raise ValueError(f"evaluated: duplicate delivery for {visit.task_id}")
                    deliveries.add(visit.task_id)
                elif visit.operation == "transfer":
                    if visit.transfer_id not in transfers:
                        raise ValueError(f"evaluated: unknown transfer {visit.transfer_id}")
                    transfers[visit.transfer_id].append((worker_id, scheduled))
        for event in self.plan.transfers:
            pair = transfers[event.id]
            if len(pair) != 2 or {worker_id for worker_id, _ in pair} != {event.worker_i, event.worker_j}:
                raise ValueError(f"evaluated: transfer {event.id} requires one visit per participant")
            first, second = pair[0][1], pair[1][1]
            if (first.start_time_s != second.start_time_s
                    or first.completion_time_s != second.completion_time_s
                    or first.position != second.position):
                raise ValueError(f"evaluated: transfer {event.id} times and positions must match")
        tardiness = _mapping(self.tardiness_s, "evaluated.tardiness_s")
        for task_id, value in tardiness.items():
            _identifier(task_id, "evaluated.tardiness_s task ID")
            tardiness[task_id] = _nonnegative(value, f"evaluated.tardiness_s[{task_id}]")
        if tardiness.keys() != deliveries:
            raise ValueError("evaluated: tardiness task IDs must match scheduled deliveries")
        object.__setattr__(self, "tardiness_s", MappingProxyType(tardiness))
        for name in ("total_tardiness_s", "total_travel_time_s", "total_wait_time_s", "total_distance_m"):
            getattr(self, name)

    @cached_property
    def completion_times_s(self) -> Mapping[str, float]:
        return MappingProxyType({scheduled.visit.task_id: scheduled.completion_time_s
                                 for route in self.routes.values() for scheduled in route
                                 if scheduled.visit.operation == "delivery"})

    @cached_property
    def transfer_start_times_s(self) -> Mapping[str, float]:
        return MappingProxyType({scheduled.visit.transfer_id: scheduled.start_time_s
                                 for route in self.routes.values() for scheduled in route
                                 if scheduled.visit.operation == "transfer"})

    @cached_property
    def transfer_completion_times_s(self) -> Mapping[str, float]:
        return MappingProxyType({scheduled.visit.transfer_id: scheduled.completion_time_s
                                 for route in self.routes.values() for scheduled in route
                                 if scheduled.visit.operation == "transfer"})

    @cached_property
    def total_tardiness_s(self) -> float:
        return _total(self.tardiness_s.values(), "evaluated.total_tardiness_s")

    @cached_property
    def late_task_count(self) -> int:
        return sum(value > 0 for value in self.tardiness_s.values())

    @cached_property
    def total_travel_time_s(self) -> float:
        return _total((scheduled.incoming_path.duration_s for route in self.routes.values()
                       for scheduled in route), "evaluated.total_travel_time_s")

    @cached_property
    def total_wait_time_s(self) -> float:
        return _total((scheduled.waiting_time_s for route in self.routes.values()
                       for scheduled in route), "evaluated.total_wait_time_s")

    @cached_property
    def total_distance_m(self) -> float:
        return _total((scheduled.incoming_path.length_m for route in self.routes.values()
                       for scheduled in route), "evaluated.total_distance_m")

    @cached_property
    def transfer_count(self) -> int:
        return len(self.plan.transfers)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "EvaluatedRecoveryPlan":
        try:
            values = _mapping(record, "evaluated recovery plan")
            values["plan"] = RecoveryPlan.from_dict(values["plan"])
            routes = {}
            for worker_id, visits in _mapping(values["routes"], "routes").items():
                route = []
                for record in _sequence(visits, "scheduled visits", Mapping):
                    scheduled = dict(record)
                    scheduled["visit"] = _read_visit(scheduled["visit"])
                    path = _mapping(scheduled["incoming_path"], "incoming_path")
                    for name in ("origin", "destination"):
                        path[name] = RoadPosition(**_mapping(path[name], name))
                    path["profile"] = MovementProfile(**_mapping(path["profile"], "profile"))
                    path["legs"] = tuple(RoadLeg(**leg) for leg in _sequence(path["legs"], "legs", Mapping))
                    scheduled["incoming_path"] = RoadPath(**path)
                    route.append(ScheduledRecoveryVisit(**scheduled))
                routes[worker_id] = tuple(route)
            values["routes"] = routes
            return cls(**values)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid evaluated recovery plan: {error}") from error
