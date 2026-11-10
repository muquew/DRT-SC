from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import json
from types import MappingProxyType
from typing import Any, Literal

from common.data.roads.schema import MovementProfile, RoadNetwork, RoadPosition
from common.initial_sequences.inputs import PlanningInput, PlanningTask, PlanningVisit, PlanningWorker, _freeze, _identifier, _ids, _mapping, _nonnegative, _number, _plain, _routes, _sequence
from common.execution.state import ExecutionSnapshot


def checked_activity_domains(item, domains=None):

    domains = dict(item.source_settings.activity_domains if domains is None else domains)
    if set(domains) - set(item.worker_map):
        raise ValueError('activity domain has unknown workers')
    for domain in domains.values():
        domain.validate_roads(item.roads)
    return domains


@dataclass(frozen=True, kw_only=True)
class SegmentPortion:


    segment_id: str
    start_m: float
    end_m: float

    def __post_init__(self):
        _identifier(self.segment_id, "segment_portion.segment_id")
        for name in ("start_m", "end_m"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), f"segment_portion.{name}"))
        if self.start_m > self.end_m:
            raise ValueError("segment_portion: start_m exceeds end_m")


@dataclass(frozen=True, kw_only=True)
class MeetingDomain:


    node_ids: tuple[str, ...] = ()
    segment_portions: tuple[SegmentPortion, ...] = ()
    all_road_positions: bool = False

    def __post_init__(self):
        if type(self.all_road_positions) is not bool:
            raise ValueError("meeting_domain.all_road_positions: expected a boolean")
        object.__setattr__(self, "node_ids", _ids(self.node_ids, "meeting_domain.node_ids"))
        object.__setattr__(self, "segment_portions", _sequence(
            self.segment_portions, "meeting_domain.segment_portions", SegmentPortion))

    def validate_roads(self, roads: RoadNetwork) -> None:
        if not isinstance(roads, RoadNetwork):
            raise ValueError("meeting_domain: expected RoadNetwork")
        for node_id in self.node_ids:
            if node_id not in roads.node_map:
                raise ValueError(f"meeting_domain: unknown node {node_id}")
        for portion in self.segment_portions:
            segment = roads.segment_map.get(portion.segment_id)
            if segment is None:
                raise ValueError(f"meeting_domain: unknown segment {portion.segment_id}")
            if portion.end_m > segment.length_m:
                raise ValueError(f"meeting_domain: interval exceeds segment {segment.id}")

    def contains(self, position: RoadPosition, roads: RoadNetwork) -> bool:
        self.validate_roads(roads)
        position = roads.normalize_position(position)
        if self.all_road_positions:
            return True
        if position.node_id is not None:
            if position.node_id in self.node_ids:
                return True
            for portion in self.segment_portions:
                segment = roads.segment_map[portion.segment_id]
                if ((portion.start_m == 0 and position.node_id == segment.u)
                        or (portion.end_m == segment.length_m and position.node_id == segment.v)):
                    return True
            return False
        return any(portion.segment_id == position.segment_id
                   and portion.start_m <= position.offset_m <= portion.end_m
                   for portion in self.segment_portions)

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "MeetingDomain":
        try:
            values = _mapping(record, "meeting domain")
            values["segment_portions"] = tuple(SegmentPortion(**portion) for portion in
                _sequence(values.get("segment_portions", ()), "segment_portions", Mapping))
            return cls(**values)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid meeting domain: {error}") from error


@dataclass(frozen=True, kw_only=True)
class RecoverySettings:

    handover_ranges_m: Mapping[str, float]
    transfer_duration_s: float
    meeting_domain: MeetingDomain
    activity_domains: Mapping[str, MeetingDomain] = field(default_factory=dict)
    def __post_init__(self):
        ranges = _mapping(self.handover_ranges_m, "settings.handover_ranges_m")
        for worker_id, value in ranges.items():
            _identifier(worker_id, "settings.handover_ranges_m worker ID")
            ranges[worker_id] = _nonnegative(value, f"settings.handover_ranges_m[{worker_id}]")
        object.__setattr__(self, "handover_ranges_m", MappingProxyType(ranges))
        object.__setattr__(self, "transfer_duration_s", _number(
            self.transfer_duration_s, "settings.transfer_duration_s", positive=True))
        if not isinstance(self.meeting_domain, MeetingDomain):
            raise ValueError("settings.meeting_domain: expected MeetingDomain")
        domains = _mapping(self.activity_domains, 'settings.activity_domains')
        for worker_id, domain in domains.items():
            _identifier(worker_id, 'settings.activity_domains worker ID')
            if not isinstance(domain, MeetingDomain):
                raise ValueError('settings.activity_domains values must be MeetingDomain')
        object.__setattr__(self, 'activity_domains', MappingProxyType(domains))

    def to_dict(self) -> dict[str, Any]:
        values = _plain(self)
        if not self.activity_domains:
            values.pop('activity_domains')
        return values

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "RecoverySettings":
        try:
            values = _mapping(record, "recovery settings")
            values["meeting_domain"] = MeetingDomain.from_dict(values["meeting_domain"])
            values['activity_domains'] = {w: MeetingDomain.from_dict(domain)
                for w, domain in values.get('activity_domains', {}).items()}
            return cls(**values)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid recovery settings: {error}") from error


@dataclass(frozen=True, kw_only=True)
class RecoveryWorker:
    id: str
    location: RoadPosition
    capacity: float
    profile: MovementProfile
    carried_task_ids: tuple[str, ...]
    handover_range_m: float
    service_mode: Literal["delivery", "handover_only"] = "delivery"

    def __post_init__(self):
        _identifier(self.id, "recovery_worker.id")
        if not isinstance(self.location, RoadPosition):
            raise ValueError("recovery_worker.location: expected RoadPosition")
        if not isinstance(self.profile, MovementProfile):
            raise ValueError("recovery_worker.profile: expected MovementProfile")
        object.__setattr__(self, "capacity", _number(self.capacity, "recovery_worker.capacity", positive=True))
        object.__setattr__(self, "carried_task_ids", _ids(self.carried_task_ids, "recovery_worker.carried_task_ids"))
        object.__setattr__(self, "handover_range_m", _nonnegative(self.handover_range_m, "recovery_worker.handover_range_m"))
        if self.service_mode not in ("delivery", "handover_only"):
            raise ValueError("recovery_worker.service_mode: expected delivery or handover_only")


@dataclass(frozen=True, kw_only=True)
class RecoveryInput:
    instance_id: str
    source_snapshot: ExecutionSnapshot
    source_settings: RecoverySettings
    roads: RoadNetwork
    tasks: tuple[PlanningTask, ...]
    workers: tuple[RecoveryWorker, ...]
    reference_sequences: Mapping[str, tuple[PlanningVisit, ...]]
    events: tuple[Mapping[str, Any], ...] = ()
    changes: tuple[Mapping[str, Any], ...] = ()
    task_map: Mapping[str, PlanningTask] = field(init=False, repr=False, compare=False)
    worker_map: Mapping[str, RecoveryWorker] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        _identifier(self.instance_id, "common.recovery.instance_id")
        for name, kind in (("source_snapshot", ExecutionSnapshot), ("source_settings", RecoverySettings),
                           ("roads", RoadNetwork)):
            if not isinstance(getattr(self, name), kind):
                raise ValueError(f"recovery.{name}: expected {kind.__name__}")
        for name, kind in (("tasks", PlanningTask), ("workers", RecoveryWorker)):
            records = _sequence(getattr(self, name), f"recovery.{name}", kind)
            index = {record.id: record for record in records}
            if len(index) != len(records):
                raise ValueError(f"recovery.{name}: duplicate IDs")
            object.__setattr__(self, name, records)
            object.__setattr__(self, "task_map" if name == "tasks" else "worker_map", MappingProxyType(index))
        object.__setattr__(self, "reference_sequences", _routes(
            self.reference_sequences, "common.recovery.reference_sequences", PlanningVisit))
        for name in ("events", "changes"):
            object.__setattr__(self, name, tuple(_freeze(record) for record in
                _sequence(getattr(self, name), f"recovery.{name}", Mapping)))
        self.validate()

    @property
    def time_s(self) -> float:
        return self.source_snapshot.time_s

    @property
    def time_origin_s(self) -> float:
        return self.source_snapshot.time_origin_s

    @property
    def load_unit(self) -> str:
        return self.source_snapshot.load_unit

    @property
    def network_id(self) -> str:
        return self.roads.network_id

    @property
    def snapshot_id(self) -> str:
        return self.roads.snapshot_id

    def validate(self) -> None:
        source = self.source_snapshot
        if self.network_id != source.network_id:
            raise ValueError("recovery: roads must preserve the source physical network")
        if set(self.task_map) != set(source.unfinished_task_ids):
            raise ValueError("recovery: tasks must contain all and only unfinished obligations")
        if set(self.worker_map) != set(source.worker_map):
            raise ValueError("recovery: workers must contain all and only source workers")
        if set(self.source_settings.handover_ranges_m) != set(source.worker_map):
            raise ValueError("recovery: handover_ranges_m must cover exactly the source workers")
        self.source_settings.meeting_domain.validate_roads(self.roads)
        if set(self.source_settings.activity_domains) - set(self.worker_map):
            raise ValueError('activity domains contain unknown workers')
        for domain in self.source_settings.activity_domains.values():
            domain.validate_roads(self.roads)

        for worker in self.workers:
            original = source.worker_map[worker.id]
            if self.roads.normalize_position(worker.location) != self.roads.normalize_position(original.location):
                raise ValueError(f"worker {worker.id}: actual location must remain unchanged")
            if set(worker.carried_task_ids) != set(original.carried_task_ids):
                raise ValueError(f"worker {worker.id}: actual carried tasks must remain unchanged")
            if worker.capacity != original.capacity or worker.profile != original.profile:
                raise ValueError(f"worker {worker.id}: capacity and movement profile must remain unchanged")
        for task in self.tasks:
            original = source.task_map[task.id].task
            for name in ("pickup", "load", "promised_delivery_time_s", "release_time_s", "known_at_s", "provenance"):
                if getattr(task, name) != getattr(original, name):
                    raise ValueError(f"task {task.id}: {name} must remain unchanged")
            self.roads.normalize_position(original.delivery)

        expected_sequences = {w.worker_id: w.remaining_visits for w in source.workers}
        if self.reference_sequences != expected_sequences:
            raise ValueError("recovery: reference_sequences must equal the source remaining visits")


        planning = PlanningInput(
            instance_id=self.instance_id, planning_time_s=self.time_s, time_origin_s=self.time_origin_s,
            network_id=self.network_id, snapshot_id=self.snapshot_id, load_unit=self.load_unit,
            tasks=self.tasks, required_task_ids=source.unfinished_task_ids,
            workers=tuple(PlanningWorker(
                id=worker.id, location=worker.location, capacity=worker.capacity,
                available_time_s=source.worker_map[worker.id].available_time_s, profile=worker.profile,
                carried_task_ids=worker.carried_task_ids, remaining_visits=self.reference_sequences[worker.id],
                fixed_prefix_length=source.worker_map[worker.id].fixed_prefix_length,
            ) for worker in self.workers),
        )
        planning.validate_roads(self.roads)
        expected = Counter((task.id, "delivery") for task in self.tasks)
        expected.update((state.task.id, "pickup") for state in source.tasks if state.status == "awaiting_pickup")
        actual = Counter((visit.task_id, visit.operation)
                         for visits in self.reference_sequences.values() for visit in visits)
        if actual != expected:
            raise ValueError("recovery: reference_sequences must cover every remaining operation exactly once")

    def to_dict(self) -> dict[str, Any]:

        source = self.source_snapshot
        return {
            "instance_id": self.instance_id,
            "time_s": self.time_s,
            "time_origin_s": self.time_origin_s,
            "load_unit": self.load_unit,
            "network_id": self.network_id,
            "snapshot_id": self.snapshot_id,
            "source_snapshot": {name: getattr(source, name) for name in (
                "instance_id", "input_fingerprint", "plan_fingerprint", "network_id", "snapshot_id", "time_s")},
            "source_settings": self.source_settings.to_dict(),
            "tasks": _plain(self.tasks),
            "workers": _plain(self.workers),
            "reference_sequences": _plain(self.reference_sequences),
            "events": _plain(self.events),
            "changes": _plain(self.changes),
        }

    @cached_property
    def fingerprint(self) -> str:


        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
