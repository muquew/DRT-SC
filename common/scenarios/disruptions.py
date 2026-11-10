from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import json

from common.data.roads.schema import RoadNetwork, RoadPosition
from common.initial_sequences.inputs import _identifier, _number, _nonnegative, _plain, _provenance
from common.execution.state import ExecutionSnapshot
from common.recovery.inputs import RecoveryInput, RecoverySettings, RecoveryWorker


KINDS = ("pickup_delay", "reception_delay", "delivery_relocation",
         "road_time_change", "worker_withdrawal")


class DisruptionError(ValueError):
    def __init__(self, code, message, *, event=None):
        super().__init__(message)
        self.code = code
        self.event_id = None if event is None else event.id
        self.target_id = None if event is None else event.target_id

    def to_dict(self):
        return {"code": self.code, "message": str(self),
                "event_id": self.event_id, "target_id": self.target_id}


@dataclass(frozen=True, kw_only=True)
class DisruptionEvent:


    id: str
    kind: str
    time_s: float
    target_id: str
    value: object = None
    provenance: Mapping = field(default_factory=dict)

    def __post_init__(self):
        _identifier(self.id, "event.id")
        _identifier(self.target_id, "event.target_id")
        if self.kind not in KINDS:
            raise ValueError(f"unknown disruption kind: {self.kind}")
        object.__setattr__(self, "time_s", _number(self.time_s, "event.time_s"))
        if self.kind in ("pickup_delay", "reception_delay", "road_time_change"):
            object.__setattr__(self, "value", _number(self.value, "event.value",
                                                    positive=self.kind == "road_time_change"))
        elif self.kind == "delivery_relocation":
            if not isinstance(self.value, RoadPosition):
                raise ValueError("delivery_relocation.value must be a RoadPosition")
        elif self.value is not None:
            object.__setattr__(self, "value", _nonnegative(self.value, "withdrawal.handover_range_m"))
        _provenance(self)

    def to_dict(self):
        return _plain(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, Mapping):
            raise ValueError("event must be a mapping")
        record = dict(value)
        if record.get("kind") == "delivery_relocation" and isinstance(record.get("value"), Mapping):
            record["value"] = RoadPosition(**record["value"])
        return cls(**record)


def apply_disruptions(snapshot, original_roads, events, recovery_settings, *, instance_id=None):


    if not isinstance(snapshot, ExecutionSnapshot) or not isinstance(original_roads, RoadNetwork):
        raise TypeError("expected an ExecutionSnapshot and original RoadNetwork")
    if (snapshot.network_id != original_roads.network_id or
            snapshot.snapshot_id != original_roads.snapshot_id):
        raise DisruptionError("source_roads", "roads do not match the execution snapshot")
    if not isinstance(recovery_settings, RecoverySettings):
        raise TypeError("expected RecoverySettings")
    if not isinstance(events, (tuple, list)) or not all(isinstance(e, DisruptionEvent) for e in events):
        raise TypeError("events must contain DisruptionEvent records")
    if len({e.id for e in events}) != len(events):
        raise DisruptionError("duplicate_event", "event IDs must be unique within the batch")
    if set(recovery_settings.handover_ranges_m) != set(snapshot.worker_map):
        raise DisruptionError("handover_ranges", "ranges must cover every snapshot worker exactly")
    recovery_settings.meeting_domain.validate_roads(original_roads)
    events = tuple(sorted(events, key=lambda e: e.id))
    tasks = {s.task.id: s.task for s in snapshot.tasks if s.status != "delivered"}
    active_services = {w.service.visit.task_id: w.service.visit.operation
                       for w in snapshot.workers if w.service is not None}
    task_updates, worker_updates, travel_times, writes, changes = {}, {}, {}, {}, []

    def update(event, entity, name, old, new):
        key = entity, event.target_id, name
        if key in writes and writes[key] != new:
            raise DisruptionError("conflicting_update", f"conflicting values for {key}", event=event)
        writes[key] = new
        changes.append({"event_id": event.id, "entity": entity, "target_id": event.target_id,
                        "field": name, "old_value": _plain(old), "new_value": _plain(new),
                        "changed": old != new})

    for event in events:
        if event.time_s != snapshot.time_s:
            raise DisruptionError("event_time", "all events must occur at the snapshot time", event=event)
        target, kind = event.target_id, event.kind
        if kind in ("pickup_delay", "reception_delay", "delivery_relocation"):
            state = snapshot.task_map.get(target)
            if state is None or state.status == "delivered":
                raise DisruptionError("task_target", "event requires an unfinished task", event=event)
            task = state.task
            active = active_services.get(target)
            if ((active == 'pickup' and kind == 'pickup_delay') or
                    (active == 'delivery' and kind in ('reception_delay', 'delivery_relocation'))):
                raise DisruptionError('service_in_progress', 'cannot change an already started service', event=event)
            if kind == "pickup_delay":
                if state.status != "awaiting_pickup":
                    raise DisruptionError("task_state", "pickup delay requires an uncollected task", event=event)
                name, new = "pickup_ready_time_s", event.value
            elif kind == "reception_delay":
                name, new = "delivery_ready_time_s", event.value
            else:
                if state.status != "carried":
                    raise DisruptionError("task_state", "delivery relocation requires a carried task", event=event)
                name = "delivery"
                try:
                    new = original_roads.normalize_position(event.value)
                except ValueError as error:
                    raise DisruptionError("relocation_position", str(error), event=event) from error
                if original_roads.normalize_position(task.delivery) == new:
                    new = task.delivery
            old = getattr(task, name)
            if kind.endswith("delay") and new < old:
                raise DisruptionError("earlier_readiness", "a delay cannot reduce the original readiness", event=event)
            update(event, "task", name, old, new)
            task_updates.setdefault(target, {})[name] = new
        elif kind == "road_time_change":
            arc = original_roads.arc_map.get(target)
            if arc is None:
                raise DisruptionError("arc_target", "event refers to an unknown directed arc", event=event)
            update(event, "arc", "travel_time_s", arc.travel_time_s, event.value)
            if arc.travel_time_s != event.value:
                travel_times[target] = event.value
        else:
            if target not in snapshot.worker_map:
                raise DisruptionError("worker_target", "event refers to an unknown worker", event=event)
            update(event, "worker", "service_mode", "delivery", "handover_only")
            worker_updates.setdefault(target, {})["service_mode"] = "handover_only"
            if event.value is not None:
                old = recovery_settings.handover_ranges_m[target]
                update(event, "worker", "handover_range_m", old, event.value)
                worker_updates[target]["handover_range_m"] = event.value

    roads = (original_roads.updated(version=f"{original_roads.version}:recovery@{snapshot.time_s!r}",
                                    travel_times_s=travel_times)
             if travel_times else original_roads)
    recovered_tasks = tuple(replace(task, **task_updates.get(identifier, {}))
                            for identifier, task in tasks.items())
    workers = tuple(RecoveryWorker(
        id=w.worker_id, location=w.location, capacity=w.capacity, profile=w.profile,
        carried_task_ids=w.carried_task_ids,
        handover_range_m=worker_updates.get(w.worker_id, {}).get(
            "handover_range_m", recovery_settings.handover_ranges_m[w.worker_id]),
        service_mode=worker_updates.get(w.worker_id, {}).get("service_mode", "delivery"),
    ) for w in snapshot.workers)
    if instance_id is None:
        names = json.dumps([e.id for e in events], ensure_ascii=True, separators=(",", ":"))
        instance_id = f"{snapshot.instance_id}:recovery@{snapshot.time_s!r}:{names}"
    return RecoveryInput(
        instance_id=instance_id, source_snapshot=snapshot, source_settings=recovery_settings,
        roads=roads, tasks=recovered_tasks, workers=workers,
        reference_sequences={w.worker_id: w.remaining_visits for w in snapshot.workers},
        events=tuple(e.to_dict() for e in events), changes=tuple(changes),
    )
