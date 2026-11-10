from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
import math
from random import Random

from common.execution.replay import PlanReplay
from common.execution.state import ExecutionSnapshot
from common.initial_sequences.inputs import _freeze, _mapping, _nonnegative, _number, _plain
from common.scenarios.disruptions import DisruptionEvent, KINDS


@dataclass(frozen=True, kw_only=True)
class EventSamplingConfig:
    kind: str
    seed: int = 0
    count: int | None = None
    fraction: float | None = None
    delay_s: float | None = None
    time_factor: float | None = None
    relocation_band_m: tuple[float, float] | None = None
    withdrawal_pool: str = "carried"
    handover_range_m: float | None = None
    location_spacing_m: float = 100.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown event sampling kind: {self.kind}")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if self.count is not None and (type(self.count) is not int or self.count < 0):
            raise ValueError("count must be a nonnegative integer or None")
        if self.count is not None and self.fraction is not None:
            raise ValueError("choose count or fraction, not both")
        if self.fraction is not None:
            value = _number(self.fraction, "fraction")
            if not 0 <= value <= 1:
                raise ValueError("fraction must be in [0, 1]")
            object.__setattr__(self, "fraction", value)
        for name, applicable in (
            ("delay_s", self.kind in ("pickup_delay", "reception_delay")),
            ("time_factor", self.kind == "road_time_change"),
            ("relocation_band_m", self.kind == "delivery_relocation"),
        ):
            if applicable != (getattr(self, name) is not None):
                requirement = "required" if applicable else "not applicable"
                raise ValueError(f"{name} is {requirement} for {self.kind}")
        if self.delay_s is not None:
            object.__setattr__(self, "delay_s", _number(self.delay_s, "delay_s", positive=True))
        if self.time_factor is not None:
            value = _number(self.time_factor, "time_factor")
            if value <= 1:
                raise ValueError("time_factor must exceed 1")
            object.__setattr__(self, "time_factor", value)
        if self.relocation_band_m is not None:
            if not isinstance(self.relocation_band_m, (tuple, list)) or len(self.relocation_band_m) != 2:
                raise ValueError("relocation_band_m must be a lower/upper pair")
            low, high = (_nonnegative(value, "relocation_band_m") for value in self.relocation_band_m)
            if low >= high:
                raise ValueError("relocation_band_m requires 0 <= lower < upper")
            object.__setattr__(self, "relocation_band_m", (low, high))
        if self.withdrawal_pool not in ("carried", "empty_busy", "all"):
            raise ValueError("withdrawal_pool must be carried, empty_busy, or all")
        if self.kind != "worker_withdrawal" and (
                self.withdrawal_pool != "carried" or self.handover_range_m is not None):
            raise ValueError("withdrawal options are only applicable to worker_withdrawal")
        if self.handover_range_m is not None:
            object.__setattr__(self, "handover_range_m", _nonnegative(self.handover_range_m, "handover_range_m"))
        spacing = _number(self.location_spacing_m, "location_spacing_m", positive=True)
        if self.kind != "delivery_relocation" and spacing != 100.0:
            raise ValueError("location_spacing_m is only applicable to delivery_relocation")
        object.__setattr__(self, "location_spacing_m", spacing)

    def to_dict(self):
        return _plain(self)

    @classmethod
    def from_dict(cls, record):
        try:
            return cls(**_mapping(record, "event sampling config"))
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"invalid event sampling config: {error}") from error


@dataclass(frozen=True, kw_only=True)
class EventSample:
    status: str
    events: tuple[DisruptionEvent, ...]
    details: Mapping

    def __post_init__(self):
        if self.status not in ("sampled", "not_generated"):
            raise ValueError("event sample status must be sampled or not_generated")
        if not isinstance(self.events, (tuple, list)) or not all(isinstance(e, DisruptionEvent) for e in self.events):
            raise ValueError("events must contain DisruptionEvent records")
        events = tuple(self.events)
        details = _mapping(self.details, "event sample details")
        requested = details.get("requested_count")
        if self.status == "not_generated" and events:
            raise ValueError("only sampled batches may contain events")
        if self.status == "sampled" and not events and (type(requested) is not int or requested != 0):
            raise ValueError("empty sampled batches require a declared zero request")
        if (self.status == "sampled" and requested is not None
                and (type(requested) is not int or requested < 0 or len(events) != requested)):
            raise ValueError("sampled event count differs from the declared request")
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "details", _freeze(details))

    def to_dict(self):
        return {"status": self.status, "events": [e.to_dict() for e in self.events],
                "details": _plain(self.details)}


def _count(config, size):

    if config.count is not None:
        return config.count
    return math.ceil(Fraction(str(config.fraction)) * size) if config.fraction is not None else 1


def _result(events, details, reason=None):
    return EventSample(status="sampled" if reason is None else "not_generated",
                       events=tuple(events) if reason is None else (), details={**details, "reason": reason})


def _weighted_index(weights, rng):
    total = _number(math.fsum(weights), "sampling weight total", positive=True)
    target, cumulative = rng.random() * total, 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if target < cumulative:
            return index
    return len(weights) - 1


def _distance(routing, origin, destination):
    value = routing.shortest_distance(origin, destination)
    if value == math.inf:
        return value
    return _nonnegative(value, "directed road distance")


def _event(config, snapshot, target, value, **provenance):
    return DisruptionEvent(
        id=f"{config.kind}:{target}", kind=config.kind, time_s=snapshot.time_s,
        target_id=target, value=value,
        provenance={"sampler": "event_sampling", "sampling_config": config.to_dict(), **provenance},
    )


def _merged(intervals):
    result = []
    for low, high in sorted(intervals):
        if result and low <= result[-1][1]:
            result[-1] = result[-1][0], max(result[-1][1], high)
        else:
            result.append((low, high))
    return result


def _remaining_legs(path, elapsed=0.0):

    def boundary(value, end, operations):
        return abs(value - end) <= operations * max(math.ulp(value), math.ulp(end), math.ulp(elapsed))

    if elapsed >= path.duration_s or (elapsed and boundary(elapsed, path.duration_s, max(1, len(path.legs)))):
        return
    remaining = elapsed
    for index, leg in enumerate(path.legs):
        if remaining >= leg.duration_s or (remaining and boundary(remaining, leg.duration_s, index + 1)):
            remaining = max(0.0, remaining - leg.duration_s)
            continue
        start = leg.start_offset_m + (leg.end_offset_m - leg.start_offset_m) * (remaining / leg.duration_s)
        remaining = 0.0
        if start != leg.end_offset_m:
            yield leg.arc_id, leg.segment_id, min(start, leg.end_offset_m), max(start, leg.end_offset_m)


def _road_events(replay, snapshot, config, rng, details):
    roads, routing = replay.routing.roads, replay.routing
    arcs, physical = defaultdict(list), defaultdict(list)
    for worker in snapshot.workers:
        for index, scheduled in enumerate(replay.plan.routes[worker.worker_id][worker.completed_visit_count:]):
            if index == 0 and worker.phase in ("waiting", "servicing"):
                continue
            elapsed = worker.travel.elapsed_time_s if index == 0 and worker.phase == "travelling" else 0.0
            for arc_id, segment_id, low, high in _remaining_legs(scheduled.incoming_path, elapsed):
                if not roads.arc_map[arc_id].enabled:
                    raise ValueError("original remaining path uses a disabled arc")
                arcs[arc_id].append((low, high))
                physical[segment_id].append((low, high))
    portions = [{"segment_id": segment_id, "start_m": low, "end_m": high}
                for segment_id in sorted(physical) for low, high in _merged(physical[segment_id])]
    details.update(source_arc_pool_size=len(arcs), remaining_portions=portions,
                   remaining_length_m=math.fsum(p["end_m"] - p["start_m"] for p in portions),
                   ranked_arc_ids=[], unreachable_arc_ids=[], center_position=None,
                   pool_size=0, requested_count=_count(config, 0), selected_ids=[])
    if not portions:
        return _result((), details, None if details["requested_count"] == 0 else "empty_pool")
    portion = portions[_weighted_index([p["end_m"] - p["start_m"] for p in portions], rng)]
    offset = portion["start_m"] + rng.random() * (portion["end_m"] - portion["start_m"])
    center = roads.segment_position(portion["segment_id"], offset)
    ranking, unreachable = [], []
    register = getattr(routing, "register_query_positions", None)
    if callable(register):
        register((roads.segment_position(roads.arc_map[arc_id].segment_id, low + (high - low) / 2)
                  for arc_id in sorted(arcs) for low, high in _merged(arcs[arc_id])),
                 metrics=("length_m",))
    for arc_id in sorted(arcs):
        segment_id = roads.arc_map[arc_id].segment_id
        distance = min(_distance(routing, center, roads.segment_position(segment_id, low + (high - low) / 2))
                       for low, high in _merged(arcs[arc_id]))
        if math.isinf(distance):
            unreachable.append(arc_id)
        else:
            ranking.append((distance, arc_id))
    ranking.sort()
    pool_size = len(ranking)
    count = _count(config, pool_size)
    details.update(center_position=_plain(center), pool_size=pool_size, requested_count=count,
                   ranked_arc_ids=[arc_id for _, arc_id in ranking], unreachable_arc_ids=unreachable,
                   center_rule="remaining physical interval union length, then uniform offset",
                   distance_rule="center to remaining directed-arc portion midpoint; minimum across disjoint portions")
    if count == 0:
        return _result((), details)
    if not pool_size or count > pool_size:
        return _result((), details, "empty_pool" if not pool_size else "insufficient_pool")
    selected = ranking[:count]
    details["selected_ids"] = [arc_id for _, arc_id in selected]
    segment_ids = sorted({roads.arc_map[arc_id].segment_id for _, arc_id in selected})
    details.update(selected_segment_ids=segment_ids,
                   affected_physical_length_m=math.fsum(roads.segment_map[s].length_m for s in segment_ids))
    events = []
    for distance, arc_id in selected:
        arc = roads.arc_map[arc_id]
        value = _number(arc.travel_time_s * config.time_factor, "sampled road time", positive=True)
        events.append(_event(config, snapshot, arc_id, value, center_position=_plain(center),
                             directed_distance_m=distance, original_travel_time_s=arc.travel_time_s,
                             application_scope="entire directed arc"))
    return _result(events, details)


def _position_key(position):
    return position.node_id or "", position.segment_id or "", position.offset_m or 0.0


def _relocation_candidates(roads, routing, origin, band, spacing):

    low, high = band
    x, y = roads.coordinates(origin)
    weights = defaultdict(list)
    for segment in sorted(roads.segments, key=lambda item: item.id):
        xs, ys = zip(*segment.geometry)

        if math.hypot(max(min(xs) - x, 0, x - max(xs)), max(min(ys) - y, 0, y - max(ys))) > high:
            continue
        quotient = _number(segment.length_m / spacing, "relocation grid size")
        count = math.ceil(quotient)
        offsets = [index * spacing for index in range(count)] + [segment.length_m]
        for index, offset in enumerate(offsets):
            position = roads.segment_position(segment.id, offset)
            px, py = roads.coordinates(position)
            if math.hypot(px - x, py - y) > high:
                continue
            left = 0.0 if index == 0 else offsets[index - 1] + (offset - offsets[index - 1]) / 2
            right = segment.length_m if index == len(offsets) - 1 else offset + (offsets[index + 1] - offset) / 2
            if right > left:
                weights[position].append(right - left)
    candidates = []
    register = getattr(routing, "register_query_positions", None)
    if callable(register):
        register(weights, metrics=("length_m",))
    for position in sorted(weights, key=_position_key):
        distance = _distance(routing, origin, position)
        if low < distance <= high:
            candidates.append((position, distance, math.fsum(weights[position])))
    return candidates


def sample_events(replay: PlanReplay, snapshot: ExecutionSnapshot, config: EventSamplingConfig) -> EventSample:

    if not isinstance(replay, PlanReplay) or not isinstance(snapshot, ExecutionSnapshot):
        raise TypeError("expected PlanReplay and ExecutionSnapshot")
    if not isinstance(config, EventSamplingConfig):
        raise TypeError("config must be an EventSamplingConfig")
    if replay.at(snapshot.time_s) != snapshot:
        raise ValueError("snapshot differs from the supplied parent plan replay")
    roads = replay.routing.roads
    if roads.network_id != snapshot.network_id or roads.snapshot_id != snapshot.snapshot_id:
        raise ValueError("replay routing does not match the snapshot roads")
    rng = Random(config.seed)
    details = {"config": config.to_dict(), "time_s": snapshot.time_s,
               "pool_size": 0, "requested_count": _count(config, 0), "selected_ids": []}
    if config.count == 0 or config.fraction == 0:
        details.update(selection_skipped=True, pool_size_evaluated=False)
        return _result((), details)
    if config.kind == "road_time_change":
        return _road_events(replay, snapshot, config, rng, details)
    if config.kind == "worker_withdrawal":
        pool = [worker.worker_id for worker in snapshot.workers if
                config.withdrawal_pool == "all" or
                (config.withdrawal_pool == "carried" and worker.carried_task_ids) or
                (config.withdrawal_pool == "empty_busy" and not worker.carried_task_ids and worker.remaining_visits)]
    else:
        statuses = ({"awaiting_pickup"} if config.kind == "pickup_delay" else
                    {"carried"} if config.kind == "delivery_relocation" else {"awaiting_pickup", "carried"})
        active = {w.service.visit.task_id: w.service.visit.operation
                  for w in snapshot.workers if w.service is not None}
        pool = [state.task.id for state in snapshot.tasks if state.status in statuses
                and not ((active.get(state.task.id) == 'pickup' and config.kind == 'pickup_delay')
                    or (active.get(state.task.id) == 'delivery' and config.kind in ('reception_delay', 'delivery_relocation')))]
    pool.sort()
    rng.shuffle(pool)
    count = _count(config, len(pool))
    details.update(pool_size=len(pool), requested_count=count)
    if count == 0:
        return _result((), details)
    if not pool or count > len(pool):
        return _result((), details, "empty_pool" if not pool else "insufficient_pool")
    selected, events = pool[:count], []
    details["selected_ids"] = selected
    if config.kind == "delivery_relocation":
        details.update(location_spacing_m=config.location_spacing_m,
                       location_rule="physical segment arclength grid including endpoints; Voronoi length weights merged at shared nodes",
                       distance_rule="old delivery to candidate by directed shortest road distance", relocations=[])
    for target in selected:
        if config.kind == "worker_withdrawal":
            events.append(_event(config, snapshot, target, config.handover_range_m,
                                 selection_rule="stable ID shuffle prefix", pool=config.withdrawal_pool))
            continue
        task = snapshot.task_map[target].task
        if config.kind == "delivery_relocation":
            candidates = _relocation_candidates(roads, replay.routing, task.delivery,
                                                config.relocation_band_m, config.location_spacing_m)
            row = {"task_id": target, "old_delivery": _plain(task.delivery), "candidate_count": len(candidates)}
            details["relocations"].append(row)
            if not candidates:
                details["failed_task_id"] = target
                return _result((), details, "no_relocation_candidate")
            position, distance, weight = candidates[_weighted_index([row[2] for row in candidates], rng)]
            row.update(new_delivery=_plain(position), directed_distance_m=distance, selected_weight_m=weight)
            events.append(_event(config, snapshot, target, position, **row))
        else:
            old = task.pickup_ready_time_s if config.kind == "pickup_delay" else task.delivery_ready_time_s
            value = _number(max(old, snapshot.time_s) + config.delay_s, "sampled readiness")
            if value <= max(old, snapshot.time_s):
                raise ValueError("delay is below the timestamp's numeric resolution")
            events.append(_event(config, snapshot, target, value, old_readiness_s=old,
                                 selection_rule="stable ID shuffle prefix", formula="max(old_readiness_s, time_s) + delay_s"))
    return _result(events, details)
