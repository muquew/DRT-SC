from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DomainSemantics:
    cell_count: int
    point_fraction: float
    normalized_interval_width: float

    relative_midpoint_mean: tuple[float, float]
    relative_midpoint_extent: tuple[float, float]


@dataclass(frozen=True)
class GeometrySemantics:
    domains: tuple[DomainSemantics, ...]
    mode_nonzero_coefficients: int
    mode_inequalities: int
    cached_leg_minima_s: tuple[float | None, ...]
    cached_leg_missing: tuple[bool, ...]


def _position_xy(roads, position, check_budget):
    if position.node_id is not None:
        node = roads.node_map[position.node_id]
        return node.x, node.y
    segment = roads.segment_map[position.segment_id]
    return _segment_xy(segment, float(position.offset_m), check_budget)


def _segment_xy(segment, offset, check_budget):
    remaining = offset
    points = segment.geometry
    for i, (left, right) in enumerate(zip(points, points[1:])):
        if i % 32 == 0: check_budget()
        length = math.hypot(right[0]-left[0], right[1]-left[1])
        if length and remaining <= length:
            ratio = max(0., remaining / length)
            return (left[0]+ratio*(right[0]-left[0]), left[1]+ratio*(right[1]-left[1]))
        remaining -= length
    return points[-1]


def summarize_node(archive, node, cell_type):
    road_snapshot = archive.instance.roads
    domains = []
    for event, domain in zip(archive._events, node.domains):
        archive.check_budget()
        workers = (archive.instance.worker_map[event.worker_i], archive.instance.worker_map[event.worker_j])
        anchors = [_position_xy(road_snapshot, worker.location, archive.check_budget) for worker in workers]
        center = (anchors[0][0]/2+anchors[1][0]/2, anchors[0][1]/2+anchors[1][1]/2)
        scale = max(*(float(w.handover_range_m) for w in workers),
                    math.hypot(anchors[0][0]-anchors[1][0], anchors[0][1]-anchors[1][1]))
        if scale == 0: scale = 1.
        points = []; widths = []; point_count = 0
        for i, cell in enumerate(domain):
            if i % 32 == 0: archive.check_budget()
            if cell.point is not None:
                point_count += 1
                xy = _position_xy(road_snapshot, cell.point, archive.check_budget)
            else:
                widths.append(float(cell.high-cell.low)/scale)
                xy = _segment_xy(road_snapshot.segment_map[cell.segment], float((cell.low+cell.high)/2), archive.check_budget)
            points.append(((xy[0]-center[0])/scale, (xy[1]-center[1])/scale))
        domains.append(DomainSemantics(len(domain), point_count/len(domain), sum(widths),
            tuple(sum(p[k] for p in points)/len(points) for k in (0, 1)),
            tuple(max(p[k] for p in points)-min(p[k] for p in points) for k in (0, 1))))
    minima = []
    event_domains = {event.id: domain for event, domain in zip(archive._events, node.domains)}
    for a, b, wid in archive._edges:
        archive.check_budget()
        activity = archive.activity_domains.get(wid)
        router = archive.roads if activity is None or activity.all_road_positions else archive._activity_routers.get(activity)
        origin, destination = archive._nodes[a][2], archive._nodes[b][2]
        speed = archive.instance.worker_map[wid].profile.speed_factor
        value = None
        if router is not None:
            if not isinstance(origin, str) and not isinstance(destination, str):
                value = archive._fixed_minima.get((router, origin, destination, speed))
            else:
                left = event_domains[origin] if isinstance(origin, str) else (cell_type(point=origin),)
                right = event_domains[destination] if isinstance(destination, str) else (cell_type(point=destination),)
                value = archive._travel_cache.get((router, left, right, speed))
        try: value = None if value is None else float(value)
        except OverflowError: value = None
        minima.append(value if value is not None and math.isfinite(value) else None)
    nonzero = inequalities = count = 0
    for alternatives in (() if node.modes is None else node.modes):
        archive.check_budget()
        for mode in alternatives:
            if count % 32 == 0: archive.check_budget()
            count += 1
            nonzero += sum(coefficient != 0 for coefficient in mode.coefficients.values())
            inequalities += len(mode.conditions)
    return GeometrySemantics(tuple(domains), nonzero, inequalities, tuple(minima), tuple(v is None for v in minima))

from fractions import Fraction
from common.config import service_value

DECISION_WINDOW=service_value("decision_window")

@dataclass(frozen=True)
class GeometryNodeCandidate:
    lower_bound_numerator: int
    lower_bound_denominator: int
    depth: int
    label: tuple
    admission_serial: int
    phase: str = 'unknown'
    unfixed_road_domains: int = 0
    unfixed_travel_modes: int = 0
    log_remaining_choices: float = 0.0
    semantics: GeometrySemantics | None = None

    @property
    def lower_bound(self):
        return Fraction(self.lower_bound_numerator, self.lower_bound_denominator)

@dataclass(frozen=True)
class GeometryNodeDecision:

    candidates: tuple[GeometryNodeCandidate, ...]
    heuristic_index: int = 0
    local_incumbent_tardiness: float | None = None
    remaining_budget_s: float | None = None
    transfer_count: int = 0
    runnable_node_count: int = 0
    primary_lp_completed: int = 0
