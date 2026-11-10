from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import cached_property
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected a nonempty string")


def _number(value, name, *, positive=False, nonnegative=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name}: expected a finite number")
    if positive and value <= 0 or nonnegative and value < 0:
        raise ValueError(f"{name}: expected {'> 0' if positive else '>= 0'}")
    return float(value)


def _xy(value, name):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name}: expected a coordinate pair")
    return tuple(_number(x, name) for x in value)


def _freeze(value):

    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("metadata: expected string keys")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or type(value) in (str, bool, int, float):
        if type(value) in (int, float):
            _number(value, "metadata number")
        return value
    raise ValueError("metadata: expected JSON-like values")


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                      ensure_ascii=True).encode()).hexdigest()


@dataclass(frozen=True, kw_only=True)
class RoadNode:
    id: str
    x: float
    y: float

    def __post_init__(self):
        _identifier(self.id, "node.id")
        object.__setattr__(self, "x", _number(self.x, "node.x"))
        object.__setattr__(self, "y", _number(self.y, "node.y"))


@dataclass(frozen=True, kw_only=True)
class RoadSegment:


    id: str
    u: str
    v: str
    geometry: tuple[tuple[float, float], ...]
    length_m: float = field(init=False)

    def __post_init__(self):
        for name in ("id", "u", "v"):
            _identifier(getattr(self, name), f"segment.{name}")
        if not isinstance(self.geometry, (tuple, list)) or len(self.geometry) < 2:
            raise ValueError("segment.geometry: expected at least two points")
        geometry = tuple(_xy(point, "segment.geometry point") for point in self.geometry)
        lengths = [math.dist(a, b) for a, b in zip(geometry, geometry[1:])]
        if any(length == 0 for length in lengths):
            raise ValueError("segment.geometry: consecutive points must differ")
        length = _number(math.fsum(lengths), "segment.length_m", positive=True)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "length_m", length)


@dataclass(frozen=True, kw_only=True)
class RoadArc:


    id: str
    segment_id: str
    forward: bool
    travel_time_s: float
    enabled: bool = True

    def __post_init__(self):
        _identifier(self.id, "arc.id")
        _identifier(self.segment_id, "arc.segment_id")
        if type(self.forward) is not bool or type(self.enabled) is not bool:
            raise ValueError("arc.forward and arc.enabled: expected booleans")
        object.__setattr__(self, "travel_time_s",
                           _number(self.travel_time_s, "arc.travel_time_s", positive=True))


@dataclass(frozen=True, kw_only=True)
class RoadPosition:


    network_id: str
    node_id: str | None = None
    segment_id: str | None = None
    offset_m: float | None = None

    def __post_init__(self):
        _identifier(self.network_id, "position.network_id")
        if (self.node_id is None) == (self.segment_id is None):
            raise ValueError("position: specify exactly one node_id or segment_id")
        if self.node_id is not None:
            _identifier(self.node_id, "position.node_id")
            if self.offset_m is not None:
                raise ValueError("position.offset_m: not used with node_id")
        else:
            _identifier(self.segment_id, "position.segment_id")
            object.__setattr__(self, "offset_m",
                               _number(self.offset_m, "position.offset_m", nonnegative=True))


@dataclass(frozen=True, kw_only=True)
class RoadNetwork:
    name: str
    version: str
    crs: str
    travel_mode: str
    nodes: tuple[RoadNode, ...]
    segments: tuple[RoadSegment, ...]
    arcs: tuple[RoadArc, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    node_map: Mapping[str, RoadNode] = field(init=False, repr=False, compare=False)
    segment_map: Mapping[str, RoadSegment] = field(init=False, repr=False, compare=False)
    arc_map: Mapping[str, RoadArc] = field(init=False, repr=False, compare=False)
    network_id: str = field(init=False)
    snapshot_id: str = field(init=False)

    def __post_init__(self):
        for name in ("name", "version", "crs", "travel_mode"):
            _identifier(getattr(self, name), f"network.{name}")
        for name, record_type, map_name in (("nodes", RoadNode, "node_map"),
                                            ("segments", RoadSegment, "segment_map"),
                                            ("arcs", RoadArc, "arc_map")):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)):
                raise ValueError(f"network.{name}: expected a sequence")
            values = tuple(values)
            if not all(isinstance(item, record_type) for item in values):
                raise ValueError(f"network.{name}: expected {record_type.__name__} records")
            records = {item.id: item for item in values}
            if len(records) != len(values):
                raise ValueError(f"network.{name}: duplicate IDs")
            object.__setattr__(self, name, values)
            object.__setattr__(self, map_name, MappingProxyType(records))
        for segment in self.segments:
            if segment.u not in self.node_map or segment.v not in self.node_map:
                raise ValueError(f"segment {segment.id}: unknown endpoint node")
            u, v = self.node_map[segment.u], self.node_map[segment.v]
            if segment.geometry[0] != (u.x, u.y) or segment.geometry[-1] != (v.x, v.y):
                raise ValueError(f"segment {segment.id}: geometry must match endpoint nodes")
        for arc in self.arcs:
            if arc.segment_id not in self.segment_map:
                raise ValueError(f"arc {arc.id}: unknown segment")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata: expected a mapping")
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        identity = [self.name, self.crs, self.travel_mode,
                    sorted((n.id, n.x, n.y) for n in self.nodes),
                    sorted((s.id, s.u, s.v, s.geometry) for s in self.segments),
                    sorted((a.id, a.segment_id, a.forward) for a in self.arcs)]
        object.__setattr__(self, "network_id", _fingerprint(identity))
        snapshot = [self.network_id, self.version,
                    sorted((a.id, a.travel_time_s, a.enabled) for a in self.arcs)]
        object.__setattr__(self, "snapshot_id", _fingerprint(snapshot))

    def node_position(self, node_id: str) -> RoadPosition:
        return self.normalize_position(RoadPosition(network_id=self.network_id, node_id=node_id))

    def segment_position(self, segment_id: str, offset_m: float) -> RoadPosition:
        return self.normalize_position(RoadPosition(network_id=self.network_id,
                                                    segment_id=segment_id, offset_m=offset_m))

    def normalize_position(self, position: RoadPosition) -> RoadPosition:
        if not isinstance(position, RoadPosition) or position.network_id != self.network_id:
            raise ValueError("position: expected a position on this network")
        if position.node_id is not None:
            if position.node_id not in self.node_map:
                raise ValueError(f"position: unknown node {position.node_id}")
            return position
        segment = self.segment_map.get(position.segment_id)
        if segment is None:
            raise ValueError(f"position: unknown segment {position.segment_id}")
        if position.offset_m > segment.length_m:
            raise ValueError("position.offset_m: beyond segment length")
        if position.offset_m == 0 or position.offset_m == segment.length_m:
            node_id = segment.u if position.offset_m == 0 else segment.v
            return RoadPosition(network_id=self.network_id, node_id=node_id)
        return position

    def coordinates(self, position: RoadPosition) -> tuple[float, float]:
        position = self.normalize_position(position)
        if position.node_id is not None:
            node = self.node_map[position.node_id]
            return node.x, node.y
        segment = self.segment_map[position.segment_id]
        remaining = position.offset_m
        for start, end in zip(segment.geometry, segment.geometry[1:]):
            length = math.dist(start, end)
            if remaining <= length:
                fraction = remaining / length
                return (start[0] + fraction * (end[0] - start[0]),
                        start[1] + fraction * (end[1] - start[1]))
            remaining -= length
        return segment.geometry[-1]

    def updated(self, *, version: str, travel_times_s: Mapping[str, float] | None = None,
                enabled: Mapping[str, bool] | None = None) -> "RoadNetwork":
        changes = ({} if travel_times_s is None else travel_times_s,
                   {} if enabled is None else enabled)
        for values in changes:
            if not isinstance(values, Mapping):
                raise ValueError("arc updates: expected a mapping")
            if any(arc_id not in self.arc_map for arc_id in values):
                raise ValueError("arc updates: unknown arc ID")
        times, flags = changes
        arcs = tuple(replace(arc, travel_time_s=times.get(arc.id, arc.travel_time_s),
                             enabled=flags.get(arc.id, arc.enabled)) for arc in self.arcs)
        return replace(self, version=version, arcs=arcs)


@dataclass(frozen=True, kw_only=True)
class MovementProfile:
    speed_factor: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "speed_factor",
                           _number(self.speed_factor, "profile.speed_factor", positive=True))


@dataclass(frozen=True, kw_only=True)
class RoadLeg:
    arc_id: str
    segment_id: str
    start_offset_m: float
    end_offset_m: float
    length_m: float
    duration_s: float

    def __post_init__(self):
        _identifier(self.arc_id, "leg.arc_id")
        _identifier(self.segment_id, "leg.segment_id")
        for name in ("start_offset_m", "end_offset_m", "length_m", "duration_s"):
            object.__setattr__(self, name,
                               _number(getattr(self, name), f"leg.{name}", nonnegative=True))
        if not math.isclose(self.length_m, abs(self.end_offset_m - self.start_offset_m),
                            rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("leg.length_m: must equal its offset span")
        if self.length_m <= 0 or self.duration_s <= 0 or self.start_offset_m == self.end_offset_m:
            raise ValueError("leg: travel requires positive length and duration; stationary paths have no legs")


@dataclass(frozen=True, kw_only=True)
class RoadPath:
    network_id: str
    snapshot_id: str
    origin: RoadPosition
    destination: RoadPosition
    profile: MovementProfile
    legs: tuple[RoadLeg, ...]

    def __post_init__(self):
        _identifier(self.network_id, "path.network_id")
        _identifier(self.snapshot_id, "path.snapshot_id")
        for position in (self.origin, self.destination):
            if not isinstance(position, RoadPosition) or position.network_id != self.network_id:
                raise ValueError("path: endpoints must belong to its network")
        if not isinstance(self.profile, MovementProfile):
            raise ValueError("path.profile: expected MovementProfile")
        if not isinstance(self.legs, (tuple, list)) or not all(
                isinstance(leg, RoadLeg) for leg in self.legs):
            raise ValueError("path.legs: expected RoadLeg records")
        object.__setattr__(self, "legs", tuple(self.legs))
        try:
            _number(self.duration_s, "path.duration_s", nonnegative=True)
            _number(self.length_m, "path.length_m", nonnegative=True)
        except OverflowError as error:
            raise ValueError("path totals exceed the finite numeric range") from error

    @cached_property
    def duration_s(self) -> float:
        return math.fsum(leg.duration_s for leg in self.legs)

    @cached_property
    def length_m(self) -> float:
        return math.fsum(leg.length_m for leg in self.legs)


@dataclass(frozen=True, kw_only=True)
class RoadMatch:


    position: RoadPosition
    distance_m: float
    input_xy: tuple[float, float]
    snapped_xy: tuple[float, float]

    def __post_init__(self):
        if not isinstance(self.position, RoadPosition):
            raise ValueError("match.position: expected RoadPosition")
        object.__setattr__(self, "distance_m",
                           _number(self.distance_m, "match.distance_m", nonnegative=True))
        object.__setattr__(self, "input_xy", _xy(self.input_xy, "match.input_xy"))
        object.__setattr__(self, "snapped_xy", _xy(self.snapped_xy, "match.snapped_xy"))
