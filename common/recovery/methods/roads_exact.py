from collections import OrderedDict
from common.config import cache_limits
from dataclasses import dataclass
from fractions import Fraction
import heapq
import math
import sys
from types import MappingProxyType

from common.data.roads.schema import RoadNetwork, RoadPosition
from common.data.roads.activity_domain import ActivityDomain
from common.recovery.inputs import MeetingDomain


def _rational(value):
    if isinstance(value, Fraction):
        return value
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError("expected a finite int, float, or Fraction")
    return Fraction(value)


@dataclass(frozen=True, kw_only=True)
class ExactPosition:


    network_id: str
    node_id: str | None = None
    segment_id: str | None = None
    offset_m: Fraction | None = None


@dataclass(frozen=True)
class Branch:


    slope: Fraction
    intercept: Fraction
    lo: Fraction
    hi: Fraction

    def value(self, offset):

        return self.slope * _rational(offset) + self.intercept


@dataclass
class _SourceSearch:
    distances: dict
    queue: list
    settled: set


@dataclass(frozen=True, eq=False)
class ExactRoadIndex:


    roads: RoadNetwork
    activity_key: tuple
    lengths: object
    scales: object
    arcs: object
    graph: object
    reverse_graph: object
    out_arcs: object
    retained_bytes: int

    @classmethod
    def freeze(cls, router):


        size = 1024
        def freeze_rows(rows, *, weighted=False):
            nonlocal size
            size += sys.getsizeof(rows) + 64
            visits = 0
            for key in rows:
                router.check_budget()
                rows[key] = tuple(rows[key])
                size += sys.getsizeof(rows[key])
                if weighted:
                    for edge in rows[key]:
                        size += sys.getsizeof(edge) + sys.getsizeof(edge[1]) + sys.getsizeof(edge[2])
                        visits += 1
                        if visits % 512 == 0:
                            router.check_budget()
            return MappingProxyType(rows)
        router.check_budget()
        size += sys.getsizeof(router._lengths) + sys.getsizeof(router._scales) + 128
        for length in router._lengths.values():
            router.check_budget()
            size += sys.getsizeof(length) + sys.getsizeof(length.numerator) + sys.getsizeof(length.denominator)
        size += sum(sys.getsizeof(x) for x in router._scales.values())
        arcs, graph = freeze_rows(router._arcs), freeze_rows(router._graph, weighted=True)
        reverse, out = freeze_rows(router._reverse_graph, weighted=True), freeze_rows(router._out_arcs)
        index = cls(router.roads,
                    (True,) if router.activity is None else router.activity.cache_key,
                    MappingProxyType(router._lengths), MappingProxyType(router._scales),
                    arcs, graph, reverse, out, size)
        router.check_budget()
        index.attach(router)
        return index

    def attach(self, router):
        key = (True,) if router.activity is None else router.activity.cache_key
        if self.roads is not router.roads or self.activity_key != key:
            raise ValueError('immutable exact-road index belongs to another road/activity snapshot')
        router._lengths, router._scales = self.lengths, self.scales
        router._arcs, router._graph = self.arcs, self.graph
        router._reverse_graph, router._out_arcs = self.reverse_graph, self.out_arcs


class ExactRoads:


    def __init__(self, roads: RoadNetwork, check_budget=lambda: None, *, source_cache_size=None,
                 cache_units=True, activity_domain=None, immutable_index=None):
        if not isinstance(roads, RoadNetwork):
            raise TypeError("roads must be a RoadNetwork")
        if source_cache_size is None:
            source_cache_size = cache_limits("exact_roads")["source_entries"]
        if type(source_cache_size) is not int or source_cache_size < 0:
            raise ValueError("source_cache_size must be a nonnegative integer")
        if type(cache_units) is not bool:
            raise ValueError("cache_units must be a boolean")
        self.roads = roads
        self.activity = (None if activity_domain is None or activity_domain.all_road_positions
                         else ActivityDomain(roads, activity_domain))
        self.check_budget = check_budget
        self.source_cache_size = source_cache_size
        self.cache_units = cache_units

        self._time_units = {}
        self._distance_unit = Fraction(1)
        self._sources = OrderedDict()
        self.source_stats = dict(hits=0, misses=0, evictions=0, bypasses=0, bypass_reasons={}, invalidations=0)
        if immutable_index is not None:
            self.check_budget()
            if not isinstance(immutable_index, ExactRoadIndex):
                raise TypeError('immutable_index must be an ExactRoadIndex')
            immutable_index.attach(self)
            self.check_budget()
            return
        self._lengths = {}
        self._scales = {'time': 1, 'distance': 1}
        self._arcs, self._graph, self._reverse_graph, self._out_arcs = {}, {}, {}, {}
        for node in roads.nodes:
            self.check_budget()
            self._graph[node.id] = []
            self._reverse_graph[node.id] = []
        for segment in roads.segments:
            self.check_budget()
            self._lengths[segment.id] = _rational(segment.length_m)
            self._scales['distance'] = max(self._scales['distance'], self._lengths[segment.id].denominator)
            self._arcs[segment.id] = []


        for arc in roads.arcs:
            self.check_budget()
            if arc.enabled:
                self._scales['time'] = max(self._scales['time'], arc.travel_time_s.as_integer_ratio()[1])
        for arc in sorted(roads.arcs, key=lambda record: record.id):
            self.check_budget()
            if arc.enabled:
                segment = roads.segment_map[arc.segment_id]
                start, end = (segment.u, segment.v) if arc.forward else (segment.v, segment.u)
                self._arcs[segment.id].append(arc)
                self._out_arcs.setdefault(start, []).append(arc)


                if self.activity is not None and not self.activity.covers_segment(segment.id):
                    continue
                numerator, denominator = arc.travel_time_s.as_integer_ratio()
                time = numerator * (self._scales['time'] // denominator)
                length = self._lengths[segment.id]
                distance = length.numerator * (self._scales['distance'] // length.denominator)
                self._graph[start].append((end, time, distance))
                self._reverse_graph[end].append((start, time, distance))

    def activity_contains(self, position):
        return self.activity is None or self.activity.contains_position(position)

    def allowed_intervals(self, segment_id):

        if self.activity is None:
            return ((Fraction(0), self._lengths[segment_id]),)
        return self.activity.merged_intervals(segment_id)

    def endpoint_interval(self, segment_id, endpoint):

        for low, high in self.allowed_intervals(segment_id):
            if low <= endpoint <= high:
                return low, high
        return None

    def allowed_span(self, segment_id, start, end):
        return self.activity is None or self.activity.contains_span(segment_id, start, end)

    def position(self, segment_id, offset):
        return self.normalize(ExactPosition(network_id=self.roads.network_id,
                                            segment_id=segment_id, offset_m=_rational(offset)))

    def normalize(self, position):
        if not isinstance(position, (RoadPosition, ExactPosition)):
            raise TypeError("expected RoadPosition or ExactPosition")
        if position.network_id != self.roads.network_id:
            raise ValueError("position belongs to a different physical road network")
        if (position.node_id is None) == (position.segment_id is None):
            raise ValueError("position must specify exactly one node or segment")
        if position.node_id is not None:
            if position.node_id not in self.roads.node_map or position.offset_m is not None:
                raise ValueError("invalid node position")
            return ExactPosition(network_id=self.roads.network_id, node_id=position.node_id)
        segment = self.roads.segment_map.get(position.segment_id)
        if segment is None:
            raise ValueError("unknown segment")
        offset = _rational(position.offset_m)
        length = self._lengths[segment.id]
        if not 0 <= offset <= length:
            raise ValueError("offset lies outside the segment")
        if offset == 0 or offset == length:
            return ExactPosition(network_id=self.roads.network_id,
                                 node_id=segment.u if offset == 0 else segment.v)
        return ExactPosition(network_id=self.roads.network_id, segment_id=segment.id, offset_m=offset)

    @staticmethod
    def _settings(metric, speed):
        if metric not in ("time", "distance"):
            raise ValueError("metric must be 'time' or 'distance'")
        speed = _rational(speed)
        if speed <= 0:
            raise ValueError("speed must be positive")
        return speed if metric == "time" else Fraction(1)

    def _node_cost(self, origin, destination, metric, *, reverse=False):
        self.check_budget()
        if self.activity is not None and (not self.activity_contains(self.roads.node_position(origin))
                or not self.activity_contains(self.roads.node_position(destination))):
            return None
        source, target = (destination, origin) if reverse else (origin, destination)
        graph = self._reverse_graph if reverse else self._graph
        key = source, metric, reverse


        state = self._sources.pop(key, None)
        if state is None:
            self.source_stats['misses'] += 1
            state = _SourceSearch({source: 0}, [(0, source)], set())
        else:
            self.source_stats['hits'] += 1
        try:
            while target not in state.settled and state.queue:
                self.check_budget()
                cost, node = heapq.heappop(state.queue)
                if node in state.settled or state.distances[node] != cost:
                    continue
                state.settled.add(node)
                for neighbor, time, distance in graph[node]:
                    self.check_budget()
                    candidate = cost + (time if metric == "time" else distance)
                    if neighbor not in state.distances or candidate < state.distances[neighbor]:
                        state.distances[neighbor] = candidate
                        heapq.heappush(state.queue, (candidate, neighbor))


                if node == target:
                    break
        except BaseException:


            self.source_stats['invalidations'] += int(bool(self.source_cache_size))
            raise
        if self.source_cache_size:
            self._sources[key] = state
            if len(self._sources) > self.source_cache_size:
                self._sources.popitem(last=False)
                self.source_stats['evictions'] += 1
        else:
            from common.cache_telemetry import bypass
            bypass(self.source_stats, 'disabled')
        return (Fraction(state.distances[target], self._scales[metric])
                if target in state.settled else None)

    def _unit(self, arc, metric):
        if metric != "time":
            return self._distance_unit if self.cache_units else Fraction(1)
        if self.cache_units:
            unit = self._time_units.get(arc)
            if unit is not None:
                return unit
        unit = _rational(arc.travel_time_s) / self._lengths[arc.segment_id]
        if self.cache_units:
            self._time_units[arc] = unit
        return unit

    def _connectors(self, position, *, leaving, metric):
        if not self.activity_contains(position):
            return ()
        if position.node_id is not None:
            return ((position.node_id, Fraction(0)),)
        segment = self.roads.segment_map[position.segment_id]
        length, offset = self._lengths[segment.id], position.offset_m
        result = []
        for arc in self._arcs[segment.id]:
            self.check_budget()
            if leaving:
                node, span = (segment.v, length - offset) if arc.forward else (segment.u, offset)
            else:
                node, span = (segment.u, offset) if arc.forward else (segment.v, length - offset)
            endpoint = length if arc.forward == leaving else Fraction(0)
            if not self.allowed_span(segment.id, offset, endpoint):
                continue
            result.append((node, span * self._unit(arc, metric)))
        return tuple(result)

    def cost(self, origin, destination, metric="time", speed=1, *, reverse=False):


        self.check_budget()
        if type(reverse) is not bool:
            raise ValueError("reverse must be a boolean")
        scale = self._settings(metric, speed)
        origin, destination = self.normalize(origin), self.normalize(destination)
        if not self.activity_contains(origin) or not self.activity_contains(destination):
            return None
        if origin == destination:
            return Fraction(0)
        best = None
        if origin.segment_id is not None and origin.segment_id == destination.segment_id:
            delta = destination.offset_m - origin.offset_m
            for arc in self._arcs[origin.segment_id]:
                self.check_budget()
                if ((arc.forward and delta > 0) or (not arc.forward and delta < 0)) and \
                        self.allowed_span(origin.segment_id, origin.offset_m, destination.offset_m):
                    candidate = abs(delta) * self._unit(arc, metric)
                    best = candidate if best is None else min(best, candidate)
        for start, first in self._connectors(origin, leaving=True, metric=metric):
            for end, last in self._connectors(destination, leaving=False, metric=metric):
                self.check_budget()
                middle = self._node_cost(start, end, metric, reverse=reverse)
                if middle is not None:
                    candidate = first + middle + last
                    best = candidate if best is None else min(best, candidate)
        return None if best is None else best / scale

    def affine_branches(self, fixed, segment_id, incoming: bool, metric="time", speed=1,
                        *, endpoint_costs=None, cache_statistics=None):


        self.check_budget()
        scale = self._settings(metric, speed)
        if type(incoming) is not bool:
            raise ValueError("incoming must be a boolean")
        fixed = self.normalize(fixed)
        if not self.activity_contains(fixed):
            return []
        segment = self.roads.segment_map[segment_id]
        length, zero = self._lengths[segment_id], Fraction(0)
        result = []
        for arc in self._arcs[segment_id]:
            self.check_budget()
            unit = self._unit(arc, metric) / scale
            if incoming:
                endpoint = self.roads.node_position(segment.u if arc.forward else segment.v)
                slope, intercept = (unit, zero) if arc.forward else (-unit, unit * length)
            else:
                endpoint = self.roads.node_position(segment.v if arc.forward else segment.u)
                slope, intercept = (-unit, unit * length) if arc.forward else (unit, zero)
            key = (self, fixed, endpoint.node_id, incoming, metric, scale) if endpoint_costs is not None else None
            if endpoint_costs is not None and key in endpoint_costs:
                base = endpoint_costs[key]
                if cache_statistics is not None:
                    cache_statistics['endpoint_cost_hits'] = cache_statistics.get('endpoint_cost_hits', 0)+1
            else:
                base = (self.cost(fixed, endpoint, metric, speed) if incoming else
                        self.cost(endpoint, fixed, metric, speed, reverse=True))
                if endpoint_costs is not None:

                    endpoint_costs[key] = base
                    if cache_statistics is not None:
                        cache_statistics['endpoint_cost_misses'] = cache_statistics.get('endpoint_cost_misses', 0)+1
                        cache_statistics['endpoint_cache_peak_entries'] = max(
                            cache_statistics.get('endpoint_cache_peak_entries', 0), len(endpoint_costs))
            if base is not None:
                endpoint_offset = zero if arc.forward == incoming else length
                interval = self.endpoint_interval(segment_id, endpoint_offset)
                if interval is not None:
                    result.append(Branch(slope, base + intercept, *interval))
            if fixed.segment_id == segment_id:
                offset = fixed.offset_m
                increasing = arc.forward == incoming
                component = self.endpoint_interval(segment_id, offset)
                if component is not None:
                    low, high = component
                    if increasing:
                        result.append(Branch(unit, -unit * offset, offset, high))
                    else:
                        result.append(Branch(-unit, unit * offset, low, offset))
        return result

    def _validate_domain(self, domain):
        if not isinstance(domain, MeetingDomain):
            raise TypeError("domain must be a MeetingDomain")
        self.check_budget()
        for node_id in domain.node_ids:
            self.check_budget()
            if node_id not in self.roads.node_map:
                raise ValueError("meeting domain contains an unknown node")
        for portion in domain.segment_portions:
            self.check_budget()
            segment = self.roads.segment_map.get(portion.segment_id)
            if segment is None or portion.end_m > segment.length_m:
                raise ValueError("meeting domain contains an invalid segment portion")

    def contains(self, domain: MeetingDomain, position):

        self._validate_domain(domain)
        position = self.normalize(position)
        if domain.all_road_positions:
            return True
        if position.node_id is not None:
            if position.node_id in domain.node_ids:
                return True
            for portion in domain.segment_portions:
                self.check_budget()
                segment = self.roads.segment_map[portion.segment_id]
                if ((portion.start_m == 0 and position.node_id == segment.u)
                        or (portion.end_m == segment.length_m and position.node_id == segment.v)):
                    return True
        else:
            for portion in domain.segment_portions:
                self.check_budget()
                if (portion.segment_id == position.segment_id
                        and _rational(portion.start_m) <= position.offset_m <= _rational(portion.end_m)):
                    return True
        return False

    def reachable_parts(self, origin, radius):


        self.check_budget()
        origin, radius = self.normalize(origin), _rational(radius)
        if radius < 0:
            raise ValueError("radius must be nonnegative")
        if not self.activity_contains(origin):
            return (), {}
        points, portions = {origin}, {}
        if radius == 0:
            return (origin,), {}

        def add_portion(segment_id, low, high):
            self.check_budget()
            points.add(self.position(segment_id, low))
            points.add(self.position(segment_id, high))
            if low < high:
                portions.setdefault(segment_id, []).append((low, high))

        if origin.segment_id is not None:
            length, offset = self._lengths[origin.segment_id], origin.offset_m
            component = self.endpoint_interval(origin.segment_id, offset)
            for arc in self._arcs[origin.segment_id]:
                self.check_budget()
                low, high = ((offset, min(component[1], offset + radius)) if arc.forward
                             else (max(component[0], offset - radius), offset))
                add_portion(origin.segment_id, low, high)
        distances, queue, settled = {}, [], set()
        for node, distance in self._connectors(origin, leaving=True, metric="distance"):
            self.check_budget()
            if distance <= radius and (node not in distances or distance < distances[node]):
                distances[node] = distance
                heapq.heappush(queue, (distance, node))
        while queue:
            self.check_budget()
            distance, node = heapq.heappop(queue)
            if node in settled or distance != distances[node]:
                continue
            settled.add(node)
            points.add(ExactPosition(network_id=self.roads.network_id, node_id=node))
            remaining = radius - distance
            for arc in self._out_arcs.get(node, ()):
                self.check_budget()
                segment = self.roads.segment_map[arc.segment_id]
                length = self._lengths[segment.id]
                component = self.endpoint_interval(segment.id, Fraction(0) if arc.forward else length)
                if component is None:
                    continue
                reachable = component[1] if arc.forward else length - component[0]
                span = min(length, remaining, reachable)
                low, high = ((Fraction(0), span) if arc.forward else (length - span, length))
                add_portion(segment.id, low, high)
                following = segment.v if arc.forward else segment.u
                candidate = distance + length
                if reachable == length and candidate <= radius and (following not in distances or candidate < distances[following]):
                    distances[following] = candidate
                    heapq.heappush(queue, (candidate, following))
        intervals = self._merge_portions(portions)
        return tuple(sorted(points, key=lambda p: (p.node_id or '', p.segment_id or '',
                                                  p.offset_m or Fraction(0)))), intervals

    def _merge_portions(self, portions):
        intervals = {}
        for segment_id in sorted(portions):
            self.check_budget()
            merged = []
            for lo, hi in sorted(portions[segment_id]):
                self.check_budget()
                if merged and lo <= merged[-1][1]:
                    merged[-1] = merged[-1][0], max(merged[-1][1], hi)
                else:
                    merged.append((lo, hi))
            intervals[segment_id] = tuple(merged)
        return intervals

    def domain_parts(self, domain: MeetingDomain):

        self._validate_domain(domain)
        points, portions = {}, {}

        def add_point(segment_id, offset):
            point = self.roads.segment_position(segment_id, offset)
            points[(point.node_id, point.segment_id, point.offset_m)] = point

        node_ids = self.roads.node_map if domain.all_road_positions else domain.node_ids
        for node_id in sorted(node_ids):
            self.check_budget()
            point = self.roads.node_position(node_id)
            points[(point.node_id, None, None)] = point
        if domain.all_road_positions:
            for segment in self.roads.segments:
                self.check_budget()
                portions[segment.id] = [(Fraction(0), self._lengths[segment.id])]
        else:
            for portion in domain.segment_portions:
                self.check_budget()
                add_point(portion.segment_id, portion.start_m)
                add_point(portion.segment_id, portion.end_m)
                if portion.start_m < portion.end_m:
                    portions.setdefault(portion.segment_id, []).append(
                        (_rational(portion.start_m), _rational(portion.end_m)))
        return tuple(points.values()), self._merge_portions(portions)
