from fractions import Fraction
from types import MappingProxyType


class ActivityDomain:


    def __init__(self, roads, domain=None):
        self.roads = roads
        if isinstance(domain, ActivityDomain):
            if (domain.roads.network_id, domain.roads.snapshot_id) != (roads.network_id, roads.snapshot_id):
                raise ValueError('activity domain belongs to another road snapshot')
            self.all_road_positions = domain.all_road_positions
            self.node_ids, self._intervals = domain.node_ids, domain._intervals
            self.cache_key, self.is_stationary = domain.cache_key, domain.is_stationary
            return
        if domain is not None:
            domain.validate_roads(roads)
        self.all_road_positions = domain is None or domain.all_road_positions
        if self.all_road_positions:

            self.node_ids = frozenset()
            self._intervals = MappingProxyType({})
            self.cache_key, self.is_stationary = (True,), False
            return
        nodes = set(domain.node_ids)
        intervals = {}
        for portion in domain.segment_portions:
            segment = roads.segment_map[portion.segment_id]
            lo, hi = Fraction(portion.start_m), Fraction(portion.end_m)
            intervals.setdefault(segment.id, []).append((lo, hi))
            if lo == 0:
                nodes.add(segment.u)
            if hi == Fraction(segment.length_m):
                nodes.add(segment.v)


        for segment in roads.segments:
            if segment.u in nodes:
                intervals.setdefault(segment.id, []).append((Fraction(0), Fraction(0)))
            if segment.v in nodes:
                length = Fraction(segment.length_m)
                intervals.setdefault(segment.id, []).append((length, length))
        merged = {}
        for sid, pieces in intervals.items():
            result = []
            for lo, hi in sorted(pieces):
                if result and lo <= result[-1][1]:
                    result[-1] = (result[-1][0], max(hi, result[-1][1]))
                else:
                    result.append((lo, hi))
            merged[sid] = tuple(result)
        self.node_ids = frozenset(nodes)
        self._intervals = MappingProxyType(merged)
        self.cache_key = (False, tuple(sorted(nodes)), tuple(sorted(merged.items())))
        interior_points = {(sid, lo) for sid, pieces in merged.items() for lo, hi in pieces
                           if lo == hi and 0 < lo < Fraction(roads.segment_map[sid].length_m)}
        self.is_stationary = (len(nodes) + len(interior_points) == 1
                              and all(lo == hi for pieces in merged.values() for lo, hi in pieces))

    def merged_intervals(self, segment_id):
        segment = self.roads.segment_map[segment_id]
        if self.all_road_positions:
            return ((Fraction(0), Fraction(segment.length_m)),)
        return self._intervals.get(segment_id, ())

    def contains_span(self, segment_id, start, end):
        if self.all_road_positions:


            return 0 <= min(start, end) and max(start, end) <= self.roads.segment_map[segment_id].length_m
        lo, hi = sorted((Fraction(start), Fraction(end)))
        return any(left <= lo and hi <= right
                   for left, right in self.merged_intervals(segment_id))

    def covers_segment(self, segment_id):
        if self.all_road_positions:
            return True
        return self.contains_span(segment_id, 0, self.roads.segment_map[segment_id].length_m)

    def contains_position(self, position):
        if position.network_id != self.roads.network_id:
            raise ValueError('activity position belongs to another road network')
        if position.node_id is not None:
            if position.node_id not in self.roads.node_map:
                raise ValueError('activity position has an unknown road node')
            return self.all_road_positions or position.node_id in self.node_ids
        if position.segment_id is None or position.offset_m is None:
            raise ValueError('activity position requires a node or segment offset')
        return self.contains_span(position.segment_id, position.offset_m, position.offset_m)
