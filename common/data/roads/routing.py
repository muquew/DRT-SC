from common.cache_telemetry import observed_lru_cache as lru_cache, retention_counts, bypass
from fractions import Fraction
import math
import sys

import networkx as nx

from common.data.roads.schema import MovementProfile, RoadLeg, RoadMatch, RoadNetwork, RoadPath
from common.data.roads.activity_domain import ActivityDomain
from common.config import cache_record


def _finite(value, label, *, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{label}: expected a finite number >= {minimum}")


class _CostOverflow(ValueError):
    pass


class _CostUnderflow(ValueError):
    pass


def _duration(length, segment_length, segment_time, speed_factor=1.0):
    ratio = length / segment_length
    product = ratio * segment_time
    value = product / speed_factor
    if (not math.isfinite(value) or value == 0
            or 0 < ratio < sys.float_info.min or 0 < product < sys.float_info.min
            or 0 < value < sys.float_info.min):

        try:
            value = float(Fraction(length) * Fraction(segment_time)
                          / (Fraction(segment_length) * Fraction(speed_factor)))
        except OverflowError as error:
            raise _CostOverflow("travel duration exceeds the finite numeric range") from error
    if not math.isfinite(value):
        raise _CostOverflow("travel duration exceeds the finite numeric range")
    if value <= 0:
        raise _CostUnderflow("positive travel duration is below the finite numeric range")
    return value


class Routing:


    def __init__(self, roads: RoadNetwork, *, cache_size: int | None = None, source_cache_size: int | None = None,
                 source_cache_node_limit: int | None = None, source_cache_byte_limit: int | None = None,
                 search_method: str = "dijkstra",
                 activity_domain=None, cache_profile='compat'):
        self.cache_configuration = cache_record('routing_' + cache_profile, dict(
            cache_size=cache_size, source_cache_size=source_cache_size,
            source_cache_node_limit=source_cache_node_limit, source_cache_byte_limit=source_cache_byte_limit))
        limits = self.cache_configuration['effective']
        cache_size, source_cache_size, source_cache_node_limit = (
            limits['cache_size'], limits['source_cache_size'], limits['source_cache_node_limit'])
        if not isinstance(roads, RoadNetwork):
            raise ValueError("roads: expected RoadNetwork")
        if type(cache_size) is not int or cache_size < 0:
            raise ValueError("cache_size: expected a nonnegative integer")
        if type(source_cache_size) is not int or source_cache_size < 0:
            raise ValueError("source_cache_size: expected a nonnegative integer")
        if type(source_cache_node_limit) is not int or source_cache_node_limit < 0:
            raise ValueError("source_cache_node_limit: expected a nonnegative integer")
        source_cache_byte_limit = limits['source_cache_byte_limit']
        if search_method not in ("dijkstra", "scipy"):
            raise ValueError("search_method: expected 'dijkstra' or 'scipy'")
        self.search_method = search_method
        self.activity_domain = ActivityDomain(roads, activity_domain)
        self._cache_size = cache_size
        self._domain_routers = {}
        self._domain_declarations = {}
        self.requested_source_cache_size = source_cache_size
        self.source_cache_node_limit = source_cache_node_limit
        self.source_cache_byte_limit = source_cache_byte_limit


        self.source_cache_size = (source_cache_size if search_method == 'scipy' else
            min(source_cache_size, source_cache_node_limit // max(1, len(roads.nodes))))
        self._roads = roads
        self._crs = None
        if not roads.crs.startswith("local:"):
            from pyproj import CRS
            from pyproj.exceptions import ProjError
            try:
                self._crs = CRS.from_user_input(roads.crs)
            except ProjError as error:
                raise ValueError(f"roads.crs: {error}") from error
            if (not self._crs.is_projected or len(self._crs.axis_info) != 2
                    or any(not math.isclose(axis.unit_conversion_factor, 1.0)
                           for axis in self._crs.axis_info)):
                raise ValueError("roads.crs: expected a projected coordinate reference in metres")
        self._graph = nx.MultiDiGraph()
        self._graph.add_nodes_from(sorted(roads.node_map))
        self._segment_arcs = {key: [] for key in roads.segment_map}
        uniform, uniform_possible = None, True
        for arc in sorted(roads.arcs, key=lambda a: a.id):
            if not arc.enabled:
                continue
            segment = roads.segment_map[arc.segment_id]


            self._segment_arcs[segment.id].append(arc)
            if not self.activity_domain.covers_segment(segment.id):
                continue
            u, v = (segment.u, segment.v) if arc.forward else (segment.v, segment.u)
            self._graph.add_edge(u, v, key=arc.id, travel_time_s=arc.travel_time_s,
                                 length_m=segment.length_m)
            if uniform_possible:
                values = segment.length_m, arc.travel_time_s
                if uniform is None:
                    uniform = values
                elif values != uniform:
                    uniform_possible = False
        self._uniform_step = (tuple(map(int, uniform)) if uniform_possible and uniform
                              and all(v.is_integer() for v in uniform)
                              and len(roads.nodes) * max(map(int, uniform)) <= 2**53 else None)
        nx.freeze(self._graph)
        if search_method == "scipy":
            self._initialize_scipy()
        self._source_tree = (lru_cache(maxsize=self.source_cache_size)(self._uncached_source_tree)
                             if self.source_cache_size and search_method != 'scipy' else None)
        self._query_targets = {metric:set() for metric in ('travel_time_s','length_m')}
        self._query_registration_bypasses = 0
        self._scipy_sources = None
        if search_method == 'scipy':
            from common.data.roads.query_cache import SourceQueryCache
            self._scipy_sources = SourceQueryCache(self,self.source_cache_size,
                source_cache_node_limit,source_cache_byte_limit)
        self._source_bypasses = retention_counts()
        self._node_cost = lru_cache(maxsize=cache_size)(self._uncached_node_cost)
        self._node_path = lru_cache(maxsize=cache_size)(self._uncached_node_path)
        self._condensation = None
        self._node_reachable = lru_cache(maxsize=cache_size)(self._uncached_node_reachable)
        self._spatial_index = None
        self._transforms = {}

    @property
    def cache_stats(self):


        from types import MappingProxyType
        result = {"pair_max_entries_each": self._cache_size,
                  "search_method": self.search_method,
                  "source_requested_entries": self.requested_source_cache_size,
                  "source_max_entries": self.source_cache_size,
                  "source_node_limit": self.source_cache_node_limit,
                  "source_byte_limit": self.source_cache_byte_limit,
                  "road_nodes": len(self.roads.nodes),
                  "domain_router_count": len(self._domain_routers)}
        for name, cache in (("node_cost", self._node_cost), ("node_path", self._node_path),
                            ("node_reachable", self._node_reachable)):
            info = cache.cache_info()
            result.update({name + "_" + k: getattr(info, k)
                           for k in ("hits", "misses", "currsize", "maxsize")})
            result.update({name+'_'+k:v for k,v in cache.retention_stats().items()})
        source = self._scipy_sources if self._scipy_sources is not None else self._source_tree
        if source is not None:
            info = source.cache_info()
            result.update({"source_" + k: getattr(info, k)
                           for k in ("hits", "misses", "currsize", "maxsize")})
            result.update({'source_'+k:v for k,v in source.retention_stats().items()})
        else:
            result.update(source_hits=0,source_misses=self._source_bypasses['bypasses'],source_currsize=0,source_maxsize=0)
            result.update({'source_'+k:v for k,v in self._source_bypasses.items()})
        result['source_capacity_nodes']=self.source_cache_node_limit
        if self._scipy_sources is not None:
            result.update({'source_'+k:v for k,v in self._scipy_sources.snapshot().items()})
            result.update(registered_query_targets=sum(map(len,self._query_targets.values())),
                registered_time_targets=len(self._query_targets['travel_time_s']),
                registered_length_targets=len(self._query_targets['length_m']),
                query_target_registration_bypasses=self._query_registration_bypasses,
                query_target_registry_bytes=sum(map(sys.getsizeof,self._query_targets.values())),
                source_retention='forward distances at registered targets; original time predecessor ancestor closure')
        result['source_bypass_unit']='source-cache query omitted; counts at query choice, not again inside Dijkstra'
        return MappingProxyType(result)

    def _record_source_bypass(self):
        reason = 'disabled' if not self.requested_source_cache_size or not self.source_cache_node_limit else 'oversized_nodes'
        bypass(self._source_bypasses, reason)

    @property
    def roads(self):
        return self._roads

    def register_query_nodes(self, node_ids, *, metrics=('travel_time_s','length_m')):


        metrics = tuple(metrics)
        if not metrics or set(metrics)-set(self._query_targets):
            raise ValueError('query registration metric must be travel_time_s or length_m')
        for node in node_ids:
            if node not in self.roads.node_map:
                raise ValueError('query target is not a node on this road network')
            if self.search_method != 'scipy':
                continue
            index = self._scipy_index[node]
            for metric in metrics:
                targets = self._query_targets[metric]
                if index in targets:
                    continue
                if len(targets) >= self.source_cache_node_limit:
                    self._query_registration_bypasses += 1
                    continue
                targets.add(index)
        return self

    def register_query_positions(self, positions, *, metrics=('travel_time_s','length_m')):

        nodes = set()
        for position in positions:
            position = self.roads.normalize_position(position)
            for leaving in (True, False):
                nodes.update(node for node, _ in self._connectors(position, leaving=leaving))
        return self.register_query_nodes(sorted(nodes),metrics=metrics)

    def register_query_input(self, item, *, plans=()):

        positions = [p for task in item.tasks for p in (task.pickup,task.delivery) if p is not None]
        positions.extend(worker.location for worker in item.workers)
        for plan in plans:
            positions.extend(event.location for event in plan.transfers)
        return self.register_query_positions(positions)

    def fork_query_session(self, *, inherit_queries=True):


        memo = {}
        def fork(parent):
            if id(parent) in memo:
                return memo[id(parent)]
            result = object.__new__(type(parent))
            result.__dict__ = parent.__dict__.copy()
            memo[id(parent)] = result
            result._query_targets = {metric:set(values) for metric,values in parent._query_targets.items()}
            result._query_registration_bypasses = 0
            result._source_bypasses = retention_counts()
            result._source_tree = (lru_cache(maxsize=result.source_cache_size)(result._uncached_source_tree)
                if result.search_method != 'scipy' and result.source_cache_size else None)
            result._scipy_sources = (parent._scipy_sources.fork(result,inherit=inherit_queries)
                if parent._scipy_sources is not None else None)
            for name in ('_node_cost','_node_path','_node_reachable'):
                setattr(result,name,lru_cache(maxsize=result._cache_size)(getattr(result,'_uncached'+name)))
            result._domain_routers = {key:fork(value) for key,value in parent._domain_routers.items()}
            result._domain_declarations = {key:fork(value) for key,value in parent._domain_declarations.items()}
            result._transforms = dict(parent._transforms)

            if isinstance(getattr(parent,'counters',None),dict):
                result.counters = {key:0 for key in parent.counters}
            return result
        return fork(self)

    def for_activity_domain(self, domain=None):


        if isinstance(domain, ActivityDomain):
            if (domain.roads.network_id, domain.roads.snapshot_id) != (self.roads.network_id, self.roads.snapshot_id):
                raise ValueError('activity domain belongs to another road snapshot')
            declaration_key = domain.cache_key
        else:
            declaration_key = domain
        cached = self._domain_declarations.get(declaration_key)
        if cached is not None:
            return cached
        compiled = ActivityDomain(self.roads, domain)
        if compiled.cache_key == self.activity_domain.cache_key:
            result = self
        else:
            result = self._domain_routers.get(compiled.cache_key)
            if result is None:
                result = Routing(self.roads, cache_size=self._cache_size,
                    source_cache_size=self.requested_source_cache_size,
                    source_cache_node_limit=self.source_cache_node_limit,
                    source_cache_byte_limit=self.source_cache_byte_limit,
                    search_method=self.search_method, activity_domain=compiled)
                if self.search_method == 'scipy':
                    for metric,targets in self._query_targets.items():
                        result.register_query_nodes((self._scipy_nodes[index] for index in targets),metrics=(metric,))
                self._domain_routers[compiled.cache_key] = result
                result.cache_configuration = self.cache_configuration
                if self.activity_domain.all_road_positions:
                    result._domain_declarations[None] = self
        self._domain_declarations[declaration_key] = result
        return result

    @staticmethod
    def _exact_time_weight(_origin, _destination, parallel):
        return min(Fraction(arc["travel_time_s"]) for arc in parallel.values())

    def _initialize_scipy(self):

        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

        self._scipy_dijkstra = dijkstra
        self._scipy_nodes = tuple(sorted(self.roads.node_map))
        self._scipy_index = {node: i for i, node in enumerate(self._scipy_nodes)}
        rows, columns = [], []
        costs = {"travel_time_s": [], "length_m": []}
        for u in self._scipy_nodes:
            for v in sorted(self._graph[u]):
                rows.append(self._scipy_index[u])
                columns.append(self._scipy_index[v])
                for metric, values in costs.items():
                    values.append(min(edge[metric] for edge in self._graph[u][v].values()))
        size = len(self._scipy_nodes)
        self._scipy_graphs = {metric: csr_matrix((values, (rows, columns)), shape=(size, size), dtype="float64")
                              for metric, values in costs.items()}

    def _scipy_lookup(self, origin, destination, metric, *, need_path=False):
        cost, predecessors = self._scipy_sources.lookup(origin, destination, metric,need_path=need_path)


        if not math.isfinite(cost) and self._node_reachable(origin, destination):
            raise _CostOverflow("path cost overflow; this is not an unreachable route")
        return cost, predecessors

    def _uncached_node_cost(self, origin, destination, metric, *, exact=False):
        if exact:
            try:
                return nx.dijkstra_path_length(self._graph, origin, destination,
                                               weight=self._exact_time_weight)
            except nx.NetworkXNoPath:
                return math.inf
        if self.search_method == "scipy":
            return self._scipy_lookup(origin, destination, metric)[0]
        if self._source_tree is not None:
            distances, _ = self._source_tree(origin, metric)
            if destination not in distances:
                return math.inf
            cost = distances[destination]
            if not math.isfinite(cost):
                raise _CostOverflow("path cost overflow; this is not an unreachable route")
            return cost
        self._record_source_bypass()
        try:
            cost = nx.dijkstra_path_length(self._graph, origin, destination, weight=metric)
        except nx.NetworkXNoPath:
            return math.inf
        if not math.isfinite(cost):
            raise _CostOverflow("path cost overflow; this is not an unreachable route")
        return cost

    def _uncached_node_path(self, origin, destination, metric, *, exact=False):
        if self.search_method == "scipy" and not exact:
            if metric == 'travel_time_s':
                cost, predecessors = self._scipy_lookup(origin, destination, metric,need_path=True)
            else:


                distances, predecessors = self._scipy_dijkstra(self._scipy_graphs[metric],
                    directed=True,indices=self._scipy_index[origin],return_predecessors=True)
                cost = float(distances[self._scipy_index[destination]])
            if cost == math.inf:
                raise nx.NetworkXNoPath(f"no path from {origin} to {destination}")
            source, current = self._scipy_index[origin], self._scipy_index[destination]
            nodes = [destination]
            while current != source:
                current = int(predecessors[current])
                if current < 0 or len(nodes) >= len(self._scipy_nodes):
                    raise ValueError("invalid SciPy predecessor chain")
                nodes.append(self._scipy_nodes[current])
            nodes.reverse()
        elif exact or self._source_tree is None:
            if not exact:self._record_source_bypass()
            nodes = nx.dijkstra_path(self._graph, origin, destination,
                                     weight=self._exact_time_weight if exact else metric)
        else:
            distances, parents = self._source_tree(origin, metric)
            if destination not in distances:
                raise nx.NetworkXNoPath(f"no path from {origin} to {destination}")
            nodes = [destination]
            while nodes[-1] != origin:
                nodes.append(parents[nodes[-1]])
            nodes.reverse()
        return tuple(min(self._graph[u][v], key=lambda key: (self._graph[u][v][key][metric], key))
                     for u, v in zip(nodes, nodes[1:]))

    def _uncached_source_tree(self, origin, metric):


        if self.search_method == "scipy":
            result = self._scipy_dijkstra(self._scipy_graphs[metric], directed=True,
                indices=self._scipy_index[origin], return_predecessors=(metric == 'travel_time_s'))
            return result if metric == 'travel_time_s' else (result,None)
        predecessors, distances = nx.dijkstra_predecessor_and_distance(self._graph, origin, weight=metric)
        return distances, {node: previous[0] for node, previous in predecessors.items() if previous}

    def _uncached_node_reachable(self, origin, destination):
        if self._condensation is None:
            self._condensation = nx.condensation(self._graph)
        components = self._condensation.graph["mapping"]
        source, target = components[origin], components[destination]
        return source == target or nx.has_path(self._condensation, source, target)

    def _part(self, arc, start, end, profile):
        length = abs(end - start)
        if length == 0:
            return None
        segment = self.roads.segment_map[arc.segment_id]
        duration = _duration(length, segment.length_m, arc.travel_time_s, profile.speed_factor)
        return RoadLeg(arc_id=arc.id, segment_id=segment.id, start_offset_m=start,
                       end_offset_m=end, length_m=length, duration_s=duration)

    def _connectors(self, position, *, leaving):
        if not self.activity_domain.contains_position(position):
            return []
        if position.node_id is not None:
            return [(position.node_id, None)]
        segment = self.roads.segment_map[position.segment_id]
        entries = []
        for arc in self._segment_arcs[segment.id]:
            start, end = (0.0, segment.length_m) if arc.forward else (segment.length_m, 0.0)
            if leaving:
                node = segment.v if arc.forward else segment.u
                piece = (arc, position.offset_m, end)
            else:
                node = segment.u if arc.forward else segment.v
                piece = (arc, start, position.offset_m)
            if self.activity_domain.contains_span(segment.id, piece[1], piece[2]):
                entries.append((node, piece))
        return entries

    def _piece_cost(self, piece, metric, *, exact=False):
        if piece is None:
            return 0
        arc, start, end = piece
        length = abs(end - start)
        if exact:
            return (Fraction(length) * Fraction(arc.travel_time_s)
                    / Fraction(self.roads.segment_map[arc.segment_id].length_m))
        return (length if metric == "length_m" else
                _duration(length, self.roads.segment_map[arc.segment_id].length_m, arc.travel_time_s))

    def _best(self, origin, destination, metric, *, exact=False):

        self.register_query_positions((origin,destination),metrics=(metric,))
        if not (self.activity_domain.contains_position(origin)
                and self.activity_domain.contains_position(destination)):
            return math.inf, None
        if origin == destination:
            return 0.0, (None, None, None)
        best_cost, best = math.inf, None
        overflow = False
        if (origin.segment_id is not None and origin.segment_id == destination.segment_id
                and self.activity_domain.contains_span(origin.segment_id, origin.offset_m, destination.offset_m)):
            for arc in self._segment_arcs[origin.segment_id]:
                delta = destination.offset_m - origin.offset_m
                if (arc.forward and delta > 0) or (not arc.forward and delta < 0):
                    piece = (arc, origin.offset_m, destination.offset_m)
                    cost = self._piece_cost(piece, metric, exact=exact)
                    if cost < best_cost:
                        best_cost, best = cost, (piece, None, None)
        for source, first in self._connectors(origin, leaving=True):
            for target, last in self._connectors(destination, leaving=False):
                try:
                    middle = (self._uncached_node_cost(source, target, metric, exact=True) if exact
                              else self._node_cost(source, target, metric))
                except _CostOverflow:
                    overflow = True
                    continue
                if middle == math.inf:
                    continue
                cost = (self._piece_cost(first, metric, exact=exact) + middle
                        + self._piece_cost(last, metric, exact=exact))
                if not exact and not math.isfinite(cost):
                    overflow = True
                    continue
                if cost < best_cost:
                    best_cost, best = cost, (first, (source, target), last)
        if best is None and overflow:
            raise _CostOverflow("reachable routes exceed the finite numeric range")
        return best_cost, best

    def _best_time(self, origin, destination):
        try:
            cost, best = self._best(origin, destination, "travel_time_s")
            if cost == 0 or cost >= sys.float_info.min:
                return cost, best, False
        except (_CostOverflow, _CostUnderflow):
            pass


        cost, best = self._best(origin, destination, "travel_time_s", exact=True)
        return cost, best, True

    @staticmethod
    def _profile(profile):
        if not isinstance(profile, MovementProfile):
            raise ValueError("profile: expected MovementProfile")
        return profile

    def travel_time(self, origin, destination, profile=MovementProfile()):

        profile = self._profile(profile)
        origin = self.roads.normalize_position(origin)
        destination = self.roads.normalize_position(destination)
        cost, best, exact = self._best_time(origin, destination)
        if best is None:
            return math.inf
        try:
            result = float(cost / Fraction(profile.speed_factor)) if exact else cost / profile.speed_factor
        except OverflowError as error:
            raise _CostOverflow("profile produces an unrepresentable travel duration") from error
        if not math.isfinite(result) or (cost > 0 and result == 0):
            raise ValueError("profile produces an unrepresentable travel duration")
        return result

    def reachable(self, origin, destination):


        origin = self.roads.normalize_position(origin)
        destination = self.roads.normalize_position(destination)
        if not (self.activity_domain.contains_position(origin)
                and self.activity_domain.contains_position(destination)):
            return False
        if origin == destination:
            return True
        if (origin.segment_id is not None and origin.segment_id == destination.segment_id
                and self.activity_domain.contains_span(origin.segment_id, origin.offset_m, destination.offset_m)):
            delta = destination.offset_m - origin.offset_m
            if any((arc.forward and delta > 0) or (not arc.forward and delta < 0)
                   for arc in self._segment_arcs[origin.segment_id]):
                return True
        return any(self._node_reachable(source, target)
                   for source, _ in self._connectors(origin, leaving=True)
                   for target, _ in self._connectors(destination, leaving=False))

    def shortest_distance(self, origin, destination):

        origin = self.roads.normalize_position(origin)
        destination = self.roads.normalize_position(destination)
        cost, _ = self._best(origin, destination, "length_m")
        return cost

    def nodes_within_distance(self, origin, radius_m):


        _finite(radius_m, "radius_m")
        origin = self.roads.normalize_position(origin)
        if not self.activity_domain.contains_position(origin):
            return {}
        if radius_m == 0:
            return {origin.node_id: 0.0} if origin.node_id is not None else {}
        connectors = {}
        for node, piece in self._connectors(origin, leaving=True):
            length = self._piece_cost(piece, "length_m")
            connectors[node] = min(length, connectors.get(node, math.inf))


        ceiling = math.nextafter(radius_m + math.ulp(radius_m), math.inf)
        result = {}
        for source, first in connectors.items():
            if first > ceiling:
                continue
            cutoff = math.nextafter(ceiling - first, math.inf)
            if self.search_method == "scipy":
                costs = self._scipy_dijkstra(self._scipy_graphs["length_m"],
                    directed=True, indices=self._scipy_index[source], limit=cutoff)
                entries = ((self._scipy_nodes[index], float(cost))
                           for index, cost in enumerate(costs) if math.isfinite(cost))
            else:
                entries = nx.single_source_dijkstra_path_length(
                    self._graph, source, cutoff=cutoff, weight="length_m").items()
            for node, middle in entries:
                distance = first + middle
                if (math.isfinite(distance)
                        and distance - radius_m <= math.ulp(max(distance, radius_m))):
                    result[node] = min(distance, result.get(node, math.inf))
        return dict(sorted(result.items()))

    def _route_parts(self, best, exact):

        first, middle, last = best
        if first is not None:
            yield first
        if middle is not None:
            identifiers = (self._uncached_node_path(*middle, "travel_time_s", exact=True) if exact
                           else self._node_path(*middle, "travel_time_s"))
            for identifier in identifiers:
                arc = self.roads.arc_map[identifier]
                length = self.roads.segment_map[arc.segment_id].length_m
                start, end = (0.0, length) if arc.forward else (length, 0.0)
                yield arc, start, end
        if last is not None:
            yield last

    def route_duration(self, origin, destination, profile=MovementProfile()):


        profile = self._profile(profile)
        origin = self.roads.normalize_position(origin)
        destination = self.roads.normalize_position(destination)
        _, best, exact = self._best_time(origin, destination)
        if best is None:
            return math.inf
        lengths, durations = [], []

        def append_part(part):
            arc, start, end = part
            length = abs(end - start)
            if length:
                lengths.append(length)
                durations.append(_duration(length, self.roads.segment_map[arc.segment_id].length_m,
                                           arc.travel_time_s, profile.speed_factor))

        uniform = self._uniform_step
        scaled = uniform[1] / profile.speed_factor if uniform else 0.0
        if (not exact and uniform and math.isfinite(scaled) and scaled > 0 and scaled.is_integer()
                and len(self.roads.nodes) * int(scaled) <= 2**53):


            first, middle, last = best
            if first is not None:
                append_part(first)
            if middle is not None:
                hops = int(self._node_cost(*middle, "travel_time_s") / uniform[1])
                lengths.append(hops * uniform[0])
                durations.append(hops * scaled)
            if last is not None:
                append_part(last)
        else:
            for part in self._route_parts(best, exact):
                append_part(part)
        try:
            duration, length = math.fsum(durations), math.fsum(lengths)
        except OverflowError as error:
            raise ValueError("path totals exceed the finite numeric range") from error
        if not math.isfinite(duration) or not math.isfinite(length):
            raise ValueError("path totals exceed the finite numeric range")
        return duration

    def route(self, origin, destination, profile=MovementProfile()):

        profile = self._profile(profile)
        origin = self.roads.normalize_position(origin)
        destination = self.roads.normalize_position(destination)
        _, best, exact = self._best_time(origin, destination)
        if best is None:
            return None
        legs = []
        for part in self._route_parts(best, exact):
            leg = self._part(*part, profile)
            if leg is not None:
                legs.append(leg)
        return RoadPath(network_id=self.roads.network_id, snapshot_id=self.roads.snapshot_id,
                        origin=origin, destination=destination, profile=profile, legs=tuple(legs))

    def position_after(self, path, elapsed_s):

        _finite(elapsed_s, "elapsed_s")
        if not isinstance(path, RoadPath) or path.network_id != self.roads.network_id:
            raise ValueError("path: expected a path on this base road network")
        current = self.roads.normalize_position(path.origin)
        destination = self.roads.normalize_position(path.destination)
        for leg in path.legs:
            arc = self.roads.arc_map.get(leg.arc_id)
            if arc is None or arc.segment_id != leg.segment_id:
                raise ValueError("path: invalid arc or segment identity")
            if (leg.end_offset_m > leg.start_offset_m) != arc.forward:
                raise ValueError("path: leg opposes the directed arc")
            start = self.roads.segment_position(leg.segment_id, leg.start_offset_m)
            end = self.roads.segment_position(leg.segment_id, leg.end_offset_m)
            if current != start or not math.isclose(abs(leg.end_offset_m - leg.start_offset_m),
                                                     leg.length_m, rel_tol=1e-10, abs_tol=1e-9):
                raise ValueError("path: discontinuous legs or inconsistent segment length")
            current = end
        if current != destination:
            raise ValueError("path: leg endpoints differ from its destination")
        if elapsed_s == 0:
            return self.roads.normalize_position(path.origin)


        def at_boundary(value, boundary, operations):
            return abs(value - boundary) <= operations * max(
                math.ulp(value), math.ulp(boundary), math.ulp(elapsed_s))

        if elapsed_s >= path.duration_s or at_boundary(elapsed_s, path.duration_s, max(1, len(path.legs))):
            return destination
        remaining = elapsed_s
        for index, leg in enumerate(path.legs):
            if at_boundary(remaining, leg.duration_s, index + 1):
                return self.roads.segment_position(leg.segment_id, leg.end_offset_m)
            if remaining < leg.duration_s:
                fraction = remaining / leg.duration_s
                offset = leg.start_offset_m + (leg.end_offset_m - leg.start_offset_m) * fraction
                return self.roads.segment_position(leg.segment_id, offset)
            remaining -= leg.duration_s
        return destination

    def _project(self, point, point_crs):
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
            raise ValueError("point: expected two finite coordinates")
        if not isinstance(point_crs, str) or not point_crs.strip():
            raise ValueError("point_crs: an explicit coordinate reference is required")
        if self.roads.crs.startswith("local:"):
            if point_crs != self.roads.crs:
                raise ValueError("local coordinates require an explicit external mapping before matching")
            return tuple(point)
        from pyproj import Transformer
        from pyproj.exceptions import ProjError

        try:
            if point_crs not in self._transforms:
                self._transforms[point_crs] = Transformer.from_crs(point_crs, self._crs, always_xy=True)
            result = self._transforms[point_crs].transform(*point, errcheck=True)
        except ProjError as error:
            raise ValueError(f"coordinate transformation failed: {error}") from error
        if not all(math.isfinite(value) for value in result):
            raise ValueError("coordinate transformation returned nonfinite values")
        return tuple(result)

    def match_point(self, point, point_crs, max_distance_m):

        _finite(max_distance_m, "max_distance_m")
        projected = self._project(point, point_crs)
        from shapely import LineString, Point, STRtree

        if self._spatial_index is None:
            identifiers = tuple(sorted(key for key, arcs in self._segment_arcs.items() if arcs))
            lines = tuple(LineString(self.roads.segment_map[key].geometry) for key in identifiers)
            self._spatial_index = (identifiers, lines, STRtree(lines))
        identifiers, lines, tree = self._spatial_index
        if not identifiers:
            return None
        query = Point(projected)
        matches = tree.query_nearest(query, all_matches=True)
        if len(matches) == 0:
            return None
        index = min(int(i) for i in matches)
        distance = float(lines[index].distance(query))
        if distance > max_distance_m:
            return None
        offset = float(lines[index].project(query))
        segment_id = identifiers[index]

        length = self.roads.segment_map[segment_id].length_m
        offset = length if offset == lines[index].length else min(length, max(0.0, offset))
        position = self.roads.segment_position(segment_id, offset)
        return RoadMatch(position=position, distance_m=distance, input_xy=projected,
                         snapped_xy=self.roads.coordinates(position))
