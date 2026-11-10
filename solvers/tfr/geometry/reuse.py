from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from dataclasses import fields, is_dataclass
from fractions import Fraction
import sys
import time
from types import MappingProxyType

from common.data.roads.schema import RoadNetwork, RoadArc, RoadNode, RoadSegment
from common.recovery.methods.roads_exact import ExactRoads, ExactRoadIndex
from common.config import cache_record, cache_limits
from common.cache_telemetry import bypass


_MISS = object()
MIB = 1024 * 1024


def cache_telemetry_snapshot(resources, path_reuse, archives):

    result = {} if resources is None else resources.snapshot()
    if path_reuse is not None:result['compiled'] = path_reuse.cache_snapshot()
    for record in archives.values():
        archive=record.get('archive')
        stats=archive.stats if archive is not None else record.get('geometry_summary') or {}
        for name,row in stats.get('cache_retention',{}).items():
            total=result.setdefault(name,dict(hits=0,misses=0,evictions=0,bypasses=0,bypass_reasons={},entries=0,
                max_entries=row['max_entries'],instances=0,scope='event totals across archives; capacity per archive; compacted storage excluded'))
            total['instances']+=1
            for key in ('hits','misses','evictions','bypasses'):total[key]+=row[key]
            if archive is not None:total['entries']+=row['entries']
            for reason,count in row['bypass_reasons'].items():total['bypass_reasons'][reason]=total['bypass_reasons'].get(reason,0)+count
    return result


def retained_size(value, check, limit):


    pending, seen, size, visits = [value], set(), 256, 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        visits += 1
        if visits % 512 == 0:
            check()
        if isinstance(item, (RoadNetwork, RoadArc, RoadNode, RoadSegment)):
            continue
        size += sys.getsizeof(item)
        if isinstance(item, Mapping):
            if isinstance(item, MappingProxyType):
                size += 256 + 96 * len(item)
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, (tuple, list, set, frozenset)):
            pending.extend(item)
        elif isinstance(item, Fraction):
            pending.extend((item.numerator, item.denominator))
        elif is_dataclass(item):
            size += sys.getsizeof(vars(item)) if hasattr(item, '__dict__') else 0
            pending.extend(getattr(item, f.name) for f in fields(item))
        if size > limit:
            return size
    check()
    return size


class _BoundedCache:
    def __init__(self, max_bytes, max_entries, check, *, evict):
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('cache byte limit must be a nonnegative integer')
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError('cache entry limit must be a nonnegative integer')
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.check, self.evict = check, evict
        self.rows = OrderedDict()
        self.stats = dict(hits=0, misses=0, insertions=0, evictions=0,
                          bypasses=0, retained_bytes=0, peak_retained_bytes=0,
                          accounting_wall_s=0., accounting_cpu_s=0., bypass_reasons={})

    def get(self, key):
        self.check()
        row = self.rows.get(key, _MISS)
        if row is _MISS:
            self.stats['misses'] += 1
            return _MISS
        self.stats['hits'] += 1
        self.rows.move_to_end(key)
        self.check()
        return row[0]

    def put(self, key, value, *, value_bytes=None, key_bytes=None):
        self.check()


        available = self.max_bytes if self.evict else self.max_bytes - self.stats['retained_bytes']
        if not available or not self.max_entries or (not self.evict and len(self.rows) >= self.max_entries):
            bypass(self.stats, 'disabled' if not self.max_bytes or not self.max_entries else 'retention_full_non_evicting')
            return
        started, cpu = time.perf_counter(), time.process_time()
        if value_bytes is not None and key_bytes is not None:
            size = value_bytes + key_bytes
        elif value_bytes is None:
            size = retained_size((key, value), self.check, available)
        else:
            size = value_bytes + retained_size(key, self.check, max(0, available-value_bytes))
        self.stats['accounting_wall_s'] += time.perf_counter()-started
        self.stats['accounting_cpu_s'] += time.process_time()-cpu
        if size > available:
            bypass(self.stats, 'oversized_bytes' if self.evict or not self.stats['retained_bytes'] else 'remaining_bytes_non_evicting')
            return
        while self.rows and (len(self.rows) >= self.max_entries or
                            self.stats['retained_bytes'] + size > self.max_bytes):
            _, (_, removed) = self.rows.popitem(last=False)
            self.stats['retained_bytes'] -= removed
            self.stats['evictions'] += 1
        self.check()
        self.rows[key] = value, size
        self.stats['insertions'] += 1
        self.stats['retained_bytes'] += size
        self.stats['peak_retained_bytes'] = max(self.stats['peak_retained_bytes'],
                                               self.stats['retained_bytes'])

    def snapshot(self):
        return dict(self.stats, entries=len(self.rows), max_entries=self.max_entries,
                    max_bytes=self.max_bytes, eviction_enabled=self.evict,
                    bypass_reasons=dict(self.stats['bypass_reasons']))


def _source_state_bytes(state, label_bytes):


    return (512 + sys.getsizeof(state.distances) + sys.getsizeof(state.queue)
            + sys.getsizeof(state.settled) + len(state.distances)*label_bytes
            + len(state.queue)*(sys.getsizeof((0, '')) + label_bytes))


class _SourceView(MutableMapping):

    def __init__(self, cache, namespace, namespace_bytes, label_bytes, statistics):
        self.cache, self.namespace = cache, namespace
        self.namespace_bytes, self.label_bytes = namespace_bytes, label_bytes
        self.statistics = statistics

    def __getitem__(self, key):
        return self.cache.rows[self.namespace, key][0]

    def pop(self, key, default=_MISS):
        self.cache.check()
        row = self.cache.rows.pop((self.namespace, key), _MISS)
        if row is _MISS:
            self.cache.stats['misses'] += 1
            if default is _MISS:raise KeyError(key)
            return default
        self.cache.stats['hits'] += 1
        self.cache.stats['retained_bytes'] -= row[1]
        return row[0]

    def __delitem__(self, key):
        _, size = self.cache.rows.pop((self.namespace, key))
        self.cache.stats['retained_bytes'] -= size

    def __setitem__(self, key, state):
        combined = self.namespace, key
        if combined in self.cache.rows:del self[key]
        evictions = self.cache.stats['evictions']
        self.cache.put(combined, state, value_bytes=_source_state_bytes(state, self.label_bytes),
                       key_bytes=self.namespace_bytes + retained_size(key, self.cache.check, self.cache.max_bytes))
        self.statistics['evictions'] += self.cache.stats['evictions']-evictions

    def __iter__(self):
        return iter(tuple(key for namespace, key in self.cache.rows if namespace == self.namespace))

    def __len__(self):
        return sum(namespace == self.namespace for namespace, _ in self.cache.rows)

    def move_to_end(self, key):
        self.cache.rows.move_to_end((self.namespace, key))

    def popitem(self, last=True):
        keys = list(self)
        if not keys:raise KeyError('empty exact source namespace')
        key = keys[-1 if last else 0]
        return key, self.pop(key)

    def clear(self):
        for key in list(self):del self[key]


class GeometryResources:
    def __init__(self, instance, check_budget, *, scope_key, state_error=ValueError,
                 index_bytes=None, domain_bytes=None, scalar_bytes=None,
                 index_entries=None, domain_entries=None, scalar_entries=None):
        self.cache_configuration = cache_record('h_geometry', dict(
            index_bytes=index_bytes, domain_bytes=domain_bytes, scalar_bytes=scalar_bytes,
            index_entries=index_entries, domain_entries=domain_entries, scalar_entries=scalar_entries))
        limits = self.cache_configuration['effective']
        self.instance, self.roads = instance, instance.roads
        self.boundary = getattr(instance, 'execution_boundary', None)
        self.road_identity = (id(self.roads), self.roads.network_id, self.roads.snapshot_id)
        self.scope_key = scope_key
        self.check_budget, self.state_error = check_budget, state_error
        self.indexes = _BoundedCache(limits['index_bytes'], limits['index_entries'], self.check, evict=False)
        self.domains = _BoundedCache(limits['domain_bytes'], limits['domain_entries'], self.check, evict=False)
        self.scalars = _BoundedCache(limits['scalar_bytes'], limits['scalar_entries'], self.check, evict=True)
        self.exact_sources = _BoundedCache(limits['scalar_bytes'], cache_limits('exact_roads')['source_entries'],
                                          self.check, evict=True)
        self._source_namespace_bytes = {}


        self._source_label_bytes = sys.getsizeof(1 << (2099 + len(self.roads.nodes).bit_length()))
        self.private_source_stats = dict(queries=0, retained_bytes=0, observed_peak_retained_bytes=0)
        self.domain_routing_stats = {}
        self.exact_source_stats = dict(hits=0, misses=0, evictions=0, bypasses=0, bypass_reasons={}, invalidations=0)
        self.exact_router_instances = 0
        self.stats = dict(index_builds=0, domain_expansions=0,
                          independent_timing_computations=0,independent_timing_hits=0,
                          fixed_cost_computations=0, domain_minimum_computations=0,
                          fixed_cost_hits=0, domain_minimum_hits=0,
                          index_build_wall_s=0., index_build_cpu_s=0., largest_index_bytes=0,
                          index_freeze_wall_s=0., index_freeze_cpu_s=0.,
                          domain_accounting_wall_s=0., domain_accounting_cpu_s=0.)

    def check(self):
        self.check_budget()
        roads = self.instance.roads
        if (roads is not self.roads or
                (id(roads), roads.network_id, roads.snapshot_id) != self.road_identity or
                getattr(self.instance, 'execution_boundary', None) is not self.boundary):
            raise self.state_error('geometry reuse context changed')

    def check_context(self, instance):
        self.check()
        if instance is not self.instance:
            raise self.state_error('geometry resources belong to another frozen input')

    def router(self, activity_domain=None, *, check_budget=None):
        self.check()


        callback = self.check_budget if check_budget is None else check_budget


        domain = None if activity_domain is None or activity_domain.all_road_positions else activity_domain
        key = self.scope_key, self.road_identity, domain
        index = self.indexes.get(key)
        if index is not _MISS:
            router = ExactRoads(self.roads, callback, activity_domain=domain, immutable_index=index)
            return self._share_sources(router)
        started, cpu = time.perf_counter(), time.process_time()
        router = ExactRoads(self.roads, callback, activity_domain=domain)
        self.stats['index_build_wall_s'] += time.perf_counter()-started
        self.stats['index_build_cpu_s'] += time.process_time()-cpu
        self.stats['index_builds'] += 1
        if self.indexes.max_bytes:
            started, cpu = time.perf_counter(), time.process_time()
            index = ExactRoadIndex.freeze(router)


            activity_bytes = retained_size(index.activity_key, self.check, self.indexes.max_bytes)
            self.stats['largest_index_bytes'] = max(self.stats['largest_index_bytes'],
                index.retained_bytes+activity_bytes)
            self.stats['index_freeze_wall_s'] += time.perf_counter()-started
            self.stats['index_freeze_cpu_s'] += time.process_time()-cpu
            self.indexes.put(key, index, value_bytes=index.retained_bytes+activity_bytes)
        else:
            bypass(self.indexes.stats, 'disabled')
        return self._share_sources(router)

    def _share_sources(self, router):
        namespace = self._router_key(router)
        if namespace not in self._source_namespace_bytes:
            self._source_namespace_bytes[namespace] = retained_size(namespace, self.check, self.exact_sources.max_bytes)
        router.source_stats = self.exact_source_stats
        router._sources = _SourceView(self.exact_sources, namespace,
            self._source_namespace_bytes[namespace], self._source_label_bytes, self.exact_source_stats)
        self.exact_router_instances += 1
        return router

    def observe_private_sources(self, states):
        size = sum(_source_state_bytes(state, self._source_label_bytes) for state in states)
        self.private_source_stats.update(queries=len(states), retained_bytes=size,
            observed_peak_retained_bytes=max(size, self.private_source_stats['observed_peak_retained_bytes']))

    def _router_key(self, roads):
        self.check()
        roads.check_budget()
        if roads.roads is not self.roads:
            raise self.state_error('geometry query uses another road snapshot')
        return (self.scope_key, self.road_identity,
                (True,) if roads.activity is None else roads.activity.cache_key)

    def domain_parts(self, roads, domain):
        key = self._router_key(roads), domain
        result = self.domains.get(key)
        if result is not _MISS:
            roads.check_budget()
            return result
        points, portions = roads.domain_parts(domain)
        self.stats['domain_expansions'] += 1
        started, cpu = time.perf_counter(), time.process_time()
        result = points, MappingProxyType(dict(portions))
        self.check()


        size = 512 + sys.getsizeof(points) + sys.getsizeof(portions)
        for point in points:
            self.check()
            size += sys.getsizeof(point) + sys.getsizeof(vars(point))
            if point.offset_m is not None:
                size += sys.getsizeof(point.offset_m)
                if isinstance(point.offset_m, Fraction):
                    size += sys.getsizeof(point.offset_m.numerator)+sys.getsizeof(point.offset_m.denominator)
        for pieces in portions.values():
            self.check()
            size += sys.getsizeof(pieces)
            for piece in pieces:
                size += sys.getsizeof(piece)
                for value in piece:
                    size += sys.getsizeof(value)
                    if isinstance(value, Fraction):
                        size += sys.getsizeof(value.numerator)+sys.getsizeof(value.denominator)
        self.stats['domain_accounting_wall_s'] += time.perf_counter()-started
        self.stats['domain_accounting_cpu_s'] += time.process_time()-cpu
        self.domains.put(key, result, value_bytes=size)
        roads.check_budget()
        return result

    def domain_parts_steps(self, roads, domain):

        from solvers.tfr.geometry.initialization import domain_parts_steps
        key = self._router_key(roads), domain
        result = self.domains.get(key)
        if result is not _MISS:
            return result
        points, portions = yield from domain_parts_steps(roads, domain)
        self.stats['domain_expansions'] += 1
        result = points, MappingProxyType(dict(portions))
        size = 512 + sys.getsizeof(points) + sys.getsizeof(portions)
        for point in points:
            self.check()
            yield
            size += sys.getsizeof(point) + sys.getsizeof(vars(point))
            if point.offset_m is not None:
                size += sys.getsizeof(point.offset_m)
                if isinstance(point.offset_m, Fraction):
                    size += sys.getsizeof(point.offset_m.numerator)+sys.getsizeof(point.offset_m.denominator)
        for pieces in portions.values():
            self.check()
            yield
            size += sys.getsizeof(pieces)
            for piece in pieces:
                yield
                size += sys.getsizeof(piece)
                for value in piece:
                    size += sys.getsizeof(value)
                    if isinstance(value, Fraction):
                        size += sys.getsizeof(value.numerator)+sys.getsizeof(value.denominator)
        self.domains.put(key, result, value_bytes=size)
        return result

    def _scalar(self, key, calculate, counter):
        result = self.scalars.get(key)
        if result is not _MISS:
            self.stats[counter.removesuffix('_computations')+'_hits'] += 1
            return result
        result = calculate()
        self.stats[counter] += 1
        self.check()


        self.scalars.put(key, result)
        return result

    def fixed_cost(self, roads, origin, destination, speed):
        from solvers.tfr.geometry.bounds import complete_steps
        return complete_steps(self.fixed_cost_steps(roads,origin,destination,speed))

    def domain_minimum(self, roads, left, right, speed, calculate):
        key = ('domain_time_forward', self._router_key(roads), left, right, Fraction(speed))
        result = self._scalar(key, calculate, 'domain_minimum_computations')
        roads.check_budget()
        return result

    def scalar_steps(self, key, calculate, counter):

        result = self.scalars.get(key)
        if result is not _MISS:
            self.stats[counter.removesuffix('_computations')+'_hits'] += 1
            return result
        result = yield from calculate()
        self.stats[counter] += 1
        self.check()
        self.scalars.put(key, result)
        return result

    def fixed_cost_steps(self, roads, origin, destination, speed, *, progress=None):
        roads._geometry_resources = self
        from solvers.tfr.geometry.initialization import cost_steps
        scale = roads._settings('time', speed)
        origin, destination = roads.normalize(origin), roads.normalize(destination)
        key = ('fixed_time_forward', self._router_key(roads), origin, destination, scale)
        value = self.scalars.get(key)
        if value is not _MISS:
            self.stats['fixed_cost_hits'] += 1
            return value
        value = yield from cost_steps(roads,origin,destination,speed,progress=progress)
        self.stats['fixed_cost_computations'] += 1
        self.check()
        self.scalars.put(key,value)
        return value

    def domain_minimum_steps(self, roads, left, right, speed, *, progress=None):
        from solvers.tfr.geometry.bounds import domain_travel_minimum_steps
        key = ('domain_time_forward', self._router_key(roads), left, right, Fraction(speed))
        if len(left)==1 and len(right)==1 and left[0].point is not None and right[0].point is not None:


            calculate = lambda: self.fixed_cost_steps(roads,left[0].point,right[0].point,speed,progress=progress)
        else:
            calculate = lambda: domain_travel_minimum_steps(roads, left, right, speed,
                                                           routing_stats=self.domain_routing_stats, progress=progress)
        return (yield from self.scalar_steps(key, calculate, 'domain_minimum_computations'))

    def snapshot(self):
        return dict(self.stats, configuration=self.cache_configuration, indexes=self.indexes.snapshot(), domains=self.domains.snapshot(),
                    exact_sources=dict(self.exact_source_stats, router_instances=self.exact_router_instances,
                        max_entries=cache_limits('exact_roads')['source_entries'],
                        scope='one shared completed-tree LRU across all routers and activity domains',
                        cache=self.exact_sources.snapshot(), private_queries=dict(self.private_source_stats)),
                    exact_domain_routing=dict(self.domain_routing_stats),
                    scalars=self.scalars.snapshot(), scope='one_frozen_solve_only',
                    retained_cache_limit_bytes=sum(c.max_bytes for c in
                        (self.indexes, self.domains, self.scalars, self.exact_sources)),
                    memory_note='Accounted cache-owned objects, not a process RSS bound; original search state and temporary work are separate.')
