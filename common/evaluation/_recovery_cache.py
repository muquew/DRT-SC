from collections import OrderedDict
from types import MappingProxyType
from weakref import ref

from common.data.roads.schema import RoadPath
from common.data.roads.routing import Routing
from common.evaluation.feasibility import checked_path
from common.evaluation.recovery_plan import _route_key as _standard_route_key
from common.config import cache_limits, cache_record
from common.cache_telemetry import retention_counts, bypass


def _nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class _WeightedLRU:


    def __init__(self, max_entries, max_weight, weight_unit='weight'):
        self.max_entries = _nonnegative_integer(max_entries, "max_entries")
        self.max_weight = _nonnegative_integer(max_weight, "max_weight")
        self.entries = OrderedDict()
        self.weight = 0
        self.weight_unit = weight_unit
        self.hits = self.misses = 0
        self.retention = retention_counts()

    def get(self, key):
        entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.entries.move_to_end(key)
        self.hits += 1
        return entry[0]

    def put(self, key, value, weight):
        _nonnegative_integer(weight, "weight")
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.weight -= previous[1]
        if self.max_entries == 0 or weight > self.max_weight:
            bypass(self.retention, 'disabled' if self.max_entries == 0 else 'oversized_'+self.weight_unit)
            return
        while self.entries and (len(self.entries) >= self.max_entries
                                or self.weight + weight > self.max_weight):
            _, (_, old_weight) = self.entries.popitem(last=False)
            self.weight -= old_weight
            self.retention['evictions'] += 1
        self.entries[key] = value, weight
        self.weight += weight


class _RouteKey:


    __slots__ = ("value", "code")

    def __init__(self, value):
        self.value, self.code = value, hash(value)

    def __hash__(self):
        return self.code

    def __eq__(self, other):
        if not isinstance(other, _RouteKey):
            return NotImplemented
        return self.value == other.value


class _RouteEntry:


    __slots__ = ("compiled", "times", "scheduled", "weight", "cacheable")

    def __init__(self, compiled, weight):
        self.compiled = compiled
        self.weight, self.cacheable = weight, False
        self.times = self.scheduled = None


def default_route_cache_entries(worker_count):

    _nonnegative_integer(worker_count, "worker_count")
    return cache_limits('evaluation')['route_max_entries']


def search_cache_limits(worker_count, environ=None):

    _nonnegative_integer(worker_count, "worker_count")
    return cache_record('evaluation', environ=environ)['effective']


class RecoveryEvaluationCache:


    def __init__(self, routing: Routing, *, path_max_entries=None, path_max_legs=None,
                 route_max_entries=None, route_max_visits=None,
                 component_max_entries=None, component_max_visits=None,
                 route_key_reuse=False):
        if not isinstance(routing, Routing):
            raise TypeError("routing must be a Routing")
        overrides = dict(path_max_entries=path_max_entries, path_max_legs=path_max_legs,
            route_max_entries=route_max_entries, route_max_visits=route_max_visits,
            component_max_entries=component_max_entries, component_max_visits=component_max_visits)

        if component_max_entries is None and route_max_entries is not None:
            overrides['component_max_entries'] = route_max_entries
        if component_max_visits is None and route_max_visits is not None:
            overrides['component_max_visits'] = route_max_visits
        self.cache_configuration = cache_record('evaluation', overrides)
        limits = self.cache_configuration['effective']
        path_max_entries, path_max_legs = limits['path_max_entries'], limits['path_max_legs']
        route_max_entries, route_max_visits = limits['route_max_entries'], limits['route_max_visits']
        component_max_entries, component_max_visits = limits['component_max_entries'], limits['component_max_visits']
        self._routing = routing
        self._roads = routing.roads
        self._network_id = self._roads.network_id
        self._snapshot_id = self._roads.snapshot_id
        self._paths = _WeightedLRU(path_max_entries, path_max_legs, 'legs')
        self._routes = _WeightedLRU(route_max_entries, route_max_visits, 'visits')
        if type(route_key_reuse) is not bool:
            raise ValueError('route_key_reuse must be boolean')
        self.route_key_reuse = route_key_reuse
        self._route_keys = _WeightedLRU(route_max_entries, route_max_visits, 'dependency_items')
        self._components = _WeightedLRU(
            route_max_entries if component_max_entries is None else component_max_entries,
            route_max_visits if component_max_visits is None else component_max_visits, 'visits')
        self._scheduled_hits = self._scheduled_misses = 0
        self._timed_nodes = 0

    @property
    def routing(self):
        return self._routing

    def check_routing(self, routing: Routing) -> None:

        if (routing is not self._routing or routing.roads is not self._roads
                or self._roads.network_id != self._network_id
                or self._roads.snapshot_id != self._snapshot_id):
            raise ValueError("recovery cache does not match the Routing object and road snapshot")

    def checked_path(self, origin, destination, profile, *, activity_domain=None, **context) -> RoadPath:

        self.check_routing(self._routing)
        routing = (self._routing if activity_domain is None else
                   self._routing.for_activity_domain(activity_domain))
        key = routing.activity_domain.cache_key, origin, destination, profile
        path = self._paths.get(key)
        if path is None:


            path = checked_path(routing, origin, destination, profile, **context)

            path.duration_s
            path.length_m
            self._paths.put(key, path, len(path.legs))
        return path

    def get_route(self, key):

        entry = self.get_route_entry(key)
        return None if entry is None else entry.compiled

    def prepare_route_key(self, item, plan, worker_id, visits, builder):


        self.check_routing(self._routing)
        if not self.route_key_reuse:
            bypass(self._route_keys.retention, 'disabled')
            return builder(item, plan, worker_id, visits)
        standard = builder is _standard_route_key
        events = (tuple(plan.transfer_map[v.transfer_id] for v in visits
                        if v.operation == 'transfer') if standard else plan.transfers)
        transfer_identity = tuple(map(id, events)) if standard else id(plan.transfers)
        identity = id(item), worker_id, id(visits), transfer_identity
        saved = self._route_keys.get(identity)
        if saved is not None:
            if not standard or saved[0]() is item:
                return saved[3]


            self._route_keys.hits -= 1
            self._route_keys.misses += 1
        key = _RouteKey(builder(item, plan, worker_id, visits))


        task_count = len(key.value[3]) if standard else len(item.tasks)
        weight = (1 + task_count + len(visits) + len(events)
                  + len(item.worker_map[worker_id].carried_task_ids))
        self._route_keys.put(identity, (ref(item) if standard else item, visits, events, key), weight)
        return key

    def get_route_entry(self, key):

        self.check_routing(self._routing)
        return self._routes.get(key if isinstance(key, _RouteKey) else _RouteKey(key))

    def put_route(self, key, template, weight):

        self.check_routing(self._routing)
        if not isinstance(template, tuple):
            raise TypeError("route template must be a tuple")
        entry = _RouteEntry(template, weight)
        self._routes.put(key if isinstance(key, _RouteKey) else _RouteKey(key), entry, weight)
        entry.cacheable = self._routes.max_entries > 0 and weight <= self._routes.max_weight
        return entry

    def get_component(self, key):

        self.check_routing(self._routing)
        routes = self._components.get(key)
        if routes is not None:
            self._scheduled_hits += len(routes)
        return routes

    def put_component(self, key, routes):

        self.check_routing(self._routing)
        entries = [entry for _, entry in key[2]]
        if all(entry.cacheable for entry in entries):
            self._components.put(key, routes, sum(entry.weight for entry in entries))
        else:
            bypass(self._components.retention, 'dependency_not_cacheable')

    def record_timing_work(self, count):
        self._timed_nodes += count

    def scheduled_route(self, entry, times):

        if entry.times == times:
            self._scheduled_hits += 1
            return entry.scheduled
        self._scheduled_misses += 1
        return None

    @staticmethod
    def remember_schedule(entry, times, scheduled):
        entry.times, entry.scheduled = times, scheduled

    @property
    def stats(self):

        return MappingProxyType({
            "configuration": self.cache_configuration,
            "route_max_entries": self._routes.max_entries,
            "route_max_visits": self._routes.max_weight,
            "component_max_entries": self._components.max_entries,
            "component_max_visits": self._components.max_weight,
            "routing": dict(self._routing.cache_stats),
            "path_max_entries": self._paths.max_entries,
            "path_max_legs": self._paths.max_weight,
            "path_hits": self._paths.hits,
            "path_misses": self._paths.misses,
            "path_entries": len(self._paths.entries),
            "path_legs": self._paths.weight,
            "route_hits": self._routes.hits,
            "route_misses": self._routes.misses,
            "route_entries": len(self._routes.entries),
            "route_visits": self._routes.weight,
            "scheduled_hits": self._scheduled_hits,
            "scheduled_misses": self._scheduled_misses,
            "component_hits": self._components.hits,
            "component_misses": self._components.misses,
            "component_entries": len(self._components.entries),
            "component_visits": self._components.weight,
            "timed_nodes": self._timed_nodes,
            **{prefix+'_'+name:value for prefix,cache in
               (('route',self._routes),('component',self._components),('path',self._paths),('route_key',self._route_keys))
               for name,value in cache.retention.items() if name!='bypass_reasons'},
            **{prefix+'_bypass_reasons':dict(cache.retention['bypass_reasons']) for prefix,cache in
               (('route',self._routes),('component',self._components),('path',self._paths),('route_key',self._route_keys))},
            'route_key_max_entries':self._route_keys.max_entries,
            'route_key_max_visits':self._route_keys.max_weight,
            'route_key_enabled':self.route_key_reuse,
            'route_key_entries':len(self._route_keys.entries),'route_key_hits':self._route_keys.hits,'route_key_misses':self._route_keys.misses,
            'retention_counting_unit':'one put rejection; get misses are not bypasses; dependency rejection counted instead of put',
            **{prefix+'_weight_unit':cache.weight_unit for prefix,cache in
               (('route',self._routes),('component',self._components),('path',self._paths),('route_key',self._route_keys))},
            **({"route_key_hits": self._route_keys.hits,
                "route_key_misses": self._route_keys.misses,
                "route_key_entries": len(self._route_keys.entries),
                "route_key_weight": self._route_keys.weight} if self.route_key_reuse else {}),
        })


def shared_cache(routing):
    root = routing.for_activity_domain(None)
    cache = getattr(root, '_recovery_evaluation_cache', None)
    if cache is None or cache.routing is not root:
        cache = RecoveryEvaluationCache(root, route_key_reuse=True)
        root._recovery_evaluation_cache = cache
    return cache
