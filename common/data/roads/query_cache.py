from collections import OrderedDict, namedtuple
from dataclasses import dataclass
import math
import sys

from common.cache_telemetry import retention_counts, bypass


@dataclass(frozen=True, slots=True)
class _SourceSlice:
    targets: object
    distances: object
    parent_nodes: object
    parents: object
    node_count: int

    @property
    def retained_nodes(self):
        return self.node_count

    @property
    def retained_bytes(self):

        return (sys.getsizeof(self) + sys.getsizeof(self.node_count)
                + sum(sys.getsizeof(a) for a in
                      (self.targets, self.distances, self.parent_nodes, self.parents)))

    def cost(self, target):
        import numpy as np
        offset = int(np.searchsorted(self.targets, target))
        if offset == len(self.targets) or int(self.targets[offset]) != target:
            raise KeyError(target)
        return float(self.distances[offset])

    def __getitem__(self, node):
        import numpy as np
        offset = int(np.searchsorted(self.parent_nodes, node))
        if offset == len(self.parent_nodes) or int(self.parent_nodes[offset]) != node:
            raise ValueError('missing retained SciPy predecessor')
        return int(self.parents[offset])


class SourceQueryCache:


    def __init__(self, routing, max_entries, max_nodes, max_bytes):
        self.routing = routing
        self.max_entries, self.max_nodes, self.max_bytes = max_entries, max_nodes, max_bytes
        self.entries = OrderedDict()
        self.nodes = self.bytes = self.peak_nodes = self.peak_bytes = 0
        self.hits = self.misses = self.extensions = self.inherited_entries = 0
        self.metric_queries = {metric:dict(hits=0,misses=0) for metric in ('travel_time_s','length_m')}
        self.query_kinds = {kind:dict(hits=0,misses=0) for kind in ('cost','path')}
        self.retention = retention_counts()

    def fork(self, routing, *, inherit=True):
        result = type(self)(routing, self.max_entries, self.max_nodes, self.max_bytes)
        if inherit:


            result.entries = self.entries.copy()
            result.nodes = result.peak_nodes = self.nodes
            result.bytes = self.bytes
            result.peak_bytes = self.bytes + result._table_bytes()
            result.inherited_entries = len(result.entries)
        return result

    def cache_info(self):
        return namedtuple('CacheInfo', 'hits misses maxsize currsize')(
            self.hits, self.misses, self.max_entries, len(self.entries))

    def retention_stats(self):
        return dict(self.retention, accounting_exact=True, accounting_reason=None,
                    counting_unit='forward source computation; union of retained target/predecessor nodes',
                    clears=0, reset_scope='this query session')

    def _slice(self, origin, metric, distances, predecessors, targets):
        import numpy as np
        size = len(self.routing._scipy_nodes)
        dtype = np.int32 if size <= np.iinfo(np.int32).max else np.int64
        targets = np.array(sorted(targets), dtype=dtype)
        values = np.array(distances[targets], dtype=np.float64, copy=True)
        closure = set()
        source = self.routing._scipy_index[origin]
        if metric == 'travel_time_s':
            for target, value in zip(targets, values):
                if not math.isfinite(float(value)):
                    continue
                current = int(target)
                while current != source and current not in closure:
                    closure.add(current)
                    current = int(predecessors[current])
                    if current < 0 or len(closure) >= size:
                        raise ValueError('invalid SciPy predecessor chain')
        nodes = np.array(sorted(closure), dtype=dtype)
        parents = (np.array(predecessors[nodes], dtype=dtype, copy=True) if len(nodes)
                   else np.empty(0, dtype=dtype))
        for array in (targets, values, nodes, parents):
            array.flags.writeable = False
        return _SourceSlice(targets, values, nodes, parents,len(set(map(int,targets)) | closure))

    def _remove(self, key, *, eviction=False):
        old = self.entries.pop(key)
        self.nodes -= old.retained_nodes
        self.bytes -= old.retained_bytes + sys.getsizeof(key)
        if eviction:
            self.retention['evictions'] += 1

    def _admit(self, key, entry):
        first_entry_table = sys.getsizeof(OrderedDict(((None,None),))) - sys.getsizeof(OrderedDict())
        reason = ('disabled' if not self.max_entries or not self.max_nodes or not self.max_bytes else
                  'oversized_nodes' if entry.retained_nodes > self.max_nodes else
                  'oversized_bytes' if entry.retained_bytes + sys.getsizeof(key)
                      + first_entry_table > self.max_bytes else None)
        if reason:
            bypass(self.retention, reason)
            return
        if key in self.entries:
            self._remove(key)
        while self.entries and (len(self.entries) >= self.max_entries or
                self.nodes + entry.retained_nodes > self.max_nodes or
                self.bytes + entry.retained_bytes + sys.getsizeof(key)
                    + self._table_bytes() > self.max_bytes):
            self._remove(next(iter(self.entries)), eviction=True)
        self.entries[key] = entry
        self.nodes += entry.retained_nodes
        self.bytes += entry.retained_bytes + sys.getsizeof(key)


        while self.entries and self.bytes + self._table_bytes() > self.max_bytes:
            self._remove(next(iter(self.entries)), eviction=True)
        if not self.entries:
            self.entries = OrderedDict()
        self.peak_nodes = max(self.peak_nodes, self.nodes)
        self.peak_bytes = max(self.peak_bytes, self.bytes + self._table_bytes())

    def _table_bytes(self):
        return sys.getsizeof(self.entries) - sys.getsizeof(OrderedDict())

    def lookup(self, origin, destination, metric, *, need_path=False):
        key = origin, metric
        query_kind = self.query_kinds['path' if need_path else 'cost']
        target = self.routing._scipy_index[destination]
        previous = self.entries.get(key)
        if previous is not None:
            try:
                cost = previous.cost(target)
            except KeyError:
                self.extensions += 1
            else:
                self.hits += 1
                self.metric_queries[metric]['hits'] += 1
                query_kind['hits'] += 1
                self.entries.move_to_end(key)
                return cost, previous
        self.misses += 1
        self.metric_queries[metric]['misses'] += 1
        query_kind['misses'] += 1
        try:
            distances, predecessors = self.routing._uncached_source_tree(origin, metric)
        except BaseException:
            bypass(self.retention,'computation_exception')
            raise
        if not self.max_entries or not self.max_nodes or not self.max_bytes:
            bypass(self.retention, 'disabled')
            return float(distances[target]), predecessors
        targets = set(self.routing._query_targets[metric])
        targets.add(target)
        if previous is not None:
            targets.update(map(int, previous.targets))
        entry = self._slice(origin, metric, distances, predecessors, targets)
        self._admit(key, entry)
        return float(distances[target]), entry

    def snapshot(self):
        result = dict(retained_nodes=self.nodes,
                    retained_bytes=self.bytes + self._table_bytes(),
                    peak_retained_nodes=self.peak_nodes, peak_retained_bytes=self.peak_bytes,
                    byte_limit=self.max_bytes, target_extensions=self.extensions,
                    inherited_entries=self.inherited_entries,
                    distance_records=sum(len(e.targets) for e in self.entries.values()),
                    predecessor_records=sum(len(e.parent_nodes) for e in self.entries.values()),
                    distance_array_bytes=sum(sys.getsizeof(e.targets)+sys.getsizeof(e.distances)
                        for e in self.entries.values()),
                    predecessor_array_bytes=sum(sys.getsizeof(e.parent_nodes)+sys.getsizeof(e.parents)
                        for e in self.entries.values()))
        for metric,counts in self.metric_queries.items():
            result.update({metric+'_'+key:value for key,value in counts.items()})
        for kind,counts in self.query_kinds.items():
            result.update({kind+'_'+key:value for key,value in counts.items()})
        return result
