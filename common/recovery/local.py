from collections import defaultdict, OrderedDict
from dataclasses import replace
import heapq
import math
import json
import sys
from time import perf_counter

from common.recovery.plan import RecoveryPlan, RecoveryVisit


def full_input(item):
    return getattr(item, 'full_projection', item)


def reference_plan(item):
    full = full_input(item)
    boundary = getattr(full, 'execution_boundary', None)
    if boundary is not None:
        plan = boundary.reference_plan
        if set(item.task_map) != set(full.task_map):
            from common.recovery.adapters._common import strip
            plan = strip(plan, set(full.task_map) - set(item.task_map))
        return plan
    return RecoveryPlan(input_id=item.instance_id, network_id=item.network_id,
        snapshot_id=item.snapshot_id, sequences={w: tuple(RecoveryVisit(
            operation=v.operation, task_id=v.task_id) for v in visits)
            for w, visits in item.reference_sequences.items()})


def project_tasks(item, included):

    included = frozenset(included)
    if included == frozenset(item.task_map):
        return item
    if not included <= set(item.task_map):
        raise ValueError('local projection contains unknown obligations')
    if getattr(item, 'execution_boundary', None) is not None:
        from solvers.tfr.interfaces.boundary import project_subset
        return project_subset(full_input(item), included)
    source = item.source_snapshot
    routes = {w: tuple(v for v in visits if v.task_id in included)
              for w, visits in item.reference_sequences.items()}
    workers = tuple(replace(w, carried_task_ids=tuple(t for t in w.carried_task_ids if t in included))
                    for w in item.workers)
    snapshot_workers = tuple(replace(source.worker_map[w.id], carried_task_ids=w.carried_task_ids,
        load=math.fsum(item.task_map[t].load for t in w.carried_task_ids), remaining_visits=routes[w.id],
        phase='waiting' if routes[w.id] else 'idle', travel=None) for w in workers)
    return replace(item, tasks=tuple(t for t in item.tasks if t.id in included), workers=workers,
        reference_sequences=routes, source_snapshot=replace(source,
            tasks=tuple(t for t in source.tasks if t.task.id in included), workers=snapshot_workers))


def recovery_search_config():

    from common.config import SEARCH_DEFAULTS
    record = dict(SEARCH_DEFAULTS)
    if not isinstance(record, dict) or set(record) != {'search_radius_m'}:
        raise ValueError('recovery search configuration requires search_radius_m')
    radius = record['search_radius_m']
    if type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0:
        raise ValueError('search_radius_m must be finite and nonnegative')
    return record


def _within(distance, radius):
    return math.isfinite(distance) and (distance <= radius
        or distance - radius <= math.ulp(max(distance, radius)))


def _merge_intervals(intervals):
    merged = []
    for low, high in sorted(intervals):
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(high, merged[-1][1]))
        else:
            merged.append((low, high))
    return tuple(merged)


class _WorkerRangeSearch:


    def __init__(self, context, nodes, portions):
        self.context = context
        self.distances, self.queue, self.settled, self.best = {}, [], set(), {}
        roads, routing = context.item.roads, context.routing
        for node in nodes:
            self._seed(node, 0.)
        for segment_id, intervals in portions.items():
            segment = roads.segment_map[segment_id]
            arcs = routing._segment_arcs[segment_id]
            forward = any(arc.forward for arc in arcs)
            backward = any(not arc.forward for arc in arcs)
            for low, high in intervals:
                if low == 0:
                    self._seed(segment.u, 0.)
                if high == segment.length_m:
                    self._seed(segment.v, 0.)
                if forward:
                    self._seed(segment.u, low)
                if backward:
                    self._seed(segment.v, segment.length_m - high)


                for wid in context._segment_workers.get(segment_id, ()):
                    offset = context._worker_positions[wid].offset_m
                    cost = (0. if low <= offset <= high else
                        low - offset if offset < low and forward else
                        offset - high if offset > high and backward else math.inf)
                    if cost < self.best.get(wid, math.inf):
                        self.best[wid] = cost

    def _seed(self, node, cost):
        if cost < self.distances.get(node, math.inf):
            self.distances[node] = cost
            heapq.heappush(self.queue, (cost, node))

    def advance(self, radius):
        context = self.context
        while self.queue and _within(self.queue[0][0], radius):
            cost, node = heapq.heappop(self.queue)
            if node in self.settled or cost != self.distances[node]:
                continue
            self.settled.add(node)
            context.stats['spatial_settled_nodes'] += 1
            for wid, prefix in context._endpoint_workers.get(node, ()):
                candidate = cost + prefix
                if candidate < self.best.get(wid, math.inf):
                    self.best[wid] = candidate
            for previous, length, arc_count in context.reverse_length_neighbors(node):
                context.stats['spatial_relaxed_arcs'] += arc_count
                self._seed(previous, cost + length)

    def exhausted(self, allowed, radius):


        return (all(w in self.best and _within(self.best[w], radius) for w in allowed)
            or not self.queue and not any(w in allowed and not _within(d, radius)
                                         for w, d in self.best.items()))


class LocalRecoveryContext:
    def __init__(self, item, routing, *, search_radius_m=None, analyze_reference=True,evaluation_cache=None):
        from common.recovery.adapters._common import strip
        started = perf_counter()
        self.item, self.routing = item, routing.for_activity_domain(None)
        self._reverse_length_rows = self.routing._graph.__dict__.setdefault('_spatial_reverse_length_rows', {})
        self.evaluation_cache=evaluation_cache
        original = getattr(full_input(item), 'original_input', full_input(item))
        self.permission_anchors = {w.id: w.location for w in original.workers}
        self.search_radius_m = (recovery_search_config()['search_radius_m']
            if search_radius_m is None else self._checked_radius(search_radius_m))
        self.reference_plan = reference_plan(item)
        self.holders = {t: w.id for w in item.workers for t in w.carried_task_ids}
        self.stats = dict(reference_checks=0, reference_rejections={}, scope_wall_s=0.,
            relation_queries=0, spatial_candidates=0, qualified_relations=0,
            initial_search_radius_m=self.search_radius_m, max_search_radius_m=self.search_radius_m,
            candidate_protocol='directed-spatial-rings', radius_expansions=0,
            spatial_queries=0, spatial_settled_nodes=0, spatial_relaxed_arcs=0,
            handover_domain_queries=0, handover_settled_nodes=0, local_empty_queries=0,
            spatial_cache_entries=0, spatial_cache_nodes=0, spatial_cache_peak_nodes=0,
            spatial_cache_evictions=0, spatial_cache_bypasses=0,
            basic_qualification_scope='delivery role and single-item carrying capacity',
            activity_domain_scope='physical validation after spatial discovery')
        from common.config import cache_record
        self.spatial_cache_configuration = cache_record('spatial_search')
        self._search_limits = self.spatial_cache_configuration['effective']
        self._searches, self._search_weights = OrderedDict(), {}
        self._active_searches, self._basic_by_load = {}, {}
        self._endpoint_workers = defaultdict(list)
        self._segment_workers = defaultdict(list)
        self._worker_positions = {}
        self._role_workers = tuple(sorted(w.id for w in item.workers if w.service_mode == 'delivery'))
        for wid in self._role_workers:
            position = self.routing.roads.normalize_position(item.worker_map[wid].location)
            self._worker_positions[wid] = position
            if position.segment_id is not None:
                self._segment_workers[position.segment_id].append(wid)
            for node, piece in self.routing._connectors(position, leaving=True):
                self._endpoint_workers[node].append((wid, self.routing._piece_cost(piece, 'length_m')))
        meeting = item.source_settings.meeting_domain
        self._meeting_nodes = set(meeting.node_ids)
        portions = defaultdict(list)
        for portion in meeting.segment_portions:
            segment = item.roads.segment_map[portion.segment_id]
            portions[segment.id].append((portion.start_m, portion.end_m))
            if portion.start_m == 0:
                self._meeting_nodes.add(segment.u)
            if portion.end_m == segment.length_m:
                self._meeting_nodes.add(segment.v)
        self._meeting_portions = {s: _merge_intervals(spans) for s, spans in portions.items()}
        self.repair_task_ids = self._repair_scope() if analyze_reference else None
        self.base_plan = strip(self.reference_plan, self.repair_task_ids) if analyze_reference else None
        self.stats.update(R=len(item.task_map), D=len(self.repair_task_ids) if analyze_reference else None,
                          scope_wall_s=perf_counter()-started)

    def reverse_length_neighbors(self, node):


        row = self._reverse_length_rows.get(node)
        if row is None:
            row = tuple((previous, min(a['length_m'] for a in arcs.values()), len(arcs))
                        for previous, arcs in self.routing._graph.pred[node].items())
            self._reverse_length_rows[node] = row
        return row

    @staticmethod
    def _checked_radius(radius):
        if type(radius) not in (int, float) or not math.isfinite(radius) or radius < 0:
            raise ValueError('radius_m must be finite and nonnegative')
        return radius

    def _repair_scope(self):
        from common.evaluation.recovery_plan import _evaluate_recovery_plan, RecoveryPlanValidationError
        from common.recovery.adapters._common import strip
        pending = set()
        for worker in self.item.workers:
            if worker.service_mode == 'handover_only':
                pending.update(worker.carried_task_ids)
                pending.update(v.task_id for v in self.reference_plan.sequences[worker.id] if v.task_id)
        for event in self.reference_plan.transfers:
            if self.item.worker_map[event.worker_j].service_mode == 'handover_only':
                pending.update(event.tasks_i_to_j)
            if self.item.worker_map[event.worker_i].service_mode == 'handover_only':
                pending.update(event.tasks_j_to_i)
        expected = {(t, 'delivery') for t in self.item.task_map}
        expected.update((t, 'pickup') for t in self.item.task_map if t not in self.holders)
        observed = {(v.task_id, v.operation) for route in self.reference_plan.sequences.values()
                    for v in route if v.operation in ('pickup', 'delivery')}
        pending.update(t for t, operation in expected-observed)
        while len(pending) < len(self.item.task_map):
            retained = set(self.item.task_map)-pending
            projected = project_tasks(self.item, retained)
            base = strip(self.reference_plan, pending)
            self.stats['reference_checks'] += 1
            try:
                _evaluate_recovery_plan(projected, base, self.routing,cache=self.evaluation_cache,
                    allow_partial_boundary=True, score_only=True)
                break
            except RecoveryPlanValidationError as error:
                rejected = self.stats['reference_rejections']
                rejected[error.code] = rejected.get(error.code, 0)+1
                affected = {error.task_id} if error.task_id in retained else set()
                event = base.transfer_map.get(error.transfer_id)
                if event is not None:
                    affected.update(event.tasks_i_to_j)
                    affected.update(event.tasks_j_to_i)
                if not affected and error.worker_id in base.sequences and error.visit_index:
                    visit = base.sequences[error.worker_id][error.visit_index-1]
                    if visit.task_id:
                        affected.add(visit.task_id)
                if not affected-pending:
                    raise
                pending.update(affected)
        return frozenset(pending)

    def original_anchor(self, worker_id):
        return self.permission_anchors[worker_id]

    def basic_workers(self,task_id):

        from common.initial_sequences.inputs import _exceeds_capacity
        load=self.item.task_map[task_id].load
        if load not in self._basic_by_load:
            self._basic_by_load[load]=tuple(w for w in self._role_workers
                if not _exceeds_capacity(load,self.item.worker_map[w].capacity))
        return self._basic_by_load[load]

    def contact_point(self, task_id):
        holder = self.holders.get(task_id)
        return self.item.task_map[task_id].pickup if holder is None else self.permission_anchors[holder]

    def _handover_region(self, holder):

        self.stats['handover_domain_queries'] += 1
        roads, routing = self.item.roads, self.routing
        anchor = roads.normalize_position(self.permission_anchors[holder])
        radius = self.item.worker_map[holder].handover_range_m
        nodes, portions, distances, queue = set(), defaultdict(list), {}, []
        if anchor.node_id is not None:
            nodes.add(anchor.node_id)
        else:
            offset = anchor.offset_m
            portions[anchor.segment_id].append((offset, offset))
            segment = roads.segment_map[anchor.segment_id]
            for arc in routing._segment_arcs[segment.id]:
                portions[segment.id].append((offset, min(segment.length_m, offset + radius))
                    if arc.forward else (max(0., offset - radius), offset))
        for node, piece in routing._connectors(anchor, leaving=True):
            cost = routing._piece_cost(piece, 'length_m')
            if _within(cost, radius) and cost < distances.get(node, math.inf):
                distances[node] = cost
                heapq.heappush(queue, (cost, node))
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distances[node]:
                continue
            self.stats['handover_settled_nodes'] += 1
            nodes.add(node)
            remaining = max(0., radius - cost)
            for successor, arcs in routing._graph.succ[node].items():
                for arc_id, data in arcs.items():
                    arc = roads.arc_map[arc_id]
                    segment = roads.segment_map[arc.segment_id]
                    extent = min(segment.length_m, remaining)
                    portions[segment.id].append((0., extent) if arc.forward else
                        (segment.length_m - extent, segment.length_m))
                    candidate = cost + data['length_m']
                    if _within(candidate, radius) and candidate < distances.get(successor, math.inf):
                        distances[successor] = candidate
                        heapq.heappush(queue, (candidate, successor))
        if self.item.source_settings.meeting_domain.all_road_positions:
            return nodes, {s: _merge_intervals(spans) for s, spans in portions.items()}
        intersected = {}
        for segment_id, spans in portions.items():
            intersections = [(max(low, start), min(high, end))
                for low, high in _merge_intervals(spans)
                for start, end in self._meeting_portions.get(segment_id, ())
                if max(low, start) <= min(high, end)]
            if intersections:
                intersected[segment_id] = _merge_intervals(intersections)
        return nodes & self._meeting_nodes, intersected

    def _task_search(self, task_id):
        holder = self.holders.get(task_id)
        point = self.contact_point(task_id)
        key = ('holder', holder) if holder is not None else ('pickup', point)
        if key in self._active_searches:
            return self._active_searches[key][0]
        if key in self._searches:
            self._searches.move_to_end(key)
            return self._searches[key]
        if key not in self._searches:
            if holder is not None:
                nodes, portions = self._handover_region(holder)
            else:
                point = self.item.roads.normalize_position(point)
                nodes = {point.node_id} if point.node_id is not None else set()
                portions = ({point.segment_id: ((point.offset_m, point.offset_m),)}
                    if point.segment_id is not None else {})
            search = _WorkerRangeSearch(self, nodes, portions)
            search.cache_key = key
            self._retain_search(search)
            self.stats['spatial_queries'] += 1
            return search

    def _retain_search(self, search):

        key = search.cache_key
        self._searches.pop(key, None)
        self._search_weights.pop(key, None)
        weight = (len(search.distances) + len(search.queue)
            + len(search.settled) + len(search.best))
        limits = self._search_limits
        if not limits['max_entries'] or weight > limits['max_nodes']:
            self.stats['spatial_cache_bypasses'] += 1
        else:
            self._searches[key] = search
            self._search_weights[key] = weight
            while (len(self._searches) > limits['max_entries']
                    or sum(self._search_weights.values()) > limits['max_nodes']):
                oldest, _ = self._searches.popitem(last=False)
                self._search_weights.pop(oldest)
                self.stats['spatial_cache_evictions'] += 1
        retained = sum(self._search_weights.values())
        self.stats['spatial_cache_entries'] = len(self._searches)
        self.stats['spatial_cache_nodes'] = retained
        self.stats['spatial_cache_peak_nodes'] = max(self.stats['spatial_cache_peak_nodes'], retained)

    def actor_pool(self, task_id, *, radius_m=None):

        radius = self.search_radius_m if radius_m is None else self._checked_radius(radius_m)
        search = self._task_search(task_id)
        return self._actors_in_range(task_id, search, radius)

    def _actors_in_range(self, task_id, search, radius):
        search.advance(radius)
        self._retain_search(search)
        allowed = set(self.basic_workers(task_id))
        selected = tuple(sorted((w for w, d in search.best.items()
            if w in allowed and _within(d, radius)), key=lambda w: (search.best[w], w)))
        self.stats['relation_queries'] += 1
        self.stats['qualified_relations'] += len(allowed)
        self.stats['spatial_candidates'] += len(selected)
        self.stats['local_empty_queries'] += not selected
        self.stats['max_search_radius_m'] = max(self.stats['max_search_radius_m'], radius)
        return selected

    def spatial_search_exhausted(self, task_id, *, radius_m=None):
        radius = self.search_radius_m if radius_m is None else self._checked_radius(radius_m)
        allowed = set(self.basic_workers(task_id))
        if not allowed:
            return True
        search = self._task_search(task_id)
        search.advance(radius)
        self._retain_search(search)
        return search.exhausted(allowed, radius)

    def next_search_radius(self, task_id, radius_m):
        radius = self._checked_radius(radius_m)
        if self.spatial_search_exhausted(task_id, radius_m=radius):
            return None
        return min(sys.float_info.max, max(1., 2. * radius))

    def iter_worker_rings(self, task_id, *, sender_id=None, radius_m=None, yield_empty=False):


        radius = self.search_radius_m if radius_m is None else self._checked_radius(radius_m)
        search = self._task_search(task_id)
        key = search.cache_key
        self._active_searches[key] = (search, self._active_searches.get(key, (None, 0))[1] + 1)
        seen = set()
        try:
            while True:
                workers = self._actors_in_range(task_id, search, radius)
                added = tuple(w for w in workers if w not in seen)
                seen.update(added)
                if added or yield_empty:
                    yield radius, added
                if search.exhausted(set(self.basic_workers(task_id)), radius):
                    return
                self.stats['radius_expansions'] += 1
                radius = min(sys.float_info.max, max(1., 2. * radius))
        finally:
            users = self._active_searches[key][1] - 1
            if users:
                self._active_searches[key] = (search, users)
            else:
                self._active_searches.pop(key)
            self._retain_search(search)

    def delivery_workers(self, task_id, *, radius_m=None):
        return self.actor_pool(task_id, radius_m=radius_m)

    def eligible_workers(self, task_id, sender_id=None, *, radius_m=None, preserve_reference=False):
        return self.actor_pool(task_id, radius_m=radius_m)

    qualified_workers = eligible_workers

    def worker_candidate_priority(self, task_id, worker_id, sender_id=None):
        search = self._task_search(task_id)
        search.advance(self.search_radius_m)
        self._retain_search(search)
        return 0, search.best.get(worker_id, math.inf), worker_id

    def ordered_workers(self, task_id, sender_id=None, *, radius_m=None):
        return self.actor_pool(task_id, radius_m=radius_m)

    def meeting_allowed(self, sender, receiver, location, *, task_ids=()):
        if sender == receiver or self.item.worker_map[receiver].service_mode != 'delivery':
            return False
        if not self.item.source_settings.meeting_domain.contains(location, self.item.roads):
            return False
        distance = self.routing.shortest_distance(self.permission_anchors[sender], location)
        radius = self.item.worker_map[sender].handover_range_m
        if not math.isfinite(distance) or distance-radius > math.ulp(max(distance, radius)):
            return False
        return True

    def legal_meeting(self, sender_id, receiver_id, position, task_id):
        return self.meeting_allowed(sender_id, receiver_id, position, task_ids=(task_id,))

    def validate_plan(self, plan):
        from common.evaluation.recovery_plan import evaluate_recovery_plan
        return evaluate_recovery_plan(self.item, plan, self.routing)

    def affected_tasks(self, plan, changed_worker_ids):

        affected_workers = set(changed_worker_ids)
        while True:
            expanded = affected_workers | {w for e in plan.transfers
                if {e.worker_i, e.worker_j} & affected_workers for w in (e.worker_i, e.worker_j)}
            if expanded == affected_workers:
                break
            affected_workers = expanded
        tasks = {v.task_id for w in affected_workers for v in plan.sequences[w] if v.task_id}
        tasks.update(t for e in plan.transfers if {e.worker_i, e.worker_j} & affected_workers
                     for t in (*e.tasks_i_to_j, *e.tasks_j_to_i))
        return frozenset(tasks)
