from collections import ChainMap, OrderedDict
from dataclasses import dataclass
import heapq
from itertools import chain
import math
from types import SimpleNamespace

from common.config import cache_limits
from common.evaluation.recovery_plan import (
    RecoveryPlanValidationError, _advance_time, _compile_route, _context,
    _evaluate_recovery_plan, _finite, _service_duration,
)
from common.recovery.plan import RecoveryPlan, _total
from common.recovery.adapters._insertion_delta import InsertionDelta


UNSUPPORTED = object()


def _metric_units(values):


    result=0
    for value in values:
        value=float(value)
        if not math.isfinite(value) or value<0:
            raise ValueError('metric terms require the original summation path')
        numerator,denominator=value.as_integer_ratio()
        result+=numerator << (1075-denominator.bit_length())
    return result


def _metric_expansion(units):

    while units:
        shift=max(0,units.bit_length()-53)
        chunk=units >> shift
        yield math.ldexp(float(chunk),shift-1074)
        units-=chunk << shift


class _MetricTotals:

    names=('total_tardiness_s','total_travel_time_s','total_wait_time_s','total_distance_m')

    def __init__(self, base):
        self.base=base
        try:
            self.by_worker={worker:tuple(_metric_units(values) for values in groups)
                for worker,groups in base.items()}
        except (ValueError,OverflowError):
            self.by_worker=None
        else:
            self.units=tuple(sum(values[i] for values in self.by_worker.values()) for i in range(4))

    def totals(self, changed):
        if self.by_worker is not None:
            try:
                updated={worker:tuple(_metric_units(values) for values in groups)
                    for worker,groups in changed.items()}
            except (ValueError,OverflowError):
                pass
            else:
                units=list(self.units)
                for worker,values in updated.items():
                    for i,value in enumerate(values):
                        units[i]+=value-self.by_worker[worker][i]
                return [_total(_metric_expansion(value),'evaluated.'+name)
                    for value,name in zip(units,self.names)]
        return [_total(chain.from_iterable(changed.get(w,self.base[w])[i] for w in self.base),
            'evaluated.'+name) for i,name in enumerate(self.names)]


class UnsupportedNumericInsertion(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class NumericInsertionScore:
    total_tardiness_s: float
    total_travel_time_s: float
    total_wait_time_s: float
    total_distance_m: float
    transfer_count: int


@dataclass(frozen=True, slots=True)
class _Row:
    node: tuple
    previous: tuple | None
    duration: float
    distance: float
    ready: float | None
    task_id: str | None
    promise: float | None
    context: dict
    service: float


@dataclass(frozen=True, slots=True)
class _Template:
    rows: tuple
    remaining: tuple
    prepared: object = None


def _row(item, worker, index, visit, path, previous):
    if visit.operation == 'transfer':
        node, ready, task_id, promise = ('transfer', visit.transfer_id, 0), None, None, None
    elif visit.operation in ('pickup', 'delivery'):
        task = item.task_map[visit.task_id]
        node = ('service', worker, index)
        ready = task.pickup_ready_time_s if visit.operation == 'pickup' else task.delivery_ready_time_s
        task_id = visit.task_id if visit.operation == 'delivery' else None
        promise = task.promised_delivery_time_s if task_id is not None else None
    else:
        raise UnsupportedNumericInsertion('private/custom visits require the native evaluator')
    return _Row(node, previous, path.duration_s, path.length_m, ready, task_id, promise,
                _context(worker, index, visit), _service_duration(item, worker, index, visit))


class _NumericEditContext:


    def __init__(self, base_item, expanded_item, base_plan, routing, *, cache=None):
        if type(base_plan) is not RecoveryPlan:
            raise UnsupportedNumericInsertion('ordinary RecoveryPlan required')
        added = set(expanded_item.task_map) - set(base_item.task_map)
        if not added or not set(base_item.task_map) < set(expanded_item.task_map):
            raise UnsupportedNumericInsertion('a nonempty added-task set is required')
        if (base_item.roads is not expanded_item.roads or
                getattr(base_item, 'execution_boundary', None) is not getattr(expanded_item, 'execution_boundary', None) or
                base_item.source_settings != expanded_item.source_settings or
                any(getattr(base_item, name) != getattr(expanded_item, name)
                    for name in ('instance_id', 'time_s', 'time_origin_s', 'load_unit')) or
                set(base_item.worker_map) != set(expanded_item.worker_map) or
                any(task != expanded_item.task_map[key] for key, task in base_item.task_map.items())):
            raise UnsupportedNumericInsertion('expanded input changes admitted physical facts')
        expanded_item.validate()
        self.added_task_ids = frozenset(added)
        self._initial_changes = set()
        for key, old in base_item.worker_map.items():
            new = expanded_item.worker_map[key]
            if any(getattr(old, name) != getattr(new, name) for name in
                   ('id', 'location', 'capacity', 'profile', 'handover_range_m', 'service_mode')):
                raise UnsupportedNumericInsertion('expanded worker physical fields changed')
            if new.carried_task_ids != old.carried_task_ids:
                if tuple(t for t in new.carried_task_ids if t not in added) != old.carried_task_ids:
                    raise UnsupportedNumericInsertion('expanded inventory changes other tasks')
                self._initial_changes.add(key)
        self.item, self.base_plan, self.cache = expanded_item, base_plan, cache
        self.routing = routing.for_activity_domain(None)
        from solvers.tfr.interfaces.boundary import PartialBoundaryRecoveryInput
        try:
            self.base = _evaluate_recovery_plan(base_item, base_plan, self.routing, cache=cache,
                allow_partial_boundary=isinstance(base_item, PartialBoundaryRecoveryInput))
        except RecoveryPlanValidationError as error:
            raise UnsupportedNumericInsertion('base plan has no admitted schedule: ' + error.code) from error
        self.worker_order = tuple(base_plan.sequences)
        self.stats = dict(scores=0, template_compiles=0, template_hits=0, template_evictions=0,
                          general_component_scores=0, independent_scores=0, unsupported=0,
                          group_hits=0, group_evictions=0, affected_hits=0, affected_evictions=0,
                          c_schedule_calls=0, python_schedule_calls=0, c_schedule_fallbacks=0)
        number = 1
        while f'transfer-{number}' in base_plan.transfer_map:
            number += 1
        self._new_transfer_id = f'transfer-{number}'
        transfer_ids = (*base_plan.transfer_map, self._new_transfer_id)
        self._kernel = None
        self.kernel_receipt = dict(available=False, reason='not_initialized')
        from common.evaluation._insertion_schedule import ScheduleKernel, ScheduleUnavailable
        self._schedule_unavailable = ScheduleUnavailable
        try:
            self._kernel = ScheduleKernel(self.worker_order, transfer_ids,
                                          expanded_item.source_settings.transfer_duration_s)
        except ScheduleUnavailable as error:
            self.kernel_receipt = dict(available=False, reason=str(error))
        else:
            self.kernel_receipt = dict(available=True, **self._kernel.receipt)
        boundary = getattr(expanded_item, 'execution_boundary', None)
        self.departures = dict(boundary.release_times) if boundary else {
            w: expanded_item.time_s for w in self.worker_order}
        self.anchors = dict(boundary.permission_anchors) if boundary else {
            w.id: w.location for w in expanded_item.workers}
        self.domains = dict(expanded_item.source_settings.activity_domains)
        self.meetings = {event.id: expanded_item.roads.normalize_position(event.location)
                         for event in base_plan.transfers}
        self._base_templates, self._base_metrics = {}, {}
        neighbors = {w: set() for w in self.worker_order}
        for event in base_plan.transfers:
            neighbors[event.worker_i].add(event.worker_j)
            neighbors[event.worker_j].add(event.worker_i)
        self._components = {}
        for worker in self.worker_order:
            if worker in self._components:
                continue
            component, pending = set(), [worker]
            while pending:
                current = pending.pop()
                if current not in component:
                    component.add(current)
                    pending.extend(neighbors[current] - component)
            for current in component:
                self._components[current] = frozenset(component)
        for worker in self.worker_order:
            rows, previous = [], None
            route = self.base.routes[worker]
            for index, scheduled in enumerate(route):
                row = _row(expanded_item, worker, index, scheduled.visit, scheduled.incoming_path, previous)
                rows.append(row)
                previous = row.node
            self._base_templates[worker] = self._prepare(worker, tuple(rows), ())
            self._base_metrics[worker] = (
                tuple(self.base.tardiness_s[s.visit.task_id] for s in route if s.visit.operation == 'delivery'),
                tuple(s.incoming_path.duration_s for s in route),
                tuple(s.waiting_time_s for s in route),
                tuple(s.incoming_path.length_m for s in route),
            )
        self._metric_totals = _MetricTotals(self._base_metrics)
        limits = cache_limits('evaluation')
        self._template_limit = limits['route_max_entries']
        self._visit_limit = limits['route_max_visits']
        self._memo_limit = cache_limits('memo')['meeting_entries']
        self._templates, self._groups, self._affected = OrderedDict(), OrderedDict(), OrderedDict()
        self._template_visits = 0

    def _prepare(self, worker, rows, remaining):
        prepared = None
        if self._kernel is not None:
            try:
                prepared = self._kernel.prepare(worker, rows, self.departures[worker])
            except self._schedule_unavailable as error:
                self.stats['c_schedule_fallbacks'] += 1
                self.kernel_receipt.update(available=False, reason=str(error))
                self._kernel = None
        return _Template(rows, remaining, prepared)

    def _retain_memo(self, cache, key, value, counter):
        if self._memo_limit:
            if key not in cache and len(cache) >= self._memo_limit:
                cache.popitem(last=False)
                self.stats[counter] += 1
            cache[key] = value
            cache.move_to_end(key)
        return value

    def _group(self, delta):
        key = (delta.kind, delta.sender, delta.receiver, delta.position,
               getattr(delta, 'reuse_event_id', None), delta.transfer_id)
        group = self._groups.get(key)
        if group is not None:
            self.stats['group_hits'] += 1
            self._groups.move_to_end(key)
            return group
        if delta.kind == 'direct' or (delta.kind == 'continuation' and not delta.carried_task_ids):
            event, events, transfer_map, meetings = None, self.base_plan.transfers, self.base_plan.transfer_map, self.meetings
        else:
            events = delta.transfer_events()
            event_id = delta.reuse_event_id if delta.kind == 'reuse' else delta.transfer_id
            event = next(e for e in events if e.id == event_id)
            context = {'transfer_id': event.id}
            try:
                meeting = self.item.roads.normalize_position(event.location)
                permitted = self.item.source_settings.meeting_domain.contains(meeting, self.item.roads)
            except ValueError as error:
                raise RecoveryPlanValidationError('meeting_domain', str(error), **context) from error
            if not permitted:
                raise RecoveryPlanValidationError('meeting_domain', 'meeting location is outside the permitted domain', **context)
            for worker_id in (event.worker_i, event.worker_j):
                worker = self.item.worker_map[worker_id]
                outgoing, incoming = ((event.tasks_i_to_j, event.tasks_j_to_i) if worker_id == event.worker_i
                                      else (event.tasks_j_to_i, event.tasks_i_to_j))
                if worker.service_mode == 'handover_only' and incoming:
                    raise RecoveryPlanValidationError('worker_withdrawal', 'withdrawn worker cannot receive items', worker_id=worker_id, **context)
                if outgoing:
                    origin = self.item.roads.normalize_position(self.anchors[worker_id])
                    try:
                        distance = self.routing.shortest_distance(origin, meeting)
                    except (ValueError, OverflowError) as error:
                        raise RecoveryPlanValidationError('routing_error', str(error), worker_id=worker_id, **context) from error
                    if distance == math.inf:
                        raise RecoveryPlanValidationError('unreachable', 'sender has no directed route to the meeting', worker_id=worker_id, **context)
                    if not math.isfinite(distance) or distance < 0:
                        raise RecoveryPlanValidationError('routing_error', 'routing returned an invalid distance', worker_id=worker_id, **context)
                    radius = worker.handover_range_m
                    if (origin != meeting if radius == 0 else distance - radius > math.ulp(max(distance, radius))):
                        raise RecoveryPlanValidationError('handover_range', 'sender cannot meet within its recovery-origin range', worker_id=worker_id, **context)
            transfer_map = ChainMap({event.id: event}, self.base_plan.transfer_map)
            meetings = ChainMap({event.id: meeting}, self.meetings)
        group = SimpleNamespace(events=events, plan=SimpleNamespace(transfer_map=transfer_map), meetings=meetings)
        return self._retain_memo(self._groups, key, group, 'group_evictions')

    def _template_key(self, delta, worker):
        if delta.kind == 'direct':
            return ('direct', worker, delta.pickup_index, delta.delivery_index)
        if worker == delta.sender:
            return (delta.kind, 'sender', worker, delta.position, delta.pickup_index,
                    delta.sender_transfer_index, delta.reuse_event_id, delta.transfer_id)
        return (delta.kind, 'receiver', worker, delta.position,
                delta.receiver_transfer_index, delta.delivery_index, delta.reuse_event_id, delta.transfer_id)

    def _compile(self, delta, worker, visits, group, key):
        self.stats['template_compiles'] += 1
        compiled, remaining = _compile_route(self.item, group.plan, worker, visits, self.routing,
                                             group.meetings, self.cache, self.domains)
        rows = tuple(_row(self.item, worker, i, visit, row['path'], row['previous'])
                     for i, (visit, row) in enumerate(zip(visits, compiled)))
        result = self._prepare(worker, rows, remaining)
        weight = max(1, len(rows))
        if self._template_limit and weight <= self._visit_limit:
            while self._templates and (len(self._templates) >= self._template_limit or
                                      self._template_visits + weight > self._visit_limit):
                _, old = self._templates.popitem(last=False)
                self._template_visits -= max(1, len(old.rows))
                self.stats['template_evictions'] += 1
            self._templates[key] = result
            self._template_visits += weight
        return result

    def _members(self, delta):
        key = delta.sender, delta.receiver
        cached = self._affected.get(key)
        if cached is not None:
            self.stats['affected_hits'] += 1
            self._affected.move_to_end(key)
            return cached
        members = self._components[delta.receiver]
        if delta.sender is not None:
            members = members | self._components[delta.sender]
        result = tuple(w for w in self.worker_order if w in members)
        return self._retain_memo(self._affected, key, result, 'affected_evictions')

    def _schedule(self, members, templates):
        if self._kernel is not None:
            prepared = tuple(templates[w].prepared for w in members)
            if all(route is not None for route in prepared):
                try:
                    self.stats['c_schedule_calls'] += 1
                    return self._kernel.run(prepared)
                except self._schedule_unavailable as error:
                    self.stats['c_schedule_fallbacks'] += 1
                    self.kernel_receipt.update(available=False, reason=str(error))
                    self._kernel = None
        self.stats['python_schedule_calls'] += 1
        return self._python_schedule(members, templates)

    def _python_schedule(self, members, templates):
        nodes, successors, predecessors = {}, {}, {}
        for worker in members:
            for row in templates[worker].rows:
                nodes.setdefault(row.node, []).append(row)
                if row.previous is not None:
                    predecessors.setdefault(row.node, set()).add(row.previous)
                    successors.setdefault(row.previous, set()).add(row.node)
        indegree = {node: len(predecessors.get(node, ())) for node in nodes}
        ready = [node for node in nodes if not indegree[node]]
        heapq.heapify(ready)
        completions, starts, arrivals = {}, {}, {}
        while ready:
            node = heapq.heappop(ready)
            incoming = []
            for row in nodes[node]:
                previous = (self.departures[row.context['worker_id']] if row.previous is None
                            else completions[row.previous])
                arrival = _advance_time(previous, row.duration, 'travel', **row.context)
                arrivals[row.context['worker_id'], row.context['visit_index']] = arrival
                incoming.append(arrival)
            start = max(incoming)
            if nodes[node][0].ready is not None:
                start = max(start, nodes[node][0].ready)
            completion = _advance_time(start, nodes[node][0].service, 'service duration', **nodes[node][0].context)
            starts[node], completions[node] = start, completion
            for row in nodes[node]:
                _finite(start - arrivals[row.context['worker_id'], row.context['visit_index']], 'waiting time', **row.context)
            for following in successors.get(node, ()):
                indegree[following] -= 1
                if not indegree[following]:
                    heapq.heappush(ready, following)
        if len(completions) != len(nodes):
            unresolved = sorted(n for n in nodes if n not in completions and n[0] == 'transfer')
            raise RecoveryPlanValidationError('transfer_cycle', 'worker visit orders contain a circular transfer dependency',
                                              transfer_id=unresolved[0][1] if unresolved else None)
        metrics = {}
        for worker in members:
            tardiness, travel, waits, distance = [], [], [], []
            for row in templates[worker].rows:
                if row.task_id is not None:
                    tardiness.append(_finite(max(0., completions[row.node] - row.promise), 'tardiness', **row.context))
                travel.append(row.duration)
                waits.append(starts[row.node] - arrivals[worker, row.context['visit_index']])
                distance.append(row.distance)
            metrics[worker] = tuple(tardiness), tuple(travel), tuple(waits), tuple(distance)
        return metrics

    def score(self, delta):
        if not self._supported(delta):
            self.stats['unsupported'] += 1
            return UNSUPPORTED
        changed = {delta.receiver} | ({delta.sender} if delta.sender is not None else set())
        if not self._initial_changes <= changed:
            self.stats['unsupported'] += 1
            return UNSUPPORTED
        group = self._group(delta)
        compiled, sequences = {}, None
        for worker in self.worker_order:
            if worker not in changed:
                continue
            key = self._template_key(delta, worker)
            template = self._templates.get(key)
            if template is None:
                if sequences is None:
                    sequences = delta.changed_sequences()
                template = self._compile(delta, worker, sequences[worker], group, key)
            else:
                self.stats['template_hits'] += 1
                self._templates.move_to_end(key)
            compiled[worker] = template
        custody = {w: t.remaining for w, t in compiled.items() if t.remaining}
        if custody:
            worker = min(custody)
            raise RecoveryPlanValidationError('remaining_custody', 'worker finishes with undelivered items',
                                              worker_id=worker, task_id=min(custody[worker]))
        members = self._members(delta)
        templates = {w: compiled.get(w, self._base_templates[w]) for w in members}
        metrics = self._schedule(members, templates)
        try:
            totals = self._metric_totals.totals(metrics)
        except (ValueError, OverflowError) as error:
            raise RecoveryPlanValidationError('numeric_range', str(error)) from error
        self.stats['scores'] += 1
        self.stats['general_component_scores'] += any(len(self._components[w]) > 1 for w in changed)
        self.stats['independent_scores'] += all(len(self._components[w]) == 1 for w in changed)
        return NumericInsertionScore(*totals, len(group.events))


def _index(value, lower, upper):
    return type(value) is int and lower <= value <= upper


class NumericInsertionContext(_NumericEditContext):


    def __init__(self, base_item, expanded_item, base_plan, routing, *, cache=None):
        added = set(expanded_item.task_map) - set(base_item.task_map)
        if len(added) != 1:
            raise UnsupportedNumericInsertion('exactly one added task required')
        self.task_id = next(iter(added))
        super().__init__(base_item, expanded_item, base_plan, routing, cache=cache)
        self._owner = next((w.id for w in expanded_item.workers if self.task_id in w.carried_task_ids), None)
        self._event_slots = {(visit.transfer_id, worker): index
                             for worker, route in base_plan.sequences.items()
                             for index, visit in enumerate(route) if visit.operation == 'transfer'}

    def _supported(self, delta):
        if (type(delta) is not InsertionDelta or delta.base_plan is not self.base_plan or
                delta.task_id != self.task_id or delta.kind not in ('direct', 'transfer', 'reuse') or
                delta.receiver not in self.item.worker_map or
                self.item.worker_map[delta.receiver].service_mode != 'delivery'):
            return False
        receiver_length = len(self.base_plan.sequences[delta.receiver])
        if delta.kind == 'direct':
            if delta.sender is not None:
                return False
            if self._owner is not None:
                return (delta.receiver == self._owner and delta.pickup_index is None and
                        _index(delta.delivery_index, 0, receiver_length))
            return (_index(delta.pickup_index, 0, receiver_length) and
                    _index(delta.delivery_index, delta.pickup_index + 1, receiver_length + 1))
        if (delta.sender not in self.item.worker_map or delta.sender == delta.receiver or
                (self._owner is not None and (delta.sender != self._owner or delta.pickup_index is not None))):
            return False
        sender_length = len(self.base_plan.sequences[delta.sender])
        if delta.kind == 'transfer':
            if delta.transfer_id != self._new_transfer_id:
                return False
            if self._owner is None:
                if not _index(delta.pickup_index, 0, sender_length):
                    return False
                start, end = delta.pickup_index + 1, sender_length + 1
            else:
                start, end = 0, sender_length
            return (_index(delta.sender_transfer_index, start, end) and
                    _index(delta.receiver_transfer_index, 0, receiver_length) and
                    _index(delta.delivery_index, delta.receiver_transfer_index + 1, receiver_length + 1))
        event = self.base_plan.transfer_map.get(delta.reuse_event_id)
        if event is None or {event.worker_i, event.worker_j} != {delta.sender, delta.receiver}:
            return False
        sender_slot = self._event_slots.get((event.id, delta.sender))
        receiver_slot = self._event_slots.get((event.id, delta.receiver))
        return (sender_slot is not None and receiver_slot is not None and
                (self._owner is not None or _index(delta.pickup_index, 0, sender_slot)) and
                _index(delta.delivery_index, receiver_slot + 1, receiver_length))


class NumericBulkContext(_NumericEditContext):


    def __init__(self, base_item, expanded_item, base_plan, routing, *, cache=None):
        super().__init__(base_item, expanded_item, base_plan, routing, cache=cache)
        self.task_ids = self.added_task_ids
        self._bulk_proof = None

    def _supported(self, delta):
        from collections import Counter
        from common.recovery.adapters._continuation_delta import ContinuationDelta
        from common.recovery.plan import RecoveryVisit
        if (type(delta) is not ContinuationDelta or delta.base_plan is not self.base_plan or
                delta.sender not in self.item.worker_map or delta.receiver not in self.item.worker_map or
                delta.sender == delta.receiver or self.item.worker_map[delta.sender].service_mode != 'handover_only' or
                self.item.worker_map[delta.receiver].service_mode != 'delivery' or
                self.base_plan.sequences[delta.sender] or
                not _index(delta.gap, 0, len(self.base_plan.sequences[delta.receiver])) or
                type(delta.block) is not tuple or type(delta.carried_task_ids) is not tuple):
            return False
        if self._bulk_proof is not None:
            sender, block, carried = self._bulk_proof
            if delta.sender != sender or (delta.block is not block and delta.block != block) or delta.carried_task_ids != carried:
                return False
        else:
            if any(type(v) is not RecoveryVisit or v.operation not in ('pickup', 'delivery') for v in delta.block):
                return False
            if delta.task_ids != self.added_task_ids:
                return False
            native_tokens = tuple((v.operation, v.task_id) for v in self.item.reference_sequences[delta.sender])
            if tuple((v.operation, v.task_id) for v in delta.block) != native_tokens:
                return False
            carried = self.item.worker_map[delta.sender].carried_task_ids
            if delta.carried_task_ids != carried or not set(carried) <= self.added_task_ids:
                return False
            holders = {t for worker in self.item.workers for t in worker.carried_task_ids}
            expected = Counter((task, 'delivery') for task in self.added_task_ids)
            expected.update((task, 'pickup') for task in self.added_task_ids if task not in holders)
            if Counter((v.task_id, v.operation) for v in delta.block) != expected:
                return False
            self._bulk_proof = delta.sender, delta.block, delta.carried_task_ids
        if delta.carried_task_ids:
            return delta.position is not None and delta.transfer_id == self._new_transfer_id
        return delta.position is None and delta.transfer_id is None

    def _template_key(self, delta, worker):
        if worker == delta.sender:
            return ('continuation', 'sender', worker, delta.position, delta.transfer_id)
        return ('continuation', 'receiver', worker, delta.position, delta.gap, delta.transfer_id)
