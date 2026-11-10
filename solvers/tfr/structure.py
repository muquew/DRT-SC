from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass, replace
from itertools import combinations, permutations
from functools import cached_property, lru_cache
from common.config import cache_limits
from math import floor, fsum, isfinite
from common.recovery.plan import RecoveryPlan, RecoveryVisit, TransferEvent
from common.initial_sequences.inputs import _exceeds_capacity


def _tick(check):
    if check is not None:
        check()


@lru_cache(maxsize=cache_limits("h_structure")["visit_key_entries"], typed=True)
def _key_visit(operation, task_id):
    return operation, task_id


_ROUTE_KEY_POOL_MAX_VISITS = cache_limits("h_structure")["route_key_visits"]
_ROUTE_KEY_POOL_MAX_ENTRIES = cache_limits("h_structure")["route_key_entries"]
_route_key_pool = OrderedDict()
_route_key_pool_visits = 0


def _key_route(worker, visits):
    global _route_key_pool_visits


    lookup = (type(worker), worker, tuple(map(id, visits)))
    cached = _route_key_pool.get(lookup)
    if cached is not None:
        _route_key_pool.move_to_end(lookup)
        return cached
    result = worker, visits
    if len(visits) > _ROUTE_KEY_POOL_MAX_VISITS:
        return result
    _route_key_pool[lookup] = result
    _route_key_pool_visits += len(visits)
    while (_route_key_pool_visits > _ROUTE_KEY_POOL_MAX_VISITS
           or len(_route_key_pool) > _ROUTE_KEY_POOL_MAX_ENTRIES):
        _, (_, old_visits) = _route_key_pool.popitem(last=False)
        _route_key_pool_visits -= len(old_visits)
    return result


def structure_key(plan):

    names = {}
    routes = []
    for w, visits in sorted(plan.sequences.items()):
        route = []
        for v in visits:
            if v.operation == 'transfer':
                names.setdefault(v.transfer_id, len(names))
                route.append(_key_visit('transfer', names[v.transfer_id]))
            else:
                route.append(_key_visit(v.operation, v.task_id))
        routes.append(_key_route(w, tuple(route)))
    meetings = []
    for e in plan.transfers:
        names.setdefault(e.id, len(names))
        a, b = e.worker_i, e.worker_j
        forward, backward = e.tasks_i_to_j, e.tasks_j_to_i
        if a > b:
            a, b, forward, backward = b, a, backward, forward
        meetings.append((names[e.id], a, b, tuple(sorted(forward)), tuple(sorted(backward))))
    return tuple(routes), tuple(sorted(meetings))


@dataclass(frozen=True)
class StructureState:
    plan: RecoveryPlan
    domain: str
    reordered: frozenset
    pending: tuple
    conditions_key: str = ''
    departure_key: str = ''

    @cached_property
    def key(self):
        return (self.domain, self.conditions_key, self.departure_key,
                structure_key(self.plan), tuple(sorted(self.reordered)), self.pending)


@dataclass(frozen=True)
class CapacityGap:
    worker_id: str
    after_index: int
    held: frozenset
    excess: float


@dataclass(frozen=True)
class StructureCheck:
    errors: tuple
    gaps: tuple

    @property
    def valid(self):
        return not self.errors and not self.gaps


def _directions(e, w):
    return (e.tasks_i_to_j, e.tasks_j_to_i) if w == e.worker_i else (e.tasks_j_to_i, e.tasks_i_to_j)


def check_structure(item, plan, *, pending=(), check_budget=None):

    pending = set(pending)
    task_ids = set(item.task_map)
    errors, gaps = [], []
    if (plan.input_id, plan.network_id, plan.snapshot_id) != (item.instance_id, item.network_id, item.snapshot_id):
        errors.append('input_identity')
    if set(plan.sequences) != set(item.worker_map):
        return StructureCheck(tuple(errors + ['worker_coverage']), ())
    holders = {t: w.id for w in item.workers for t in w.carried_task_ids}
    seen, pairs = Counter(), Counter()
    graph, indegree = defaultdict(set), {}
    for w, visits in plan.sequences.items():
        _tick(check_budget)
        worker = item.worker_map[w]
        held = set(worker.carried_task_ids)
        previous = None
        def capacity(index):
            if held - task_ids:
                return
            load = fsum(item.task_map[t].load for t in held)
            if _exceeds_capacity(load, worker.capacity):
                gaps.append(CapacityGap(w, index, frozenset(held), load - worker.capacity))
        capacity(-1)
        for index, v in enumerate(visits):
            _tick(check_budget)
            node = ('transfer', v.transfer_id) if v.operation == 'transfer' else ('service', w, index)
            indegree.setdefault(node, 0)
            if previous is not None and node not in graph[previous]:
                graph[previous].add(node)
                indegree[node] += 1
            previous = node
            if v.operation == 'transfer':
                e = plan.transfer_map.get(v.transfer_id)
                if e is None or w not in (e.worker_i, e.worker_j):
                    errors.append('transfer_membership')
                    continue
                pairs[(e.id, w)] += 1
                outgoing, incoming = map(set, _directions(e, w))
                if (outgoing | incoming) - task_ids:
                    errors.append('unknown_task')
                if outgoing - held or incoming & held:
                    errors.append('custody')
                if worker.service_mode == 'handover_only' and incoming:
                    errors.append('withdrawn_receives')
                held = (held - outgoing) | incoming
            elif v.operation in ('pickup', 'delivery'):
                t = v.task_id
                seen[(t, v.operation)] += 1
                if t not in item.task_map:
                    errors.append('unknown_task')
                if worker.service_mode != 'delivery':
                    errors.append('withdrawn_services')
                if v.operation == 'pickup':
                    if t in holders or t in held:
                        errors.append('custody')
                    held.add(t)
                else:
                    if t not in held:
                        errors.append('custody')
                    held.discard(t)
            else:
                errors.append('unsupported_visit')
            capacity(index)
        if held - pending:
            errors.append('undelivered_inventory')
    expected = Counter((t, 'delivery') for t in item.task_map if t not in pending)
    expected.update((t, 'pickup') for t in item.task_map if t not in holders and t not in pending)
    if seen != expected:
        errors.append('service_coverage')
    expected_pairs = Counter((e.id, w) for e in plan.transfers for w in (e.worker_i, e.worker_j))
    if pairs != expected_pairs:
        errors.append('transfer_pairing')
    queue = deque(n for n, degree in indegree.items() if degree == 0)
    processed = 0
    while queue:
        _tick(check_budget)
        n = queue.popleft()
        processed += 1
        for successor in graph[n]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
    if processed != len(indegree):
        errors.append('event_cycle')
    return StructureCheck(tuple(sorted(set(errors))), tuple(gaps))


def release_tasks(plan, tasks):

    tasks = set(tasks)
    events = []
    for e in plan.transfers:
        a, b = tuple(t for t in e.tasks_i_to_j if t not in tasks), tuple(t for t in e.tasks_j_to_i if t not in tasks)
        if a or b:
            events.append(replace(e, tasks_i_to_j=a, tasks_j_to_i=b))
    ids = {e.id for e in events}
    sequences = {w: tuple(v for v in visits if
                         (v.transfer_id in ids if v.operation == 'transfer' else v.task_id not in tasks))
                 for w, visits in plan.sequences.items()}
    return replace(plan, sequences=sequences, transfers=tuple(events))


def ordered_pending(item, tasks):
    holders = {t: w for w in item.workers for t in w.carried_task_ids}
    def key(t):
        holder = holders.get(t)
        rank = 0 if holder and holder.service_mode == 'handover_only' else 1 if holder else 2
        return rank, item.task_map[t].promised_delivery_time_s, t
    return tuple(sorted(tasks, key=key))


def fixed_handover_bounds(item, u0, original_plan=None, departure_times=None,
                          common_release_lower_bound=None):
    if not isfinite(u0) or u0 < 0:
        raise ValueError('U0 must be a certified nonnegative finite total tardiness')
    delta = item.source_settings.transfer_duration_s
    if delta <= 0:
        raise ValueError('positive handover duration required')
    departure_times = departure_times or {w.id: item.time_s for w in item.workers}
    common_release = (min(departure_times.values()) if common_release_lower_bound is None
                      else common_release_lower_bound)
    if not isfinite(common_release) or any(t < common_release for t in departure_times.values()):
        raise ValueError('common release bound must be finite and no later than every worker release')
    holders = {t: w.id for w in item.workers for t in w.carried_task_ids}
    old = Counter(t for e in (original_plan.transfers if original_plan else ())
                  for t in (*e.tasks_i_to_j, *e.tasks_j_to_i))
    bounds = {}
    for t, task in item.task_map.items():
        available = departure_times[holders[t]] if t in holders else max(
            value for value in (common_release, task.pickup_ready_time_s, task.release_time_s)
            if value is not None)
        quotient = floor((task.promised_delivery_time_s + u0 - available) / delta)
        if quotient < 0:
            raise ValueError(f'inconsistent certified U0 or departure state for {t}')
        bounds[t] = max(old[t], len(item.workers) - 1, quotient)
    return bounds


def _nonstationary_walks(start, eligible, hops, check_budget):

    if hops == 0:
        yield ()
        return
    path = [start]
    stack = [iter(w for w in eligible if w != start)]
    while stack:
        _tick(check_budget)
        try:
            worker = next(stack[-1])
        except StopIteration:
            stack.pop()
            if len(path) > len(stack):
                path.pop()
            continue
        if len(stack) == hops:
            yield tuple(path[1:]) + (worker,)
        else:
            path.append(worker)
            stack.append(iter(tuple(w for w in eligible if w != worker)))


def custody_chains(item, task_id, *, domain='rebuild', bounds=None, check_budget=None, worker_order=None):

    eligible = tuple(sorted(w.id for w in item.workers if w.service_mode == 'delivery'))
    if worker_order is not None:
        ordered = tuple(worker_order)
        if len(ordered) != len(eligible) or set(ordered) != set(eligible):
            raise ValueError('worker ordering must preserve every eligible worker exactly once')
        eligible = ordered
    holder = next((w.id for w in item.workers if task_id in w.carried_task_ids), None)
    starts = (holder,) if holder is not None else eligible
    if domain not in ('rebuild', 'improvement'):
        raise ValueError('unknown chain domain')
    maximum = len(item.workers) - 1 if domain == 'rebuild' else bounds[task_id]
    for hops in range(maximum + 1):
        for start in starts:
            _tick(check_budget)
            if domain == 'rebuild':
                tails = permutations(tuple(w for w in eligible if w != start), hops)
            else:
                tails = _nonstationary_walks(start, eligible, hops, check_budget)
            for tail in tails:
                _tick(check_budget)
                chain = (start,) + tail
                if chain[-1] not in eligible or any(a == b for a, b in zip(chain, chain[1:])):
                    continue
                yield chain


def _insert(plan, w, index, visit):
    sequences = dict(plan.sequences)
    sequences[w] = sequences[w][:index] + (visit,) + sequences[w][index:]
    return replace(plan, sequences=sequences)


def _new_id(plan):
    i = 0
    while f'ts_{i}' in plan.transfer_map:
        i += 1
    return f'ts_{i}'


def _chain_insertions(item, plan, task_id, chain, check_budget, ordering=None, *, meeting_locations=None):

    held = any(task_id in w.carried_task_ids for w in item.workers)
    steps = ([] if held else [('pickup', chain[0])])
    steps += [('transfer', (a, b)) for a, b in zip(chain, chain[1:])]
    steps += [('delivery', chain[-1])]
    if meeting_locations is not None and len(meeting_locations) != len(chain)-1:
        raise ValueError('one witnessed position required per handover')
    def choices(current, step, step_index):
        operation, target = step
        if operation != 'transfer':
            visit = RecoveryVisit(operation=operation, task_id=task_id)
            indices = (range(len(current.sequences[target]) + 1) if ordering is None else
                       ordering.gap_order(current, target, visit, task_id))
            for index in indices:
                _tick(check_budget)
                yield current, _insert(current, target, index, visit)
            return
        a, b = target
        location = (item.worker_map[a].location if meeting_locations is None else
                    meeting_locations[step_index - int(not held)])
        for e in current.transfers:
            _tick(check_budget)
            if {e.worker_i, e.worker_j} != {a, b} or task_id in (*e.tasks_i_to_j, *e.tasks_j_to_i):
                continue
            if meeting_locations is not None and e.location != location:
                continue
            field = 'tasks_i_to_j' if e.worker_i == a else 'tasks_j_to_i'
            changed = replace(e, **{field: getattr(e, field) + (task_id,)})
            yield current, replace(current, transfers=tuple(changed if x.id == e.id else x for x in current.transfers))
        identifier = _new_id(current)
        event = TransferEvent(id=identifier, worker_i=a, worker_j=b,
                              location=location, tasks_i_to_j=(task_id,))
        base = replace(current, transfers=current.transfers + (event,))
        visit = RecoveryVisit(operation='transfer', transfer_id=identifier)
        left = (range(len(base.sequences[a]) + 1) if ordering is None else
                ordering.gap_order(base, a, visit, task_id))
        right = (range(len(base.sequences[b]) + 1) if ordering is None else
                 ordering.gap_order(base, b, visit, task_id))
        for i in left:
            for j in right:
                _tick(check_budget)
                yield base, _insert(_insert(base, a, i, visit), b, j, visit)
    stack = [iter(choices(plan, steps[0], 0))]
    while stack:
        _tick(check_budget)
        try:
            parent, candidate = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if len(stack) == len(steps):
            yield candidate
        else:
            if ordering is not None and getattr(ordering, 'child_score_reuse_active', False):


                ordering.warm_intermediate_child_estimate(parent, candidate, *steps[len(stack)])
            stack.append(iter(choices(candidate, steps[len(stack)], len(stack))))


def task_insertions(item, plan, task_id, *, domain='rebuild', bounds=None, pending=(), check_budget=None, raw_candidates=False, ordering=None):

    seen = set()
    worker_order = (None if ordering is None else ordering.worker_order(plan, task_id,
                    tuple(w.id for w in item.workers if w.service_mode == 'delivery')))
    for chain in custody_chains(item, task_id, domain=domain, bounds=bounds, check_budget=check_budget,
                                worker_order=worker_order):
        for candidate in _chain_insertions(item, plan, task_id, chain, check_budget, ordering):
            if raw_candidates:
                yield candidate
                continue
            result = check_structure(item, candidate, pending=set(pending) - {task_id}, check_budget=check_budget)
            if result.errors:
                continue
            key = structure_key(candidate)
            if key not in seen:
                seen.add(key)
                yield candidate


def capacity_extensions(item, gaps, reordered, *, check_budget=None):

    background = sorted(set().union(*(g.held for g in gaps)) - set(reordered)) if gaps else []
    for size in range(1, len(background) + 1):
        for values in combinations(background, size):
            _tick(check_budget)
            chosen = frozenset(values)
            if all(not _exceeds_capacity(fsum(item.task_map[t].load for t in g.held - chosen), item.worker_map[g.worker_id].capacity) for g in gaps):
                yield chosen


def background_state(item, state, tasks):
    tasks = frozenset(tasks) - state.reordered
    return replace(state, plan=release_tasks(state.plan, tasks),
                   reordered=state.reordered | tasks,
                   pending=ordered_pending(item, set(state.pending) | tasks))


def time_background_extensions(evaluated, reordered, *, tolerance=1e-7,
                               check_budget=None):


    reordered = frozenset(reordered)
    rows = defaultdict(list)
    predecessors = defaultdict(set)
    deliveries = {}
    def node(w, i, row):
        return ('transfer', row.visit.transfer_id) if row.visit.operation == 'transfer' else ('service', w, i)
    for w, route in evaluated.routes.items():
        for i, row in enumerate(route):
            _tick(check_budget)
            current = node(w, i, row)
            rows[current].append(row)
            if row.visit.operation == 'delivery':
                deliveries[row.visit.task_id] = current
            if i and abs(row.arrival_time_s - row.start_time_s) <= tolerance:
                previous = route[i - 1]
                if abs(previous.completion_time_s + row.incoming_path.duration_s - row.start_time_s) <= tolerance:
                    predecessors[current].add(node(w, i - 1, previous))
    releases = {}
    for task, lateness in sorted(evaluated.tardiness_s.items()):
        if lateness <= 0:
            continue
        frontier = deque(((deliveries[task], 0),))
        visited = set()
        while frontier:
            _tick(check_budget)
            current, distance = frontier.popleft()
            if current in visited:
                continue
            visited.add(current)


            visit = rows[current][0].visit
            if visit.operation == 'transfer':
                event = evaluated.plan.transfer_map[visit.transfer_id]
                group = frozenset(event.tasks_i_to_j + event.tasks_j_to_i) - reordered
            elif visit.task_id is not None:
                group = frozenset((visit.task_id,)) - reordered
            else:
                group = frozenset()
            if group:
                nearest, affected = releases.setdefault(group, (distance, set()))
                releases[group] = (min(nearest, distance), affected)
                affected.add(task)
            frontier.extend((previous, distance + 1)
                            for previous in sorted(predecessors[current]))
    return tuple(sorted(releases, key=lambda group: (
        releases[group][0], -len(releases[group][1]), len(group), tuple(sorted(group)))))


def time_background_tasks(evaluated, reordered, *, tolerance=1e-7):

    groups = time_background_extensions(evaluated, reordered, tolerance=tolerance)
    return frozenset().union(*groups)
