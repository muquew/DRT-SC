from collections import defaultdict
import heapq
import math

from common.data.roads.routing import Routing
from common.initial_sequences.inputs import _exceeds_capacity
from common.recovery.inputs import RecoveryInput, checked_activity_domains
from common.recovery.plan import RecoveryPlan, ScheduledRecoveryVisit, EvaluatedRecoveryPlan, WaypointVisit
from common.recovery._search_visits import TransferStop
from common.evaluation.feasibility import PlanValidationError, checked_path


class RecoveryPlanValidationError(ValueError):


    def __init__(self, code, message, *, worker_id=None, visit_index=None, task_id=None, transfer_id=None):
        super().__init__(message)
        self.code, self.message = code, message
        self.worker_id, self.visit_index = worker_id, visit_index
        self.task_id, self.transfer_id = task_id, transfer_id

    def to_dict(self):
        return {"code": self.code, "message": self.message, "worker_id": self.worker_id,
                "visit_index": self.visit_index, "task_id": self.task_id, "transfer_id": self.transfer_id}


def _context(worker_id, index, visit):
    return {"worker_id": worker_id, "visit_index": index + 1,
            "task_id": visit.task_id, "transfer_id": visit.transfer_id}


def _finite(value, name, **context):
    if not math.isfinite(value):
        raise RecoveryPlanValidationError("numeric_range", f"{name} exceeds the finite numeric range", **context)
    return value


def _advance_time(start, duration, name, **context):
    completion = start + duration
    if not math.isfinite(completion):
        _finite(completion, name, **context)
    if duration > 0 and completion <= start:


        following = math.nextafter(start, math.inf)
        if not math.isfinite(following) or following - start > 1e-9:
            raise RecoveryPlanValidationError(
                "numeric_range", f"{name} is below the time base's resolution", **context)
        completion = following
    return completion


def _load(item, held, **context):
    try:
        return _finite(math.fsum(item.task_map[t].load for t in sorted(held)), "load", **context)
    except OverflowError as error:
        raise RecoveryPlanValidationError("numeric_range", "load exceeds the finite numeric range", **context) from error


def _directions(event, worker_id):
    return ((event.tasks_i_to_j, event.tasks_j_to_i) if worker_id == event.worker_i
            else (event.tasks_j_to_i, event.tasks_i_to_j))


def _active_service(item, worker_id):
    if getattr(item, 'execution_boundary', None) is not None:
        return None
    source = getattr(item, 'source_snapshot', None)
    return None if source is None else source.worker_map[worker_id].service


def _service_duration(item, worker_id, index, visit):
    if visit.operation == 'transfer':
        settings = getattr(item, 'source_settings', None)
        return 0. if settings is None else settings.transfer_duration_s
    if visit.operation not in ('pickup', 'delivery'):
        return 0.
    active = _active_service(item, worker_id)
    if (index == 0 and active is not None and active.visit.operation == visit.operation
            and active.visit.task_id == visit.task_id):
        return active.remaining_time_s
    return getattr(item.task_map[visit.task_id], visit.operation + '_service_time_s', 0.)


def _structure(item, plan, routing, *, allow_transfer_stops=False, allow_native_waypoints=False,
               permission_anchors=None, activity_domains=None):
    if plan.input_id != item.instance_id:
        raise RecoveryPlanValidationError("input_id", "plan belongs to a different recovery input")
    if (plan.network_id != item.network_id or plan.snapshot_id != item.snapshot_id
            or routing.roads.network_id != item.network_id or routing.roads.snapshot_id != item.snapshot_id):
        raise RecoveryPlanValidationError("road_snapshot", "plan or routing belongs to a different road snapshot")
    if set(plan.sequences) != set(item.worker_map):
        raise RecoveryPlanValidationError("worker_coverage", "sequences must include every worker, including idle workers")
    activity_domains = checked_activity_domains(item, activity_domains)
    for wid, domain in activity_domains.items():
        if not domain.contains(item.worker_map[wid].location, item.roads):
            raise RecoveryPlanValidationError('activity_domain', 'departure lies outside activity domain', worker_id=wid)
    permission_anchors = permission_anchors or {w.id: w.location for w in item.workers}
    locations = {}
    for event in plan.transfers:
        context = {"transfer_id": event.id}
        if event.worker_i not in item.worker_map or event.worker_j not in item.worker_map:
            raise RecoveryPlanValidationError("transfer_membership", "transfer has an unknown participant", **context)
        unknown = (set(event.tasks_i_to_j) | set(event.tasks_j_to_i)) - set(item.task_map)
        if unknown:
            raise RecoveryPlanValidationError("transfer_membership", "transfer contains an unknown task",
                                              task_id=min(unknown), **context)
        try:
            meeting = item.roads.normalize_position(event.location)
            permitted = item.source_settings.meeting_domain.contains(meeting, item.roads)
        except ValueError as error:
            raise RecoveryPlanValidationError("meeting_domain", str(error), **context) from error
        if not permitted:
            raise RecoveryPlanValidationError("meeting_domain", "meeting location is outside the permitted domain", **context)
        locations[event.id] = meeting
        for worker_id in (event.worker_i, event.worker_j):
            worker = item.worker_map[worker_id]
            outgoing, incoming = _directions(event, worker_id)
            if worker.service_mode == "handover_only" and incoming:
                raise RecoveryPlanValidationError("worker_withdrawal", "withdrawn worker cannot receive items",
                                                  worker_id=worker_id, **context)
            if outgoing:
                origin = item.roads.normalize_position(permission_anchors[worker_id])
                try:
                    distance = routing.shortest_distance(origin, meeting)
                except (ValueError, OverflowError) as error:
                    raise RecoveryPlanValidationError("routing_error", str(error), worker_id=worker_id, **context) from error
                if distance == math.inf:
                    raise RecoveryPlanValidationError("unreachable", "sender has no directed route to the meeting",
                                                      worker_id=worker_id, **context)
                if not math.isfinite(distance) or distance < 0:
                    raise RecoveryPlanValidationError("routing_error", "routing returned an invalid distance",
                                                      worker_id=worker_id, **context)
                radius = worker.handover_range_m
                outside = (origin != meeting if radius == 0 else
                           distance - radius > math.ulp(max(distance, radius)))
                if outside:
                    raise RecoveryPlanValidationError("handover_range", "sender cannot meet within its recovery-origin range",
                                                      worker_id=worker_id, **context)
    holders = {t: w.id for w in item.workers for t in w.carried_task_ids}
    seen, pairs = set(), defaultdict(list)
    for worker_id, visits in plan.sequences.items():
        worker = item.worker_map[worker_id]
        active = _active_service(item, worker_id)
        if active is not None and (not visits or visits[0].operation != active.visit.operation
                or visits[0].task_id != active.visit.task_id):
            raise RecoveryPlanValidationError('service_in_progress', 'an already started service must complete first',
                                              worker_id=worker_id)
        if active is not None:
            task = item.task_map[active.visit.task_id]
            if item.roads.normalize_position(worker.location) != item.roads.normalize_position(
                    getattr(task, active.visit.operation)):
                raise RecoveryPlanValidationError('service_in_progress', 'active service must retain its location',
                                                  worker_id=worker_id)
        for index, visit in enumerate(visits):
            context = _context(worker_id, index, visit)
            if visit.operation == "waypoint":
                if not allow_native_waypoints:
                    raise RecoveryPlanValidationError("private_visit", "native travel controls must be projected to a service plan", **context)
                if not isinstance(visit, WaypointVisit):
                    raise RecoveryPlanValidationError("private_visit", "waypoints require WaypointVisit", **context)
                if visit.for_task_id is not None and visit.for_task_id not in item.task_map:
                    raise RecoveryPlanValidationError("service_membership", "waypoint provenance is outside the task set", **context)
                try:
                    item.roads.normalize_position(visit.location)
                except ValueError as error:
                    raise RecoveryPlanValidationError("routing_error", str(error), **context) from error
                continue
            if isinstance(visit, TransferStop) or visit.operation == "transfer_stop":
                if not allow_transfer_stops or not isinstance(visit, TransferStop):
                    raise RecoveryPlanValidationError(
                        "private_visit", "empty transfer stops are only valid in private search states", **context)
                try:
                    stop = item.roads.normalize_position(visit.location)
                    permitted = item.source_settings.meeting_domain.contains(stop, item.roads)
                except ValueError as error:
                    raise RecoveryPlanValidationError("meeting_domain", str(error), **context) from error
                if not permitted:
                    raise RecoveryPlanValidationError(
                        "meeting_domain", "empty transfer stop is outside the permitted domain", **context)
                continue
            if visit.operation == "transfer":
                event = plan.transfer_map.get(visit.transfer_id)
                if event is None or worker_id not in (event.worker_i, event.worker_j):
                    raise RecoveryPlanValidationError("transfer_membership", "visit does not belong to a declared participant", **context)
                pairs[event.id].append(worker_id)
            else:
                if visit.task_id not in item.task_map:
                    raise RecoveryPlanValidationError("service_membership", "visit is outside the unfinished task set", **context)
                if worker.service_mode == "handover_only" and not (index == 0 and active is not None):
                    raise RecoveryPlanValidationError("worker_withdrawal", "withdrawn worker cannot pick up or deliver", **context)
                key = visit.task_id, visit.operation
                if key in seen:
                    raise RecoveryPlanValidationError("duplicate_visit", "task operation appears more than once", **context)
                seen.add(key)
                if visit.operation == "pickup" and visit.task_id in holders:
                    raise RecoveryPlanValidationError("carried_pickup", "a carried item cannot be picked up again", **context)
    for event in plan.transfers:
        if sorted(pairs[event.id]) != sorted((event.worker_i, event.worker_j)):
            raise RecoveryPlanValidationError("transfer_pairing", "transfer must occur once in each participant's sequence",
                                              transfer_id=event.id)
    expected = {(t.id, "delivery") for t in item.tasks}
    expected.update((t.id, "pickup") for t in item.tasks if t.id not in holders)
    if seen != expected:
        raise RecoveryPlanValidationError("service_coverage", f"missing service operations: {sorted(expected - seen)}")
    return locations


def _route_key(item, plan, worker_id, visits):

    worker = item.worker_map[worker_id]
    tasks = set(worker.carried_task_ids)
    events = []
    for visit in visits:
        if visit.operation == "transfer":
            event = plan.transfer_map[visit.transfer_id]
            events.append(event)
            tasks.update(event.tasks_i_to_j)
            tasks.update(event.tasks_j_to_i)
        elif visit.operation not in ("transfer_stop", "waypoint"):
            tasks.add(visit.task_id)
    attributes = tuple((t, item.task_map[t].pickup, item.task_map[t].delivery,
                        item.task_map[t].load, item.task_map[t].pickup_ready_time_s,
                        item.task_map[t].delivery_ready_time_s,
                        item.task_map[t].pickup_service_time_s,
                        item.task_map[t].delivery_service_time_s) for t in sorted(tasks))
    boundary = getattr(item, 'execution_boundary', None)
    anchor = worker.location if boundary is None else boundary.permission_anchors[worker_id]
    domain = item.source_settings.activity_domains.get(worker_id)
    return worker, visits, tuple(events), attributes, anchor, domain, _active_service(item, worker_id)


def _compile_route(item, plan, worker_id, visits, routing, meeting_locations, cache, activity_domains=None):
    worker = item.worker_map[worker_id]
    worker_routing = routing.for_activity_domain((activity_domains or {}).get(worker_id))
    activity = worker_routing.activity_domain
    location = item.roads.normalize_position(worker.location)
    held, previous = set(worker.carried_task_ids), None
    templates = []
    for index, visit in enumerate(visits):
        context = _context(worker_id, index, visit)
        service = _service_duration(item, worker_id, index, visit)
        if visit.operation == "transfer":
            event = plan.transfer_map[visit.transfer_id]
            node = ("transfer", event.id, 0)
            destination, ready = meeting_locations[event.id], None
            outgoing, incoming = map(set, _directions(event, worker_id))
            invalid_items = (outgoing - held) | (incoming & held)
            if invalid_items:
                raise RecoveryPlanValidationError("custody", "transfer sends absent items or receives items already held",
                                                  **{**context, "task_id": min(invalid_items)})

            held = (held - outgoing) | incoming
        elif visit.operation == "transfer_stop":


            node = ("transfer_stop", worker_id, index)
            destination, ready = item.roads.normalize_position(visit.location), None
        elif visit.operation == "waypoint":
            node = ("waypoint", worker_id, index)
            destination, ready = item.roads.normalize_position(visit.location), visit.not_before_s
        else:
            task = item.task_map[visit.task_id]
            node = ("service", worker_id, index)
            if visit.operation == "pickup":
                destination, ready = task.pickup, task.pickup_ready_time_s
                if task.id in held:
                    raise RecoveryPlanValidationError("custody", "worker already holds the pickup item", **context)
                held.add(task.id)
            else:
                destination, ready = task.delivery, task.delivery_ready_time_s
                if task.id not in held:
                    raise RecoveryPlanValidationError("custody", "worker cannot deliver an item it does not hold", **context)
                held.remove(task.id)
            destination = item.roads.normalize_position(destination)
        load = _load(item, held, **context)
        if _exceeds_capacity(load, worker.capacity):
            raise RecoveryPlanValidationError("capacity", "carrying capacity exceeded after visit", **context)
        if not activity.contains_position(destination):
            label = 'stationary activity domain' if activity.is_stationary else 'activity domain'
            raise RecoveryPlanValidationError('activity_domain', f'route leaves {label}', **context)
        try:
            if cache is None:
                path = checked_path(worker_routing, location, destination, worker.profile,
                                    worker_id=worker_id, visit_index=index + 1, task_id=visit.task_id)
            else:
                path = cache.checked_path(location, destination, worker.profile,
                                          activity_domain=(activity_domains or {}).get(worker_id),
                                          worker_id=worker_id, visit_index=index + 1, task_id=visit.task_id)
        except PlanValidationError as error:
            raise RecoveryPlanValidationError(error.code, error.message, **context) from error
        if any(not activity.contains_span(leg.segment_id, leg.start_offset_m, leg.end_offset_m)
               for leg in path.legs):
            raise RecoveryPlanValidationError('activity_domain', 'route crosses outside activity domain', **context)
        row = {"visit": visit, "node": node, "previous": previous, "path": path,
               "ready": ready, "service": service, "load": load, "held": tuple(sorted(held)), "context": context}
        templates.append(row)
        previous, location = node, destination
    return tuple(templates), tuple(sorted(held))


def _worker_components(plan):

    parents = {worker: worker for worker, visits in plan.sequences.items() if visits}

    def root(worker):
        while parents[worker] != worker:
            parents[worker] = parents[parents[worker]]
            worker = parents[worker]
        return worker

    for event in plan.transfers:
        parents[root(event.worker_j)] = root(event.worker_i)
    groups = {}
    for worker in parents:
        groups.setdefault(root(worker), []).append(worker)
    return tuple(tuple(group) for group in groups.values())


def evaluate_recovery_plan(recovery_input, plan, routing=None, *, cache=None, execution_boundary=None):

    return _evaluate_recovery_plan(recovery_input, plan, routing, cache=cache,
                                   allow_transfer_stops=False, execution_boundary=execution_boundary)


def _score_recovery_plan(recovery_input, plan, routing=None, *, cache=None):

    return _evaluate_recovery_plan(recovery_input, plan, routing, cache=cache,
                                   allow_transfer_stops=False, score_only=True)


def _evaluate_recovery_plan(recovery_input, plan, routing=None, *, cache=None,
                            allow_transfer_stops=False, allow_native_waypoints=False, score_only=False, execution_boundary=None,
                            allow_partial_boundary=False):


    if not isinstance(recovery_input, RecoveryInput) or not isinstance(plan, RecoveryPlan):
        raise TypeError("expected RecoveryInput and RecoveryPlan")
    item = recovery_input
    attached_boundary = getattr(item, 'execution_boundary', None)
    if execution_boundary is not None and execution_boundary is not attached_boundary:
        raise ValueError('execution boundary must be bound to the projected recovery input')
    boundary = attached_boundary
    domains = dict(item.source_settings.activity_domains)
    if boundary is not None:
        from solvers.tfr.interfaces.boundary import PartialBoundaryRecoveryInput
        if isinstance(item, PartialBoundaryRecoveryInput):
            if not allow_partial_boundary:
                raise ValueError('internal partial boundary input cannot enter public complete-plan validation')
            boundary.validate_partial_for(item)
        else:
            boundary.validate_for(item)
        departure_times = dict(boundary.release_times)
        permission_anchors = dict(boundary.permission_anchors)
        for wid, domain in boundary.activity_domains.items():
            if wid in domains and domains[wid] != domain:
                raise ValueError('activity domain may not be reset at execution boundary')
            domains[wid] = domain
        if set(departure_times) != set(item.worker_map) or set(permission_anchors) != set(item.worker_map):
            raise ValueError('boundary times and permission anchors must cover every worker')
        if any(not math.isfinite(t) or t < item.time_s for t in departure_times.values()):
            raise ValueError('invalid asynchronous boundary release time')
        for anchor in permission_anchors.values():
            item.roads.normalize_position(anchor)
    else:
        departure_times = {w.id: item.time_s for w in item.workers}
        permission_anchors = {w.id: w.location for w in item.workers}
    domains = checked_activity_domains(item, domains)


    if routing is None:
        routing = Routing(item.roads)
    if not isinstance(routing, Routing):
        raise TypeError("routing must be a Routing")


    routing = routing.for_activity_domain(None)
    if cache is not None:
        from common.evaluation._recovery_cache import RecoveryEvaluationCache
        if not isinstance(cache, RecoveryEvaluationCache):
            raise TypeError("cache must be a RecoveryEvaluationCache")
        cache.check_routing(routing)
    meeting_locations = _structure(item, plan, routing, allow_transfer_stops=allow_transfer_stops,
                                   allow_native_waypoints=allow_native_waypoints,
                                   permission_anchors=permission_anchors, activity_domains=domains)
    remaining_custody, route_entries, templates_by_worker = {}, {}, {}
    for worker_id, visits in plan.sequences.items():
        if cache is None:
            key = None
        elif cache.route_key_reuse:
            key = cache.prepare_route_key(item, plan, worker_id, visits, _route_key)
        else:
            key = _route_key(item, plan, worker_id, visits)
        entry = None if cache is None else cache.get_route_entry(key)
        compiled = None if entry is None else entry.compiled
        if compiled is None:
            compiled = _compile_route(item, plan, worker_id, visits, routing, meeting_locations, cache, domains)
            if cache is not None:
                entry = cache.put_route(key, compiled, len(visits) + len(item.worker_map[worker_id].carried_task_ids))
        route_entries[worker_id] = entry
        templates, held = compiled
        templates_by_worker[worker_id] = templates
        if held:
            remaining_custody[worker_id] = held

    if remaining_custody:
        worker_id = min(remaining_custody)
        raise RecoveryPlanValidationError("remaining_custody", "worker finishes with undelivered items",
                                          worker_id=worker_id, task_id=min(remaining_custody[worker_id]))

    cached_routes, missing_components = {}, []
    if cache is not None:
        for members in _worker_components(plan):
            key = (tuple((worker, departure_times[worker]) for worker in members),
                   item.source_settings.transfer_duration_s,
                   tuple((worker, route_entries[worker]) for worker in members))
            cached = cache.get_component(key)
            if cached is None:
                missing_components.append((key, members))
            else:
                cached_routes.update(cached)

    nodes, predecessors, successors, rows = {}, defaultdict(set), defaultdict(set), {}
    for worker_id, templates in templates_by_worker.items():
        if worker_id in cached_routes:
            continue
        for index, template in enumerate(templates):


            row = dict(template)
            rows[worker_id, index] = row
            node, previous = row["node"], row["previous"]
            nodes.setdefault(node, []).append(row)
            if previous is not None:
                predecessors[node].add(previous)
                successors[previous].add(node)
    indegree = {node: len(predecessors[node]) for node in nodes}
    ready_nodes = [node for node, degree in indegree.items() if degree == 0]
    heapq.heapify(ready_nodes)
    completions, starts = {}, {}
    while ready_nodes:
        node = heapq.heappop(ready_nodes)
        if cache is not None:
            cache.record_timing_work(1)
        arrivals = []
        for row in nodes[node]:
            previous = departure_times[row["context"]["worker_id"]] if row["previous"] is None else completions[row["previous"]]
            arrival = _advance_time(previous, row["path"].duration_s, "travel", **row["context"])
            row["arrival"] = arrival
            arrivals.append(arrival)
        start = max(arrivals)
        if nodes[node][0]["ready"] is not None:
            start = max(start, nodes[node][0]["ready"])
        duration = nodes[node][0]["service"]
        completion = _advance_time(start, duration, "service duration", **nodes[node][0]["context"])
        starts[node], completions[node] = start, completion
        for row in nodes[node]:
            _finite(start - row["arrival"], "waiting time", **row["context"])
        for following in successors[node]:
            indegree[following] -= 1
            if indegree[following] == 0:
                heapq.heappush(ready_nodes, following)
    if len(completions) != len(nodes):
        unresolved = sorted(n for n in nodes if n not in completions and n[0] == "transfer")
        raise RecoveryPlanValidationError("transfer_cycle", "worker visit orders contain a circular transfer dependency",
                                          transfer_id=unresolved[0][1] if unresolved else None)

    routes, tardiness = {}, {}
    for worker_id, visits in plan.sequences.items():
        if worker_id in cached_routes:
            routes[worker_id] = cached_routes[worker_id]
            for index, scheduled in enumerate(routes[worker_id]):
                if scheduled.visit.operation == "delivery":
                    task = item.task_map[scheduled.visit.task_id]
                    tardiness[task.id] = _finite(max(0.0, scheduled.completion_time_s - task.promised_delivery_time_s),
                                                 "tardiness", **_context(worker_id, index, scheduled.visit))
            continue
        reused = None
        if cache is not None:
            times = tuple((rows[worker_id, i]["arrival"], starts[rows[worker_id, i]["node"]],
                           completions[rows[worker_id, i]["node"]]) for i in range(len(visits)))
            reused = cache.scheduled_route(route_entries[worker_id], times)
        scheduled = []
        for index, visit in enumerate(visits):
            row = rows[worker_id, index]
            completion = completions[row["node"]]
            if visit.operation == "delivery":
                tardiness[visit.task_id] = _finite(max(0.0, completion - item.task_map[visit.task_id].promised_delivery_time_s),
                                                  "tardiness", **row["context"])
            if reused is None:
                scheduled.append(ScheduledRecoveryVisit(
                    visit=visit, incoming_path=row["path"], arrival_time_s=row["arrival"],
                    start_time_s=starts[row["node"]], completion_time_s=completion,
                    load_after=row["load"], carried_task_ids=row["held"],
                ))
        routes[worker_id] = tuple(scheduled) if reused is None else reused
        if cache is not None and reused is None:
            cache.remember_schedule(route_entries[worker_id], times, routes[worker_id])
    try:
        if score_only:
            from common.evaluation._recovery_score import _CandidateEvaluation
            result = _CandidateEvaluation(plan=plan, routes=routes, tardiness_s=tardiness)
        else:
            result = EvaluatedRecoveryPlan(plan=plan, routes=routes, tardiness_s=tardiness)
    except (ValueError, OverflowError) as error:
        raise RecoveryPlanValidationError("numeric_range", str(error)) from error
    if cache is not None:
        for key, members in missing_components:
            cache.put_component(key, tuple((worker, routes[worker]) for worker in members))
    return result
