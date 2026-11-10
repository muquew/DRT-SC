from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import math
from types import MappingProxyType

from common.data.roads.schema import RoadPosition

from common.initial_sequences.generator import GenerationResult
from common.initial_sequences.inputs import PlanningVisit, _exceeds_capacity


@dataclass(frozen=True)
class RouteEvaluation:
    travel_s: float
    tardiness_s: float
    finish_s: float
    delivery_times: Mapping[str, float]
    start_times: tuple[float, ...]
    loads_after: tuple[float, ...]
    end_position: RoadPosition


def _sum(values):
    try:
        value = math.fsum(values)
    except OverflowError as error:
        raise ValueError("route total exceeds the finite numeric range") from error
    if not math.isfinite(value):
        raise ValueError("route total exceeds the finite numeric range")
    return value


class RouteEvaluator:


    def __init__(self, planning_input, routing):
        planning_input.validate_roads(routing.roads)
        self.input, self.routing = planning_input, routing
        self.stats = {"route_evaluations": 0, "route_cache_hits": 0, "road_queries": 0}
        self.initial_holders = {t: w.id for w in self.input.workers for t in w.carried_task_ids}
        self._travel = lru_cache(maxsize=131072)(self._travel_duration)
        self._cached_evaluate = lru_cache(maxsize=4096)(self._evaluate)

    def travel(self, worker_id, origin, destination):
        worker = self.input.worker_map[worker_id]
        origin = self.routing.roads.normalize_position(origin)
        destination = self.routing.roads.normalize_position(destination)
        return self._travel(origin, destination, worker.profile)

    def _travel_duration(self, origin, destination, profile):

        self.stats["road_queries"] += 1
        return self.routing.route_duration(origin, destination, profile)

    def evaluate(self, worker_id, visits, complete=True):
        if type(complete) is not bool:
            raise ValueError("complete must be boolean")
        visits = tuple(visits)
        if worker_id not in self.input.worker_map:
            raise ValueError("unknown worker ID")
        if any(not isinstance(v, PlanningVisit) or v.task_id not in self.input.task_map for v in visits):
            raise ValueError("route must contain known PlanningVisit records")
        hits = self._cached_evaluate.cache_info().hits
        result = self._cached_evaluate(worker_id, visits, complete)
        self.stats["route_cache_hits"] += self._cached_evaluate.cache_info().hits - hits
        return result

    def _evaluate(self, worker_id, visits, complete):
        self.stats["route_evaluations"] += 1
        worker = self.input.worker_map[worker_id]
        fixed = worker.fixed_prefix_length
        if complete and len(visits) < fixed:
            return None
        if visits[:fixed] != worker.remaining_visits[:min(fixed, len(visits))]:
            return None
        onboard = set(worker.carried_task_ids)
        time, location = worker.available_time_s, worker.location
        durations, waits, lateness, starts, loads, deliveries = [], [], [], [], [], {}
        seen = set()
        for visit in visits:
            key = visit.task_id, visit.operation
            if key in seen:
                return None
            seen.add(key)
            task = self.input.task_map[visit.task_id]
            if visit.operation == "pickup":
                if task.id in self.initial_holders or task.id in onboard:
                    return None
                position, ready = task.pickup, task.pickup_ready_time_s
                onboard.add(task.id)
            else:
                if task.id not in onboard:
                    return None
                position, ready = task.delivery, task.delivery_ready_time_s
                onboard.remove(task.id)
            load = _sum(self.input.task_map[t].load for t in sorted(onboard))
            if _exceeds_capacity(load, worker.capacity):
                return None
            duration = self.travel(worker_id, location, position)
            if not math.isfinite(duration):
                return None
            arrival = time + duration
            if not math.isfinite(arrival) or (duration > 0 and arrival <= time):
                raise ValueError("route time exceeds the supported numeric resolution")
            time = max(arrival, ready)
            start = time
            wait = time - arrival
            if not math.isfinite(wait):
                raise ValueError("waiting time exceeds the finite numeric range")
            waits.append(wait)
            service = task.service_duration_s(visit.operation)
            if service:
                time += service
            if not math.isfinite(time) or (service > 0 and time <= start):
                raise ValueError("service time exceeds the supported numeric resolution")
            if visit.operation == "delivery":
                deliveries[task.id] = time
                lateness.append(max(0.0, time - task.promised_delivery_time_s))
            durations.append(duration)
            starts.append(start)
            loads.append(load)
            location = position
        if complete and onboard:
            return None
        travel, tardy = _sum(durations), _sum(lateness)
        _sum(waits)
        if not all(math.isfinite(value) for value in (travel, tardy, time)):
            raise ValueError("route score exceeds the finite numeric range")
        return RouteEvaluation(travel, tardy, time, MappingProxyType(deliveries),
                               tuple(starts), tuple(loads), location)


def base_routes(evaluator, held_route=None):

    result = {}
    for worker in evaluator.input.workers:
        visits = worker.remaining_visits
        if not visits and worker.carried_task_ids:
            if held_route is None:
                raise ValueError("the adapter must specify how it sequences carried deliveries")
            visits = held_route(worker.id)
            if visits is None:
                return None
        if evaluator.evaluate(worker.id, visits) is None:
            return None
        result[worker.id] = tuple(visits)
    return result


def pending_tasks(evaluator, routes):
    present = {v.task_id for route in routes.values() for v in route}
    required = set(evaluator.input.required_task_ids)
    return tuple(t.id for t in sorted(
        (t for t in evaluator.input.tasks if t.id not in present),
        key=lambda t: (t.id not in required,
                       t.pickup_ready_time_s if t.pickup_ready_time_s is not None else -math.inf, t.id),
    ))


def paired_insertions(route, task_id, min_gap=0):

    route = tuple(route)
    pickup = PlanningVisit(task_id=task_id, operation="pickup")
    delivery = PlanningVisit(task_id=task_id, operation="delivery")
    for i in range(min_gap, len(route) + 1):
        for j in range(i, len(route) + 1):
            yield route[:i] + (pickup,) + route[i:j] + (delivery,) + route[j:]


def delivery_insertions(route, task_id, min_gap=0):

    route = tuple(route)
    delivery = PlanningVisit(task_id=task_id, operation="delivery")
    for i in range(min_gap, len(route) + 1):
        yield route[:i] + (delivery,) + route[i:]


def finish_result(planning_input, config, routes, diagnostics=None, rejected_task_ids=()):
    if set(routes) - planning_input.worker_map.keys():
        raise ValueError("constructed routes contain unknown workers")
    required, candidates = set(planning_input.required_task_ids), set(planning_input.candidate_task_ids)
    held = {i for w in planning_input.workers for i in w.carried_task_ids}
    visits = [v for route in routes.values() for v in route]
    counts = Counter((v.task_id, v.operation) for v in visits)
    present = {v.task_id for v in visits}
    covered = {i for i in present if counts[(i, "delivery")] == 1
               and counts[(i, "pickup")] == (0 if i in held else 1)}
    if present - planning_input.task_map.keys():
        raise ValueError("constructed route contains unknown tasks")
    rejected = set(rejected_task_ids)
    if rejected - candidates or rejected & covered:
        raise ValueError("only unserved candidate tasks may be rejected")
    uncovered = required - covered
    complete = not uncovered and not present - covered
    return GenerationResult(
        input_fingerprint=planning_input.fingerprint, method_config=config,
        status="proposed" if complete else "partial",
        sequences={w.id: tuple(routes.get(w.id, ())) for w in planning_input.workers},
        accepted_task_ids=tuple(sorted(candidates & covered)),
        rejected_task_ids=tuple(sorted(rejected)),
        deferred_task_ids=tuple(sorted(candidates - covered - rejected)),
        uncovered_task_ids=tuple(sorted(uncovered)),
        termination_reason="complete proposal" if complete else "construction did not cover all required services",
        diagnostics={} if diagnostics is None else diagnostics,
    )


def unsupported_result(planning_input, config, reason):
    return GenerationResult(
        input_fingerprint=planning_input.fingerprint, method_config=config, status="unsupported",
        sequences={w.id: () for w in planning_input.workers},
        deferred_task_ids=planning_input.candidate_task_ids,
        uncovered_task_ids=planning_input.required_task_ids, termination_reason=reason,
    )
