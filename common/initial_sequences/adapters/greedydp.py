from bisect import insort
from dataclasses import replace
import math

from common.initial_sequences.generator import GeneratorConfig
from common.initial_sequences.inputs import PlanningVisit, _exceeds_capacity
from common.initial_sequences.adapters._common import RouteEvaluator, base_routes, finish_result, pending_tasks, unsupported_result


def default_config(information_mode="batch", seed=0, **parameters):

    if set(parameters) - {"worker_candidates"}:
        raise ValueError("GreedyDP parameters: only worker_candidates is supported")
    values = {"worker_candidates": None, **parameters}
    limit = values["worker_candidates"]
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("worker_candidates must be None or a positive integer")
    return GeneratorConfig(
        name="greedydp_waiting", information_mode=information_mode,
        objective="minimum incremental travel time per insertion", seed=seed, parameters=values,
        references=("https://www.vldb.org/pvldb/vol11/p1633-tong.pdf",
                    "https://github.com/BUAA-BDA/ridesharing-GreedyDP/tree/"
                    "b6ec2900047e77efbbf4352f97945cac856ecfba"),
    )


def _count(statistics, name, amount=1):
    statistics[name] = statistics.get(name, 0) + amount


def _frontier(labels, candidate, statistics):

    if candidate is not None:
        insort(labels, candidate)
        _count(statistics, "pickup_labels_created")
    result, earliest = [], math.inf
    for label in labels:
        if label[1] < earliest:
            result.append(label)
            earliest = label[1]
        else:
            _count(statistics, "dominated_labels")
    statistics["max_frontier"] = max(statistics.get("max_frontier", 0), len(result))
    return result


def _advance(time, travel, ready, service=0.0):
    if not math.isfinite(travel):
        return None
    arrival = time + travel
    if not math.isfinite(arrival) or travel > 0 and arrival <= time:
        raise ValueError("GreedyDP arrival time exceeds the finite numeric range or time resolution")
    start = max(arrival, ready)
    if not math.isfinite(start - arrival):
        raise ValueError("GreedyDP waiting time exceeds the finite numeric range")
    completion = start + service if service else start
    if not math.isfinite(completion) or (service > 0 and completion <= start):
        raise ValueError("GreedyDP service time exceeds the finite numeric range or time resolution")
    return completion


def _cost(*terms):
    try:
        return math.fsum(terms)
    except OverflowError as error:
        raise ValueError("GreedyDP travel increment exceeds the finite numeric range") from error


def best_insertion(evaluator, worker_id, route, task_id, statistics=None):


    statistics = {} if statistics is None else statistics
    _count(statistics, "insertion_calls")
    route = tuple(route)
    planning_input = evaluator.input
    worker, task = planning_input.worker_map[worker_id], planning_input.task_map[task_id]
    if task.pickup is None or task.pickup_ready_time_s is None:
        raise ValueError("new insertion requires an uncollected task")
    if task_id in evaluator.initial_holders:
        raise ValueError("an initially held task cannot be inserted as a pickup-delivery pair")
    if any(visit.task_id == task_id for visit in route):
        raise ValueError("inserted task already occurs in the route")
    baseline = evaluator.evaluate(worker_id, route)
    if baseline is None or _exceeds_capacity(task.load, worker.capacity):
        return None

    def travel(origin, destination):
        return evaluator.travel(worker_id, origin, destination)

    def room(onboard):
        try:
            load = math.fsum([planning_input.task_map[i].load for i in sorted(onboard)] + [task.load])
        except OverflowError:
            return False
        return not _exceeds_capacity(load, worker.capacity)

    n = len(route)
    positions = [worker.location]
    ready, edges = [worker.available_time_s], [0.0]
    services = [0.0]
    times = [worker.available_time_s]
    onboard = set(worker.carried_task_ids)
    rooms = [room(onboard)]
    for visit in route:
        old = planning_input.task_map[visit.task_id]
        pickup = visit.operation == "pickup"
        positions.append(old.pickup if pickup else old.delivery)
        edges.append(travel(positions[-2], positions[-1]))
        ready.append(old.pickup_ready_time_s if pickup else old.delivery_ready_time_s)
        service = old.service_duration_s(visit.operation)
        services.append(service)
        start = baseline.start_times[len(services) - 2]
        times.append(start + service if service else start)
        if pickup:
            onboard.add(visit.task_id)
        else:
            onboard.remove(visit.task_id)
        rooms.append(room(onboard))

    pickup, delivery = task.pickup, task.delivery
    direct = travel(pickup, delivery)
    labels, best = [], None
    for j in range(worker.fixed_prefix_length, n + 1):
        here = positions[j]
        to_pickup = travel(here, pickup)


        to_delivery = travel(here, delivery) if labels else math.inf
        if j < n:
            from_delivery = travel(delivery, positions[j + 1])
            removed_edge = edges[j + 1]
        else:
            from_delivery = removed_edge = 0.0

        def consider(cost, finish, pickup_gap):
            nonlocal best
            _count(statistics, "delivery_candidates")
            if finish is None or not math.isfinite(cost):
                return
            if not math.isfinite(baseline.travel_s + cost):
                raise ValueError("GreedyDP route travel exceeds the finite numeric range")
            if j < n:
                _count(statistics, "suffix_checks")
                next_time = _advance(finish, from_delivery, ready[j + 1], services[j + 1])
                if next_time is None:
                    return
            score = cost, j, pickup_gap
            if best is None or score < best:
                best = score

        pickup_time = _advance(times[j], to_pickup, task.pickup_ready_time_s, task.pickup_service_time_s)
        if rooms[j] and pickup_time is not None:
            _count(statistics, "adjacent_candidates")
            finish = _advance(pickup_time, direct, task.delivery_ready_time_s, task.delivery_service_time_s)
            consider(_cost(-removed_edge, to_pickup, direct, from_delivery), finish, j)

        for cost, time, i in labels:
            finish = _advance(time, to_delivery, task.delivery_ready_time_s, task.delivery_service_time_s)
            consider(_cost(cost, -removed_edge, to_delivery, from_delivery), finish, i)

        if j == n:
            break
        if not rooms[j + 1]:
            _count(statistics, "capacity_resets")
            labels = []
            continue
        advanced = []
        for cost, time, i in labels:
            _count(statistics, "label_extensions")
            time = _advance(time, edges[j + 1], ready[j + 1], services[j + 1])
            if time is not None:
                advanced.append((cost, time, i))
        candidate = None
        if rooms[j] and pickup_time is not None:
            from_pickup = travel(pickup, positions[j + 1])
            arrival = _advance(pickup_time, from_pickup, ready[j + 1], services[j + 1])
            cost = _cost(-removed_edge, to_pickup, from_pickup)
            if arrival is not None and math.isfinite(cost):
                candidate = cost, arrival, j
        labels = _frontier(advanced, candidate, statistics)

    if best is None:
        return None
    cost, j, i = best
    visits = (route[:i] + (PlanningVisit(task_id=task_id, operation="pickup"),) + route[i:j]
              + (PlanningVisit(task_id=task_id, operation="delivery"),) + route[j:])
    if evaluator.evaluate(worker_id, visits) is None:

        _count(statistics, "winning_route_validation_failures")
        return None
    return {"route": visits, "added_travel_time": cost, "pickup_gap": i, "delivery_gap": j}


def _held_delivery_route(evaluator, worker_id, statistics):

    worker = evaluator.input.worker_map[worker_id]
    order = tuple(sorted(worker.carried_task_ids, key=lambda identifier: (
        evaluator.input.task_map[identifier].promised_delivery_time_s, identifier)))
    statistics["held_initialization_order"][worker_id] = order
    route = ()
    for identifier in order:
        baseline = evaluator.evaluate(worker_id, route, complete=False)
        best = None
        for gap in range(len(route) + 1):
            candidate = route[:gap] + (PlanningVisit(task_id=identifier, operation="delivery"),) + route[gap:]
            _count(statistics, "held_delivery_candidates")
            value = evaluator.evaluate(worker_id, candidate, complete=False)
            if value is not None:
                score = value.travel_s - baseline.travel_s, gap
                if best is None or score < best[0]:
                    best = score, candidate
        if best is None:
            statistics["held_initialization_failures"][worker_id] = identifier
            return None
        route = best[1]
    return route


class GreedyDPAdapter:


    def __init__(self, progress=None):
        self.progress = progress

    def generate(self, planning_input, routing, method_config):
        from common.evaluation.feasibility import validate_generation_input

        validate_generation_input(planning_input, routing, method_config)
        parameters = method_config.parameters
        worker_limit = parameters.get("worker_candidates")
        if (method_config.name != "greedydp_waiting"
                or method_config.objective != "minimum incremental travel time per insertion"
                or set(parameters) != {"worker_candidates"}
                or worker_limit is not None and (type(worker_limit) is not int or worker_limit < 1)):
            return unsupported_result(planning_input, method_config, "unsupported GreedyDP method configuration")
        routing.register_query_input(planning_input)
        evaluator = RouteEvaluator(planning_input, routing)
        statistics = {"max_frontier": 0, "request_order": [], "failed_insertions": {},
                      "worker_candidates": worker_limit,
                      "shortlist": ("all workers" if worker_limit is None else
                          "estimated terminal arrival using road-coordinate Euclidean lower bound, then worker ID"),
                      "worker_selection": "minimum travel increment, then worker ID",
                      "order_rule": "required first, then pickup readiness and task ID",
                      "held_initialization": "delivery-only insertion in promise/ID order; minimum travel increment, ties by earliest gap",
                      "held_initialization_order": {}, "held_initialization_failures": {},
                      "adaptation": "Pareto pickup labels with two-endpoint waiting; preserved old visit order"}
        issued = {w.id: w.remaining_visits or tuple(PlanningVisit(task_id=i, operation="delivery")
                  for i in w.carried_task_ids) for w in planning_input.workers}
        ordered_tasks = pending_tasks(evaluator, issued)
        terminal = {}

        def report(count, phase, **details):
            if self.progress is not None:
                self.progress(dict(phase=phase, tasks_done=count, tasks_total=len(ordered_tasks),
                    failed_tasks=len(statistics["failed_insertions"]),
                    route_evaluations=evaluator.stats["route_evaluations"],
                    road_queries=evaluator.stats["road_queries"], **details))

        completed = 0
        routes = base_routes(evaluator, held_route=lambda wid: _held_delivery_route(evaluator, wid, statistics))
        if routes is None:
            statistics.update(evaluator.stats)
            result = finish_result(planning_input, method_config,
                                   {w.id: w.remaining_visits for w in planning_input.workers},
                                   diagnostics=statistics)
            return replace(result, status="no_plan_found",
                           termination_reason="existing routes could not be continued under the declared constraints")
        for index, (wid, route) in enumerate(routes.items(), start=1):
            score = evaluator.evaluate(wid, route)
            if score is None:
                raise ValueError("Restored initial-preparation route failed physical validation")
            terminal[wid] = score.finish_s, score.end_position

        for index, task_id in enumerate(ordered_tasks[completed:], start=completed):
            statistics["request_order"].append(task_id)
            if worker_limit is None:
                candidates = sorted(routes)
            else:
                pickup_xy = routing.roads.coordinates(planning_input.task_map[task_id].pickup)
                candidates = sorted(routes, key=lambda wid: (terminal[wid][0] + math.dist(
                    routing.roads.coordinates(terminal[wid][1]), pickup_xy)
                    / (4.0 * planning_input.worker_map[wid].profile.speed_factor), wid))[:worker_limit]
            best = None
            for worker_id in candidates:
                insertion = best_insertion(evaluator, worker_id, routes[worker_id], task_id, statistics)
                if insertion is not None:
                    score = insertion["added_travel_time"], worker_id
                    if best is None or score < best[0]:
                        best = score, insertion
            if best is None:
                statistics["failed_insertions"][task_id] = "no feasible insertion found"
            else:
                (_, worker_id), insertion = best
                routes[worker_id] = insertion["route"]
                score = evaluator.evaluate(worker_id, routes[worker_id])
                terminal[worker_id] = score.finish_s, score.end_position
            report(index + 1, "constructing_initial_plan")
        statistics.update(evaluator.stats)
        return finish_result(planning_input, method_config, routes, diagnostics=statistics)
