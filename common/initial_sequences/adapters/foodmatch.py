from dataclasses import replace
from heapq import heapify, heappop, heappush
from itertools import combinations
import math
import random

from common.initial_sequences.generator import GeneratorConfig
from common.initial_sequences.inputs import PlanningInput, PlanningVisit, PlanningWorker
from common.initial_sequences.adapters._common import RouteEvaluator, base_routes, delivery_insertions, finish_result, pending_tasks, unsupported_result


_COMMIT = "3270e0319e88c88134f9bb0100c41a024c2345ad"
_OBJECTIVE = "minimum marginal extra delivery time after maximum-cardinality batch matching"
_DEFAULTS = {"max_batch_size": 3, "batching_quality_cutoff_s": 60.0}
_ADAPTATION = {
    "mechanism": "agglomerative order graph with marginal XDT edges, then global batch-worker matching",
    "batch_reference": "virtual empty workers at each first pickup, fastest supplied profile and largest capacity",
    "xdt_reference": "earliest direct delivery completion from max(planning time, pickup readiness), including both service durations and customer readiness, using the shared fastest profile",
    "route_search": "enumerate precedence-valid orders within each batch; maximum supported batch size is four",
    "quality_cutoff": "maximum average batch XDT, checked before a proposed merge",
    "existing_routes": "preserve the entire existing remaining route, including held deliveries and fixed prefix; append batches",
    "initial_custody": "without an issued route, insert held deliveries in task-ID order, minimizing total delivery completion time over every insertion gap; partial checks retain all initial items",
    "capacity": "actual onboard load after each visit, not total assigned load",
    "matching": "maximum cardinality in batches, then minimum total marginal XDT; exact integer scaling of binary64 edge costs",
    "scope": "static repeated matching rounds on a fixed road snapshot and fixed worker set; required tasks before candidates",
    "outside_scope": "dynamic simulation, angular prediction, best-first sparse graph, original rejection penalties, and existing-route interleaving",
}


def _parameters(values):
    if set(values) != set(_DEFAULTS):
        raise ValueError("FoodMatch requires the parameter set produced by default_config")
    p = dict(values)
    if type(p["max_batch_size"]) is not int or not 1 <= p["max_batch_size"] <= 4:
        raise ValueError("max_batch_size must be an integer in [1, 4] for bounded exact enumeration")
    cutoff = p["batching_quality_cutoff_s"]
    if type(cutoff) not in (int, float) or not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("batching_quality_cutoff_s must be finite and nonnegative")
    return p


def default_config(information_mode="batch", seed=0, **parameters):

    p = _parameters({**_DEFAULTS, **parameters})
    if type(seed) is not int:
        raise ValueError("FoodMatch requires an integer tie-breaking seed")
    return GeneratorConfig(
        name="foodmatch_batch", information_mode=information_mode,
        objective=_OBJECTIVE,
        references=("https://arxiv.org/abs/2008.12905",
                    f"https://github.com/idea-iitd/FoodMatch/blob/{_COMMIT}/Swiggy/include/vehicle_assignment.cpp",
                    f"https://github.com/idea-iitd/FoodMatch/blob/{_COMMIT}/Swiggy/include/routeplan.cpp"),
        seed=seed, parameters=p,
    )


def _sum(values):
    value = math.fsum(values)
    if not math.isfinite(value):
        raise ValueError("FoodMatch cost exceeds the finite numeric range")
    return value


def _route_key(route):
    return tuple((visit.task_id, visit.operation) for visit in route)


def _initial_carried_route(evaluator, worker_id):


    route = ()
    for task_id in sorted(evaluator.input.worker_map[worker_id].carried_task_ids):
        best = None
        for candidate in delivery_insertions(route, task_id):
            score = evaluator.evaluate(worker_id, candidate, complete=False)
            if score is None:
                continue
            key = (_sum(score.delivery_times.values()), score.travel_s, _route_key(candidate))
            if best is None or key < best[0]:
                best = key, candidate
        if best is None:
            return None
        route = best[1]
    return route


def _legal_orders(ids):

    def extend(prefix, states):
        if all(state == 2 for state in states):
            yield prefix
            return
        for i, state in enumerate(states):
            if state < 2:
                operation = "pickup" if state == 0 else "delivery"
                changed = states[:i] + (state + 1,) + states[i + 1:]
                yield from extend(prefix + (PlanningVisit(task_id=ids[i], operation=operation),), changed)
    yield from extend((), (0,) * len(ids))


class _BatchCosts:


    def __init__(self, evaluator, ids):
        item = evaluator.input
        fastest = max(item.workers, key=lambda worker: worker.profile.speed_factor).profile
        capacity = max(worker.capacity for worker in item.workers)
        tasks = tuple(item.task_map[i] for i in ids)
        workers = tuple(PlanningWorker(id=f"batch-reference-{n}", location=task.pickup,
                                       available_time_s=item.planning_time_s, capacity=capacity,
                                       profile=fastest) for n, task in enumerate(tasks))
        reference = PlanningInput(
            instance_id=item.instance_id + ":batch-reference", planning_time_s=item.planning_time_s,
            time_origin_s=item.time_origin_s, network_id=item.network_id, snapshot_id=item.snapshot_id,
            load_unit=item.load_unit, tasks=tasks, workers=workers, required_task_ids=tuple(ids),
        )
        self.evaluator = RouteEvaluator(reference, evaluator.routing)
        self.starts = {task.id: worker.id for task, worker in zip(tasks, workers)}
        self.lower = {}
        for task in tasks:
            travel = self.evaluator.travel(self.starts[task.id], task.pickup, task.delivery)
            self.lower[task.id] = (max(max(item.planning_time_s, task.pickup_ready_time_s)
                                       + task.pickup_service_time_s + travel,
                                       task.delivery_ready_time_s) + task.delivery_service_time_s
                                   if math.isfinite(travel) else None)
        self.orders, self.costs = {}, {}

    def sequences(self, batch):
        if batch not in self.orders:
            self.orders[batch] = tuple(_legal_orders(batch))
        return self.orders[batch]

    def extra(self, evaluation, batch):
        if any(self.lower[i] is None for i in batch):
            return None
        return _sum(evaluation.delivery_times[i] - self.lower[i] for i in batch)

    def cost(self, batch):
        if batch not in self.costs:
            best = None
            for route in self.sequences(batch):
                score = self.evaluator.evaluate(self.starts[route[0].task_id], route)
                if score is not None:
                    cost = self.extra(score, batch)
                    if cost is not None and (best is None or cost < best):
                        best = cost
            self.costs[batch] = best
        return self.costs[batch]


def _cluster(ids, costs, options, worker_ids, parameters, rank):


    batches = [(i,) for i in ids if any(options(wid, (i,)) is not None for wid in worker_ids)]
    merges = []
    active = set(batches)
    batch_rank = {batch: tuple(sorted(rank[i] for i in batch)) for batch in batches}

    def proposal(left, right):
        combined = tuple(sorted(left + right))
        if len(combined) > parameters["max_batch_size"]:
            return None
        merged_cost = costs.cost(combined)
        if merged_cost is None:
            return None
        marginal = _sum((merged_cost, -costs.cost(left), -costs.cost(right)))
        tie = tuple(sorted(rank[i] for i in combined))
        return marginal, tie, combined, left, right


    proposals = []
    for left, right in combinations(batches, 2):
        candidate = proposal(left, right)
        if candidate is not None:
            proposals.append(candidate)
    heapify(proposals)
    while proposals:
        deferred = []
        while proposals:
            candidate = heappop(proposals)
            marginal, tie, combined, left, right = candidate
            if left not in active or right not in active:
                continue


            other_cost = _sum(costs.cost(b) for b in batches if b != left and b != right)
            average = _sum((other_cost, costs.cost(combined))) / (len(batches) - 1)
            if average > parameters["batching_quality_cutoff_s"]:
                deferred.append(candidate)
                continue
            if not any(options(wid, combined) is not None for wid in worker_ids):
                continue
            break
        else:
            break
        active.difference_update((left, right))
        active.add(combined)
        batch_rank[combined] = tie
        batches = [batch for batch in batches if batch != left and batch != right] + [combined]
        batches.sort(key=batch_rank.__getitem__)
        merges.append({"left": left, "right": right, "batch": combined, "marginal_xdt_s": marginal})
        if len(merges) == 1:


            proposals.extend(deferred)
            proposals = [
                (m, t, batch, a, b) if batch_rank[a] < batch_rank[b] else (m, t, batch, b, a)
                for m, t, batch, a, b in proposals if a in active and b in active
            ]
            heapify(proposals)
        else:
            for candidate in deferred:
                if candidate[3] in active and candidate[4] in active:
                    heappush(proposals, candidate)
        for batch in batches:
            if batch == combined:
                continue
            left, right = ((batch, combined) if batch_rank[batch] < tie else (combined, batch))
            candidate = proposal(left, right)
            if candidate is not None:
                heappush(proposals, candidate)
    return tuple(batches), merges


def _matching(worker_ids, batches, options):
    import networkx as nx

    edges = []
    denominator = 1
    for wid in worker_ids:
        for batch in batches:
            candidate = options(wid, batch)
            if candidate is not None:
                numerator, divisor = float(candidate[0]).as_integer_ratio()
                denominator = max(denominator, divisor)
                edges.append((wid, batch, numerator, divisor))
    graph = nx.Graph()
    graph.add_nodes_from(("worker", wid) for wid in worker_ids)
    graph.add_nodes_from(("batch", batch) for batch in batches)
    for wid, batch, numerator, divisor in edges:
        graph.add_edge(("worker", wid), ("batch", batch), weight=-numerator * (denominator // divisor))
    selected = []
    for left, right in nx.max_weight_matching(graph, maxcardinality=True, weight="weight"):
        worker_node, batch_node = (left, right) if left[0] == "worker" else (right, left)
        selected.append((worker_node[1], batch_node[1]))
    return tuple(sorted(selected)), len(edges)


class FoodMatchAdapter:
    def generate(self, planning_input, routing, method_config):
        from common.evaluation.feasibility import validate_generation_input

        validate_generation_input(planning_input, routing, method_config)
        if (method_config.name != "foodmatch_batch"
                or method_config.objective != _OBJECTIVE):
            return unsupported_result(planning_input, method_config,
                                      "expected foodmatch_batch with its declared XDT objective")
        try:
            parameters = _parameters(method_config.parameters)
            if type(method_config.seed) is not int:
                raise ValueError("FoodMatch requires an integer tie-breaking seed")
        except ValueError as error:
            return unsupported_result(planning_input, method_config, str(error))
        evaluator = RouteEvaluator(planning_input, routing)
        routes = base_routes(evaluator, held_route=lambda wid: _initial_carried_route(evaluator, wid))
        diagnostics = {"adaptation": _ADAPTATION, "parameters": parameters, "rounds": []}
        if routes is None:
            reason = "issued routes or FoodMatch carried-delivery initialization could not be scheduled"
            diagnostics["termination"] = reason
            result = finish_result(planning_input, method_config,
                                   {worker.id: worker.remaining_visits for worker in planning_input.workers},
                                   diagnostics=diagnostics)
            return replace(result, status="no_plan_found", termination_reason=reason)
        ids = pending_tasks(evaluator, routes)
        if ids and planning_input.workers:
            costs = _BatchCosts(evaluator, ids)
            worker_ids = tuple(sorted(routes))
            shuffled = sorted(ids)
            random.Random(method_config.seed).shuffle(shuffled)
            rank = {identifier: index for index, identifier in enumerate(shuffled)}
            while ids:
                required = tuple(i for i in ids if i in planning_input.required_task_ids)
                active = required or ids
                cache = {}

                def options(wid, batch):
                    key = (wid, batch)
                    if key not in cache:
                        best, best_key = None, None
                        for suffix in costs.sequences(batch):
                            route = tuple(routes[wid]) + suffix
                            score = evaluator.evaluate(wid, route)
                            if score is None:
                                continue
                            marginal = costs.extra(score, batch)
                            if marginal is None:
                                continue
                            candidate_key = (marginal, score.finish_s, score.travel_s, _route_key(route))
                            if best_key is None or candidate_key < best_key:
                                best_key, best = candidate_key, (marginal, route)
                        cache[key] = best
                    return cache[key]

                batches, merges = _cluster(active, costs, options, worker_ids, parameters, rank)
                matched, edge_count = _matching(worker_ids, batches, options)
                record = {"phase": "required" if required else "candidate", "batches": batches,
                          "merges": merges, "feasible_edges": edge_count, "matches": []}
                diagnostics["rounds"].append(record)
                if not matched:
                    diagnostics["termination"] = "no feasible batch-worker edge remains"
                    break
                for wid, batch in matched:
                    marginal, route = options(wid, batch)
                    routes[wid] = route
                    record["matches"].append({"worker_id": wid, "batch": batch, "marginal_xdt_s": marginal})
                ids = pending_tasks(evaluator, routes)
            diagnostics["batch_evaluation_stats"] = dict(costs.evaluator.stats)
        diagnostics["route_evaluation_stats"] = dict(evaluator.stats)
        return finish_result(planning_input, method_config, routes, diagnostics=diagnostics)
