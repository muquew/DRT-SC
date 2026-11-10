from dataclasses import asdict
import math
import random

from common.initial_sequences.generator import GenerationResult, GeneratorConfig
from common.initial_sequences.adapters._common import RouteEvaluator, base_routes, delivery_insertions, finish_result, paired_insertions, pending_tasks, unsupported_result


_DESTROY = ("random", "late", "long_segment", "similar", "cluster")
_REPAIR = ("greedy", "regret2")
_DEFAULTS = {
    "travel_weight": 1.0, "tardiness_weight": 1.0,
    "iterations": 16, "remove_fraction": .25, "repair_restarts": 2,
    "reaction": .2, "weight_floor": .05, "temperature_ratio": .05,
    "cooling": .95, "tabu_iterations": 1, "tabu_neighbors": 6, "tabu_tenure": 3,
    "cluster_iterations": 5, "destroy_operators": _DESTROY, "repair_operators": _REPAIR,
}
_ADAPTATION = {
    "source": "Tu et al., doi:10.1109/JIOT.2019.2930984, Section V / Algorithms 1-2",
    "scope": "one invocation with supplied batch or online-visible input",
    "matching": "greedy worker-to-provider travel-time matching until provider demand is covered; a construction preference, not a permanent ownership constraint",
    "matching_capacity": "follows the prose demand-coverage rule rather than Algorithm 1's inconsistent strict capacity-sum comparison",
    "visits": "explicit paired pickup/delivery operations; initial custody and fixed prefixes retained",
    "held_initialization": "insert held deliveries by weighted travel/tardiness cost when no remaining route is supplied",
    "cluster": "fixed-budget k-means on range-standardized delivery x, y and promise; sampled cluster members are capped at the removal count; differs from the original road-cost metric",
    "weights": "positive relative gain EWMA with an explicit weight floor; avoids zero-denominator updates",
    "tabu": "complete-task relocation and pair exchange; road-time neighbours replace Voronoi neighbours",
    "budget": "fixed iterations, bounded repair restarts and tabu neighbourhoods",
    "objective_units": "travel and tardiness are measured in seconds with explicit objective weights",
    "outside_scope": "online event simulation and entropy-based rider predispatch",
}


def _parameters(values):
    if set(values) != set(_DEFAULTS):
        raise ValueError("OCD requires exactly the parameter set produced by default_config")
    p = dict(values)
    for name in ("iterations", "tabu_iterations"):
        if type(p[name]) is not int or p[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("repair_restarts", "tabu_neighbors", "tabu_tenure", "cluster_iterations"):
        if type(p[name]) is not int or p[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("travel_weight", "tardiness_weight", "remove_fraction", "reaction",
                 "weight_floor", "temperature_ratio", "cooling"):
        if type(p[name]) not in (int, float) or not math.isfinite(p[name]) or p[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if p["travel_weight"] == p["tardiness_weight"] == 0:
        raise ValueError("at least one objective weight must be positive")
    for name in ("remove_fraction", "reaction", "weight_floor", "cooling"):
        if not 0 < p[name] <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    for name, choices in (("destroy_operators", _DESTROY), ("repair_operators", _REPAIR)):
        value = p[name]
        if (not isinstance(value, (tuple, list)) or not value
                or any(not isinstance(item, str) or item not in choices for item in value)
                or len(set(value)) != len(value)):
            raise ValueError(f"{name} must contain distinct known operators")
        p[name] = tuple(value)
    return p


def default_config(information_mode="batch", seed=0, **parameters):
    values = {**_DEFAULTS, **parameters}
    values = _parameters(values)
    if type(seed) is not int:
        raise ValueError("OCD requires an integer random seed")
    return GeneratorConfig(
        name="ocd_alns_tabu", information_mode=information_mode,
        objective="travel_weight * travel_s + tardiness_weight * tardiness_s",
        references=("https://doi.org/10.1109/JIOT.2019.2930984",), seed=seed, parameters=values,
    )


def _route_key(route):
    return tuple((visit.task_id, visit.operation) for visit in route)


class _Search:
    def __init__(self, evaluator, config, parameters):
        self.ev, self.config, self.p = evaluator, config, parameters
        self.input, self.rng = evaluator.input, random.Random(config.seed)
        self.worker_ids = tuple(sorted(self.input.worker_map))
        self.protected = {identifier for worker in self.input.workers for identifier in worker.carried_task_ids}
        self.protected.update(visit.task_id for worker in self.input.workers
                              for visit in worker.remaining_visits[:worker.fixed_prefix_length])
        self.weights = [{name: 1.0 for name in parameters["destroy_operators"]},
                        {name: 1.0 for name in parameters["repair_operators"]}]
        self.stats = {"adaptation": _ADAPTATION, "provider_matches": {},
                      "destroy_counts": {name: 0 for name in parameters["destroy_operators"]},
                      "repair_counts": {name: 0 for name in parameters["repair_operators"]},
                      "repair_attempts": 0, "failed_repairs": 0, "accepted_alns": 0,
                      "accepted_uphill_alns": 0, "tabu_relocation_evaluations": 0,
                      "tabu_exchange_evaluations": 0, "tabu_moves": 0, "tabu_blocked": 0,
                      "tabu_aspirations": 0, "weight_updates": 0, "iterations_completed": 0}
        self.tabu, self.tabu_step = {}, 0

    def cost(self, evaluation):
        value = math.fsum((self.p["travel_weight"] * evaluation.travel_s,
                           self.p["tardiness_weight"] * evaluation.tardiness_s))
        if not math.isfinite(value):
            raise ValueError("weighted OCD objective exceeds the finite numeric range")
        return value

    def total(self, routes):
        value = math.fsum(self.cost(self.ev.evaluate(wid, routes[wid])) for wid in self.worker_ids)
        if not math.isfinite(value):
            raise ValueError("total OCD objective exceeds the finite numeric range")
        return value

    def options(self, identifier, routes, worker_ids=None):

        choices = []
        for wid in self.worker_ids if worker_ids is None else worker_ids:
            old = self.ev.evaluate(wid, routes[wid])
            if old is None:
                continue
            best = None
            for route in paired_insertions(routes[wid], identifier,
                                           min_gap=self.input.worker_map[wid].fixed_prefix_length):
                evaluated = self.ev.evaluate(wid, route)
                if evaluated is None:
                    continue
                delta = self.cost(evaluated) - self.cost(old)
                choice = (delta, wid, _route_key(route), route)
                if best is None or choice[:3] < best[:3]:
                    best = choice
            if best is not None:
                choices.append(best)
        return sorted(choices, key=lambda choice: choice[:3])

    def provider_matching(self, tasks):
        demand = {}
        for task in tasks:
            demand[task.pickup] = demand.get(task.pickup, 0) + task.load

        providers = sorted(demand, key=lambda pos: repr(asdict(pos)))
        edges = []
        for wid in self.worker_ids:
            worker = self.input.worker_map[wid]
            for j, provider in enumerate(providers):
                travel = self.ev.travel(wid, worker.location, provider)
                if math.isfinite(travel):
                    edges.append((travel, wid, j))
        assigned, covered = {}, {position: 0 for position in providers}
        for _, wid, j in sorted(edges):
            provider = providers[j]
            if wid not in assigned and covered[provider] < demand[provider]:
                assigned[wid] = provider
                covered[provider] += self.input.worker_map[wid].capacity
        self.stats["provider_matches"] = {wid: asdict(position) for wid, position in assigned.items()}
        return assigned

    def held_route(self, worker_id):

        route = ()
        worker = self.input.worker_map[worker_id]
        identifiers = sorted(worker.carried_task_ids,
                             key=lambda i: (self.input.task_map[i].promised_delivery_time_s, i))
        for identifier in identifiers:
            choices = []
            for candidate in delivery_insertions(route, identifier):
                evaluation = self.ev.evaluate(worker_id, candidate, complete=False)
                if evaluation is not None:
                    choices.append((self.cost(evaluation), _route_key(candidate), candidate))
            if not choices:
                return None
            route = min(choices, key=lambda value: value[:2])[2]
        return route

    def construct(self):
        routes = base_routes(self.ev, self.held_route)
        if routes is None:
            return None
        tasks = [self.input.task_map[identifier] for identifier in pending_tasks(self.ev, routes)]
        required = set(self.input.required_task_ids)
        tasks.sort(key=lambda task: (task.id not in required, task.promised_delivery_time_s, task.id))
        matching = self.provider_matching(tasks)
        for task in tasks:
            preferred = [wid for wid in self.worker_ids if matching.get(wid) == task.pickup]
            choices = self.options(task.id, routes, preferred) if preferred else []
            if not choices:
                choices = self.options(task.id, routes)
            if choices:
                _, wid, _, route = choices[0]
                routes[wid] = route
        return routes

    def owners(self, routes):
        return {visit.task_id: wid for wid, route in routes.items() for visit in route if visit.operation == "delivery"}

    def remove(self, routes, identifiers):
        identifiers = set(identifiers)
        return {wid: tuple(visit for visit in route if visit.task_id not in identifiers)
                for wid, route in routes.items()}

    def cluster(self, identifiers, count):
        raw = [(*self.ev.routing.roads.coordinates(self.input.task_map[i].delivery),
                self.input.task_map[i].promised_delivery_time_s) for i in identifiers]
        scales = [max(1.0, *(abs(point[j]) for point in raw)) for j in range(3)]
        raw = [tuple(point[j] / scales[j] for j in range(3)) for point in raw]
        minima = [min(point[j] for point in raw) for j in range(3)]
        spans = [max(point[j] for point in raw) - minima[j] for j in range(3)]
        points = [tuple((point[j] - minima[j]) / spans[j] if spans[j] else 0 for j in range(3)) for point in raw]
        k = max(1, math.ceil(len(identifiers) / count))
        centers = [points[i] for i in self.rng.sample(range(len(points)), k)]
        groups = []
        for _ in range(self.p["cluster_iterations"]):
            groups = [[] for _ in centers]
            for index, point in enumerate(points):
                target = min(range(k), key=lambda j: (math.dist(point, centers[j]), j))
                groups[target].append(index)
            centers = [tuple(math.fsum(points[i][j] for i in group) / len(group) for j in range(3))
                       if group else centers[c] for c, group in enumerate(groups)]
        chosen = self.rng.choice([group for group in groups if group])
        chosen = self.rng.sample(chosen, min(count, len(chosen)))
        return tuple(identifiers[i] for i in chosen)

    def destroy(self, name, routes, identifiers, count):
        if name == "random":
            return tuple(self.rng.sample(identifiers, count))
        if name == "cluster":
            return self.cluster(identifiers, count)
        owners = self.owners(routes)
        if name == "late":
            scores = {i: max(0, self.ev.evaluate(owners[i], routes[owners[i]]).delivery_times[i]
                             - self.input.task_map[i].promised_delivery_time_s) for i in identifiers}
            return tuple(sorted(identifiers, key=lambda i: (-scores[i], i))[:count])
        if name == "long_segment":
            scores = dict.fromkeys(identifiers, 0)
            for wid, route in routes.items():
                previous, previous_task = self.input.worker_map[wid].location, None
                for visit in route:
                    task = self.input.task_map[visit.task_id]
                    position = task.pickup if visit.operation == "pickup" else task.delivery
                    duration = self.ev.travel(wid, previous, position)
                    for identifier in (previous_task, task.id):
                        if identifier in scores:
                            scores[identifier] = max(scores[identifier], duration)
                    previous, previous_task = position, task.id
            return tuple(sorted(identifiers, key=lambda i: (-scores[i], i))[:count])
        seed = self.rng.choice(identifiers)
        task = self.input.task_map[seed]
        wid = owners[seed]
        def similarity(identifier):
            other = self.input.task_map[identifier]
            travel = self.ev.travel(wid, task.delivery, other.delivery)
            spatial = self.p["travel_weight"] * travel if self.p["travel_weight"] else 0
            temporal = (self.p["tardiness_weight"] * abs(task.promised_delivery_time_s - other.promised_delivery_time_s)
                        if self.p["tardiness_weight"] else 0)
            return spatial + temporal
        return tuple(sorted(identifiers, key=lambda i: (similarity(i), i))[:count])

    def repair(self, name, reduced, removed):
        best, best_cost = None, math.inf
        for _ in range(min(self.p["repair_restarts"], len(removed))):
            self.stats["repair_attempts"] += 1
            routes, pending = dict(reduced), list(removed)
            self.rng.shuffle(pending)
            while pending:
                if name == "greedy":
                    identifier = pending[0]
                    options = self.options(identifier, routes)
                else:
                    candidates = []
                    for order, identifier in enumerate(pending):
                        options = self.options(identifier, routes)
                        if not options:
                            continue
                        regret = options[1][0] - options[0][0] if len(options) > 1 else math.inf
                        candidates.append((-regret, order, identifier, options))
                    if not candidates:
                        break
                    _, _, identifier, options = min(candidates, key=lambda value: value[:2])
                if not options:
                    break
                _, wid, _, route = options[0]
                routes[wid] = route
                pending.remove(identifier)
            if pending:
                continue
            cost = self.total(routes)
            if cost < best_cost:
                best, best_cost = routes, cost
        if best is None:
            self.stats["failed_repairs"] += 1
        return best, best_cost

    def update_weights(self, destroy, repair, improvement, old_cost):
        reward = max(0.0, improvement) / max(1.0, abs(old_cost))
        for weights, operator in zip(self.weights, (destroy, repair)):
            weights[operator] = max(self.p["weight_floor"],
                                    (1 - self.p["reaction"]) * weights[operator] + self.p["reaction"] * reward)
        self.stats["weight_updates"] += 1

    def tabu_search(self, current, current_cost, best, best_cost, identifiers):
        for _ in range(self.p["tabu_iterations"]):
            self.tabu_step += 1
            owners = self.owners(current)
            seed = self.rng.choice(identifiers)
            point = self.input.task_map[seed].delivery
            nearby = sorted(identifiers, key=lambda identifier: (
                self.ev.travel(owners[seed], point, self.input.task_map[identifier].delivery), identifier))[:self.p["tabu_neighbors"]]
            moves = []
            for identifier in nearby:
                reduced = self.remove(current, (identifier,))
                for delta, wid, key, route in self.options(identifier, reduced):
                    candidate = dict(reduced)
                    candidate[wid] = route
                    if candidate == current:
                        continue
                    self.stats["tabu_relocation_evaluations"] += 1
                    moves.append((self.total(candidate), "relocate", (identifier,), candidate))
            for index, left in enumerate(nearby):
                for right in nearby[index + 1:]:
                    candidate = dict(current)
                    for wid in {owners[left], owners[right]}:
                        candidate[wid] = tuple(type(visit)(
                            task_id=right if visit.task_id == left else left if visit.task_id == right else visit.task_id,
                            operation=visit.operation) for visit in current[wid])
                    self.stats["tabu_exchange_evaluations"] += 1
                    if all(self.ev.evaluate(wid, candidate[wid]) is not None for wid in {owners[left], owners[right]}):
                        moves.append((self.total(candidate), "exchange", (left, right), candidate))
            chosen = None
            for move in sorted(moves, key=lambda move: move[:3]):
                forbidden = any(self.tabu.get(identifier, 0) > self.tabu_step for identifier in move[2])
                if forbidden and move[0] >= best_cost:
                    self.stats["tabu_blocked"] += 1
                    continue
                if forbidden:
                    self.stats["tabu_aspirations"] += 1
                chosen = move
                break
            if chosen is None:
                continue
            current_cost, _, changed, current = chosen
            for identifier in changed:
                self.tabu[identifier] = self.tabu_step + self.p["tabu_tenure"]
            self.stats["tabu_moves"] += 1
            if current_cost < best_cost:
                best, best_cost = dict(current), current_cost
        return current, current_cost, best, best_cost

    def run(self):
        routes = self.construct()
        self.stats["route_evaluation_stats"] = dict(self.ev.stats)
        if routes is None:
            return GenerationResult(input_fingerprint=self.input.fingerprint, method_config=self.config,
                                    status="no_plan_found", sequences={wid: () for wid in self.worker_ids},
                                    deferred_task_ids=self.input.candidate_task_ids,
                                    uncovered_task_ids=self.input.required_task_ids,
                                    termination_reason="existing worker state has no valid base route", diagnostics=self.stats)
        initial = finish_result(self.input, self.config, routes, diagnostics=self.stats)
        if initial.uncovered_task_ids:
            return initial
        current, best = dict(routes), dict(routes)
        current_cost = best_cost = self.total(routes)
        self.stats["initial_objective"] = current_cost
        self.stats["initial_sequences"] = {wid: [asdict(v) for v in route] for wid, route in routes.items()}
        self.stats["best_history"] = [best_cost]
        identifiers = sorted(set(self.owners(routes)) - self.protected)
        temperature = self.p["temperature_ratio"] * max(1.0, current_cost)
        for _ in range(self.p["iterations"] if identifiers else 0):
            destroy, repair = [self.rng.choices(tuple(weights), weights=tuple(weights.values()))[0] for weights in self.weights]
            self.stats["destroy_counts"][destroy] += 1
            self.stats["repair_counts"][repair] += 1
            count = max(1, math.ceil(len(identifiers) * self.p["remove_fraction"]))
            removed = self.destroy(destroy, current, identifiers, count)
            candidate, candidate_cost = self.repair(repair, self.remove(current, removed), removed)
            delta = candidate_cost - current_cost
            self.update_weights(destroy, repair, -delta if candidate is not None else 0, current_cost)
            accepted = candidate is not None and (delta <= 0 or (
                temperature > 0 and self.rng.random() < math.exp(-delta / temperature)))
            if accepted:
                self.stats["accepted_alns"] += 1
                self.stats["accepted_uphill_alns"] += delta > 0
                current, current_cost = candidate, candidate_cost
                if current_cost < best_cost:
                    best, best_cost = dict(current), current_cost
            current, current_cost, best, best_cost = self.tabu_search(current, current_cost, best, best_cost, identifiers)
            temperature *= self.p["cooling"]
            self.stats["iterations_completed"] += 1
            self.stats["best_history"].append(best_cost)
        self.stats.update(best_objective=best_cost, destroy_weights=self.weights[0], repair_weights=self.weights[1],
                          route_evaluation_stats=dict(self.ev.stats))
        return finish_result(self.input, self.config, best, diagnostics=self.stats)


class OCDAdapter:
    def generate(self, planning_input, routing, method_config):
        from common.evaluation.feasibility import validate_generation_input
        validate_generation_input(planning_input, routing, method_config)
        if method_config.name != "ocd_alns_tabu":
            return unsupported_result(planning_input, method_config, "expected ocd_alns_tabu")
        try:
            parameters = _parameters(method_config.parameters)
            if type(method_config.seed) is not int:
                raise ValueError("OCD requires an integer random seed")
        except ValueError as error:
            return unsupported_result(planning_input, method_config, str(error))
        if method_config.objective != "travel_weight * travel_s + tardiness_weight * tardiness_s":
            return unsupported_result(planning_input, method_config, "objective must declare the configured weighted seconds cost")
        evaluator = RouteEvaluator(planning_input, routing)
        return _Search(evaluator, method_config, parameters).run()
