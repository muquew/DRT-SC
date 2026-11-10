import math

from common.data.roads.schema import RoadPath
from common.data.roads.routing import Routing
from common.data.roads.routing import _duration
from common.initial_sequences.inputs import PlanningInput, _exceeds_capacity
from common.initial_sequences.generator import GeneratorConfig, GenerationResult
from common.initial_sequences.plan import InitialPlan, ScheduledVisit


class PlanValidationError(ValueError):


    def __init__(self, code, message, *, worker_id=None, visit_index=None, task_id=None, result=None):
        super().__init__(message)
        self.code, self.message = code, message
        self.worker_id, self.visit_index, self.task_id = worker_id, visit_index, task_id
        self.result = result

    def to_dict(self):
        return {"code": self.code, "message": self.message, "worker_id": self.worker_id,
                "visit_index": self.visit_index, "task_id": self.task_id}


def validate_generation_input(planning_input, routing, method_config):


    if not isinstance(planning_input, PlanningInput) or not isinstance(routing, Routing):
        raise PlanValidationError("input_type", "expected PlanningInput and Routing")
    if not isinstance(method_config, GeneratorConfig):
        raise PlanValidationError("method_config", "expected GeneratorConfig")
    try:
        planning_input.validate_roads(routing.roads)
    except (ValueError, TypeError) as error:
        raise PlanValidationError("road_snapshot", str(error)) from error
    if method_config.information_mode == "online":
        for task in planning_input.tasks:
            if task.known_at_s is None or task.known_at_s > planning_input.planning_time_s:
                raise PlanValidationError("online_information", "task attributes are not known at planning time",
                                          task_id=task.id)
            if task.release_time_s is not None and task.release_time_s > planning_input.planning_time_s:
                raise PlanValidationError("online_information", "task has not been released at planning time",
                                          task_id=task.id)


def _structure(planning_input, result):

    if result.input_fingerprint != planning_input.fingerprint:
        raise PlanValidationError("input_fingerprint", "result belongs to a different planning input")
    if result.status != "proposed":
        raise PlanValidationError("generation_status", f"generator returned {result.status}: {result.termination_reason}")
    if result.uncovered_task_ids:
        raise PlanValidationError("uncovered_tasks", "a complete proposal cannot declare uncovered obligations")
    if set(result.sequences) != set(planning_input.worker_map):
        raise PlanValidationError("worker_coverage", "sequences must include every input worker, including idle workers")
    candidates = set(planning_input.candidate_task_ids)
    decisions = [set(result.accepted_task_ids), set(result.rejected_task_ids), set(result.deferred_task_ids)]
    if set.union(*decisions) != candidates or sum(map(len, decisions)) != len(candidates):
        raise PlanValidationError("candidate_partition", "accepted, rejected and deferred IDs must partition candidate requests")
    service = set(planning_input.required_task_ids) | decisions[0]
    holders = {task_id: worker.id for worker in planning_input.workers for task_id in worker.carried_task_ids}
    owners, seen = dict(holders), set()
    for worker in planning_input.workers:
        visits = result.sequences[worker.id]
        n = worker.fixed_prefix_length
        if visits[:n] != worker.remaining_visits[:n]:
            raise PlanValidationError("fixed_prefix", "the worker's fixed prefix was changed", worker_id=worker.id)
        for k, visit in enumerate(visits, 1):
            context = {"worker_id": worker.id, "visit_index": k, "task_id": visit.task_id}
            if visit.task_id not in service:
                raise PlanValidationError("service_membership", "visit is outside the final service set", **context)
            key = (visit.task_id, visit.operation)
            if key in seen:
                raise PlanValidationError("duplicate_visit", "task operation appears more than once", **context)
            seen.add(key)
            if visit.operation == "pickup" and visit.task_id in holders:
                raise PlanValidationError("carried_pickup", "an initially carried item must not be picked up again", **context)
            if owners.setdefault(visit.task_id, worker.id) != worker.id:
                raise PlanValidationError("task_owner", "pickup and delivery must retain one worker; no transfer was proposed", **context)
    expected = {(identifier, "delivery") for identifier in service}
    expected.update((identifier, "pickup") for identifier in service if identifier not in holders)
    if seen != expected:
        missing = sorted(expected - seen)
        raise PlanValidationError("service_coverage", f"missing required service operations: {missing}")
    return service


def _load(onboard, planning_input, context):
    try:
        return math.fsum(planning_input.task_map[i].load for i in sorted(onboard))
    except OverflowError as error:
        raise PlanValidationError("numeric_range", "carried load exceeds the finite numeric range", **context) from error


def _finite(value, label, context):
    if not math.isfinite(value):
        raise PlanValidationError("numeric_range", f"{label} exceeds the finite numeric range", **context)
    return value


def checked_path(routing, origin, destination, profile, **context):

    try:
        path = routing.route(origin, destination, profile)
    except (ValueError, OverflowError) as error:
        raise PlanValidationError("routing_error", str(error), **context) from error
    if path is None:
        raise PlanValidationError("unreachable", "no directed path connects consecutive visits", **context)
    if (not isinstance(path, RoadPath) or path.network_id != routing.roads.network_id
            or path.snapshot_id != routing.roads.snapshot_id or path.profile != profile
            or path.origin != origin or path.destination != destination):
        raise PlanValidationError("path_identity", "routing returned an inconsistent path", **context)
    try:
        for leg in path.legs:
            arc = routing.roads.arc_map.get(leg.arc_id)
            if arc is None or not arc.enabled or arc.segment_id != leg.segment_id:
                raise PlanValidationError("path_arc", "path uses an unavailable or inconsistent road arc", **context)
            segment = routing.roads.segment_map[arc.segment_id]
            expected_time = _duration(leg.length_m, segment.length_m, arc.travel_time_s, profile.speed_factor)
            if not math.isclose(leg.duration_s, expected_time, rel_tol=1e-12, abs_tol=0):
                raise PlanValidationError("path_cost", "path duration differs from the current road cost", **context)
        if routing.position_after(path, path.duration_s) != destination:
            raise ValueError("path does not reach the service location")
    except PlanValidationError:
        raise
    except (ValueError, OverflowError) as error:
        raise PlanValidationError("schedule", str(error), **context) from error
    return path


def _schedule(planning_input, result, routing):
    routes, tardiness = {}, {}
    for worker in planning_input.workers:
        location = routing.roads.normalize_position(worker.location)
        completion = worker.available_time_s
        onboard, scheduled = set(worker.carried_task_ids), []
        for k, visit in enumerate(result.sequences[worker.id], 1):
            task = planning_input.task_map[visit.task_id]
            context = {"worker_id": worker.id, "visit_index": k, "task_id": task.id}
            if visit.operation == "pickup":
                if task.id in onboard:
                    raise PlanValidationError("custody", "worker already holds the picked-up item", **context)
                destination, ready = task.pickup, task.pickup_ready_time_s
                onboard.add(task.id)
            else:
                if task.id not in onboard:
                    raise PlanValidationError("custody", "worker cannot deliver an item it does not hold", **context)
                destination, ready = task.delivery, task.delivery_ready_time_s
                onboard.remove(task.id)
            load_after = _load(onboard, planning_input, context)
            if _exceeds_capacity(load_after, worker.capacity):
                raise PlanValidationError("capacity", "carrying capacity exceeded after visit", **context)
            destination = routing.roads.normalize_position(destination)
            path = checked_path(routing, location, destination, worker.profile, **context)
            try:
                arrival = _finite(completion + path.duration_s, "arrival time", context)
                if path.duration_s > 0 and arrival <= completion:
                    raise PlanValidationError("numeric_range", "travel is below the time base's numeric resolution", **context)
                start = max(arrival, ready)
                _finite(start - arrival, "waiting time", context)
                service = task.service_duration_s(visit.operation)
                completion = _finite(start + service if service else start, "service completion time", context)
                if service > 0 and completion <= start:
                    raise PlanValidationError("numeric_range", "service is below the time base's numeric resolution", **context)
                if visit.operation == "delivery":
                    tardiness[task.id] = _finite(max(0.0, completion - task.promised_delivery_time_s), "tardiness", context)
                scheduled.append(ScheduledVisit(
                    visit=visit, incoming_path=path, arrival_time_s=arrival, start_time_s=start,
                    load_after=load_after, carried_task_ids=tuple(sorted(onboard)), service_time_s=service,
                ))
            except PlanValidationError:
                raise
            except (ValueError, OverflowError) as error:
                raise PlanValidationError("schedule", str(error), **context) from error
            location = destination
        if onboard:
            raise PlanValidationError("remaining_custody", "worker finishes with undelivered items", worker_id=worker.id)
        routes[worker.id] = tuple(scheduled)
    try:
        return InitialPlan(
            input_fingerprint=planning_input.fingerprint, network_id=planning_input.network_id,
            snapshot_id=planning_input.snapshot_id, generation_result=result, routes=routes, tardiness_s=tardiness,
        )
    except (ValueError, OverflowError) as error:
        raise PlanValidationError("plan_structure", str(error)) from error


def evaluate_generation(planning_input, result, routing):


    if not isinstance(result, GenerationResult):
        raise PlanValidationError("result_type", "expected GenerationResult")
    try:
        validate_generation_input(planning_input, routing, result.method_config)
        _structure(planning_input, result)
        plan = _schedule(planning_input, result, routing)


        object.__setattr__(plan, "_validated_for", (planning_input.fingerprint, routing.roads.snapshot_id))
        return plan
    except PlanValidationError as error:
        error.result = result
        raise
