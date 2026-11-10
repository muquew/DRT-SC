from collections import Counter, defaultdict
from fractions import Fraction
import math

from common.evaluation.recovery_plan import RecoveryPlanValidationError, evaluate_recovery_plan
from common.execution.replay import PlanReplay
from common.recovery.inputs import RecoveryInput
from common.scenarios.event_sampling import _remaining_legs


def _floor(value):

    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError("scenario diagnostic exceeds the finite numeric range") from error
    if not math.isfinite(number):
        raise ValueError("scenario diagnostic exceeds the finite numeric range")
    return math.nextafter(number, -math.inf) if Fraction(number) > value else number


def _add_lower(start, duration):
    return _floor(Fraction(start) + Fraction(duration))


def _late_lower(completion, promise):
    return _floor(max(Fraction(0), Fraction(completion) - Fraction(promise)))


def _sum_lower(values):
    return _floor(sum((Fraction(value) for value in values), Fraction(0)))


def _grouped_lower(values, boundary):


    observed = _sum_lower(values[t] for t in boundary.delivered_observed)
    committed = _sum_lower(values[t] for t in boundary.delivered_committed)
    suffix = _sum_lower(values[t] for t in boundary.remaining_task_ids)
    return _add_lower(_add_lower(observed, committed), suffix)


def _bounds(item, boundary):
    snapshot, tau = item.source_snapshot, item.time_s
    active = {w.service.visit.task_id: w.service for w in snapshot.workers if w.service is not None}
    clock, ready = {}, {}
    for task in item.tasks:
        clock[task.id] = _late_lower(tau, task.promised_delivery_time_s)
        service = active.get(task.id)
        if service is not None and service.visit.operation == "delivery":

            completion = service.completion_time_s
        else:
            state = snapshot.task_map[task.id]
            if service is not None and service.visit.operation == "pickup":
                picked = service.completion_time_s
            elif state.status == "awaiting_pickup":
                picked = _add_lower(max(tau, task.pickup_ready_time_s), task.pickup_service_time_s)
            else:
                picked = tau
            completion = _add_lower(max(picked, task.delivery_ready_time_s), task.delivery_service_time_s)
        ready[task.id] = _late_lower(completion, task.promised_delivery_time_s)
    return {
        "clock_s": _grouped_lower(clock, boundary),
        "readiness_service_s": _grouped_lower(ready, boundary),
        "definition": "independent_ready_service_relaxation",
    }


def _exposure(replay, item):
    snapshot = item.source_snapshot
    remaining = set(snapshot.unfinished_task_ids)
    changed = {row["event_id"] for row in item.changes if row["changed"]}
    by_kind = defaultdict(set)
    counts, targets = Counter(), set()
    for event in item.events:
        kind, target = event["kind"], event["target_id"]
        counts[kind] += 1
        entity = ("worker" if kind == "worker_withdrawal" else
                  "arc" if kind == "road_time_change" else "task")
        targets.add((entity, target))
        if event["id"] not in changed:
            continue
        exposed = by_kind[kind]
        if entity == "task":
            exposed.add(target)
        elif entity == "worker":
            exposed.update(visit.task_id for visit in snapshot.worker_map[target].remaining_visits
                           if visit.operation == "delivery")
        else:
            for worker in snapshot.workers:
                route = replay.plan.routes[worker.worker_id][worker.completed_visit_count:]
                for index, scheduled in enumerate(route):
                    if index == 0 and worker.phase in ("waiting", "servicing"):
                        continue
                    elapsed = worker.travel.elapsed_time_s if index == 0 and worker.travel is not None else 0.
                    if any(arc_id == target for arc_id, *_ in _remaining_legs(scheduled.incoming_path, elapsed)):
                        exposed.update(row.visit.task_id for row in route[index:]
                                       if row.visit.operation == "delivery")
                        break
    exposed = set().union(*by_kind.values()) if by_kind else set()
    return ({
        "total": len(item.events), "changed": len(changed),
        "by_kind": dict(sorted(counts.items())), "unique_target_count": len(targets),
    }, {
        "definition": "original_sequence_potential_exposure",
        "task_count": len(exposed & remaining), "task_ids": sorted(exposed & remaining),
        "by_kind_counts": {kind: len(ids & remaining) for kind, ids in sorted(by_kind.items())},
    })


def scenario_diagnostics(replay, item, *, projected, routing):


    if not isinstance(replay, PlanReplay) or not isinstance(item, RecoveryInput):
        raise TypeError("scenario diagnostics require a PlanReplay and RecoveryInput")
    snapshot = item.source_snapshot
    if replay.at(snapshot.time_s) != snapshot:
        raise ValueError("scenario snapshot differs from the original replay")
    if getattr(projected, "original_input", None) is not item:
        raise ValueError("scenario projection belongs to a different recovery input")
    boundary = projected.execution_boundary
    boundary.validate_for(projected)
    if routing.roads != item.roads:
        raise ValueError("scenario diagnostics routing differs from the changed roads")
    events, exposure = _exposure(replay, item)
    fixed = boundary.observed_tardiness_s + boundary.committed_tardiness_upper_s
    after = dict(physical_valid=False, T_s=None, suffix_T_s=None, fixed_T_s=fixed, reason=None)
    try:
        evaluated = evaluate_recovery_plan(projected, boundary.reference_plan, routing)
    except RecoveryPlanValidationError as error:
        after["reason"] = error.to_dict()
    else:
        after.update(physical_valid=True, T_s=evaluated.total_tardiness_s + fixed,
                     suffix_T_s=evaluated.total_tardiness_s)
    return {
        "scope": "original_unfinished_obligations", "time_s": snapshot.time_s,
        "remaining_task_count": len(snapshot.unfinished_task_ids), "worker_count": len(snapshot.workers),
        "event_counts": events, "potential_task_exposure": exposure,
        "lower_bounds": _bounds(item, boundary),
        "original_reference": {
            "before_events_T_s": math.fsum(replay.plan.tardiness_s[t] for t in snapshot.unfinished_task_ids),
            "after_events": after,
        },
        "boundary": {"suffix_task_count": len(projected.tasks),
                     "completed_task_count": len(snapshot.unfinished_task_ids) - len(projected.tasks)},
    }
