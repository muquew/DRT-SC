from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path
from common.data.roads.routing import Routing
from common.config import routing_options as configured_routing
from common.evaluation.recovery_plan import evaluate_recovery_plan, RecoveryPlanValidationError
from common.scenarios.disruptions import DisruptionEvent, KINDS, apply_disruptions
from common.recovery.plan import RecoveryPlan, RecoveryVisit

def _base_evaluation(item, roads):
    base = apply_disruptions(item.source_snapshot, roads, (), item.source_settings)
    plan = RecoveryPlan(input_id=base.instance_id, network_id=base.network_id, snapshot_id=base.snapshot_id,
                        sequences={w: tuple(RecoveryVisit(operation=v.operation, task_id=v.task_id)
                                            for v in visits) for w, visits in base.reference_sequences.items()})
    try:
        return evaluate_recovery_plan(base, plan, Routing(roads, cache_profile='formal'))
    except RecoveryPlanValidationError:
        return None

def _event_data(item, roads, event):

    source, kind = item.source_snapshot, event.kind
    if kind in ('pickup_delay', 'reception_delay', 'delivery_relocation'):
        task = source.task_map[event.target_id].task
        location = task.pickup if kind == 'pickup_delay' else task.delivery
        if kind == 'delivery_relocation':
            duration = Routing(roads, cache_profile='formal').travel_time(task.delivery, event.value)
            severity = duration if math.isfinite(duration) else math.dist(roads.coordinates(task.delivery), roads.coordinates(event.value))
        else:
            old = task.pickup_ready_time_s if kind == 'pickup_delay' else task.delivery_ready_time_s
            severity = max(0.0, event.value - old)
        seconds = severity
    elif kind == 'road_time_change':
        arc = roads.arc_map[event.target_id]
        segment = roads.segment_map[arc.segment_id]
        location = roads.segment_position(segment.id, segment.length_m / 2)
        severity = max(0.0, event.value / arc.travel_time_s - 1)
        seconds = max(0.0, event.value - arc.travel_time_s)
    else:
        location = source.worker_map[event.target_id].location
        severity, seconds = 1.0, None
    return (*roads.coordinates(location), severity, seconds)

def features(item, roads):

    grouped = defaultdict(list)
    for record in item.events:
        event = DisruptionEvent.from_dict(record)
        grouped[event.kind].append(_event_data(item, roads, event))
    vector = []
    for kind in KINDS:
        values = grouped[kind]
        vector.extend((len(values), sum(v[0] for v in values) / len(values) if values else 0,
                       sum(v[1] for v in values) / len(values) if values else 0,
                       max((v[2] for v in values), default=0)))
    return vector

def disrupted_requests(item, roads):

    base = _base_evaluation(item, roads)
    horizon = max((route[-1].completion_time_s - item.time_s for route in base.routes.values() if route),
                  default=0) + 1 if base is not None else 1.0
    result = {}
    for record in item.events:
        event = DisruptionEvent.from_dict(record)
        *_, seconds = _event_data(item, roads, event)
        amount = horizon if seconds is None else seconds
        starts = {}
        if event.kind == 'worker_withdrawal':
            starts[event.target_id] = 0
        elif event.kind == 'road_time_change':
            if base is None:
                starts = dict.fromkeys(item.reference_sequences, 0)
            else:
                for worker, route in base.routes.items():
                    for index, visit in enumerate(route):
                        if any(leg.arc_id == event.target_id for leg in visit.incoming_path.legs):
                            starts[worker] = index
                            break
        else:
            operation = 'pickup' if event.kind == 'pickup_delay' else 'delivery'
            for worker, route in item.reference_sequences.items():
                for index, visit in enumerate(route):
                    if visit.task_id == event.target_id and visit.operation == operation:
                        starts[worker] = index
                        break
        for worker, index in starts.items():
            for visit in item.reference_sequences[worker][index:]:
                result[visit.task_id] = max(result.get(visit.task_id, 0.0), amount)
    return result

def _slack(route, item):

    limits = []
    for start in range(len(route)):
        wait, remaining = 0.0, []
        for visit in route[start:]:
            wait += visit.waiting_time_s
            if visit.visit.operation == 'delivery':
                remaining.append(item.task_map[visit.visit.task_id].promised_delivery_time_s
                                 - visit.completion_time_s + wait)
        if remaining:
            limits.append(min(remaining))
    return max(limits, default=0.0)

def routing_batches(item, original_roads, multiplier):

    affected = set(disrupted_requests(item, original_roads))
    disrupted = {w for w, visits in item.reference_sequences.items()
                 if any(v.task_id in affected for v in visits)}
    if not disrupted:
        return [tuple(sorted(item.worker_map))]
    base = _base_evaluation(item, original_roads)
    held = {t for worker in item.workers for t in worker.carried_task_ids}
    pickups = {item.roads.normalize_position(item.task_map[t].pickup) for t in affected
               if t not in held and item.task_map[t].pickup is not None}


    period_end = max((item.task_map[t].promised_delivery_time_s for t in affected), default=item.time_s)
    preferred = []
    for worker in sorted(set(item.worker_map) - disrupted):
        route = () if base is None else base.routes[worker]
        intersects = any(v.visit.operation == 'pickup'
                         and item.roads.normalize_position(item.task_map[v.visit.task_id].pickup) in pickups
                         and item.time_s <= v.start_time_s <= period_end for v in route)
        if intersects:
            preferred.append((-_slack(route, item), worker))
    ordered = [w for _, w in sorted(preferred)]
    ordered.extend(sorted(set(item.worker_map) - disrupted - set(ordered)))
    width = max(1, multiplier * len(disrupted))
    return [tuple(sorted(disrupted | set(ordered[index:index + width])))
            for index in range(0, len(ordered), width)] or [tuple(sorted(disrupted))]

def _remove(engine, current, candidates, q, name):
    candidates = sorted(candidates)
    if name == 'random':
        return engine.rng.sample(candidates, q)
    if name == 'pickup':
        groups = defaultdict(list)
        for task in candidates:
            position = (engine.item.worker_map[engine.holders[task]].location if task in engine.holders
                        else engine.item.task_map[task].pickup)
            groups[engine.item.roads.normalize_position(position)].append(task)
        removed = []
        while len(removed) < q:
            key = engine.rng.choice(list(groups))
            group = groups.pop(key)
            removed.extend(group if len(group) <= q - len(removed)
                           else engine.rng.sample(group, q - len(removed)))
        return removed
    if name == 'worst':
        savings = []
        for task in candidates:
            value = engine.evaluate(engine.strip(current.plan, (task,)), engine.all_tasks - {task})
            savings.append((math.inf if value is None else value.total_tardiness_s, task))
        return [task for _, task in sorted(savings)[:q]]
    times = {(v.visit.task_id, v.visit.operation): v.start_time_s
             for route in current.routes.values() for v in route if v.visit.operation != 'transfer'}
    first = engine.rng.choice(candidates)
    selected = [first]
    while len(selected) < q:
        anchor = engine.rng.choice(selected)
        def similarity(task):
            a, b = engine.item.task_map[anchor], engine.item.task_map[task]
            pa = engine.item.worker_map[engine.holders[anchor]].location if anchor in engine.holders else a.pickup
            pb = engine.item.worker_map[engine.holders[task]].location if task in engine.holders else b.pickup
            dist = sum(math.dist(engine.item.roads.coordinates(x), engine.item.roads.coordinates(y))
                       for x, y in ((pa, pb), (a.delivery, b.delivery)))
            return (6 * dist + 5 * abs(current.tardiness_s[anchor] - current.tardiness_s[task])
                    + 9 * sum(abs(times.get((anchor, op), engine.item.time_s)
                                  - times.get((task, op), engine.item.time_s)) for op in ('pickup', 'delivery')))
        selected.append(min(set(candidates) - set(selected), key=lambda t: (similarity(t), t)))
    return selected

def match_scenario(library, item, original_roads):
    target = features(item, original_roads)
    scale = library['feature_scale']
    def distance(scenario):
        return math.fsum(((x - y) / s) ** 2 for x, y, s in zip(target, scenario['features'], scale))
    selected = min(library['scenarios'], key=lambda scenario: (distance(scenario), scenario['id']))
    return selected, math.sqrt(distance(selected))


def save_scenario_library(library, path):
    from common.io import write_json
    write_json(path, library)


def load_scenario_library(path):
    library = json.loads(Path(path).read_text())
    if not isinstance(library, dict) or not isinstance(library.get('scenarios'), list) or not library['scenarios']:
        raise ValueError('expected a nonempty Hu scenario library')
    width = 4 * len(KINDS)
    if len(library['feature_scale']) != width or any(not math.isfinite(x) or x <= 0 for x in library['feature_scale']):
        raise ValueError('invalid Hu feature scale')
    for scene in library['scenarios']:
        if len(scene['features']) != width or any(not math.isfinite(x) for x in scene['features']):
            raise ValueError('invalid Hu scene features')
        for plan in scene['candidates']:
            RecoveryPlan.from_dict(plan)
    return library


def original_roads(library, item):
    conditions = library['source_arc_conditions']
    roads = replace(item.roads, version=library['road_state_label'],
        arcs=tuple(replace(arc, travel_time_s=conditions[arc.id][0], enabled=conditions[arc.id][1])
                   for arc in item.roads.arcs))
    if roads.snapshot_id != item.source_snapshot.snapshot_id:
        raise ValueError('Hu road conditions differ from the execution snapshot')
    return roads
