from copy import copy
from dataclasses import replace
import json
import math
from common.recovery.plan import RecoveryVisit, TransferEvent
from solvers.tfr.structure import release_tasks, structure_key, check_structure
from solvers.tfr.interfaces.revisions import Candidate, FieldChoice, RevisionScript, FieldDecision, ProposalSlot, MacroAction, DecodeFailure, FEATURE_DIM, _custody_gaps, activity_allowed, handover_possible

def _features(*values):
    values = tuple((float(v) for v in values))
    return values + (0.0,) * (FEATURE_DIM - len(values))

def _group_id(group):
    return json.dumps(tuple(group), separators=(',', ':'))

def _visit_id(visit):
    return visit.operation + ':' + (visit.transfer_id if visit.operation == 'transfer' else visit.task_id)

def _visit_position(obs, plan, worker, visit):
    if visit.operation == 'transfer':
        return plan.transfer_map[visit.transfer_id].location
    task = obs.item.task_map[visit.task_id]
    return task.pickup if visit.operation == 'pickup' else task.delivery

def _insert(plan, worker, index, visit):
    routes = dict(plan.sequences)
    routes[worker] = routes[worker][:index] + (visit,) + routes[worker][index:]
    return replace(plan, sequences=routes)

def worker_pool(obs, task):
    holders = {t: w.id for w in obs.item.workers for t in w.carried_task_ids}
    origin = obs.anchor(holders[task]) if task in holders else obs.item.task_map[task].pickup
    (x, y) = obs.item.roads.coordinates(origin)

    def priority(worker):
        (px, py) = obs.item.roads.coordinates(obs.item.worker_map[worker].location)
        return (math.hypot(px - x, py - y), worker)
    return tuple(sorted(obs.local.basic_workers(task), key=priority))

def field_envelope(obs):
    return 2 + max((sum((4 * obs.handover_bounds[t] + 4 for t in group)) for group in obs.groups), default=0)

def gap_candidates(obs, plan, worker, location, *, indices=None):
    route = plan.sequences[worker]
    (x, y) = obs.item.roads.coordinates(location)
    output = []
    for index in range(len(route) + 1) if indices is None else indices:
        before = '^' if index == 0 else _visit_id(route[index - 1])
        after = '$' if index == len(route) else _visit_id(route[index])
        origin = obs.item.worker_map[worker].location if index == 0 else _visit_position(obs, plan, worker, route[index - 1])
        (px, py) = obs.item.roads.coordinates(origin)
        output.append(Candidate(before + '|' + after, index, _features(index / max(1, len(route)), math.log1p(math.hypot(px - x, py - y)), len(route) / 100.0, 0.0), 'worker:' + worker))
    return tuple(output)

def graph_observation(obs):
    from .network import graph_observation as native_graph_observation
    view = copy(obs)
    view.worker_candidates = {}
    view.worker_pools = {}
    view.search_radii = dict(obs.search_radii)
    view.distance_cache = dict(obs.distance_cache)
    view.tried = set(obs.tried)
    view.expansion_tasks = set(obs.expansion_tasks)
    for task in {t for group in obs.groups for t in group}:
        view.worker_pools[task] = worker_pool(obs, task)
    return native_graph_observation(view)

def decode_structure(obs, choose, *, max_fields=None, replay_domains=None, validate_structure=True):
    max_fields = field_envelope(obs) if max_fields is None else max_fields
    (fields, domains) = ([], [])
    active_task = None
    pending = set()

    def pick(kind, candidates, budget_worker=None):
        obs.tick()
        if len(fields) >= max_fields:
            raise DecodeFailure('field_limit')
        candidates = tuple(candidates)
        if not any((c.legal for c in candidates)):
            raise DecodeFailure('all_masked:' + kind)
        selected = choose(kind, candidates)
        if type(selected) is not int or not 0 <= selected < len(candidates) or (not candidates[selected].legal):
            raise DecodeFailure('invalid_choice:' + kind)
        candidate = candidates[selected]
        fields.append(FieldChoice(kind, candidate.identity))
        domains.append(tuple((c.identity for c in candidates if c.legal)))
        return candidate.value

    def gaps(plan, worker, position, *, receiving=False):
        indices = _custody_gaps(obs, plan, worker, active_task, pending, receiving=receiving)
        if replay_domains is None:
            return gap_candidates(obs, plan, worker, position, indices=indices)
        allowed = replay_domains[len(fields)]
        route = plan.sequences[worker]
        output = []
        for index in indices:
            before = '^' if index == 0 else _visit_id(route[index - 1])
            after = '$' if index == len(route) else _visit_id(route[index])
            identity = before + '|' + after
            if identity in allowed:
                output.append(Candidate(identity, index, _features(0), 'worker:' + worker))
        return tuple(output)

    def workers(task, sender=None):
        if replay_domains is None:
            return worker_pool(obs, task)
        return tuple((w for w in replay_domains[len(fields)] if w != 'END'))

    def eligible_candidates(task, make, sender=None):
        return tuple(make(workers(task, sender)))
    groups = tuple((Candidate(_group_id(group), tuple(group), _features(len(group) / 8.0, sum((obs.evaluated.tardiness_s[t] for t in group)) / max(obs.initial_T, 1.0), obs.temporal.group_score(group)[0] / max(obs.initial_T, 1.0), obs.temporal.group_score(group)[1] / 3600.0), tuple(('task:' + task for task in group))) for group in obs.groups))
    group = pick('group', groups)
    plan = release_tasks(obs.plan, group)
    pending.update(group)
    holders = {t: w.id for w in obs.item.workers for t in w.carried_task_ids}
    for task in sorted(group):
        active_task = task
        data = obs.item.task_map[task]
        if task not in holders:
            worker = pick('pickup_worker', eligible_candidates(task, lambda pool: (Candidate(w, w, _features(obs.item.worker_map[w].capacity, obs.item.worker_map[w].profile.speed_factor), 'worker:' + w) for w in pool if activity_allowed(obs.item, w, data.pickup) and _custody_gaps(obs, plan, w, task, pending, receiving=True))))
            gap = pick('pickup_gap', gaps(plan, worker, data.pickup, receiving=True), worker)
            plan = _insert(plan, worker, gap, RecoveryVisit(operation='pickup', task_id=task))
        else:
            worker = holders[task]
        bound = obs.handover_bounds[task]
        for hop in range(bound + 1):
            closing = _custody_gaps(obs, plan, worker, task, pending, receiving=False)
            can_end = bool(closing) and obs.item.worker_map[worker].service_mode == 'delivery' and activity_allowed(obs.item, worker, data.delivery)

            def receiver_options(receivers):
                return [Candidate('END', None, _features(1, math.log1p(hop)), legal=can_end)] + [Candidate(w, w, _features(0, math.log1p(hop), obs.item.worker_map[w].capacity), 'worker:' + w, legal=w != worker and obs.item.worker_map[w].service_mode == 'delivery' and bool(closing) and handover_possible(obs.item, worker, w, obs.anchor(worker)) and bool(_custody_gaps(obs, plan, w, task, pending, receiving=True))) for w in receivers]
            options = receiver_options(()) if hop == bound else eligible_candidates(task, receiver_options, worker)
            receiver = pick('receiver', options, worker)
            if receiver is None:
                break
            receiver_gaps = set(_custody_gaps(obs, plan, receiver, task, pending, receiving=True))
            reusable = [event for event in plan.transfers if {event.worker_i, event.worker_j} == {worker, receiver} and task not in event.tasks_i_to_j + event.tasks_j_to_i and (next((i for (i, v) in enumerate(plan.sequences[worker]) if v.operation == 'transfer' and v.transfer_id == event.id)) in closing) and (next((i + 1 for (i, v) in enumerate(plan.sequences[receiver]) if v.operation == 'transfer' and v.transfer_id == event.id)) in receiver_gaps)]
            mode = pick('meeting', [Candidate('NEW', None, _features(1))] + [Candidate(event.id, event.id, _features(0, len(event.tasks_i_to_j + event.tasks_j_to_i) / 8.0), 'meeting:' + event.id) for event in sorted(reusable, key=lambda e: e.id)], receiver)
            if mode is None:
                left = pick('sender_gap', gaps(plan, worker, obs.anchor(worker)), receiver)
                right = pick('receiver_gap', gaps(plan, receiver, obs.anchor(worker), receiving=True), receiver)
                serial = 0
                event_id = f'trail:{task}:{hop}:{serial}'
                while event_id in plan.transfer_map:
                    serial += 1
                    event_id = f'trail:{task}:{hop}:{serial}'
                event = TransferEvent(id=event_id, worker_i=worker, worker_j=receiver, location=obs.anchor(worker), tasks_i_to_j=(task,))
                plan = replace(plan, transfers=plan.transfers + (event,))
                visit = RecoveryVisit(operation='transfer', transfer_id=event_id)
                plan = _insert(_insert(plan, worker, left, visit), receiver, right, visit)
            else:
                old = plan.transfer_map[mode]
                if task in old.tasks_i_to_j + old.tasks_j_to_i:
                    raise DecodeFailure('duplicate_cargo')
                event = replace(old, tasks_i_to_j=old.tasks_i_to_j + (task,)) if worker == old.worker_i else replace(old, tasks_j_to_i=old.tasks_j_to_i + (task,))
                plan = replace(plan, transfers=tuple((event if e.id == mode else e for e in plan.transfers)))
            worker = receiver
        else:
            raise DecodeFailure('unclosed_chain')
        gap = pick('delivery_gap', gaps(plan, worker, data.delivery), worker)
        plan = _insert(plan, worker, gap, RecoveryVisit(operation='delivery', task_id=task))
        pending.remove(task)
    active_task = None
    pick('end', (Candidate('END', None, _features(1)),))
    script = RevisionScript(obs.source_witness_digest, tuple(fields), tuple(domains))
    if validate_structure:
        outcome = check_structure(obs.item, plan, check_budget=obs.check)
        if not outcome.valid:
            raise DecodeFailure('structure:' + ','.join(outcome.errors) + (':capacity' if outcome.gaps else ''))
    return (script, plan)

def extract_structure_script(obs, target_plan, group):
    group = tuple(sorted(group))
    target_key = structure_key(target_plan)
    if group not in obs.groups:
        raise DecodeFailure('teacher_group_outside_domain')
    if structure_key(release_tasks(obs.plan, group)) != structure_key(release_tasks(target_plan, group)):
        raise DecodeFailure('teacher_changed_background')
    holders = {t: w.id for w in obs.item.workers for t in w.carried_task_ids}
    current = {'task': None, 'worker': None, 'hop': 0, 'event': None, 'phase': None}
    tasks = iter(sorted(group))
    target_new_to_generated = {}

    def target_visit(visit):
        if visit.operation != 'transfer':
            return visit
        transfer = target_new_to_generated.get(visit.transfer_id, visit.transfer_id)
        return RecoveryVisit(operation='transfer', transfer_id=transfer)

    def matching_gap(candidates, worker, desired):
        target = tuple(map(target_visit, target_plan.sequences[worker]))
        desired = target_visit(desired)
        position = target.index(desired)
        preceding = {_visit_id(v) for v in target[:position]}
        following = {_visit_id(v) for v in target[position + 1:]}
        for (index, candidate) in enumerate(candidates):
            (before, after) = candidate.identity.split('|', 1)
            if (before == '^' or before in preceding) and (after == '$' or after in following):
                return index
        raise DecodeFailure('teacher_gap_outside_domain')

    def choose(kind, candidates):

        def by_id(identity):
            matches = [i for (i, c) in enumerate(candidates) if c.identity == identity and c.legal]
            if not matches:
                raise DecodeFailure('teacher_choice_outside_domain:' + kind + ':' + identity)
            return matches[0]
        if kind == 'group':
            return by_id(_group_id(group))
        if kind == 'end':
            return by_id('END')
        if current['task'] is None:
            current['task'] = next(tasks)
            current['worker'] = holders.get(current['task'])
            current['hop'] = 0
        task = current['task']
        if kind == 'pickup_worker':
            current['worker'] = next((w for (w, route) in target_plan.sequences.items() if RecoveryVisit(operation='pickup', task_id=task) in route))
            return by_id(current['worker'])
        if kind == 'pickup_gap':
            return matching_gap(candidates, current['worker'], RecoveryVisit(operation='pickup', task_id=task))
        if kind == 'receiver':
            worker = current['worker']
            events = []
            for visit in target_plan.sequences[worker]:
                if visit.operation != 'transfer':
                    continue
                event = target_plan.transfer_map[visit.transfer_id]
                outgoing = event.tasks_i_to_j if worker == event.worker_i else event.tasks_j_to_i
                if task in outgoing:
                    events.append(event)
            chain_events = []
            w = holders.get(task)
            if w is None:
                w = next((w for (w, route) in target_plan.sequences.items() if RecoveryVisit(operation='pickup', task_id=task) in route))
            offsets = {w: 0}
            for _ in range(len(target_plan.transfers) + 1):
                route = target_plan.sequences[w]
                found = None
                for j in range(offsets.get(w, 0), len(route)):
                    v = route[j]
                    if v.operation == 'delivery' and v.task_id == task:
                        break
                    if v.operation == 'transfer':
                        e = target_plan.transfer_map[v.transfer_id]
                        if task in (e.tasks_i_to_j if w == e.worker_i else e.tasks_j_to_i):
                            found = e
                            offsets[w] = j + 1
                            break
                if found is None:
                    break
                chain_events.append(found)
                w = found.worker_j if w == found.worker_i else found.worker_i
                offsets[w] = next((j + 1 for (j, v) in enumerate(target_plan.sequences[w]) if v.operation == 'transfer' and v.transfer_id == found.id))
            hop = current['hop']
            if hop >= len(chain_events):
                current['event'] = None
                return by_id('END')
            current['event'] = chain_events[hop]
            event = current['event']
            current['receiver'] = event.worker_j if worker == event.worker_i else event.worker_i
            return by_id(current['receiver'])
        if kind == 'meeting':
            event = current['event']
            identity = target_new_to_generated.get(event.id, event.id)
            if any((c.identity == identity for c in candidates)):
                current['worker'] = current['receiver']
                current['hop'] += 1
                return by_id(identity)
            return by_id('NEW')
        if kind == 'sender_gap':
            return matching_gap(candidates, current['worker'], RecoveryVisit(operation='transfer', transfer_id=current['event'].id))
        if kind == 'receiver_gap':
            result = matching_gap(candidates, current['receiver'], RecoveryVisit(operation='transfer', transfer_id=current['event'].id))
            serial = 0
            name = f"trail:{task}:{current['hop']}:{serial}"
            existing = set(obs.plan.transfer_map) | set(target_new_to_generated.values())
            while name in existing:
                serial += 1
                name = f"trail:{task}:{current['hop']}:{serial}"
            target_new_to_generated[current['event'].id] = name
            current['worker'] = current['receiver']
            current['hop'] += 1
            return result
        if kind == 'delivery_gap':
            result = matching_gap(candidates, current['worker'], RecoveryVisit(operation='delivery', task_id=task))
            current['task'] = None
            return result
        raise DecodeFailure('teacher_unknown_field')
    try:
        (script, replay) = decode_structure(obs, choose)
    except (StopIteration, ValueError) as error:
        if isinstance(error, DecodeFailure):
            raise
        raise DecodeFailure('teacher_not_expressible:' + str(error)) from error
    if structure_key(replay) != target_key:
        raise DecodeFailure('teacher_roundtrip_structure_mismatch')
    return script

def construct_structure(obs, script):
    if script.source_witness_digest != obs.source_witness_digest:
        raise DecodeFailure('structure_script_source_mismatch')
    cursor = iter(script.choices)

    def choose(kind, candidates):
        choice = next(cursor, None)
        if choice is None or choice.kind != kind:
            raise DecodeFailure('structure_script_field_mismatch')
        matches = [i for (i, c) in enumerate(candidates) if c.identity == choice.identity and c.legal]
        if len(matches) != 1:
            raise DecodeFailure('structure_script_candidate_missing')
        return matches[0]
    (_, plan) = decode_structure(obs, choose, replay_domains=script.domains)
    if next(cursor, None) is not None:
        raise DecodeFailure('structure_script_trailing_fields')
    return plan

def teacher_macro(obs, target_plan, group, *, graph=None):
    script = extract_structure_script(obs, target_plan, group)
    cursor = iter(script.choices)
    fields = []

    def choose(kind, candidates):
        choice = next(cursor, None)
        if choice is None or choice.kind != kind:
            raise DecodeFailure('teacher_field_mismatch')
        matches = [i for (i, c) in enumerate(candidates) if c.identity == choice.identity and c.legal]
        if len(matches) != 1:
            raise DecodeFailure('teacher_candidate_missing')
        index = matches[0]
        fields.append(FieldDecision(kind, tuple(candidates), index, 0.0, sum((c.legal for c in candidates)) == 1))
        return index
    (replay, plan) = decode_structure(obs, choose)
    if next(cursor, None) is not None or structure_key(plan) != structure_key(target_plan):
        raise DecodeFailure('full_teacher_structure_roundtrip_mismatch')
    return MacroAction(obs.source_witness_digest, (ProposalSlot(replay, tuple(fields), 'demonstration', plan),), graph_observation(obs) if graph is None else graph, search_radii=dict(obs.search_radii))
