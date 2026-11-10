import heapq
import math
from time import perf_counter
from contextlib import contextmanager
from dataclasses import replace

from common.evaluation.recovery_plan import RecoveryPlanValidationError
from common.recovery.adapters._insertion_delta import InsertionDelta
from common.recovery.plan import RecoveryVisit


class _TransientFrame:
    def __init__(self):
        self.data, self.done, self.result = {}, False, None


    def finish(self, result):
        self.result, self.done = result, True
        return result


@contextmanager
def search_frame(engine, name, arguments=None):
    yield _TransientFrame()


class InsertionBudget:
    def __init__(self, engine, task_ids=(), *, state=None):
        self.engine = engine
        self.by_task = {} if state is None else state
        for task in task_ids:
            self.by_task.setdefault(task, {})

    def _row(self, task, worker):
        rows = self.by_task.setdefault(task, {})
        return rows.setdefault(worker,dict(generated=0,evaluated=0,emitted=0))

    def charge_generation(self, task, worker):
        self.engine.check_budget()
        row = self._row(task, worker)
        if row['generated'] >= 64:
            return False
        row['generated'] += 1
        self.engine.stats['generated_branches'] = self.engine.stats.get('generated_branches', 0) + 1
        return True

    def charge_evaluation(self, task, worker):
        self.engine.check_budget()
        row = self._row(task, worker)
        if row['evaluated'] >= 8:
            return False
        row['evaluated'] += 1
        return True

    def can_emit(self, task, worker):
        return self._row(task, worker)['emitted'] < 8

    def emit(self, task, worker):
        self._row(task, worker)['emitted'] += 1


def ranked_gaps(engine, plan, worker, route, position, *, start=0, stop=None, limit=8):

    stop = len(route) + 1 if stop is None else min(stop, len(route) + 1)
    origin = engine.item.worker_map[worker].location
    def rank(index):
        engine.check_budget()
        a = origin if index == 0 else engine.visit_location(plan, route[index - 1])
        b = None if index == len(route) else engine.visit_location(plan, route[index])
        distance = engine.routing.shortest_distance(a, position)
        if b is not None:
            distance += engine.routing.shortest_distance(position, b) - engine.routing.shortest_distance(a, b)

        if not math.isfinite(distance):
            distance = math.inf
        return distance, index
    return tuple(index for _, index in heapq.nsmallest(limit,
        (rank(index) for index in range(start, stop))))


def _meeting_positions(engine, sender, receiver, task_id, locations=None):


    values = engine.meeting_positions(sender, receiver) if locations is None else locations
    legal = (p for p in values if engine.local.meeting_allowed(
        sender, receiver, p, task_ids=(task_id,)))
    target = engine.item.task_map[task_id].delivery
    return tuple(heapq.nsmallest(8, legal, key=lambda p: (
        engine.routing.shortest_distance(engine.item.worker_map[receiver].location, p)
        + engine.routing.shortest_distance(p, target), str(p))))


def _priority_insertions(engine, plan, task_id, ledger, *, workers=None,
                       mode='all', transfer_locations=None, emission_limit=8, continuous=False,
                       _actors=None,_receiver=None,_senders=None):

    if mode not in ('all', 'direct', 'transfer', 'reuse'):
        raise ValueError('unknown insertion mode')
    owner = engine.holders.get(task_id)
    available = _actors
    if available is None:
        available = engine.local.ordered_workers(task_id, sender_id=owner)
        if workers is not None:
            allowed = set(workers)
            available = tuple(w for w in available if w in allowed)
        if engine.config.worker_candidates is not None:
            available = available[:engine.config.worker_candidates]
    task = engine.item.task_map[task_id]
    pickup = RecoveryVisit(operation='pickup', task_id=task_id)
    serial = 0
    number = 1
    while f'transfer-{number}' in plan.transfer_map:
        number += 1
    transfer_id = f'transfer-{number}'
    for receiver in available if _receiver is None else (_receiver,):
        if not ledger.can_emit(task_id, receiver):
            continue
        if _receiver is None and not ledger.charge_generation(task_id,receiver):
            continue
        call_emitted = [0]
        def can_emit():
            return ledger.can_emit(task_id, receiver) and call_emitted[0] < emission_limit
        def emit():
            ledger.emit(task_id, receiver)
            call_emitted[0] += 1
        rr = plan.sequences[receiver]
        direct_emitted = 0
        if mode in ('all', 'direct') and (owner is None or owner == receiver):
            pickups = (None,) if owner else ranked_gaps(engine, plan, receiver, rr, task.pickup)

            direct_limit = min(emission_limit,4 if mode == 'all' and owner is None and len(available) > 1 else 8)
            for pi in pickups:
                if not ledger.charge_generation(task_id, receiver):
                    break
                route = rr if pi is None else rr[:pi] + (pickup,) + rr[pi:]
                deliveries = ranked_gaps(engine, plan, receiver, route, task.delivery,
                    start=0 if pi is None else pi + 1)
                for di in deliveries:
                    if direct_emitted >= direct_limit or not can_emit():
                        break
                    if not ledger.charge_generation(task_id, receiver):
                        break
                    emit()
                    yield InsertionDelta(plan, serial, 'direct', task_id, receiver,
                        pickup_index=pi, delivery_index=di)
                    serial += 1
                    direct_emitted += 1
                if direct_emitted >= direct_limit:
                    break
        if mode == 'direct' or not can_emit():
            continue
        senders = (owner,) if owner else tuple(w for w in available if w != receiver)
        if _senders is not None:
            senders = tuple(w for w in senders if w in _senders)
        if mode in ('all', 'transfer', 'reuse'):
            for event in plan.transfers:
                for sender, target in ((event.worker_i, event.worker_j), (event.worker_j, event.worker_i)):
                    if target != receiver or sender not in senders or not can_emit():
                        continue
                    if not ledger.charge_generation(task_id, receiver):
                        break
                    if not continuous and not engine.local.meeting_allowed(sender, receiver, event.location, task_ids=(task_id,)):
                        continue
                    sr = plan.sequences[sender]
                    si = next(i for i, v in enumerate(sr) if v.transfer_id == event.id)
                    ri = next(i for i, v in enumerate(rr) if v.transfer_id == event.id)
                    pickups = (None,) if owner else ranked_gaps(engine, plan, sender, sr, task.pickup, stop=si + 1)
                    for pi in pickups:
                        if not ledger.charge_generation(task_id, receiver):
                            break
                        for di in ranked_gaps(engine, plan, receiver, rr, task.delivery, start=ri + 1):
                            if not can_emit() or not ledger.charge_generation(task_id, receiver):
                                break
                            emit()
                            yield InsertionDelta(plan, serial, 'reuse', task_id, receiver, sender,
                                event.location, pi, delivery_index=di, reuse_event_id=event.id)
                            serial += 1
        if mode == 'reuse':
            continue
        for sender in senders:
            if sender == receiver or not can_emit():
                continue
            if not ledger.charge_generation(task_id, receiver):
                break
            sr = plan.sequences[sender]
            positions = (engine.local.original_anchor(sender),) if continuous else _meeting_positions(engine,sender,receiver,task_id,transfer_locations)
            for position in positions:
                if not ledger.charge_generation(task_id, receiver):
                    break
                pickups = (None,) if owner else ranked_gaps(engine, plan, sender, sr, task.pickup)
                for pi in pickups:
                    if not ledger.charge_generation(task_id, receiver):
                        break
                    route = sr if pi is None else sr[:pi] + (pickup,) + sr[pi:]
                    for si in ranked_gaps(engine, plan, sender, route, position,
                                          start=0 if pi is None else pi + 1):
                        if not ledger.charge_generation(task_id, receiver):
                            break
                        for ri in ranked_gaps(engine, plan, receiver, rr, position):
                            if not ledger.charge_generation(task_id, receiver):
                                break
                            visit = RecoveryVisit(operation='transfer', transfer_id=transfer_id)


                            from common.recovery.plan import WaypointVisit
                            proxy = rr[:ri] + (WaypointVisit(location=position),) + rr[ri:]
                            for di in ranked_gaps(engine, plan, receiver, proxy, task.delivery, start=ri + 1):
                                if not can_emit() or not ledger.charge_generation(task_id, receiver):
                                    break
                                emit()
                                yield InsertionDelta(plan, serial, 'transfer', task_id, receiver, sender,
                                    position, pi, si, ri, di, transfer_id=transfer_id)
                                serial += 1


def _tail_insertion(engine,plan,task_id,receiver,actors,ledger,mode,locations,senders=None):

    owner=engine.holders.get(task_id)
    route=plan.sequences[receiver]
    def charge(count):
        return all(ledger.charge_generation(task_id,receiver) for _ in range(count))
    if mode in ('all','direct') and owner in (None,receiver):
        if not charge(2 if owner is None else 1):return None
        return InsertionDelta(plan,0,'direct',task_id,receiver,
            pickup_index=len(route) if owner is None else None,
            delivery_index=len(route)+(owner is None))
    if mode not in ('all','transfer'):return None
    candidates=(owner,) if owner is not None else tuple(w for w in actors if w!=receiver)
    if senders is not None:
        candidates=tuple(w for w in candidates if w in senders)
    for sender in candidates:
        if sender==receiver:continue
        if not charge(1):return None
        positions=_meeting_positions(engine,sender,receiver,task_id,locations)
        if not positions:continue
        if not charge(1):return None
        sr=plan.sequences[sender]
        left=engine.item.worker_map[sender].location if not sr else engine.visit_location(plan,sr[-1])
        right=engine.item.worker_map[receiver].location if not route else engine.visit_location(plan,route[-1])
        position=min(positions,key=lambda p:(engine.routing.shortest_distance(left,p)+
            engine.routing.shortest_distance(right,p)+engine.routing.shortest_distance(p,engine.item.task_map[task_id].delivery),str(p)))
        if not charge(3+(owner is None)):return None
        number=1
        while f'transfer-{number}' in plan.transfer_map:number+=1
        return InsertionDelta(plan,0,'transfer',task_id,receiver,sender,position,
            pickup_index=len(sr) if owner is None else None,
            sender_transfer_index=len(sr)+(owner is None),receiver_transfer_index=len(route),
            delivery_index=len(route)+1,transfer_id=f'transfer-{number}')
    return None


def _delta_identity(delta):
    return (delta.kind,delta.receiver,delta.sender,delta.position,delta.pickup_index,
        delta.sender_transfer_index,delta.receiver_transfer_index,delta.delivery_index,
        delta.reuse_event_id,delta.transfer_id)


def bounded_insertions(engine,plan,task_id,ledger,*,workers=None,mode='all',
                       transfer_locations=None,emission_limit=8,continuous=False,
                       _actors=None,_receivers=None,_senders=None):
    _receivers = None if _receivers is None else frozenset(_receivers)
    _senders = None if _senders is None else frozenset(_senders)
    if continuous:
        yield from _priority_insertions(engine,plan,task_id,ledger,workers=workers,mode=mode,
            transfer_locations=transfer_locations,emission_limit=emission_limit,continuous=True,
            _actors=_actors,_senders=_senders)
        return
    owner=engine.holders.get(task_id)
    actors=engine.local.ordered_workers(task_id,sender_id=owner) if _actors is None else _actors
    if workers is not None:
        allowed=set(workers)
        actors=tuple(w for w in actors if w in allowed)
    if _actors is None and engine.config.worker_candidates is not None:
        actors=actors[:engine.config.worker_candidates]
    engine.stats['local_insertion_policy']='at most7 ranked joint proposals plus1 reserved append-only closure per worker'
    serial=0
    for receiver in actors:
        if _receivers is not None and receiver not in _receivers:continue
        if not ledger.can_emit(task_id,receiver) or not ledger.charge_generation(task_id,receiver):continue
        available=min(emission_limit,8-ledger._row(task_id,receiver)['emitted'])
        if available<=0:continue
        closing=_tail_insertion(engine,plan,task_id,receiver,actors,ledger,mode,transfer_locations,_senders)
        priority=available-int(closing is not None)
        closing_key=None if closing is None else _delta_identity(closing)
        if priority:
            for delta in _priority_insertions(engine,plan,task_id,ledger,workers=workers,mode=mode,
                    transfer_locations=transfer_locations,emission_limit=priority,_actors=actors,
                    _receiver=receiver,_senders=_senders):
                if _delta_identity(delta)==closing_key:continue
                yield replace(delta,serial=serial);serial+=1
        if closing is not None and ledger.can_emit(task_id,receiver):
            ledger.emit(task_id,receiver)
            yield replace(closing,serial=serial);serial+=1


_REJECTIONS = {'custody', 'capacity', 'worker_withdrawal', 'meeting_domain',
    'handover_range', 'transfer_cycle', 'unreachable', 'remaining_custody', 'service_range', 'activity_domain'}


def options(engine, plan, task_id, included, *, mode='all', blink=0., limit=2,
            workers=None, grouping='worker', noise_s=0., base_cost=None,
            transfer_locations=None, first_feasible=False, ledger=None, per_worker_limit=8):
    from common.recovery.adapters._common import InsertionOption,SearchEngine
    from common.evaluation._insertion_numeric import NumericInsertionContext, UnsupportedNumericInsertion, UNSUPPORTED
    included = frozenset(included)
    enlarged = included | {task_id}
    if task_id in included or task_id not in engine.all_tasks:
        raise ValueError('inserted task must be known and not already included')
    ledger = ledger or getattr(engine, '_local_ledger', None) or InsertionBudget(engine, (task_id,))
    engine.stats['options_calls'] += 1
    choices = {}
    expanded = engine._project(enlarged)
    context = None
    use_numeric=(engine.candidate_evaluator=='incremental' and type(engine) is SearchEngine
        and 'evaluate' not in engine.__dict__
        and engine.config.objective_mode=='total_tardiness' and engine.transfer_policy=='candidate_points')
    if use_numeric:
        try:
            context = NumericInsertionContext(engine._project(included), expanded, plan,
                engine.routing, cache=engine.evaluation_cache)
        except (UnsupportedNumericInsertion, RecoveryPlanValidationError):
            pass
    work = engine.stats.setdefault('candidate_evaluation', dict(backend='bounded-local-' + engine.candidate_evaluator + '',
        numeric_scores=0, native_scores=0, selected_full_checks=0))
    retained = []
    def checked_choices():
        accepted = []
        for rank, delta, evaluated in sorted(choices.values(), key=lambda row: row[0]):
            if evaluated is None:
                work['selected_full_checks'] += 1
                try:
                    evaluated = engine._evaluate_candidate(expanded, delta.materialize())
                except RecoveryPlanValidationError as error:
                    if error.code not in _REJECTIONS:
                        raise
                    engine.stats['rejections'][error.code] = engine.stats['rejections'].get(error.code, 0) + 1
                    continue
            accepted.append(InsertionOption(evaluated.plan, evaluated,
                (evaluated.total_tardiness_s, evaluated.total_travel_time_s, evaluated.transfer_count),
                delta.worker_id, rank[0]))
            if limit and len(accepted) >= limit:
                break
        return accepted
    def spatial_insertions():
        owner = engine.holders.get(task_id)
        allowed = None if workers is None else set(workers)
        if allowed == set():
            return
        seen, actors = set(), ()
        maximum = engine.config.worker_candidates


        if (mode in ('all', 'direct') and owner in engine.local.basic_workers(task_id)
                and (allowed is None or owner in allowed)):
            yield from bounded_insertions(engine, plan, task_id, ledger, mode='direct',
                transfer_locations=transfer_locations, emission_limit=per_worker_limit,
                _actors=(owner,), _receivers=(owner,))
            seen.add(owner)
        for radius, ring in engine.local.iter_worker_rings(task_id, sender_id=owner,yield_empty=True):
            fresh = tuple(w for w in ring if w not in seen and (allowed is None or w in allowed))
            seen.update(ring)
            if maximum is not None:
                fresh = fresh[:max(0, maximum-len(actors))]
            previous = actors
            actors += fresh
            if fresh:
                engine.stats['last_spatial_insertion'] = dict(task_id=task_id,
                    radius_m=radius, workers_considered=len(actors))
                yield from bounded_insertions(engine, plan, task_id, ledger, mode=mode,
                    transfer_locations=transfer_locations, emission_limit=per_worker_limit,
                    _actors=actors, _receivers=fresh)
                if previous and owner is None and mode != 'direct':
                    yield from bounded_insertions(engine, plan, task_id, ledger,
                        mode='transfer' if mode == 'all' else mode,
                        transfer_locations=transfer_locations, emission_limit=per_worker_limit,
                        _actors=actors, _receivers=previous, _senders=fresh)


            retained[:] = checked_choices()
            if retained or maximum is not None and len(actors) >= maximum:
                return
            choices.clear()
    for delta in spatial_insertions():
        if not ledger.charge_evaluation(task_id, delta.worker_id):
            continue
        engine.stats['generated_moves'] += 1

        if blink and engine.rng.random() < blink:
            engine.stats['blink_skips'] += 1
            continue
        engine.check_budget()
        evaluated = None
        try:
            if context is not None:
                engine.stats['evaluations'] += 1
            numeric=context.score(delta) if context is not None else UNSUPPORTED
            if numeric is UNSUPPORTED:
                work['native_scores'] += 1


                if context is None:
                    evaluated=engine.evaluate(delta.materialize(),enlarged)
                    if evaluated is None:continue
                else:
                    evaluated=engine._evaluate_candidate(expanded,delta.materialize())
                result=evaluated
            else:
                work['numeric_scores'] += 1
                result=numeric
        except RecoveryPlanValidationError as error:
            if error.code not in _REJECTIONS:
                raise
            engine.stats['rejections'][error.code] = engine.stats['rejections'].get(error.code, 0) + 1
            continue
        if context is not None and enlarged==engine.all_tasks:
            engine.stats['complete_evaluations'] += 1
        cost=result.total_tardiness_s
        if noise_s:
            base = 0. if base_cost is None else base_cost
            cost = base + max(0., cost - base + engine.rng.uniform(-noise_s, noise_s))
        rank = (cost, result.total_travel_time_s, delta.worker_id, delta.serial)
        key = delta.worker_id if grouping == 'worker' else (delta.worker_id, delta.serial)
        if key not in choices or rank < choices[key][0]:
            choices[key] = (rank, delta, evaluated)
        if first_feasible:
            retained[:] = checked_choices()
            if retained:
                break
            choices.clear()
    if context is not None:
        counts = work.setdefault('numeric_execution',{})
        for key,value in context.stats.items():
            counts[key] = counts.get(key,0)+value
        work['numeric_kernel'] = context.kernel_receipt
    rows = ledger.by_task.get(task_id,{})
    budget = dict(task_count=int(task_id in ledger.by_task), worker_task_pairs=len(rows),
        generated=0, evaluated=0, emitted=0,
        max_generated_per_worker=0, max_evaluated_per_worker=0, max_emitted_per_worker=0)
    for row in rows.values():
        for field in ('generated', 'evaluated', 'emitted'):
            budget[field] += row[field]
            maximum = 'max_' + field + '_per_worker'
            budget[maximum] = max(budget[maximum], row[field])
    engine.stats['last_task_budget'] = budget
    return retained


def repair(engine, plan, bank, *, rule='greedy', mode='all', order=None,
           blink=0., alpha=None, regret_k=2, regret_group='worker', noise_s=0.,
           rcl_scope='task', first_feasible=False, rcl_pool_limit=None):

    bank = frozenset(bank)
    with search_frame(engine, 'repair', dict(plan=engine.solution_key(plan),bank=sorted(bank),
            rule=rule,mode=mode,order=order,blink=blink,alpha=alpha,rcl_pool_limit=rcl_pool_limit)) as frame:
        if frame.done:
            return frame.result
        state = frame.data
        if not state:
            if order is None:
                ranking = sorted(bank,key=lambda t:(engine.item.task_map[t].promised_delivery_time_s,t))
                if rule == 'longest':
                    ranking.sort(key=lambda t:(-engine.task_distance(t),t))
                elif rule == 'random':
                    engine.rng.shuffle(ranking)
            else:
                ranking = list(order)+sorted(bank-set(order))
            state.update(plan=plan,remaining=set(bank),ranking=ranking,ledger={})
            engine._selected_evaluation = None
        ledger = InsertionBudget(engine,bank,state=state['ledger'])
        previous = getattr(engine,'_local_ledger',None)
        engine._local_ledger = ledger
        try:
            while state['remaining']:
                included = engine.all_tasks-state['remaining']
                tasks = [t for t in state['ranking'] if t in state['remaining']]
                candidates = []
                for task_id in tasks if rule == 'regret' else tasks[:1]:
                    choices = options(engine,state['plan'],task_id,included,mode=mode,
                        blink=blink,limit=(rcl_pool_limit or 0) if alpha is not None else regret_k if rule == 'regret' else 1,
                        grouping='insertion' if alpha is not None else regret_group,
                        noise_s=noise_s,first_feasible=first_feasible,ledger=ledger,
                        per_worker_limit=1 if rule == 'regret' else 8)
                    if alpha is not None and rcl_pool_limit is not None:
                        work=engine.stats.setdefault('grasp_rcl_pool',dict(limit=rcl_pool_limit,
                            domain='first ranked physically feasible candidates within the current spatial ring',
                            full_candidate_distribution_equivalent=False,calls=0,total_candidates=0,max_candidates=0))
                        work['calls']+=1;work['total_candidates']+=len(choices)
                        work['max_candidates']=max(work['max_candidates'],len(choices))
                    if choices:
                        choice = engine.rng.choice(choices[:max(1,math.ceil(alpha*len(choices)))]) if alpha is not None else choices[0]
                        regret = math.inf if len(choices)<regret_k else sum(c.selection_cost-choices[0].selection_cost for c in choices[1:regret_k])
                        candidates.append((task_id,choice,regret))
                if not candidates:
                    return frame.finish((state['plan'],frozenset(state['remaining'])))
                task_id,chosen,_ = min(candidates,key=lambda row:(-row[2] if rule=='regret' else 0,row[1].selection_cost,row[0]))
                state['plan'] = chosen.plan
                state['remaining'].remove(task_id)
                engine._selected_evaluation = (chosen.plan,included|{task_id},chosen.evaluation)
                engine.stats['repair_rounds'] += 1
            return frame.finish((state['plan'],frozenset()))
        finally:
            engine._local_ledger = previous


def construct_local(engine, *, rule='best', bank=None, base=None, alpha=None, order=None,
                    rcl_pool_limit=None):
    beginning = perf_counter()
    bank = engine.local.repair_task_ids if bank is None else frozenset(bank)
    base = engine.local.base_plan if base is None else base
    engine.phase = 'initialization'
    try:
        proposal, missing = repair(engine, base, bank, rule='greedy', order=order,
            alpha=alpha, first_feasible=rule == 'greedy',rcl_pool_limit=rcl_pool_limit)
        result = None if missing else engine.evaluate_selected(proposal)
        if result is not None:
            engine.seed = result
            engine.remember(result, source='reference_revalidated' if not bank else
                engine.config.method + '_local_construction')
        engine.stats['initialization'] = dict(method_constructor_attempted=True,
            method_constructor_succeeded=result is not None, common_fallback_used=False,
            repair_task_count=len(bank), preserved_task_count=len(engine.all_tasks - bank))
        engine.stats['seed_source'] = 'reference_revalidated' if not bank else engine.config.method + '_local_construction'
        engine._initialized = True
        return result
    finally:
        engine.initialization_s += perf_counter() - beginning
        engine.phase = 'search'
