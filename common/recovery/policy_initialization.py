from dataclasses import replace
from time import perf_counter, process_time
import math
from common.io import read_json
from common.recovery.plan import RecoveryVisit, TransferEvent

def _number(value, name, *, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError(f'{name} must be finite'+(' and nonnegative' if nonnegative else ''))
    return float(value)

def history_from_replay(replay, tau):

    tau = _number(tau, 'history time')
    source = replay.planning_input
    if tau < source.planning_time_s:
        raise ValueError('history time precedes the original plan')
    workers = {worker.id: {'position_samples': []} for worker in source.workers}
    for offset in (-90., -60., -30., 0.):
        time = tau + offset
        if time < source.planning_time_s:
            continue
        snapshot = replay.at(time)
        for worker in snapshot.workers:
            workers[worker.worker_id]['position_samples'].append(dict(
                time_s=source.time_origin_s+time,
                xy_m=list(replay.routing.roads.coordinates(worker.location))))
    return dict(as_of_s=source.time_origin_s+tau,
                source_kind='pre_tau_plan_replay', workers=workers)


def load_history(path, item, *, require='position'):

    if require != 'position':
        raise ValueError('DECO uses position history')
    projection = None
    if getattr(item, 'execution_boundary', None) is not None:
        projection = dict(history_observed_at_s=item.original_input.time_origin_s+item.original_input.time_s,
                          suffix_activation_at_s=item.time_origin_s+item.time_s)
        item = item.original_input
    data = read_json(path)
    now = item.time_origin_s+item.time_s
    if not isinstance(data, dict):
        raise ValueError('expected a position history object')
    data.setdefault('source_kind','observed')
    data.setdefault('as_of_s',now)
    if _number(data['as_of_s'],'history observation time') > now:
        raise ValueError('history observation time exceeds tau')
    if not isinstance(data.get('workers'),dict) or set(data['workers']) != set(item.worker_map):
        raise ValueError('history must cover every recovery worker')
    for worker, streams in data['workers'].items():
        if not isinstance(streams,dict) or not isinstance(streams.get('position_samples'),list):
            raise ValueError('missing position samples for worker '+worker)
        previous = None
        for row in streams['position_samples']:
            if not isinstance(row,dict):
                raise ValueError('position samples must be objects')
            time = _number(row['time_s'],'position observation time')
            if time > now or previous is not None and time <= previous:
                raise ValueError('position times must increase and cannot exceed tau')
            previous = time
            point = row.get('xy_m')
            if not isinstance(point,(list,tuple)) or len(point) != 2:
                raise ValueError('position samples require two coordinates')
            for coordinate in point:
                _number(coordinate,'position coordinate')
    if projection is not None:
        data = dict(data,execution_projection=projection)
    return data


def initial_policy_plan(engine, method):

    start, cpu_started = perf_counter(), process_time()
    before = engine.stats['evaluations']
    current = engine.construct_local(rule='greedy')
    info = engine.stats.setdefault(method, {})
    info['physical_continuation_wrapper'] = bool(engine.local.repair_task_ids)
    info['continuation_wrapper'] = dict(wall_s=perf_counter()-start,
        cpu_s=process_time()-cpu_started, evaluations=engine.stats['evaluations']-before,
        source=engine.stats.get('seed_source'), scope='local D repair; legal background retained')
    if current is not None:
        engine.commit_policy(current, source=method+'_initial_complete')
    return current

def owners(plan):
    return {visit.task_id: worker for worker, route in plan.sequences.items()
            for visit in route if visit.operation == 'delivery'}

def focal_move(engine, plan, task_id, receiver, location=None, *, prefix='policy', evaluate=True):

    if engine.item.worker_map[receiver].service_mode != 'delivery':
        return None
    base = engine.strip(plan, {task_id})
    routes = dict(base.sequences)
    holder = engine.holders.get(task_id)
    delivery = RecoveryVisit(operation='delivery', task_id=task_id)
    events = base.transfers
    if holder is None:
        routes[receiver] += (RecoveryVisit(operation='pickup', task_id=task_id), delivery)
    elif holder == receiver:
        routes[receiver] += (delivery,)
    else:
        if location is None:
            raise ValueError('held-task reassignment requires an explicit physical location')
        number = 1
        while f'{prefix}-{number}' in base.transfer_map:
            number += 1
        event = TransferEvent(id=f'{prefix}-{number}', worker_i=holder, worker_j=receiver,
                              location=location, tasks_i_to_j=(task_id,))
        visit = RecoveryVisit(operation='transfer', transfer_id=event.id)

        existing = next((e for e in events if {e.worker_i, e.worker_j} == {holder, receiver}
                         and e.location == location and routes[holder] and routes[receiver]
                         and routes[holder][0].transfer_id == e.id and routes[receiver][0].transfer_id == e.id), None)
        if existing is not None:
            field = 'tasks_i_to_j' if existing.worker_i == holder else 'tasks_j_to_i'
            event = replace(existing, **{field: (*getattr(existing, field), task_id)})
            events = tuple(event if e.id == existing.id else e for e in events)
            routes[receiver] += (delivery,)
        else:
            events = (*events, event)
            routes[holder] = (visit,)+routes[holder]
            routes[receiver] = (visit,)+routes[receiver]+(delivery,)
    proposal = replace(base, sequences=routes, transfers=events)
    engine.stats['generated_moves'] += 1
    return engine.evaluate(proposal) if evaluate else proposal
