import math
import numpy as np

def _normalize_columns(values):

    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("B11 features must be a nonempty finite matrix")
    scales = np.max(np.abs(values), axis=0)
    scaled = values / np.where(scales == 0, 1, scales)
    minimum, span = np.min(scaled, axis=0), np.ptp(scaled, axis=0)
    return np.divide(scaled - minimum, span, out=np.zeros_like(scaled), where=span != 0)


def _features(engine, identifiers):
    rows, distances = [], []
    for identifier in identifiers:
        engine.check_budget()
        rows.append(engine.task_features(identifier)[:7])
        distance = engine.task_distance(identifier)
        if math.isnan(distance) or distance < 0:
            raise ValueError("B11 remaining travel must be nonnegative or positive infinity")
        distances.append(distance)
    normalized = _normalize_columns(rows)
    centered = normalized - normalized.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(identifiers) - 1)
    inverse = np.linalg.pinv(covariance, hermitian=True)
    difficulty = _normalize_columns([[engine.item.task_map[i].load] for i in identifiers]).ravel()
    finite = [i for i, distance in enumerate(distances) if math.isfinite(distance)]
    travel = np.full(len(identifiers), math.inf)
    if finite:
        travel[finite] = _normalize_columns([[distances[i]] for i in finite]).ravel()
    engine.stats["rlns_infinite_nominal_travel"] = tuple(
        identifier for identifier, distance in zip(identifiers, distances) if math.isinf(distance))
    difficulty = difficulty + travel
    return normalized, inverse, dict(zip(identifiers, difficulty.tolist()))


def _related_removal(identifiers, normalized, inverse, count, rng):
    seed = rng.randrange(len(identifiers))
    delta = normalized - normalized[seed]
    squared = np.maximum(0, np.einsum("ij,jk,ik->i", delta, inverse, delta))
    order = sorted((i for i in range(len(identifiers)) if i != seed),
                   key=lambda i: (float(squared[i]), identifiers[i]))
    return (identifiers[seed], *(identifiers[i] for i in order[:count - 1]))


def _late_accept(candidate, current, lagged):
    return candidate < current or candidate < lagged


def validate_parameters(parameters):
    names = {'remove_min','remove_max','history_length','blink_probability','protocol','paper_restarts','paper_no_improve','max_rounds'}
    if set(parameters) - names:
        raise ValueError('unknown local RLNS parameter')
    for name in names - {'blink_probability','protocol'}:
        if name in parameters and (type(parameters[name]) is not int or parameters[name] <= 0):
            raise ValueError(name + ' must be positive integer')
    minimum=parameters.get('remove_min',5)
    maximum=parameters.get('remove_max',8)
    if not 1<=minimum<=maximum<=8:
        raise ValueError('remove_min/remove_max must satisfy 1 <= min <= max <= 8 in the fixed local neighborhood')
    if 'blink_probability' in parameters and not 0 <= parameters['blink_probability'] <= 1:
        raise ValueError('blink_probability must be in [0,1]')
    if parameters.get('protocol', 'paper') not in ('paper','iterations'):
        raise ValueError('unknown RLNS protocol')
    return dict(parameters)


def solve(engine):
    from common.recovery.adapters._local_search import search_frame
    p = dict(remove_min=5, remove_max=8, history_length=20, blink_probability=.05,
        protocol='paper', paper_restarts=2, paper_no_improve=10, max_rounds=50)
    if set(engine.config.parameters) - set(p):
        raise ValueError('unknown local RLNS parameter')
    p.update(engine.config.parameters)
    if not 0 <= p['blink_probability'] <= 1:
        raise ValueError('blink probability must be in [0,1]')
    with search_frame(engine, 'rlns') as frame:
        if frame.done:
            return frame.result
        state = frame.data
        if not state:
            order = sorted(engine.local.repair_task_ids, key=lambda t: (
                len(engine.local.eligible_workers(t)), engine.item.task_map[t].promised_delivery_time_s, t))
            seed = engine.construct_local(rule='best', order=order)
            if seed is None:
                return frame.finish(None)
            ids = tuple(sorted(engine.all_tasks))
            normalized, inverse, difficulty = _features(engine, ids) if ids else (None,None,{})
            state.update(seed=seed, current=seed, identifiers=ids, normalized=normalized,
                inverse=inverse, difficulty=difficulty, restart=0, round=0, no_improve=0, phase='start',
                restart_best=seed.total_tardiness_s, history=[seed.total_tardiness_s]*p['history_length'])
        while state['restart'] < p['paper_restarts']:
            if engine.config.iterations is not None and engine.stats['iterations'] >= engine.config.iterations:
                break
            current = state['current']
            if not state['identifiers'] or current.total_tardiness_s == 0:
                engine.stats['local_stop_reason'] = 'zero_tardiness'
                break
            if state['round'] >= p['max_rounds'] or state['no_improve'] >= p['paper_no_improve']:
                state.update(restart=state['restart']+1, round=0, no_improve=0, current=state['seed'],
                    restart_best=state['seed'].total_tardiness_s,
                    history=[state['seed'].total_tardiness_s]*p['history_length'])
                continue
            if state['phase'] == 'start':
                engine.stats['iterations_started'] += 1
                count = min(8,len(state['identifiers']),engine.rng.randint(p['remove_min'],max(p['remove_min'],p['remove_max'])))
                state['removed'] = _related_removal(state['identifiers'],state['normalized'],state['inverse'],count,engine.rng)
                state['order'] = sorted(state['removed'],key=lambda t:(-state['difficulty'][t],t))
                state['phase'] = 'repair'
            removed = state['removed']
            plan, bank = engine.repair(engine.strip(current.plan, removed), removed,
                order=state['order'], blink=p['blink_probability'])
            candidate = None if bank else engine.evaluate_selected(plan)
            state['no_improve'] += 1
            if candidate is not None:
                engine.remember(candidate, source='rlns_local_repair')
                value = candidate.total_tardiness_s
                if value < state['restart_best']:
                    state['restart_best'], state['no_improve'] = value, 0
                if _late_accept(value,current.total_tardiness_s,state['history'][state['round'] % p['history_length']]):
                    state['current'] = candidate
            state['history'][state['round'] % p['history_length']] = state['current'].total_tardiness_s
            state['round'] += 1
            state['phase'] = 'start'
            engine.stats['iterations'] += 1
            engine.stats['rlns_restarts_completed'] = state['restart']
        engine.stats['rlns_restarts_completed'] = state['restart']
        return frame.finish(engine.best)
