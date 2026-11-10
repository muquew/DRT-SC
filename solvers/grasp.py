from itertools import combinations, islice
import math


def validate_parameters(parameters):
    if set(parameters) - {'alpha','swap_passes','local_actions','rcl_pool_limit'}:
        raise ValueError('unknown local GRASP parameter')
    if 'alpha' in parameters and not 0 < parameters['alpha'] <= 1:
        raise ValueError('alpha must be in (0,1]')
    for name in ('swap_passes','local_actions'):
        if name in parameters and (type(parameters[name]) is not int or parameters[name] < 0):
            raise ValueError(name + ' must be nonnegative integer')
    if 'rcl_pool_limit' in parameters and (
            type(parameters['rcl_pool_limit']) is not int or parameters['rcl_pool_limit']<1):
        raise ValueError('rcl_pool_limit must be a positive integer')
    return dict(parameters)


def improve(engine, current, passes=5, action_limit=64, source='grasp'):
    from common.recovery.adapters._local_search import InsertionBudget
    for _ in range(passes):
        ledger = InsertionBudget(engine)
        previous = getattr(engine, '_local_ledger', None)
        engine._local_ledger = ledger
        improved = current
        identifiers = sorted(current.tardiness_s, key=lambda t: (-current.tardiness_s[t], t))

        identifiers = identifiers[:16]
        try:
            for first, second in islice(combinations(identifiers, 2), action_limit):
                engine.check_budget()
                engine.stats['local_actions'] = engine.stats.get('local_actions', 0) + 1
                removed = {first, second}
                partial = engine.strip(current.plan, removed)
                options = engine.options(partial, first, engine.all_tasks - removed, limit=1)
                if not options:
                    continue
                tail = engine.options(options[0].plan, second, engine.all_tasks - {second}, limit=1)
                if tail and tail[0].evaluation.total_tardiness_s < improved.total_tardiness_s:
                    improved = tail[0].evaluation
        finally:
            engine._local_ledger = previous
        if improved.total_tardiness_s >= current.total_tardiness_s:
            break
        current = improved
        engine.remember(current, source=source + '_local_exchange')
    return current


def solve(engine):
    from common.recovery.adapters._local_search import search_frame
    from solvers.parameters import load_algorithm_config
    p = {**load_algorithm_config(None,'grasp')['parameters'], **engine.config.parameters}
    validate_parameters(p)
    with search_frame(engine, 'grasp') as frame:
        if frame.done:
            return frame.result
        state = frame.data
        state.setdefault('restart', 0)
        state.setdefault('phase','start')
        engine.stats['method_parameters'] = p
        while state['restart'] < engine.config.iterations:
            if state['phase'] == 'start':
                engine.stats['iterations_started'] += 1
                state['phase'] = 'construction'
            current = engine.construct_local(rule='randomized', alpha=p['alpha'],
                rcl_pool_limit=p['rcl_pool_limit'])
            if current is not None:
                improve(engine, current, p['swap_passes'], p['local_actions'])
            state['restart'] += 1
            state['phase'] = 'start'
            engine.stats['iterations'] = state['restart']
        return frame.finish(engine.best)
