def solve(engine):
    if engine.config.parameters:
        raise ValueError('local Greedy has no method-specific parameters')
    from common.recovery.adapters._local_search import search_frame
    with search_frame(engine, 'greedy') as frame:
        if frame.done:
            return frame.result
        state = frame.data
        if 'result' not in state:
            engine.stats['method_parameters'] = dict(rule='nearest_feasible', construction_scope='D')
            engine.stats['iterations_started'] = 1
            state['result'] = engine.construct_local(rule='greedy')
            engine.stats.update(iterations=1, constructive_complete_constructions=int(state['result'] is not None),
                constructive_incomplete_constructions=int(state['result'] is None),
                constructive_seed_fallback_used=False)
        return frame.finish(state['result'])
