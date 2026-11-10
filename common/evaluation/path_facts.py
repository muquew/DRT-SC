from common.evaluation._recovery_cache import RecoveryEvaluationCache

_ENTRY_FACTS = '_drt_entry_input_path_facts'


def _check_snapshot(routing, facts):
    if (not isinstance(facts,tuple) or len(facts)!=3
            or facts[:2]!=(routing.roads.network_id,routing.roads.snapshot_id)):
        raise ValueError('input path facts belong to another road snapshot')


def capture_input_paths(routing):

    existing=getattr(routing,_ENTRY_FACTS,None)
    if existing is not None:
        _check_snapshot(routing,existing)
        return existing
    from common.evaluation._recovery_cache import shared_cache
    source=shared_cache(routing)
    if source is None:return None
    if source._routes.entries or source._components.entries:
        raise ValueError('input paths must be captured before search starts')
    facts=(routing.roads.network_id,routing.roads.snapshot_id,
        tuple((key,path,weight)for key,(path,weight)in source._paths.entries.items()))
    setattr(routing,_ENTRY_FACTS,facts)
    return facts


def fresh_path_cache(routing, facts):

    if facts is None:return None
    _check_snapshot(routing,facts)
    cache=RecoveryEvaluationCache(routing)
    for key,path,weight in facts[2]:
        if ((path.network_id,path.snapshot_id)!=facts[:2]
                or path.origin!=key[1]or path.destination!=key[2]or path.profile!=key[3]
                or len(path.legs)!=weight):
            raise ValueError('invalid immutable input path')
        cache._paths.put(key,path,weight)
    return cache


def validate_with_input_paths(item, plan, routing, facts):

    from common.evaluation.recovery_plan import evaluate_recovery_plan
    cache=fresh_path_cache(routing,facts)
    return (evaluate_recovery_plan(item,plan,routing) if cache is None else
            evaluate_recovery_plan(item,plan,routing,cache=cache))
