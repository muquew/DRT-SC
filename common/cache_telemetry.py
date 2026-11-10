from functools import lru_cache, wraps


def retention_counts():
    return dict(evictions=0, bypasses=0, bypass_reasons={})


def bypass(counts, reason):

    counts['bypasses'] += 1
    reasons = counts['bypass_reasons']
    reasons[reason] = reasons.get(reason, 0) + 1


def observed_lru_cache(maxsize=128, typed=False):


    def decorate(fn):
        state = dict(successes=0, errors=0, active=0, overlapping=False, clears=0)
        @wraps(fn)
        def compute(*args, **kwargs):
            try:
                value = fn(*args, **kwargs)
            except BaseException:
                state['errors'] += 1
                raise
            state['successes'] += 1
            return value
        cached = lru_cache(maxsize=maxsize, typed=typed)(compute)
        @wraps(fn)
        def call(*args, **kwargs):
            if state['active']:state['overlapping'] = True
            state['active'] += 1
            try:return cached(*args, **kwargs)
            finally:state['active'] -= 1
        def clear():
            cached.cache_clear()
            state['clears'] += 1
            state.update(successes=0, errors=0, overlapping=bool(state['active']))
        def snapshot():
            info = cached.cache_info()
            reasons = {}
            if state['errors']:reasons['computation_exception'] = state['errors']
            if maxsize == 0 and state['successes']:reasons['disabled'] = state['successes']
            return dict(evictions=None if state['overlapping'] else (max(0,state['successes']-info.currsize) if maxsize else 0),
                bypasses=sum(reasons.values()),bypass_reasons=reasons,
                accounting_exact=not state['overlapping'],
                clears=state['clears'],reset_scope='hit/miss/retention since cache_clear; clears since wrapper construction',
                accounting_reason='overlapping/reentrant native calls; duplicate insertion cannot be attributed' if state['overlapping'] else None,
                counting_unit='completed uncached call; one primary bypass reason; reset with cache_clear')
        call.cache_info=cached.cache_info;call.cache_clear=clear
        call.cache_parameters=cached.cache_parameters;call.retention_stats=snapshot
        return call
    return decorate


def lru_snapshot(cache, scope):
    info=cache.cache_info()
    return dict(cache.retention_stats(),hits=info.hits,misses=info.misses,entries=info.currsize,
                max_entries=info.maxsize,scope=scope)


def prefixed_snapshot(stats, prefix, **metadata):

    fields=('hits','misses','entries','evictions','bypasses','bypass_reasons','clears')
    row={name:stats[prefix+name] for name in fields}
    row['bypass_reasons']=dict(row['bypass_reasons'])
    return dict(row,**metadata)
