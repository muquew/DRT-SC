from dataclasses import replace
import math
from pathlib import Path
from time import perf_counter

from common.recovery.plan import RecoveryPlan
from common.recovery.adapters._local_search import InsertionBudget


def insertion_options(engine, plan, task_id, included, workers=None, *, noise_s=0.):
    return engine.options(plan, task_id, included, workers=workers, noise_s=noise_s, limit=0)

def repair(engine, plan, bank, *, workers=None, regret_k=1, noise_s=0., order=None, preferred_targets=None):
    bank = frozenset(bank)
    pending = set(bank)
    ledger = InsertionBudget(engine, bank)
    previous = getattr(engine, '_local_ledger', None)
    engine._local_ledger = ledger
    result = None
    ranking = list(order or sorted(bank, key=lambda t: (engine.item.task_map[t].promised_delivery_time_s,t)))
    ranking.extend(sorted(bank-set(ranking)))
    try:
        def options(task):
            included = engine.all_tasks - pending
            preferred = None if preferred_targets is None else preferred_targets.get(task)
            kwargs = dict(noise_s=noise_s,limit=max(1,regret_k))
            if regret_k > 1:


                kwargs['per_worker_limit'] = 1
            choices = engine.options(plan, task, included,
                workers=(preferred,) if preferred else workers, **kwargs)
            if not choices and preferred:
                choices = engine.options(plan, task, included, workers=workers, **kwargs)
            return choices
        while pending:
            tasks = [task for task in ranking if task in pending]
            candidates = []
            for task in tasks if regret_k > 1 else tasks[:1]:
                choices = options(task)
                if not choices:
                    continue
                regret = (math.inf if len(choices) < regret_k else
                    sum(choice.selection_cost-choices[0].selection_cost for choice in choices[1:regret_k]))
                candidates.append((task,choices[0],regret))
            if not candidates:
                return None
            task,selected,_ = min(candidates,key=lambda row:(
                -row[2],row[1].selection_cost,row[0]))
            plan, result = selected.plan, selected.evaluation
            pending.remove(task)
        return result if result is not None else engine.evaluate(plan)
    finally:
        engine._local_ledger = previous

def _reference(engine):
    return engine.local.reference_plan

def reconcile(engine, stored):

    source = RecoveryPlan.from_dict(stored)


    stored_tasks={v.task_id for route in source.sequences.values() for v in route
        if v.operation in ('pickup','delivery')}
    source=engine.strip(source,stored_tasks-engine.all_tasks)
    source=replace(source,sequences={w:tuple(v for v in route
        if not (v.operation=='pickup' and v.task_id in engine.holders))
        for w,route in source.sequences.items()})
    targets = {v.task_id:w for w,route in source.sequences.items() for v in route
        if v.operation == 'delivery' and v.task_id in engine.all_tasks}
    bank = engine.local.repair_task_ids


    source = replace(source,input_id=engine.item.instance_id,network_id=engine.item.network_id,
        snapshot_id=engine.item.snapshot_id)
    if set(source.sequences) != set(engine.item.worker_map):
        return None
    partial = engine.strip(source, bank)
    if engine.evaluate(partial, engine.all_tasks-bank) is None:
        engine.stats['hu_reconcile_incompatible_background'] = engine.stats.get('hu_reconcile_incompatible_background',0)+1
        return None
    result = repair(engine, partial, bank, preferred_targets=targets)
    if result is not None:
        engine.stats['hu_reconciled_candidates'] = engine.stats.get('hu_reconciled_candidates',0)+1
    return result

def online_work(parameters):
    from solvers.parameters import load_algorithm_config
    defaults=load_algorithm_config(None,'hu')['parameters']
    values={name:parameters.get(name,defaults[name]) for name in (
        'online_local_passes','online_local_actions')}
    for name,value in values.items():
        if type(value)is not int or value<1:
            raise ValueError(name+' must be a positive integer')
    return values

def improve(engine, current, severity):
    from solvers.grasp import improve as local_improve
    work=online_work(engine.config.parameters)
    engine.stats['hu_candidates_local_searched']=engine.stats.get('hu_candidates_local_searched',0)+1
    return local_improve(engine,current,passes=work['online_local_passes'],
        action_limit=work['online_local_actions'],source='hu')

def fallback(engine,reason):

    if engine.stats.get('hu_fallback_attempts',0):
        raise RuntimeError('Hu fallback was already attempted in this invocation')
    engine.stats.update(hu_used_library=False,hu_library_unavailable_reason=reason,
        hu_fallback='one_local_greedy_construction',hu_fallback_attempts=1)
    result=engine.construct_local(rule='greedy')
    engine._initialized=True
    return result

def solve(engine):
    from solvers.hu.library import load_scenario_library, match_scenario, original_roads, disrupted_requests
    parameters = engine.config.parameters
    if not parameters.get('library_path'):
        raise ValueError('Hu requires library_path')
    engine.stats['hu_online_configured_work'] = online_work(parameters)
    started = perf_counter()
    library = load_scenario_library(parameters['library_path'])
    roads = original_roads(library, engine.item)
    matched, distance = match_scenario(library, engine.item, roads)
    engine.stats.update(hu_library_scenarios=len(library['scenarios']), hu_reconciled_candidates=0,
        hu_used_library=False, hu_offline_runtime_s=library.get('offline_runtime_s'),
        hu_matched_scenario=matched['id'], hu_match_distance=distance,
        hu_candidate_count=min(3, len(matched['candidates'])))
    severity = disrupted_requests(engine.item, roads)
    for stored in matched['candidates'][:3]:
        engine.check_budget()
        current = reconcile(engine, stored)
        if current is None:
            continue
        engine.stats['hu_used_library'] = True
        if engine.seed is None:
            engine.seed = current
            engine.stats['seed_source'] = 'hu_scenario_local_reconciliation'
            engine.initialization_s = perf_counter() - started
        engine.remember(current, source='hu_scenario_local_reconciliation')
        improve(engine, current, severity)
    if engine.best is None:
        return fallback(engine, 'matched_scene_has_no_usable_complete_candidate')
    engine._initialized = True
    return engine.best
