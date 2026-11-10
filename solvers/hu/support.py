from copy import deepcopy
import math
from time import perf_counter, process_time
from common.config import routing_options as configured_routing
from common.data.roads.routing import Routing
from common.evaluation.recovery_plan import evaluate_recovery_plan
from common.execution.state import ExecutionSnapshot
from common.scenarios.disruptions import DisruptionEvent, apply_disruptions
from common.recovery.inputs import RecoverySettings
from common.recovery.solver import SolverConfig
from common.recovery.adapters._common import SearchEngine
from solvers.alns import AdaptiveWeights, initial_temperature, anneal_accept
from solvers.hu.solver import repair
from solvers.hu.library import _remove, routing_batches, features, save_scenario_library
from solvers.hu.protocol import resolve, IterationBudget

class CandidatePool:


    def __init__(self, capacity, key):
        if type(capacity) is not int or capacity < 1:raise ValueError('Hu pool capacity must be positive')
        self.capacity, self.key = capacity, key
        self.entries = {}
        self.observed = 0
        self.duplicates = self.replacements = self.rejected_worse_or_equal = 0

    def offer(self, result):
        self.observed += 1
        key = self.key(result.plan)
        if key in self.entries:
            self.duplicates += 1
            return False
        if len(self.entries) == self.capacity:
            worst = max(self.entries, key=lambda k: self.entries[k].total_tardiness_s)
            if result.total_tardiness_s >= self.entries[worst].total_tardiness_s:
                self.rejected_worse_or_equal += 1
                return False
            del self.entries[worst]
            self.replacements += 1
        self.entries[key] = result
        return True

    def results(self):
        return sorted(self.entries.values(), key=lambda r: r.total_tardiness_s)

    def state_record(self):
        return {name:getattr(self,name) for name in ('entries','observed','duplicates',
            'replacements','rejected_worse_or_equal')}

    def restore_state_record(self,state):
        for name,value in state.items():setattr(self,name,value)

def _alns(engine, original_roads, settings, *, candidate_observer=None):

    from common.recovery.adapters._local_search import search_frame
    with search_frame(engine, 'hu_offline_alns', settings) as frame:
        if frame.done:return frame.result
        state=frame.data
        if not state:
            engine.stats['hu_iteration_budget']=IterationBudget(settings['total_iterations']).record('initializing')
            engine.stats.update(hu_reconcile_rebuilds=0,hu_reconciled_candidates=0)
            best=engine.construct_local(rule='best')
            engine.stats['seed_source']='hu_local_alns_construction'
            if best is None:
                engine.stats['hu_iteration_budget']=IterationBudget(settings['total_iterations']).record('no_feasible_initial_plan')
                return frame.finish([])
            pool=CandidatePool(settings.get('candidate_count',1),engine.solution_key)
            pool.offer(best);engine.seed=best
            engine.stats['seed_tardiness_s']=best.total_tardiness_s
            engine.stats.update(hu_offline_q_histogram={},hu_offline_batches=[],hu_offline_reheats=0,
                                hu_offline_iterations=0,hu_offline_operators=[])
            state.update(best=best,pool=pool.state_record(),used=0,batch_index=0,
                pass_start=0,batch=None,reason='total_iteration_budget_complete',
                batches=routing_batches(engine.item,original_roads,settings['route_batch_multiplier']))
        pool=CandidatePool(settings.get('candidate_count',1),engine.solution_key)
        pool.restore_state_record(state['pool'])
        budget=IterationBudget(settings['total_iterations']);budget.used=state['used']
        batches=state['batches']
        def publish(active_batch=None):
            if candidate_observer is None:return
            record=deepcopy(engine.stats)
            record['hu_iteration_budget']=budget.record('candidate_snapshot')
            record['hu_offline_iterations']=budget.used
            record.update(hu_pool_observed=pool.observed,hu_pool_retained=len(pool.entries),
                hu_pool_shortfall=pool.capacity-len(pool.entries),hu_pool_duplicates=pool.duplicates,
                hu_pool_replacements=pool.replacements,hu_pool_rejected_worse_or_equal=pool.rejected_worse_or_equal)
            if active_batch is not None:record.setdefault('hu_batch_timing',[]).append(active_batch)
            candidate_observer(pool.results(),record)
        if not budget.used:publish()
        while budget.remaining or state['batch'] is not None:
            index=state['batch_index']
            if index==len(batches):
                if budget.used==state['pass_start']:
                    state['reason']='no_available_actions_in_any_batch';break
                state.update(batch_index=0,pass_start=budget.used)
                index=0
            workers=batches[index]
            if state['batch'] is None:
                allowance=(budget.remaining+len(batches)-index-1)//(len(batches)-index)
                current=state['best']
                removals=AdaptiveWeights(('random','worst','shaw','pickup'),settings['reaction_factor'],settings['segment_size'])
                insertions=AdaptiveWeights(('greedy','regret_2','regret_3','regret_4'),settings['reaction_factor'],settings['segment_size'])
                temperature0=initial_temperature(current,settings['initial_temperature'])
                state['batch']=dict(current=current,allowance=allowance,used_before=budget.used,
                    temperature0=temperature0,temperature=temperature0,removals=vars(removals),
                    insertions=vars(insertions),wall_s=0.,cpu_s=0.,stop='allocated_share_complete')
                engine.stats['hu_offline_batches'].append(list(workers))
            batch=state['batch']
            removals=AdaptiveWeights.__new__(AdaptiveWeights);vars(removals).update(batch['removals'])
            insertions=AdaptiveWeights.__new__(AdaptiveWeights);vars(insertions).update(batch['insertions'])
            while budget.used-batch['used_before']<batch['allowance'] and budget.remaining:
                iteration_wall,iteration_cpu=perf_counter(),process_time()
                current=batch['current']
                candidates={v.task_id for w in workers for v in current.plan.sequences[w] if v.operation=='delivery'}
                if not candidates:
                    batch['stop']='no_available_tasks_in_batch';break
                count=len(candidates)
                if settings['removal_protocol']=='algorithm':
                    q=engine.rng.randint(min(8,count,math.floor(.2*count)+1),min(8,count))
                else:q=min(8,count,max(1,math.floor(.2*count)))
                histogram=engine.stats['hu_offline_q_histogram'];histogram[str(q)]=histogram.get(str(q),0)+1
                removing,inserting=removals.choose(engine.rng),insertions.choose(engine.rng)
                shortlist=sorted(candidates,key=lambda t:(-current.tardiness_s[t],t))[:8]
                removed=_remove(engine,current,set(shortlist),q,removing)
                partial=engine.strip(current.plan,removed)
                candidate=repair(engine,partial,removed,workers=workers,
                    regret_k=1 if inserting=='greedy' else int(inserting[-1]),noise_s=settings['noise_s'])
                reward=0;pool_changed=False
                if candidate is not None:
                    pool_changed=pool.offer(candidate)
                    if candidate.total_tardiness_s<state['best'].total_tardiness_s:
                        state['best']=current=candidate;reward=settings['rewards'][0]
                    elif candidate.total_tardiness_s<current.total_tardiness_s:
                        current=candidate;reward=settings['rewards'][1]
                    else:
                        reward=settings['rewards'][2]
                        if anneal_accept(current,candidate,batch['temperature'],engine.rng):current=candidate
                removals.record(removing,reward);insertions.record(inserting,reward)
                removals.end_iteration();insertions.end_iteration()
                batch['temperature']*=settings['cooling_rate']
                if batch['temperature']<settings['reheat_threshold']:
                    batch['temperature']=batch['temperature0'];engine.stats['hu_offline_reheats']+=1
                budget.consume()
                batch.update(current=current,removals=vars(removals),insertions=vars(insertions),
                    wall_s=batch['wall_s']+perf_counter()-iteration_wall,
                    cpu_s=batch['cpu_s']+process_time()-iteration_cpu)
                state.update(used=budget.used,pool=pool.state_record())
                engine.stats['hu_offline_iterations']=budget.used
                engine.stats['hu_iteration_budget']=budget.record('running')
                if pool_changed:
                    publish(dict(workers=list(workers),wall_s=batch['wall_s'],cpu_s=batch['cpu_s'],
                        allocated=batch['allowance'],used=budget.used-batch['used_before'],
                        remaining_global=budget.remaining,batch_index=index,stop='candidate_snapshot'))
            engine.stats['hu_offline_operators'].append(dict(removal=removals.summary(),insertion=insertions.summary()))
            engine.stats.setdefault('hu_batch_timing',[]).append(dict(workers=list(workers),
                wall_s=batch['wall_s'],cpu_s=batch['cpu_s'],allocated=batch['allowance'],
                used=budget.used-batch['used_before'],remaining_global=budget.remaining,
                batch_index=index,stop=batch['stop']))
            state.update(batch=None,batch_index=index+1,pool=pool.state_record())
        engine.stats['hu_iteration_budget']=budget.record(state['reason'])
        engine.stats.update(hu_pool_observed=pool.observed,hu_pool_retained=len(pool.entries),
            hu_pool_shortfall=pool.capacity-len(pool.entries),hu_pool_duplicates=pool.duplicates,
            hu_pool_replacements=pool.replacements,hu_pool_rejected_worse_or_equal=pool.rejected_worse_or_equal)
        return frame.finish(pool.results())

def validate_preparation_parameters(parameters):

    from inspect import signature
    names={'candidate_count','total_iterations','seed','removal_protocol','route_batch_multiplier',
        'segment_size','reaction_factor','initial_temperature','cooling_rate','reheat_threshold',
        'noise_fraction','rewards','objective_mode','meeting_pool'}
    if not isinstance(parameters,dict)or set(parameters)-names:
        raise ValueError('Unknown Hu preparation search parameters')
    declared=signature(prepare_support_library).parameters
    values={name:declared[name].default for name in names}
    values.update(parameters)
    resolved=resolve(values['total_iterations'],values['candidate_count'])
    values.update(total_iterations=resolved['total_iterations_per_scenario'],candidate_count=resolved['candidate_count'])
    if values['objective_mode']!='total_tardiness':
        raise ValueError('Hu preparation requires total_tardiness')
    for name in ('candidate_count','total_iterations','route_batch_multiplier','segment_size'):
        lower=0 if name=='total_iterations'else 1
        if type(values[name])is not int or values[name]<lower:
            raise ValueError(f'{name} must be an integer >= {lower}')
    if type(values['seed'])is not int or values['removal_protocol']not in ('algorithm','tuned'):
        raise ValueError('invalid seed or removal_protocol')
    if values['meeting_pool']not in ('legacy','road_nodes'):
        raise ValueError('meeting_pool must be legacy or road_nodes')
    for name in ('reaction_factor','cooling_rate','reheat_threshold','noise_fraction'):
        value=values[name]
        if type(value)not in (int,float)or not math.isfinite(value)or value<0:
            raise ValueError(f'{name} must be finite and nonnegative')
    if not 0<=values['reaction_factor']<=1 or not 0<values['cooling_rate']<1 or values['reheat_threshold']<=0:
        raise ValueError('invalid adaptive/annealing parameter range')
    temperature=values['initial_temperature']
    if temperature is not None and (type(temperature)not in (int,float)or not math.isfinite(temperature)or temperature<=0):
        raise ValueError('initial_temperature must be positive or None')
    rewards=values['rewards']
    if (not isinstance(rewards,(tuple,list))or len(rewards)!=3 or
            any(type(v)not in (int,float)or not math.isfinite(v)or v<0 for v in rewards)or
            not rewards[0]>rewards[1]>rewards[2]):
        raise ValueError('rewards must be three finite nonnegative strictly decreasing values')
    return values


def assemble_library(snapshot, roads, settings, scenes, parameters):
    width = len(scenes[0]['features']) if scenes else 0
    low = [min(scene['features'][i] for scene in scenes) for i in range(width)]
    scale = [max(scene['features'][i] for scene in scenes)-low[i] or 1. for i in range(width)]
    return dict(road_state_label=roads.version,
        source_arc_conditions={arc.id:[arc.travel_time_s,arc.enabled] for arc in roads.arcs},
        feature_min=low, feature_scale=scale, scenarios=scenes, parameters=parameters,
        offline_runtime_s=sum(scene['preparation_runtime_s'] for scene in scenes),
        offline_cpu_s=sum(scene['preparation_cpu_s'] for scene in scenes))


def prepare_support_library(snapshot, roads, recovery_settings, predeclared_event_specs, *,
                             candidate_count=None, total_iterations=None, seed=0,
                             removal_protocol='algorithm', route_batch_multiplier=2,
                             segment_size=100, reaction_factor=.1, initial_temperature=None,
                             cooling_rate=.995, reheat_threshold=.01, noise_fraction=.025,
                             rewards=(33,9,3), objective_mode='total_tardiness',
                             routing_options=None, meeting_pool='legacy', scenario_indices=None):
    if not isinstance(snapshot, ExecutionSnapshot) or not isinstance(recovery_settings, RecoverySettings):
        raise TypeError('Hu preparation requires an execution snapshot and recovery settings')
    resolved = resolve(total_iterations, candidate_count)
    total_iterations = resolved['total_iterations_per_scenario']
    candidate_count = resolved['candidate_count']
    settings = dict(total_iterations=total_iterations,candidate_count=candidate_count,seed=seed,
        removal_protocol=removal_protocol,route_batch_multiplier=route_batch_multiplier,
        segment_size=segment_size,reaction_factor=reaction_factor,initial_temperature=initial_temperature,
        cooling_rate=cooling_rate,reheat_threshold=reheat_threshold,rewards=rewards)
    validate_preparation_parameters(settings)
    if not math.isfinite(noise_fraction) or noise_fraction < 0:
        raise ValueError('noise_fraction must be finite and nonnegative')
    specs = tuple(predeclared_event_specs)
    if not specs:
        raise ValueError('Hu preparation requires hypothetical event scenes')
    indices = tuple(range(len(specs))) if scenario_indices is None else tuple(scenario_indices)
    if len(indices) != len(specs):
        raise ValueError('scene indices must match scenes')
    scenes = []
    for index, specification in zip(indices, specs):
        started, cpu = perf_counter(), process_time()
        events = tuple(e if isinstance(e, DisruptionEvent) else DisruptionEvent.from_dict(e)
                       for e in specification['events'])
        item = apply_disruptions(snapshot, roads, events, recovery_settings)
        routing = Routing(item.roads, **configured_routing(overrides=routing_options))
        holders = {task:worker for worker in item.workers for task in worker.carried_task_ids}
        distances = (routing.travel_time(holders[t.id].location if t.id in holders else t.pickup,t.delivery)
                     for t in item.tasks)
        noise_scale = max((value for value in distances if math.isfinite(value)),default=0.) if noise_fraction else 0.
        engine = SearchEngine(item,routing,SolverConfig(method='hu_scenario',
            seed=seed+104729*index,iterations=None,worker_candidates=None,meeting_candidates=8,
            objective_mode=objective_mode,meeting_pool=meeting_pool))
        results = _alns(engine,roads,{**settings,'noise_s':noise_fraction*noise_scale})
        plans=[]
        for value in results:
            checked=evaluate_recovery_plan(item,value.plan,Routing(item.roads,**configured_routing(overrides=routing_options)))
            if checked.to_dict()!=value.to_dict():
                raise ValueError('Hu candidate independent physical evaluation differs')
            plans.append(value.plan.to_dict())
        scenes.append(dict(id=specification['id'],global_index=index,
            events=[event.to_dict() for event in events],features=features(item,roads),candidates=plans,
            runs=[dict(engine.stats)],preparation_runtime_s=perf_counter()-started,
            preparation_cpu_s=process_time()-cpu))
    return assemble_library(snapshot,roads,recovery_settings,scenes,
        dict(settings,noise_fraction=noise_fraction,objective_mode=objective_mode,meeting_pool=meeting_pool))
