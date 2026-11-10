from pathlib import Path
import json
import math
import shutil
from time import perf_counter, process_time
from common.io import read_json, write_json
from common.data.roads.routing import Routing
from common.config import routing_options as configured_routing
from common.evaluation.recovery_plan import evaluate_recovery_plan
from common.recovery.solver import SolverConfig
from common.recovery.adapters._common import SearchEngine
from solvers.deco.policy import DecoComponent
from solvers.deco.recovery import policy_pass, total_tardiness_reward
from common.recovery.policy_initialization import initial_policy_plan
from solvers.deco.data import EpisodeStore, resolve

def inspect_training(config_path, *, device=None):
    path=Path(config_path).resolve()
    config=read_json(path)
    config=dict(config)
    config['device']=device or config.get('device','cpu')
    if config['device'] not in ('cpu','auto','gpu'):
        raise ValueError('device must be cpu, auto or gpu')
    config.setdefault('seeds',[2,7])
    for key,default in [('patience',5),('checkpoint_every_episodes',50),('validation_seed',0),('improvement_threshold_s',1e-6)]:
        config.setdefault(key,default)
    for key in ('max_epochs','patience','checkpoint_every_episodes'):
        if type(config.get(key))is not int or config[key]<1:
            raise ValueError(key+' must be positive integer')
    if not config['seeds'] or len(set(config['seeds']))!=len(config['seeds']) or any(type(s)is not int or not 0<=s<2**31 for s in config['seeds']):
        raise ValueError('seeds must be distinct nonnegative int32 values')
    if not math.isfinite(config['improvement_threshold_s']) or config['improvement_threshold_s']<0:
        raise ValueError('improvement_threshold_s must be finite and nonnegative')
    store=EpisodeStore(resolve(path.parent,config['episodes']))
    selected={split:store.rows(split,config.get('datasets')) for split in ('train','validation','learning_test')}
    if not selected['train'] or not selected['validation']:
        raise ValueError('training and validation episodes are required')
    return config,store,selected


def run_episode(store, row, choose, on_transition=None):
    started, cpu = perf_counter(), process_time()
    item, history = store.load(row)
    seed = getattr(store, 'episode_routing', None)
    seed = Routing(item.roads, **configured_routing(overrides=store.routing_options)) if seed is None else seed
    from common.evaluation.path_facts import capture_input_paths, fresh_path_cache
    input_paths = capture_input_paths(seed)
    seed.register_query_input(item)
    routing = seed.fork_query_session()
    engine = SearchEngine(item, routing,
        SolverConfig(method='deco', iterations=1, seed=0, transfer_policy='candidate_points',
                     objective_mode='total_tardiness'))
    engine.stats['deco'] = dict(decisions=[])
    before = initial_policy_plan(engine, 'deco')
    if before is None:
        return dict(id=row['id'], status='no_complete_plan', high_actions=0, low_actions=0,
                    routing_cache=dict(engine.routing.cache_stats), evaluation_cache=dict(engine.evaluation_cache.stats),
                    distribution=row['distribution'], wall_s=perf_counter()-started, cpu_s=process_time()-cpu)
    def update(state, action, old, new, following, task, pair):
        if on_transition is not None:


            completed = item.time_origin_s + max(item.time_s, *old.completion_times_s.values(), *new.completion_times_s.values())
            on_transition(state, action, total_tardiness_reward(item, old, new), following, completed)
    after = policy_pass(engine, before, history, choose, on_transition=update if on_transition else None)
    checked = evaluate_recovery_plan(item, after.plan, engine.routing,
        cache=fresh_path_cache(engine.routing, input_paths))
    if checked.to_dict() != after.to_dict() or set(checked.completion_times_s) != set(item.task_map):
        raise ValueError('final policy plan failed independent complete-plan validation')
    decisions = engine.stats['deco']['decisions']
    boundary = getattr(item, 'execution_boundary', None)
    fixed = 0. if boundary is None else boundary.observed_tardiness_s+boundary.committed_tardiness_upper_s
    return dict(id=row['id'], status='feasible', distribution=row['distribution'],
        routing_cache=dict(engine.routing.cache_stats), evaluation_cache=dict(engine.evaluation_cache.stats),
        total_tardiness_s=checked.total_tardiness_s+fixed,
        suffix_tardiness_s=checked.total_tardiness_s, fixed_tardiness_s=fixed,
        reference_total_tardiness_s=before.total_tardiness_s+fixed,
        high_actions=len(decisions), low_actions=sum(len(r['low']) for r in decisions),
        filter_counts=dict(engine.stats['deco'].get('filter_counts', {})),
        transfer_count=len(checked.plan.transfers),
        complete_tasks=len(checked.completion_times_s), physical_continuation_wrapper=engine.stats['deco'].get('physical_continuation_wrapper', False),
        wall_s=perf_counter()-started, cpu_s=process_time()-cpu)

def evaluate_checkpoint(store, rows, checkpoint, *, seed=0, device='cpu'):
    model=DecoComponent.load_training(checkpoint,device=device)
    model.rng.seed(seed)
    results=[]
    try:
        choose=model.choose_action if all(model.updates.values()) else model.training_action
        for row in rows:
            results.append(run_episode(store,row,choose))
    finally:
        model.close()
    feasible=[r for r in results if r['status']=='feasible']
    total=math.fsum(r['total_tardiness_s'] for r in feasible)
    return dict(episodes=len(rows),complete_feasible=len(feasible),
        feasible_episode_ids=sorted(r['id'] for r in feasible),
        mean_full_tardiness_s=total/len(rows) if len(feasible)==len(rows) and rows else None,
        mean_feasible_full_tardiness_s=total/len(feasible) if feasible else None,records=results)


def numeric_update(stats, row):

    stats['transitions'][row['level']] += 1
    for name in ('reward', 'td_error', 'actor_loss', 'critic_loss'):
        value = row[name]
        if not math.isfinite(value):
            raise FloatingPointError(f'nonfinite {name}')
        record = stats['numeric'].setdefault(name, dict(count=0, sum=0., min=value, max=value))
        record.update(count=record['count']+1, sum=record['sum']+value,
                      min=min(record['min'], value), max=max(record['max'], value))
    reward = row['reward']
    for threshold in (10, 100, 1000, 10000):
        key = str(threshold)
        stats['reward_above'][key] = stats['reward_above'].get(key, 0) + int(reward > threshold)

def empty_stats():
    return dict(episode_visits=0, complete_feasible=0, zero_actor_episodes=0,
        high_actions=0, low_actions=0, transitions={'high': 0, 'low': 0}, observed_transition_events={'high': 0, 'low': 0},
        numeric={}, reward_above={}, filter_counts={}, distributions={}, wall_s=0., cpu_s=0.)

def validation_quality(validation, state, updates):

    feasible_ids = validation.get('feasible_episode_ids')
    if feasible_ids is not None:
        previous_ids = state.setdefault('validation_feasible_ids', feasible_ids)
        if previous_ids != feasible_ids:
            raise ValueError('validation initialization-success sets changed across checkpoints; selection blocked')
    if not all(updates.values()):
        return None
    return validation.get('mean_feasible_full_tardiness_s', validation['mean_full_tardiness_s'])

def persist(model, root, state):
    number=state.get('checkpoint_sequence',0)+1
    destination=root/f'checkpoint-{number:06d}'
    model.save_training(destination)
    previous=state.get('latest_checkpoint')
    state.update(checkpoint_sequence=number,latest_checkpoint=destination.name,previous_checkpoint=previous)
    write_json(root/'progress.json',state)
    keep={state.get('latest_checkpoint'),state.get('previous_checkpoint'),state.get('best_checkpoint')}
    for path in root.glob('checkpoint-*'):
        if path.is_dir() and not path.is_symlink() and path.name.removeprefix('checkpoint-').isdigit() and path.name not in keep:
            shutil.rmtree(path)
    return destination


def train_seed(config, store, selected, root, seed, *, resume=False):
    root.mkdir(parents=True,exist_ok=True)
    progress=root/'progress.json'
    if progress.exists():
        if not resume:
            raise FileExistsError('training output exists; use --resume or another output')
        state=read_json(progress)
        model=DecoComponent.load_training(root/state['latest_checkpoint'],device=config['device'])
    else:
        if any(root.iterdir()):
            raise FileExistsError('training output is not empty')
        model=DecoComponent(seed=seed,tensorflow_seed=seed,reward_mode='total_tardiness',device=config['device'])
        state=dict(epoch=0,next_episode=0,bad_validations=0,best_mean_tardiness_s=None,
            best_checkpoint=None,training=empty_stats(),current_epoch=empty_stats(),history=[])
        persist(model,root,state)
    started=perf_counter()
    try:
        while state['epoch']<config['max_epochs']:
            for index in range(state['next_episode'],len(selected['train'])):
                row=selected['train'][index]
                def update(first,action,reward,following,completed):
                    loss=model.train_transition(first,action,reward,following,
                        outcome_observed_at_s=completed,offline_replay=True)
                    numeric_update(state['training'],loss)
                    numeric_update(state['current_epoch'],loss)
                result=run_episode(store,row,model.training_action,update)
                for target in (state['training'],state['current_epoch']):
                    target['episode_visits']+=1
                    target['complete_feasible']+=result['status']=='feasible'
                    target['zero_actor_episodes']+=not result['high_actions']
                    for key in ('high_actions','low_actions','wall_s','cpu_s'):
                        target[key]+=result[key]
                state['next_episode']=index+1
                if state['next_episode']%config['checkpoint_every_episodes']==0:
                    persist(model,root,state)
            checkpoint=persist(model,root,state)
            validation=evaluate_checkpoint(store,selected['validation'],checkpoint,
                seed=config['validation_seed'],device=config['device'])
            quality=validation_quality(validation,state,model.updates)
            better=quality is not None and (state['best_mean_tardiness_s'] is None or
                quality<state['best_mean_tardiness_s']-config['improvement_threshold_s'])
            if better:
                state.update(best_mean_tardiness_s=quality,best_checkpoint=checkpoint.name,bad_validations=0)
            else:
                state['bad_validations']+=1
            state['history'].append(dict(epoch=state['epoch']+1,validation_mean_T=quality,
                training=state['current_epoch'],selected=better))
            print(json.dumps(dict(seed=seed,epoch=state['epoch']+1,validation_mean_T=quality,selected=better)),flush=True)
            state.update(epoch=state['epoch']+1,next_episode=0,current_epoch=empty_stats())
            persist(model,root,state)
            if state['bad_validations']>=config['patience']:
                break
        result=dict(seed=seed,epochs_completed=state['epoch'],
            selected_checkpoint=str(root/state['best_checkpoint']) if state['best_checkpoint'] else None,
            validation_mean_T=state['best_mean_tardiness_s'],updates=dict(model.updates),
            stop_reason='validation_patience' if state['bad_validations']>=config['patience'] else 'max_epochs',
            wall_s=perf_counter()-started)
        write_json(root/'training.json',result)
        return result
    finally:
        model.close()


def train(config_path, output, *, resume=False, device=None):
    config,store,selected=inspect_training(config_path,device=device)
    output=Path(output).resolve()
    return [train_seed(config,store,selected,output/f'seed-{seed}',seed,resume=resume)
            for seed in config['seeds']]
