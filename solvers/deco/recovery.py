from contextlib import nullcontext
from dataclasses import replace
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import signal
import select
import resource
import shutil
import tempfile
from time import perf_counter, process_time

from solvers.deco.policy import DecoState, REWARDS, checkpoint_metadata, state_source_id
from common.recovery.policy_initialization import focal_move, initial_policy_plan, load_history, owners
from common.recovery.adapters._common import position_key, BudgetExhausted
from common.initial_sequences.inputs import _exceeds_capacity


def total_tardiness_reward(item, before, after):

    tasks = set(item.task_map)
    if set(before.tardiness_s) != tasks or set(after.tardiness_s) != tasks:
        raise ValueError('DECO tardiness reward requires complete before/after plans')
    boundary = getattr(item, 'execution_boundary', None)
    fixed = 0. if boundary is None else boundary.observed_tardiness_s+boundary.committed_tardiness_upper_s
    return (1.+before.total_tardiness_s+fixed)/(1.+after.total_tardiness_s+fixed)


class PolicyWorker:
    def __init__(self, python, checkpoint, engine):
        engine.check_budget()
        self.engine = engine
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]),
            DRT_SC_DECO_DEVICE=engine.config.parameters.get('device','cpu'),
            TF_CPP_MIN_LOG_LEVEL='2',TF_ENABLE_ONEDNN_OPTS='0')
        self.errors = tempfile.TemporaryFile(mode='w+t')
        self.cpu_s = self.wall_s = 0.
        self._output = bytearray()
        self._closed = self._abort = False
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.children_cpu_before = usage.ru_utime+usage.ru_stime
        self.process_started = perf_counter()
        try:
            self.process = subprocess.Popen([str(python),'-m','solvers.deco.worker',str(checkpoint),str(engine.config.seed)],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.errors,text=True,env=env)
        except BaseException:
            self.errors.close()
            raise
        try:
            response = self._receive()
            if response.get('status') != 'ready':
                raise RuntimeError('DECO worker did not load a checkpoint')
            self.training = response['training']
            engine.stats.setdefault('deco',{}).update(model_load_wall_s=response['wall_s'],model_load_cpu_s=response['cpu_s'])
        except BaseException:
            self._abort = True
            self.close()
            raise

    def _receive(self):


            try:
                while b'\n' not in self._output:
                    self.engine.check_budget()

                    if not select.select([self.process.stdout], [], [], .25)[0]:
                        continue
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        self.errors.seek(0)
                        raise RuntimeError('DECO worker terminated: '+self.errors.read()[-2000:])
                    self._output.extend(chunk)
                line, _, rest = self._output.partition(b'\n')
                self._output = bytearray(rest)
                value = json.loads(line)
            except BaseException:
                self._abort = True
                raise
            self.cpu_s += value.get('cpu_s', 0)
            self.wall_s += value.get('wall_s', 0)
            return value

    def choose(self, state):
            self.engine.check_budget()
            self.process.stdin.write(json.dumps({'state': state.to_dict()}, allow_nan=False)+'\n')
            self.process.stdin.flush()
            result = self._receive()
            if result.get('status') != 'ok':
                raise RuntimeError(result.get('error', 'DECO inference failed'))
            self.engine.check_budget()
            return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.process.poll() is None:
                if self._abort:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                else:
                    try:
                        self.process.stdin.write('{"close":true}\n')
                        self.process.stdin.flush()
                    except BrokenPipeError:
                        pass
                    self.process.wait()
        finally:
            self.process.stdin.close()
            self.process.stdout.close()
            self.errors.close()
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            self.total_cpu_s = usage.ru_utime+usage.ru_stime-self.children_cpu_before
            self.total_wall_s = perf_counter()-self.process_started
            self.engine.stats.setdefault('deco',{}).update(native_worker_cpu_s=self.total_cpu_s,
                native_worker_wall_s=self.total_wall_s,native_request_cpu_s=self.cpu_s,
                native_request_wall_s=self.wall_s,worker_returncode=self.process.returncode)

def _xy(engine, position):
    return engine.item.roads.coordinates(position)


def _ll(engine, position):
    x, y = _xy(engine, position)
    if engine.item.roads.crs.startswith('local:'):
        return y/111194.92664455873, x/111194.92664455873
    from pyproj import Transformer
    transformer = getattr(engine, '_deco_coordinates', None)
    if transformer is None:
        transformer = engine._deco_coordinates = Transformer.from_crs(engine.item.roads.crs, 'EPSG:4326', always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon


def current_position(engine, worker):

    item = engine.item
    boundary = getattr(item, 'execution_boundary', None)
    original = boundary.original_input if boundary is not None else getattr(item, 'original_input', item)
    return original.worker_map[worker].location


def _prefix_position(engine, worker, at, boundary):

    roads = engine.item.roads
    location = current_position(engine, worker)
    for execution in sorted((row for row in boundary.prefix if worker in row.block.participants),
                            key=lambda row: row.start_s):
        block = execution.block
        if at < execution.start_s:
            return location
        if at < execution.end_s:
            if block.kind != 'travel':
                return block.destination
            state = boundary.original_input.source_snapshot.worker_map[worker]
            travel = state.travel
            if travel is not None and execution.start_s == boundary.H:
                elapsed = travel.elapsed_time_s
                for leg in travel.path.legs:
                    if elapsed >= leg.duration_s:
                        elapsed -= leg.duration_s
                        continue
                    segment = roads.segment_map[leg.segment_id]
                    origin = block.origins[block.participants.index(worker)]
                    start = (origin.offset_m if origin.segment_id == segment.id else
                        0. if origin.node_id == segment.u else
                        segment.length_m if origin.node_id == segment.v else None)
                    if (start is not None and origin == current_position(engine, worker)
                            and roads.segment_position(segment.id, leg.end_offset_m) == block.destination):
                        fraction = (at-execution.start_s)/(execution.end_s-execution.start_s)
                        return roads.segment_position(segment.id,
                            start+(leg.end_offset_m-start)*fraction)
                    break


            padding = engine.stats.setdefault('deco', {}).setdefault('forecast_padding',
                dict(reason='prefix_travel_path_unavailable', position='block_origin', uses=0))
            padding['uses'] += 1
            return block.origins[block.participants.index(worker)]
        location = block.destination
    return location


def predicted_position(engine, evaluation, worker, at):
    location = engine.item.worker_map[worker].location
    boundary = getattr(engine.item, 'execution_boundary', None)
    previous = engine.item.time_s if boundary is None else boundary.release_times[worker]
    if boundary is not None and at < previous:
        return _prefix_position(engine, worker, at, boundary)
    for visit in evaluation.routes[worker]:
        if at < visit.arrival_time_s:
            return engine.routing.position_after(visit.incoming_path, max(0, at-previous))
        location = visit.position
        if at < visit.completion_time_s:
            return location
        previous = visit.completion_time_s
    return location


def midpoint_on_road(engine, evaluation, pair):
    roads, domain = engine.item.roads, engine.item.source_settings.meeting_domain

    points = [_xy(engine, predicted_position(engine, evaluation, w, engine.item.time_s+30)) for w in pair]
    target = tuple((a+b)/2 for a, b in zip(*points))
    choices = []
    for node in roads.nodes:
        position = roads.node_position(node.id)
        if domain.contains(position, roads):
            choices.append((math.dist(target, (node.x, node.y)), position_key(position), position))
    portions = {}
    if not domain.all_road_positions:
        for portion in domain.segment_portions:
            portions.setdefault(portion.segment_id, []).append((portion.start_m, portion.end_m))
    for segment in roads.segments:
        engine.check_budget()
        intervals = [(0., segment.length_m)] if domain.all_road_positions else portions.get(segment.id, ())
        offset = 0.
        for first, second in zip(segment.geometry, segment.geometry[1:]):
            length = math.dist(first, second)
            for lo, hi in intervals:
                start, end = max(lo, offset), min(hi, offset+length)
                if start > end or length == 0:
                    continue
                amount = sum((q-p)*(r-p) for q, p, r in zip(target, first, second))/length
                along = min(end, max(start, offset+amount))
                point = roads.segment_position(segment.id, along)
                choices.append((math.dist(target, roads.coordinates(point)), position_key(point), point))
            offset += length
    return min(choices)[2] if choices else None


def _cos(a, b):
    norm = math.hypot(*a)*math.hypot(*b)
    return sum(x*y for x, y in zip(a, b))/norm if norm else 0.


def nearest_next_routes(engine, plan, workers):


    routes = dict(plan.sequences)
    for worker_id in workers:
        worker = engine.item.worker_map[worker_id]
        route = routes[worker_id]
        end = max((i+1 for i, v in enumerate(route) if v.operation == 'transfer'), default=0)
        prefix, remaining = route[:end], route[end:]
        held = set(worker.carried_task_ids)
        for visit in prefix:
            if visit.operation == 'pickup':
                held.add(visit.task_id)
            elif visit.operation == 'delivery':
                held.discard(visit.task_id)
            else:
                event = plan.transfer_map[visit.transfer_id]
                outgoing, incoming = (event.tasks_i_to_j, event.tasks_j_to_i) if worker_id == event.worker_i else (event.tasks_j_to_i, event.tasks_i_to_j)
                held.difference_update(outgoing)
                held.update(incoming)
        pending = {}
        for visit in remaining:
            pending.setdefault(visit.task_id, []).append(visit)
        origin = engine.visit_location(plan, prefix[-1]) if prefix else worker.location
        tail = []
        while pending:
            engine.check_budget()
            options = []
            load = math.fsum(engine.item.task_map[t].load for t in held)
            for index, (task_id, visits) in enumerate(pending.items()):
                visit = visits[0]
                if visit.operation == 'delivery' and task_id not in held:
                    continue
                if visit.operation == 'pickup' and (task_id in held or _exceeds_capacity(load+engine.item.task_map[task_id].load, worker.capacity)):
                    continue
                target = engine.visit_location(plan, visit)
                cost = engine.routing.travel_time(origin, target, worker.profile)
                if math.isfinite(cost):
                    options.append((cost, index, task_id, visit, target))
            if not options:
                return None
            _, _, task_id, visit, origin = min(options)
            tail.append(visit)
            if visit.operation == 'pickup':
                held.add(task_id)
            else:
                held.remove(task_id)
            pending[task_id].pop(0)
            if not pending[task_id]:
                del pending[task_id]
        routes[worker_id] = prefix+tuple(tail)
    return replace(plan, sequences=routes)


def _vectors(engine, evaluation, worker):
    origin = _xy(engine, current_position(engine, worker))
    result = {}
    for visit in evaluation.plan.sequences[worker]:
        if visit.operation == 'transfer' or visit.task_id in result:
            continue
        target = _xy(engine, engine.visit_location(evaluation.plan, visit))
        result[visit.task_id] = tuple(b-a for a, b in zip(origin, target))
    return result


def _summary(engine, evaluation, worker):
    visits = evaluation.routes[worker]
    tasks = [v.visit.task_id for v in visits if v.visit.operation == 'delivery']
    now = engine.item.time_s
    next_stop = visits[0].position if visits else current_position(engine, worker)
    return dict(tasks=tasks, next=next_stop,
                stage=(visits[0].completion_time_s-now)/60 if visits else 0.,
                maximum=max((evaluation.completion_times_s[t]-now for t in tasks), default=0.)/60,
                slack=min((engine.item.task_map[t].promised_delivery_time_s-evaluation.completion_times_s[t] for t in tasks), default=0.)/60,
                vectors=_vectors(engine, evaluation, worker))


def high_state(engine, current, pair, history):
    summaries = [_summary(engine, current, w) for w in pair]
    now = engine.item.time_origin_s+engine.item.time_s
    features = [engine.item.time_s/3600]
    for worker in pair:
        features.extend(_ll(engine, current_position(engine, worker)))
    for info in summaries:
        features.extend(_ll(engine, info['next']))
    features.extend(info['stage'] for info in summaries)
    for index, info in enumerate(summaries):
        vectors = list(info['vectors'].values())
        other = list(summaries[1-index]['vectors'].values())
        own = [_cos(a, b) for i, a in enumerate(vectors) for j, b in enumerate(vectors) if i != j]
        cross = [_cos(a, b) for a in vectors for b in other]
        features.extend((info['maximum'], info['slack'], len(info['tasks']),
                         sum(own)/len(own) if own else 0., sum(cross)/len(cross) if cross else 0.))
    filled = []
    for index, time in enumerate(now+offset for offset in (-90, -60, -30, 0)):
        points = []
        for worker in pair:
            samples = {row['time_s']: row['xy_m'] for row in history['workers'][worker]['position_samples']}
            if time in samples:
                points.append(samples[time])
            else:

                points.append(_xy(engine, current_position(engine, worker)))
                filled.append(index)
        features.append(math.dist(*points)/1000)
    provenance = state_source_id(engine.item, history)
    return DecoState('high', tuple(features), now, provenance,
                     tuple(now+x for x in (-90, -60, -30, 0)), tuple(sorted(set(filled))))


def low_state(engine, current, pair, task_id, history):
    owner = owners(current.plan)[task_id]
    task = engine.item.task_map[task_id]
    stop = task.delivery if task_id in engine.holders else task.pickup
    vector = _vectors(engine, current, owner)[task_id]
    features = [*_ll(engine, stop), (current.completion_times_s[task_id]-engine.item.time_s)/60]
    for worker in pair:
        info = _summary(engine, current, worker)
        features.extend((*_ll(engine, info['next']), math.dist(_xy(engine, stop), _xy(engine, info['next']))/1000,
                         len(info['tasks']), sum(_cos(vector, v) > .8 for v in info['vectors'].values()), info['maximum']))
    provenance = state_source_id(engine.item, history)
    return DecoState('low', tuple(features), engine.item.time_origin_s+engine.item.time_s, provenance)


def policy_pass(engine, current, history, choose, radius=10., remaining=1200., count=6, on_transition=None):

    stats = engine.stats['deco']
    gates = stats.setdefault('filter_counts', {})
    def record_gate(name):
        gates[name] = gates.get(name, 0)+1
    workers = sorted(w.id for w in engine.item.workers if w.service_mode == 'delivery')
    initial_routes = nearest_next_routes(engine, current.plan, workers)
    if initial_routes is not None and initial_routes != current.plan:
        routed = engine.evaluate(initial_routes)
        if routed is not None:
            current = routed
            engine.commit_policy(current, source='deco_nearest_next_routes')
    hops = {task: sum(task in e.tasks_i_to_j or task in e.tasks_j_to_i for e in current.plan.transfers) for task in engine.all_tasks}
    positions = {worker: _xy(engine, current_position(engine, worker)) for worker in workers}
    assigned, owned = {}, {}
    def refresh_ownership():
        nonlocal assigned, owned
        assigned = owners(current.plan)
        owned = {worker: [] for worker in workers}
        for task in sorted(engine.all_tasks):
            owned[assigned[task]].append(task)
        stats['pass_ownership_rebuilds'] = stats.get('pass_ownership_rebuilds', 0)+1
        stats['pass_ownership_task_rows'] = stats.get('pass_ownership_task_rows', 0)+len(engine.all_tasks)
    refresh_ownership()
    stats['potential_worker_pairs'] = len(workers)*(len(workers)-1)//2
    for pair in itertools.combinations(workers, 2):
        record_gate('pairs_checked')
        engine.check_budget()
        if max(len(owned[worker]) for worker in pair) <= 1:
            record_gate('pair_order_gate')
            continue
        if math.dist(positions[pair[0]], positions[pair[1]]) >= radius:
            record_gate('distance_gate')
            continue
        task_groups = [list(owned[worker]) for worker in pair]
        record_gate('current_encounter_high_gate')
        before_high, high = current, high_state(engine, current, pair, history)
        decision = choose(high)
        high_action = decision['action']
        row = dict(pair=pair, high=decision, low=[])
        stats['decisions'].append(row)
        if high_action:
            location = midpoint_on_road(engine, current, pair)
            if location is None:
                record_gate('no_permitted_midpoint_projection')
                row['rejection'] = 'no_permitted_midpoint_projection'
            else:
                order = sorted(range(2), key=lambda i: (-len(task_groups[i]), i))
                for task_id in (t for index in order for t in task_groups[index]):
                    if hops[task_id] > 1:
                        record_gate('task_two_hop_gate')
                        continue
                    if current.completion_times_s[task_id]-engine.item.time_s <= remaining:
                        record_gate('task_20_min_gate')
                        continue
                    if assigned[task_id] not in pair or engine.holders.get(task_id, assigned[task_id]) not in pair:
                        record_gate('task_ownership_or_custody_gate')
                        continue
                    before_low, low = current, low_state(engine, current, pair, task_id, history)
                    decision = choose(low)
                    record_gate('low_action_queries')
                    receiver = pair[decision['action']]
                    details = dict(task_id=task_id, decision=decision, receiver=receiver)
                    row['low'].append(details)
                    if sum(w == receiver for w in assigned.values()) >= count:
                        record_gate('receiver_six_order_gate')
                        details['rejection'] = 'source_six_order_gate'
                        continue
                    if receiver != assigned[task_id]:
                        proposal = focal_move(engine, current.plan, task_id, receiver, location, prefix='deco', evaluate=False)
                        proposal = nearest_next_routes(engine, proposal, pair) if proposal is not None else None
                        candidate = engine.evaluate(proposal) if proposal is not None else None
                        if candidate is None:
                            record_gate('physical_validation')
                            details['rejection'] = 'physical_validation'
                        else:
                            record_gate('accepted_transfer')
                            details.update(before_tardiness_s=current.total_tardiness_s, after_tardiness_s=candidate.total_tardiness_s)
                            current = candidate
                            refresh_ownership()
                            hops[task_id] += 1
                            engine.commit_policy(current, source='deco')
                    else:
                        record_gate('same_owner_action')
                    record_gate('low_transition_eligible')
                    if on_transition is not None:
                        on_transition(low, decision['action'], before_low, current,
                                      low_state(engine, current, pair, task_id, history), task_id, pair)
        else:
            record_gate('high_declined')
        if on_transition is not None:
            on_transition(high, high_action, before_high, current, high_state(engine, current, pair, history), None, pair)
    return current


def solve(engine):
    parameters = dict(engine.config.parameters)
    required = {'checkpoint_path','history_path','python_executable'}
    allowed = required | {'encounter_radius_m','min_remaining_time_s','max_assigned_tasks','reward_mode','device'}
    if set(parameters)-allowed or not required <= set(parameters):
        raise ValueError('DECO requires checkpoint_path, history_path and python_executable')
    reward_mode = parameters.get('reward_mode','total_tardiness')
    if reward_mode not in REWARDS:
        raise ValueError('unsupported DECO reward mode')
    radius = parameters.get('encounter_radius_m',10.)
    remaining = parameters.get('min_remaining_time_s',1200.)
    count = parameters.get('max_assigned_tasks',6)
    if any(type(x)not in(int,float)or not math.isfinite(x)or x<0 for x in(radius,remaining)) or type(count)is not int or count<1:
        raise ValueError('invalid DECO encounter parameters')
    history = load_history(parameters['history_path'],engine.item,require='position')
    metadata = checkpoint_metadata(parameters['checkpoint_path'])
    if metadata['reward_mode'] != reward_mode:
        raise ValueError('DECO checkpoint and configured reward modes differ')
    executable = parameters['python_executable']
    if not isinstance(executable,str)or not executable or shutil.which(executable)is None:
        raise FileNotFoundError('DECO Python interpreter unavailable')
    stats = engine.stats['deco'] = dict(decisions=[],encounter_radius_m=radius,
        min_remaining_time_s=remaining,max_assigned_tasks=count,reward_mode=reward_mode,
        output_semantics='final_policy_state')
    current = initial_policy_plan(engine,'deco')
    if current is None or engine.config.iterations == 0:
        return current
    process = PolicyWorker(executable,parameters['checkpoint_path'],engine)
    try:
        stats['training'] = process.training
        engine.phase='policy_decision'
        current = policy_pass(engine,current,history,process.choose,radius,remaining,count)
        engine.stats['iterations']=engine.stats['iterations_started']=1
        return current
    finally:
        process.close()
