import math
import hashlib

class AdaptiveWeights:


    def __init__(self, names, reaction_factor=0.2, segment_size=20):
        self.weights = dict.fromkeys(names, 1.0)
        self.uses = dict.fromkeys(names, 0)
        self._scores = dict.fromkeys(names, 0.0)
        self._counts = dict.fromkeys(names, 0)
        self._reaction, self._segment = reaction_factor, segment_size
        self._iterations, self.updates = 0, 0

    def choose(self, rng):
        names = tuple(self.weights)
        weights = tuple(self.weights.values())
        if not any(weights):
            return rng.choice(names)
        return rng.choices(names, weights=weights, k=1)[0]

    def record(self, name, reward):
        self.uses[name] += 1
        self._counts[name] += 1
        self._scores[name] += reward

    def end_iteration(self):
        self._iterations += 1
        if self._iterations % self._segment:
            return False
        for name, weight in self.weights.items():
            if self._counts[name]:
                self.weights[name] = ((1 - self._reaction) * weight
                                      + self._reaction * self._scores[name] / self._counts[name])
            self._scores[name] = 0.0
            self._counts[name] = 0
        self.updates += 1
        return True

    def summary(self):
        return {"weights": dict(self.weights), "uses": dict(self.uses), "updates": self.updates}


def initial_temperature(seed, configured=None):

    if configured is not None:
        return configured
    return max(1.0, seed.total_tardiness_s * (0.05 / -math.log(0.5)))


def anneal_accept(current, candidate, temperature, rng):

    delta = candidate.total_tardiness_s - current.total_tardiness_s
    if delta <= 0:
        return True
    return temperature > 0 and rng.random() < math.exp(-delta / temperature)


def validate_parameters(parameters):
    names = {'removal_fraction','segment_size','reaction_factor','cooling_rate','initial_temperature'}
    if set(parameters) - names:
        raise ValueError('unknown local ALNS parameter')
    for key in ('removal_fraction','cooling_rate','reaction_factor'):
        if key in parameters and (type(parameters[key]) not in (int,float) or not math.isfinite(parameters[key]) or not 0 < parameters[key] <= 1):
            raise ValueError(key + ' must be in (0,1]')
    if 'segment_size' in parameters and (type(parameters['segment_size']) is not int or parameters['segment_size'] < 1):
        raise ValueError('segment_size must be positive')
    if parameters.get('initial_temperature') is not None and parameters['initial_temperature'] <= 0:
        raise ValueError('initial_temperature must be positive')
    return dict(parameters)


def novelty_reward(candidate,current,best,accepted,novel):
    if not novel:
        return 0.
    if candidate < best:
        return 33.
    if accepted and candidate < current:
        return 9.
    if accepted and candidate > current:
        return 13.
    return 0.


def accepted_key(engine,plan):
    return hashlib.sha256(repr(engine.solution_key(plan)).encode()).hexdigest()


def local_removal(engine, current, count, name):


    ids = sorted(current.tardiness_s, key=lambda t: (-current.tardiness_s[t], t))
    count = min(8, count, len(ids))
    if not count:
        return frozenset()
    if name == 'random':
        return frozenset(engine.rng.sample(ids, count))
    pool = ids[:max(count, 8)]
    if name in ('related', 'shaw'):
        anchor = engine.rng.choice(pool)
        a = engine.task_features(anchor)
        pool.sort(key=lambda t: (math.dist(engine.task_features(t)[:2], a[:2]) +
            math.dist(engine.task_features(t)[2:4], a[2:4]), t))
    elif name == 'worst':
        ranked = []
        for task in pool:
            stripped = engine.strip(current.plan, (task,))
            partial = engine.evaluate(stripped, engine.all_tasks - {task})
            if partial is not None:
                ranked.append((current.total_tardiness_s - partial.total_tardiness_s, task))
        pool = [t for _, t in sorted(ranked, key=lambda x: (-x[0], x[1]))] or pool
    return frozenset(pool[:count])


def solve(engine):
    from common.recovery.adapters._local_search import search_frame
    from solvers.parameters import load_algorithm_config
    p = {**load_algorithm_config(None,'alns')['parameters'], **engine.config.parameters}
    p.setdefault('initial_temperature',None)
    validate_parameters(p)
    engine.stats['method_parameters'] = dict(p)
    with search_frame(engine, 'alns') as frame:
        if frame.done:
            return frame.result
        state = frame.data
        if not state:
            current = engine.construct_local(rule='best')
            if current is None:
                return frame.finish(None)
            state.update(current=current, iteration=0, phase='start',
                accepted_keys={accepted_key(engine,current.plan)},
                temperature=initial_temperature(current, p['initial_temperature']),
                removals=dict(weights=dict.fromkeys(('random','related','worst'), 1.),
                    scores=dict.fromkeys(('random','related','worst'), 0.), counts=dict.fromkeys(('random','related','worst'), 0)),
                insertions=dict(weights=dict.fromkeys(('greedy','regret'), 1.),
                    scores=dict.fromkeys(('greedy','regret'), 0.), counts=dict.fromkeys(('greedy','regret'), 0)))
            engine.stats['alns'] = dict(accepted=0, rejected=0, incomplete_repairs=0, removal_max=8,
                operator_weight_updates=0,weight_updates_used=0,
                initial_temperature_s=state['temperature'],temperature_s=state['temperature'])
        while state['iteration'] < engine.config.iterations:
            engine.check_budget()
            current = state['current']
            if current.total_tardiness_s == 0:
                engine.stats['local_stop_reason'] = 'zero_tardiness'
                break
            if state['phase'] == 'start':
                engine.stats['iterations_started'] += 1
                engine.stats['alns']['weight_updates_used'] = state['iteration']//p['segment_size']
                state['remove'] = engine.rng.choices(tuple(state['removals']['weights']),
                    weights=tuple(state['removals']['weights'].values()))[0]
                state['insert'] = engine.rng.choices(tuple(state['insertions']['weights']),
                    weights=tuple(state['insertions']['weights'].values()))[0]
                q = min(8,len(engine.all_tasks),max(1,math.ceil(p['removal_fraction']*len(engine.all_tasks))))
                state['removed'] = local_removal(engine,current,q,state['remove'])
                state['phase'] = 'repair'
            remove,insert,removed = state['remove'],state['insert'],state['removed']
            proposal, bank = engine.repair(engine.strip(current.plan, removed), removed,
                rule=insert, regret_k=2)
            candidate = None if bank else engine.evaluate_selected(proposal)
            reward = 0.
            if candidate is None:
                engine.stats['alns']['incomplete_repairs'] += 1
            else:
                previous_best = engine.best.total_tardiness_s
                engine.remember(candidate, source='alns_local_repair')
                accepted = anneal_accept(current, candidate, state['temperature'], engine.rng)
                if accepted:
                    state['current'] = candidate
                    engine.stats['alns']['accepted'] += 1
                key=accepted_key(engine,candidate.plan)
                reward=novelty_reward(candidate.total_tardiness_s,current.total_tardiness_s,
                    previous_best,accepted,key not in state['accepted_keys'])
                if accepted:
                    state['accepted_keys'].add(key)
                else:
                    engine.stats['alns']['rejected'] += 1
            for section, choice in (('removals',remove),('insertions',insert)):
                d = state[section]
                d['scores'][choice] += reward
                d['counts'][choice] += 1
                if (state['iteration']+1) % p['segment_size'] == 0:
                    for operator in d['weights']:
                        if d['counts'][operator]:
                            d['weights'][operator] = (1-p['reaction_factor'])*d['weights'][operator] + p['reaction_factor']*d['scores'][operator]/d['counts'][operator]
                        d['scores'][operator], d['counts'][operator] = 0., 0
            state['temperature'] *= p['cooling_rate']
            engine.stats['alns']['temperature_s'] = state['temperature']
            if (state['iteration']+1) % p['segment_size'] == 0:
                engine.stats['alns']['operator_weight_updates'] += 1
            state['iteration'] += 1
            state['phase'] = 'start'
            engine.stats['iterations'] = state['iteration']
            engine.stats['alns']['operator_weights'] = {k: dict(state[k]['weights']) for k in ('removals','insertions')}
        return frame.finish(engine.best)
