from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import random
import time
import torch
from common.config import TRAIL_DEFAULTS
from solvers.tfr.local_solver import LocalConfig, LocalEnvironment, NORMAL
from solvers.tfr.interfaces.revisions import DecodeFailure, MacroAction, ProposalSlot
from .decoder import graph_observation
from .environment import StructureEnvironment, geometry_turn, settle_environment
from .network import StructurePolicy
from .solver import solve_policy

@dataclass(frozen=True)
class TrainingConfig:
    device: str = TRAIL_DEFAULTS['device']
    bc_limit: int = TRAIL_DEFAULTS['bc_limit']
    bc_epochs: int = TRAIL_DEFAULTS['bc_epochs']
    batch_size: int = TRAIL_DEFAULTS['batch_size']
    bc_learning_rate: float = TRAIL_DEFAULTS['bc_learning_rate']
    transitions: int = TRAIL_DEFAULTS['transitions']
    startup_limit: int = TRAIL_DEFAULTS['startup_limit']
    rollout_size: int = TRAIL_DEFAULTS['rollout_size']
    ppo_epochs: int = TRAIL_DEFAULTS['ppo_epochs']
    ppo_batch_size: int = TRAIL_DEFAULTS['ppo_batch_size']
    learning_rate: float = TRAIL_DEFAULTS['learning_rate']
    clip: float = TRAIL_DEFAULTS['clip']
    gae_lambda: float = TRAIL_DEFAULTS['gae_lambda']
    value_coefficient: float = TRAIL_DEFAULTS['value_coefficient']
    entropy_coefficient: float = TRAIL_DEFAULTS['entropy_coefficient']
    max_gradient_norm: float = TRAIL_DEFAULTS['max_gradient_norm']
    target_kl: float = TRAIL_DEFAULTS['target_kl']
    anchor_coefficient: float = TRAIL_DEFAULTS['anchor_coefficient']

    def __post_init__(self):
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')
        for name in ('bc_limit', 'bc_epochs', 'batch_size', 'transitions', 'startup_limit', 'rollout_size', 'ppo_epochs', 'ppo_batch_size'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + ' must be a positive integer')
        if not 0 < self.clip < 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError('invalid PPO clip or GAE lambda')
        for name in ('bc_learning_rate', 'learning_rate', 'value_coefficient', 'entropy_coefficient', 'max_gradient_norm', 'target_kl', 'anchor_coefficient'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(name + ' must be finite and nonnegative')
        if min(self.bc_learning_rate, self.learning_rate, self.max_gradient_norm, self.target_kl) <= 0:
            raise ValueError('learning rates, gradient norm and target KL must be positive')

@dataclass
class Transition:
    macro: object
    reward: float
    done: bool
    next_value: float
    trajectory_id: object = None

def advantages(transitions, gae_lambda=0.95):
    (values, following) = ([], {})
    for transition in reversed(transitions):
        stream = transition.trajectory_id
        delta = transition.reward + (0.0 if transition.done else transition.next_value) - transition.macro.value
        advantage = delta + (0.0 if transition.done else gae_lambda * following.get(stream, 0.0))
        following[stream] = advantage
        values.append(advantage)
    values.reverse()
    returns = [a + t.macro.value for (a, t) in zip(values, transitions)]
    tensor = torch.tensor(values, dtype=torch.float32)
    if len(tensor) > 1:
        tensor = (tensor - tensor.mean()) / (tensor.std(unbiased=False) + 1e-08)
    return (tensor, torch.tensor(returns, dtype=torch.float32))

def _input(sample):
    if hasattr(sample, 'projected'):
        return (sample.projected, sample.routing)
    if isinstance(sample, dict):
        return (sample['item'], sample['routing'])
    return sample

def _teacher_targets(sample, config):
    (item, routing) = _input(sample)
    env = LocalEnvironment(item, routing, config=config, training=True)
    obs = env.reset()
    (sources, emitted, targets) = ({}, set(), [])
    publish = env.geometry_pool.publish

    def capture(value):
        previous = None if env.best is None else env.best.total_tardiness_s
        context = env.geometry_pool.candidate_context or {}
        key = context.get('entry_key') or env.geometry_pool.key(value.plan)
        if publish is not None:
            publish(value)
        if key not in sources or previous is None or value.total_tardiness_s >= previous:
            return
        (issued, group) = sources[key]
        identity = (issued.source_witness_digest, key)
        if identity not in emitted and value.total_tardiness_s < min(issued.evaluated.total_tardiness_s, issued.best_T):
            emitted.add(identity)
            targets.append((issued, value.plan, group))
    env.geometry_pool.publish = capture
    try:
        while obs is not None:
            if settle_environment(env):
                yield from targets
                env.finalize()
                return
            if env.status is not None:
                raise RuntimeError('H teacher search stopped: ' + str(env.status))
            obs = env.begin_decision()
            built = env.build_rules()
            if built['status'] != 'returned':
                raise RuntimeError('H teacher reconstruction did not finish: ' + built['status'])
            proposals = built['result']['proposals']
            for proposal in proposals:
                if proposal is not None:
                    sources[env.geometry_pool.key(proposal.plan)] = (obs, tuple(proposal.group))
            slots = tuple((ProposalSlot(None, (), 'construction_failed') if proposal is None else ProposalSlot(None, (), 'candidate', structure=proposal.plan) for proposal in proposals))
            action = MacroAction(obs.source_witness_digest, slots, search_radii=built['result'].get('search_radii', {}), expand_tasks=tuple(built['result'].get('expand_tasks', ())))
            (_, _, _, info) = env.step(action)
            if not info.get('closed'):
                raise RuntimeError('H teacher action did not close')
            if env.geometry_pool:
                geometry_turn(env)
            done = settle_environment(env)
            yield from targets
            targets.clear()
            if done:
                env.finalize()
                return
            obs = env.observation()
    finally:
        env.geometry_pool.publish = publish

def _episode(sample, config, policy, *, source=0):
    (item, routing) = _input(sample)
    env = StructureEnvironment(item, routing, config=config, training=True)
    obs = env.reset()
    while obs is not None:
        if settle_environment(env):
            env.finalize()
            return
        if env.status is not None:
            raise RuntimeError('Structure training search stopped: ' + str(env.status))
        obs = env.begin_decision()
        macro = policy.generate(obs, deterministic=False)
        (next_obs, reward, done, info) = env.step_policy(macro)
        if not info.get('closed'):
            raise RuntimeError('Structure training action did not close')
        if env.status is not None and env.status not in NORMAL:
            raise RuntimeError('Structure training search stopped: ' + str(env.status))
        next_value = 0.0
        if not done:
            with torch.no_grad():
                next_value = float(policy.encode(graph_observation(next_obs))[2])
        yield (source, macro, reward, done, next_value, reward > 0)
        if done:
            env.finalize()
            return
        obs = next_obs

def bc_update(policy, examples, config, rng):
    grouped = defaultdict(list)
    for (source, macro) in examples:
        grouped[source].append(macro)
    policy.fit_normalization([macro for (_, macro) in examples])
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.bc_learning_rate)
    losses = []
    for _ in range(config.bc_epochs):
        order = list(examples)
        rng.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            terms = [-policy.evaluate_macro(macro)[0] for (source, macro) in order[start:start + config.batch_size]]
            loss = torch.stack(terms).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite complete-structure imitation loss')
            optimizer.zero_grad()
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_gradient_norm)
            if not torch.isfinite(gradient):
                raise FloatingPointError('Nonfinite complete-structure imitation gradient')
            optimizer.step()
            losses.append(float(loss.detach()))
    return (grouped, losses)

def ppo_update(policy, optimizer, transitions, *, config, rng, anchors):
    if not transitions:
        return dict(updates=0, samples=0)
    (advantage, returns) = advantages(transitions, config.gae_lambda)
    (updates, losses) = (0, [])
    for _ in range(config.ppo_epochs):
        order = list(range(len(transitions)))
        rng.shuffle(order)
        for start in range(0, len(order), config.ppo_batch_size):
            indices = order[start:start + config.ppo_batch_size]
            evaluated = [policy.evaluate_macro(transitions[i].macro) for i in indices]
            (logp, value, entropy) = [torch.stack([row[k] for row in evaluated]) for k in range(3)]
            old = torch.tensor([transitions[i].macro.old_log_probability for i in indices], device=policy.device)
            logratio = logp - old
            ratio = logratio.exp()
            if not torch.isfinite(ratio).all() or float((ratio - 1 - logratio).mean().detach()) > config.target_kl:
                policy.update_step += 1
                return dict(updates=updates, samples=len(transitions), kl_stopped=True)
            adv = advantage[indices].to(policy.device)
            loss = -torch.minimum(ratio * adv, ratio.clamp(1 - config.clip, 1 + config.clip) * adv).mean()
            loss += config.value_coefficient * (value - returns[indices].to(policy.device)).square().mean()
            loss -= config.entropy_coefficient * entropy.mean()
            if config.anchor_coefficient:
                if not anchors:
                    raise ValueError('Complete-structure PPO requires H imitation anchors')
                chosen = [rng.choice(anchors[source]) for source in sorted(anchors) for _ in range(2)]
                loss += config.anchor_coefficient * torch.stack([-policy.evaluate_macro(m)[0] for m in chosen]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite complete-structure PPO loss')
            optimizer.zero_grad()
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(policy.parameters(), config.max_gradient_norm)
            if not torch.isfinite(gradient):
                raise FloatingPointError('Nonfinite complete-structure PPO gradient')
            optimizer.step()
            updates += 1
            losses.append(float(loss.detach()))
    policy.update_step += 1
    return dict(updates=updates, samples=len(transitions), loss=sum(losses) / max(1, len(losses)))

def train(samples, output, *, validation_samples=(), config=None, h_config=None, seed=0):
    (started, cpu_started) = (time.perf_counter(), time.process_time())
    config = config or TrainingConfig()
    if config.device == 'cuda' and (not torch.cuda.is_available()):
        raise ValueError('CUDA was requested but is unavailable')
    samples = list(samples)
    validation_samples = list(validation_samples)
    if not samples:
        raise ValueError('At least one training input is required')
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    h_config = h_config or LocalConfig(seed=seed)
    model = StructurePolicy(proposal_slots=h_config.proposal_slots).to(config.device)
    (examples, replay_failures) = ([], defaultdict(int))
    for (source, sample) in enumerate(samples):
        for (obs, target, group) in _teacher_targets(sample, h_config):
            try:
                macro = model.demonstration(obs, target, group)
            except (DecodeFailure, ValueError) as error:
                raise ValueError(
                    f'H teacher replay failed for training input {source} '
                    f'({obs.item.instance_id}), group {tuple(group)}: {error}'
                ) from error
            examples.append((source, macro))
            if len(examples) >= config.bc_limit:
                break
        if len(examples) >= config.bc_limit:
            break
    if not examples:
        raise ValueError('H produced no replayable strict improving full-structure demonstrations: ' + str(dict(replay_failures)))
    (anchors, bc_losses) = bc_update(model, examples, config, rng)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    (transitions, buffer, updates) = (0, [], [])
    output = Path(output)
    checkpoint = output if output.suffix in ('.pt', '.pth') else output / 'selected.pt'
    best_score = math.inf
    selected_this_run = False

    def select():
        nonlocal best_score, selected_this_run
        score = 0.0
        for sample in validation_samples:
            (item, routing) = _input(sample)
            result = solve_policy(item, routing, model, h_config)
            if result.plan is None or result.status not in NORMAL:
                return
            boundary = getattr(item, 'execution_boundary', None)
            fixed = 0.0 if boundary is None else boundary.observed_tardiness_s + boundary.committed_tardiness_upper_s
            score += result.plan.total_tardiness_s + fixed
        if not validation_samples or score < best_score:
            best_score = score
            model.save(checkpoint)
            selected_this_run = True
    select()
    for episode in range(config.startup_limit):
        source = episode % len(samples)
        for (_, macro, reward, done, next_value, _) in _episode(samples[source], h_config, model, source=source):
            buffer.append(Transition(macro, reward, done, next_value, episode))
            transitions += 1
            if len(buffer) >= config.rollout_size or transitions >= config.transitions:
                updates.append(ppo_update(model, optimizer, buffer, config=config, rng=rng, anchors=anchors))
                buffer.clear()
                select()
            if transitions >= config.transitions:
                break
        if transitions >= config.transitions:
            break
    if buffer:
        updates.append(ppo_update(model, optimizer, buffer, config=config, rng=rng, anchors=anchors))
        select()
    if not selected_this_run or not checkpoint.is_file():
        raise ValueError('No complete validated model was selected in this training run')
    return dict(model=str(checkpoint), teacher_methods=['h'], teacher_decisions=len(examples), teacher_replay_failures=dict(replay_failures), bc_updates=len(bc_losses), transitions=transitions, ppo_updates=sum((row['updates'] for row in updates)), training_complete=transitions >= config.transitions, terminal_status='complete' if transitions >= config.transitions else 'startup_limit_reached', validation_T_s=None if not validation_samples else best_score, training_wall_s=time.perf_counter() - started, training_cpu_s=time.process_time() - cpu_started)
