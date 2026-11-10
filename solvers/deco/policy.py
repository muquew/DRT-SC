from dataclasses import dataclass
import json
import math
from pathlib import Path
HIGH_FEATURES = ('encounter_time_hours', 'courier1_lat', 'courier1_lon', 'courier2_lat', 'courier2_lon', 'courier1_next_lat', 'courier1_next_lon', 'courier2_next_lat', 'courier2_next_lon', 'courier1_current_stage_remaining_min', 'courier2_current_stage_remaining_min', 'courier1_max_order_remaining_min', 'courier1_min_promise_variance_remaining_min', 'courier1_order_count', 'courier1_self_similarity', 'courier1_cross_similarity', 'courier2_max_order_remaining_min', 'courier2_min_promise_variance_remaining_min', 'courier2_order_count', 'courier2_self_similarity', 'courier2_cross_similarity', 'distance_t_minus_90_km', 'distance_t_minus_60_km', 'distance_t_minus_30_km', 'distance_t_km')
LOW_FEATURES = ('order_next_lat', 'order_next_lon', 'order_owner_remaining_min', 'courier1_next_lat', 'courier1_next_lon', 'courier1_order_next_distance_km', 'courier1_order_count', 'courier1_same_direction_count', 'courier1_all_orders_remaining_min', 'courier2_next_lat', 'courier2_next_lon', 'courier2_order_next_distance_km', 'courier2_order_count', 'courier2_same_direction_count', 'courier2_all_orders_remaining_min')
FEATURES = {'high': HIGH_FEATURES, 'low': LOW_FEATURES}
REWARDS = {'total_tardiness': '(1 + full_remaining_total_tardiness_before)/(1 + full_remaining_total_tardiness_after); both hierarchy levels', 'source_proxy': '0.1 delivery-time ratio + 0.9 revenue ratio; focal order at low level, pair aggregates at high level'}

def _finite(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(value)

@dataclass(frozen=True)
class DecoState:
    level: str
    values: tuple
    observed_at_s: float
    source_id: str
    history_times_s: tuple = ()
    history_filled_indices: tuple = ()

    def __post_init__(self):
        if self.level not in FEATURES:
            raise ValueError('level must be high or low')
        values = tuple((_finite(x, 'state value') for x in self.values))
        if len(values) != len(FEATURES[self.level]):
            raise ValueError(f'{self.level} state requires {len(FEATURES[self.level])} explicit features')
        now = _finite(self.observed_at_s, 'observation time')
        times = tuple((_finite(x, 'history time') for x in self.history_times_s))
        filled = tuple(self.history_filled_indices)
        if self.level == 'high':
            if times != tuple((now + offset for offset in (-90, -60, -30, 0))):
                raise ValueError('high state must explicitly declare T-90/T-60/T-30/T history')
            if any((x < 0 for x in values[-4:])):
                raise ValueError('historical distances must be nonnegative kilometers')
            if len(set(filled)) != len(filled) or any((type(x) is not int or not 0 <= x < 4 for x in filled)):
                raise ValueError('filled history indices must be distinct integers in 0..3')
        elif times or filled:
            raise ValueError('low state has no four-distance history block')
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'observed_at_s', now)
        object.__setattr__(self, 'history_times_s', times)
        object.__setattr__(self, 'history_filled_indices', filled)

    def to_dict(self):
        return dict(level=self.level, values=list(self.values), observed_at_s=self.observed_at_s, source_id=self.source_id, history_times_s=list(self.history_times_s), history_filled_indices=list(self.history_filled_indices))

def state_from_features(level, feature_values, **metadata):

    if level not in FEATURES or set(feature_values) != set(FEATURES[level]):
        raise ValueError('all and only the named source features are required')
    return DecoState(level, tuple((feature_values[name] for name in FEATURES[level])), **metadata)

def state_source_id(item, history):

    return item.instance_id

def checkpoint_metadata(directory):
    path = Path(directory)
    metadata = json.loads((path / 'metadata.json').read_text())
    if metadata.get('features') != {level: list(names) for level, names in FEATURES.items()}:
        raise ValueError('DECO checkpoint feature dimensions differ')
    mode = metadata['reward_mode']
    if mode not in REWARDS:
        raise ValueError('unsupported DECO reward mode')
    for name in ('model.index', 'model.data-00000-of-00001'):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    return dict(metadata, reward_mode=mode, reward_formula=REWARDS[mode])

def history_distances(time_s, courier1, courier2, *, current_coordinates=None):


    now = _finite(time_s, 'time_s')
    times = tuple((now + offset for offset in (-90, -60, -30, 0)))
    distances, filled = ([], [])

    def coordinate(value):
        if len(value) != 2:
            raise ValueError('coordinates must be latitude/longitude pairs')
        lat, lon = (_finite(x, 'coordinate') for x in value)
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError('source history uses geographical latitude/longitude')
        return (lat, lon)
    if current_coordinates is not None:
        if len(current_coordinates) != 2:
            raise ValueError('current coordinates require both couriers')
        current_coordinates = tuple((coordinate(x) for x in current_coordinates))
    for index, time in enumerate(times):
        points, padded = ([], False)
        for worker, history in enumerate((courier1, courier2)):
            if time in history:
                points.append(coordinate(history[time]))
            elif current_coordinates is not None:
                points.append(current_coordinates[worker])
                padded = True
            else:
                raise ValueError(f'missing courier {worker + 1} history at {time}')
        (lat0, lon0), (lat1, lon1) = [tuple(map(math.radians, point)) for point in points]
        h = math.sin((lat0 - lat1) / 2) ** 2 + math.cos(lat0) * math.cos(lat1) * math.sin((lon0 - lon1) / 2) ** 2
        distances.append(2 * 6371 * math.asin(math.sqrt(min(1.0, h))))
        if padded:
            filled.append(index)
    return dict(distances_km=tuple(distances), history_times_s=times, history_filled_indices=tuple(filled))

class DecoComponent:


    def __init__(self, *, seed=2, tensorflow_seed=1, reward_mode='total_tardiness', device='cpu'):
        if type(seed) is not int or type(tensorflow_seed) is not int:
            raise ValueError('seeds must be integers')
        if reward_mode not in REWARDS:
            raise ValueError('reward_mode must be total_tardiness or source_proxy')
        self.reward_mode = reward_mode
        import numpy as np
        import solvers.deco.network as network
        self.np, self.network = (np, network)
        self.rng = np.random.RandomState(seed)
        self.seed, self.tensorflow_seed = (seed, tensorflow_seed)
        session_config, self.device = network.device_config(device)
        self.graph = network.tf.Graph()
        with self.graph.as_default():
            network.tf.set_random_seed(tensorflow_seed)
            self.session = network.tf.Session(graph=self.graph, config=session_config)
            self.device['session_devices'] = [dict(name=d.name, type=d.device_type) for d in self.session.list_devices()]
            if self.device['selected'] == 'gpu' and (not any((d['type'] == 'GPU' for d in self.device['session_devices']))):
                self.session.close()
                if device == 'gpu':
                    raise ValueError('TensorFlow enumerated a GPU but Session could not initialize it')
                session_config, self.device = network.device_config('cpu')
                self.device.update(requested='auto', fallback_reason='Session could not initialize an enumerated GPU')
                self.session = network.tf.Session(graph=self.graph, config=session_config)
                self.device['session_devices'] = [dict(name=d.name, type=d.device_type) for d in self.session.list_devices()]
            self.actors = {'high': network.High_Level_Network(self.session, 25, 2, 0.0001), 'low': network.Actor(self.session, 15, 2, 0.0001)}
            self.critics = {'high': network.High_Level_Critic(self.session, 25, 0.0001), 'low': network.Critic(self.session, 15, 0.0001)}
            self.saver = network.tf.train.Saver()
            self.session.run(network.tf.global_variables_initializer())
            if self.device['selected'] == 'gpu':
                with self.graph.device('/GPU:0'):
                    probe_input = network.tf.placeholder(network.tf.float32, (2, 2))
                    probe = network.tf.matmul(probe_input, probe_input, name='deco_device_probe')
                metadata = network.tf.RunMetadata()
                self.session.run(probe, {probe_input: np.ones((2, 2), dtype=np.float32)}, options=network.tf.RunOptions(trace_level=network.tf.RunOptions.FULL_TRACE), run_metadata=metadata)
                self.device['gpu_probe_devices'] = [d.device for d in metadata.step_stats.dev_stats if 'GPU' in d.device.upper() and any(('deco_device_probe' in n.node_name for n in d.node_stats))]
                if not self.device['gpu_probe_devices']:
                    self.session.close()
                    raise ValueError('DECO GPU matmul did not execute on a GPU')
        self.updates = {'high': 0, 'low': 0}
        self.padded_training_states = 0
        self.offline_replay_updates = 0
        self.restored = False

    def close(self):
        self.session.close()

    def train_transition(self, state, action, reward, next_state, *, outcome_observed_at_s, offline_replay=False, execution_metadata=None):


        if self.restored:
            raise ValueError('checkpoint loading is inference-only; replay is not silently reset for resumed training')
        if not isinstance(state, DecoState) or not isinstance(next_state, DecoState) or state.level != next_state.level:
            raise ValueError('transition requires two explicit states of the same level')
        if type(action) is not int or action not in (0, 1):
            raise ValueError('source action must be integer 0 or 1')
        reward = _finite(reward, 'observed reward')
        if reward <= 0:
            raise ValueError('the declared positive-ratio reward must be positive')
        completed = _finite(outcome_observed_at_s, 'outcome observation time')
        if not state.observed_at_s <= next_state.observed_at_s <= completed:
            raise ValueError('transition observations must precede the observed outcome')
        arrays = self.np.asarray([state.values, next_state.values], dtype=self.np.float32)
        if not self.np.isfinite(arrays).all() or abs(reward) > self.np.finfo(self.np.float32).max:
            raise ValueError('state/reward exceeds finite float32 range')
        actor, critic = (self.actors[state.level], self.critics[state.level])
        critic.store_transition(arrays[0], action, reward, arrays[1])
        indices = self.rng.choice(min(critic.memory_counter, critic.memory_size), size=critic.batch_size)
        batch = critic.memory[indices, :]
        n = critic.n_features
        critic_metadata = None if execution_metadata is None else execution_metadata.setdefault('critic', self.network.tf.RunMetadata())
        actor_metadata = None if execution_metadata is None else execution_metadata.setdefault('actor', self.network.tf.RunMetadata())
        td = critic.learn(batch[:, :n], batch[:, n + 1], batch[:, -n:], run_metadata=critic_metadata)
        actor_loss = actor.learn(batch[:, :n], batch[:, n].astype(self.np.int32), td, run_metadata=actor_metadata)
        if not math.isfinite(float(td)) or not math.isfinite(float(actor_loss)):
            raise FloatingPointError('source network update produced a nonfinite loss')
        self.updates[state.level] += 1
        self.offline_replay_updates += bool(offline_replay)
        self.padded_training_states += bool(state.history_filled_indices) + bool(next_state.history_filled_indices)
        receipt = dict(level=state.level, reward=reward, td_error=float(td), actor_loss=float(actor_loss), critic_loss=0.5 * float(td) ** 2, replay_size=min(critic.memory_counter, critic.memory_size), sampled_batch_size=256)
        return receipt

    def _require_trained(self):
        if any((self.updates[level] < 1 for level in ('high', 'low'))):
            raise ValueError('trained weights for both source policy levels are required')

    def probabilities(self, state):
        self._require_trained()
        if not isinstance(state, DecoState):
            raise TypeError('an explicit DecoState is required')
        return self._probabilities(state)

    def _probabilities(self, state):
        values = self.np.asarray([state.values] * 256, dtype=self.np.float32)
        if not self.np.isfinite(values).all():
            raise ValueError('query state exceeds finite float32 range')
        actor = self.actors[state.level]
        probabilities = self.session.run(actor.all_act_prob, {actor.s: values})[0]
        if not self.np.isfinite(probabilities).all() or not self.np.isclose(probabilities.sum(), 1):
            raise FloatingPointError('source actor returned invalid probabilities')
        return tuple((float(x) for x in probabilities))

    def training_action(self, state):


        if self.restored or not isinstance(state, DecoState):
            raise ValueError('training action requires a fresh training component and explicit state')
        probabilities = self._probabilities(state)
        return dict(level=state.level, action=int(self.rng.choice(2, p=self.np.asarray(probabilities, dtype=self.np.float32))), probabilities=probabilities, role='training simulator action')

    def choose_action(self, state):
        probabilities = self.probabilities(state)
        action = int(self.rng.choice(2, p=self.np.asarray(probabilities, dtype=self.np.float32)))
        return dict(level=state.level, action=action, probabilities=probabilities, meaning=('consider_exchange' if action else 'do_not_exchange') if state.level == 'high' else f'assign_to_encounter_courier_{action}', role='source policy choice; not a validated recovery plan', history_filled_indices=state.history_filled_indices)

    def metadata(self):
        return dict(device=dict(self.device), seed=self.seed, tensorflow_seed=self.tensorflow_seed,
            features={level: list(names) for level, names in FEATURES.items()},
            batch_size=256, replay_capacity=100000, gamma=0.9, learning_rate=0.0001,
            training_updates=dict(self.updates), padded_training_states=self.padded_training_states,
            offline_replay_updates=self.offline_replay_updates, reward_mode=self.reward_mode,
            reward_formula=REWARDS[self.reward_mode])

    def save(self, directory):
        self._require_trained()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.saver.save(self.session, str(path / 'model'))
        (path / 'metadata.json').write_text(json.dumps(self.metadata(), indent=2, allow_nan=False) + '\n')

    @classmethod
    def load(cls, directory, *, device='cpu'):
        path = Path(directory)
        metadata = checkpoint_metadata(path)
        reward_mode = metadata['reward_mode']
        instance = cls(seed=metadata['seed'], tensorflow_seed=metadata['tensorflow_seed'], reward_mode=reward_mode, device=device)
        try:
            instance.saver.restore(instance.session, str(path / 'model'))
        except BaseException:
            instance.close()
            raise
        instance.updates = dict(metadata['training_updates'])
        instance.padded_training_states = metadata['padded_training_states']
        instance.offline_replay_updates = metadata['offline_replay_updates']
        instance.restored = True
        return instance

    def save_training(self, directory):


        if self.restored:
            raise ValueError('inference-only loads cannot export a training continuation')
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.saver.save(self.session, str(path / 'model'))
        metadata = self.metadata()
        metadata['checkpoint_scope'] = 'weights/Adam + FIFO replay + NumPy RNG; TF dropout stream restarts on resume'
        (path / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        arrays = {level: critic.memory[:min(critic.memory_counter, critic.memory_size)] for level, critic in self.critics.items()}
        self.np.savez_compressed(path / 'replay.npz', **arrays)
        rng = self.rng.get_state()
        state = dict(memory_counters={k: c.memory_counter for k, c in self.critics.items()}, rng=[rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]], resume_rng_scope='NumPy exact; TF1 stateful dropout stream restarts')
        (path / 'training_state.json').write_text(json.dumps(state, allow_nan=False) + '\n')

    @classmethod
    def load_training(cls, directory, *, device='cpu'):

        path = Path(directory)
        metadata = json.loads((path / 'metadata.json').read_text())
        state = json.loads((path / 'training_state.json').read_text())
        if metadata.get('features') != {k: list(v) for k, v in FEATURES.items()}:
            raise ValueError('DECO training feature dimensions differ')
        if not isinstance(state.get('memory_counters'), dict) or not isinstance(state.get('rng'), list):
            raise ValueError('DECO training requires replay counters and RNG state')
        if metadata.get('reward_mode') not in REWARDS:
            raise ValueError('unsupported DECO reward mode')
        instance = cls(seed=metadata['seed'], tensorflow_seed=metadata['tensorflow_seed'], reward_mode=metadata['reward_mode'], device=device)
        try:
            instance.saver.restore(instance.session, str(path / 'model'))
            with instance.np.load(path / 'replay.npz', allow_pickle=False) as replay:
                for level, critic in instance.critics.items():
                    counter = state['memory_counters'][level]
                    rows = replay[level]
                    if type(counter) is not int or counter < 0 or rows.shape != (min(counter, critic.memory_size), critic.n_features * 2 + 2) or (not instance.np.isfinite(rows).all()) or (metadata['training_updates'][level] != counter):
                        raise ValueError('invalid replay counter/shape/content')
                    critic.memory[:len(rows)] = rows
                    critic.memory_counter = counter
            rng = state['rng']
            instance.rng.set_state((rng[0], instance.np.array(rng[1], dtype=instance.np.uint32), *rng[2:]))
            instance.updates = dict(metadata['training_updates'])
            instance.padded_training_states = metadata['padded_training_states']
            instance.offline_replay_updates = metadata['offline_replay_updates']
            return instance
        except BaseException:
            instance.close()
            raise
