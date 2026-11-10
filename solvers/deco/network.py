import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
GAMMA = .9
BATCH_SIZE = 256


def device_config(requested='cpu'):

    if requested not in ('cpu', 'auto', 'gpu'):
        raise ValueError('DECO device must be cpu, auto or gpu')
    visible = tf.config.list_physical_devices('GPU') if requested != 'cpu' else []
    if requested == 'gpu' and not visible:
        raise ValueError('DECO device=gpu requested but TensorFlow has no usable GPU')
    selected = 'gpu' if visible else 'cpu'
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
                            device_count={'GPU': 0} if selected == 'cpu' else {},
                            allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config, dict(requested=requested, selected=selected,
        fallback_reason='TensorFlow reported no usable GPU' if requested == 'auto' and not visible else None,
        physical_gpus=[d.name for d in visible])


def _attention(sequence):

    length = int(sequence.shape[1])
    def project():
        projected = tf.keras.layers.Dense(32, use_bias=False)(sequence)
        split = tf.reshape(projected, (BATCH_SIZE, length, 4, 8))
        return tf.reshape(tf.transpose(split, (0, 2, 1, 3)), (-1, length, 8))
    query, key, value = project(), project(), project()
    logits = tf.matmul(query, key, transpose_b=True)/tf.sqrt(tf.constant(8., tf.float32))
    mask = tf.sequence_mask(tf.ones((BATCH_SIZE*4, length), dtype=tf.int32), length)
    logits = tf.where(mask, logits, tf.fill(tf.shape(logits), tf.constant(-1e6, tf.float32)))
    weights = tf.keras.layers.Dropout(.5)(tf.nn.softmax(logits, axis=-1), training=True)
    attended = tf.matmul(weights, value)
    merged = tf.reshape(tf.transpose(tf.reshape(attended, (BATCH_SIZE, 4, length, 8)), (0, 2, 1, 3)),
                        (BATCH_SIZE, length, 32))
    return tf.keras.layers.Dense(32, use_bias=False)(merged)


def _dense(values, width, name):
    return tf.layers.dense(values, width, activation=tf.nn.relu, name=name,
                           kernel_initializer=tf.random_normal_initializer(0., .1),
                           bias_initializer=tf.constant_initializer(.1))


def _encoder(states, high, dense_names, output_width):

    if high:
        recurrent = tf.keras.layers.RNN(tf.keras.layers.GRUCell(32, kernel_initializer='glorot_uniform'),
                                        time_major=True, return_sequences=True, return_state=True)
        _, hidden = recurrent(states[:, -4:][None, :])
        features = tf.concat((states[:, :-4], hidden), axis=1)
    else:
        features = states

    channels = tf.tile(features[:, :, None], (1, 1, 32))
    encoded = tf.reshape(_attention(channels), (BATCH_SIZE, int(features.shape[1])*32))
    sequence = _dense(encoded, 512, dense_names[0])[None, :]
    for incoming, outgoing in ((512, 256), (256, 128), (128, 64)):

        convolved = tf.nn.conv1d(sequence, tf.ones((1, incoming, outgoing)), 1, 'VALID')
        activated = tf.nn.relu(tf.layers.batch_normalization(convolved, training=True))
        sequence = tf.layers.max_pooling1d(activated, pool_size=1, strides=1)
    hidden = _dense(sequence[0], 64, dense_names[1])
    hidden = _dense(hidden, 32, dense_names[2])
    return _dense(hidden, output_width, dense_names[3])


class _Actor:
    def __init__(self, session, feature_count, action_count, lr, high):
        self.sess, self.n_features, self.n_actions = session, feature_count, action_count
        self.batch_size, self.lr, self.gamma = BATCH_SIZE, lr, GAMMA
        self.s = tf.placeholder(tf.float32, (BATCH_SIZE, feature_count))
        self.a = tf.placeholder(tf.int32, (BATCH_SIZE,))
        self.td_error = tf.placeholder(tf.float32, None)
        self.cost_his = []
        names = ('h_att_dense1', 'dense1', 'dense2', 'h_acts_prob') if high else ('a_dense1', 'dense1_act', 'dense2_act', 'acts_prob')
        with tf.variable_scope('h_Actor' if high else 'Actor'):
            self.logits = _encoder(self.s, high, names, action_count)
            self.all_act_prob = tf.clip_by_value(tf.nn.softmax(self.logits), 0., 1.)


        chosen_log_probability = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.a)
        self.loss = tf.reduce_mean(tf.reduce_sum(chosen_log_probability*self.td_error))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.loss)

    def learn(self, states, actions, td, *, run_metadata=None):
        _, loss = self.sess.run((self.train_op, self.loss),
                               {self.s: states, self.a: actions, self.td_error: td},
                               options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if run_metadata is not None else None,
                               run_metadata=run_metadata)
        self.cost_his.append(loss)
        return loss


class _Critic:
    def __init__(self, session, feature_count, lr, high):
        self.sess, self.n_features, self.lr = session, feature_count, lr
        self.batch_size, self.memory_size, self.gamma = BATCH_SIZE, 100000, GAMMA
        self.memory = np.zeros((self.memory_size, feature_count*2+2))
        self.memory_counter, self.cost_his = 0, []
        self.s = tf.placeholder(tf.float32, (BATCH_SIZE, feature_count))
        self.v_ = tf.placeholder(tf.float32, (BATCH_SIZE, 1))
        self.r = tf.placeholder(tf.float32, None)
        names = ('hc_dense1', 'dense1_cr', 'dense2_cr', 'V') if high else ('c_dense1', 'l1', 'l2', 'V')
        with tf.variable_scope('HL_Critic' if high else 'Critic'):
            self.v = _encoder(self.s, high, names, 1)
        self.td_error = tf.reduce_mean(self.r+GAMMA*self.v_-self.v)
        self.loss = .5*tf.reduce_sum(tf.square(self.td_error))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def store_transition(self, state, action, reward, next_state):
        self.memory[self.memory_counter % self.memory_size] = np.hstack((state, action, reward, next_state))
        self.memory_counter += 1

    def learn(self, states, rewards, next_states, *, run_metadata=None):
        following = self.sess.run(self.v, {self.s: next_states})
        td, _ = self.sess.run((self.td_error, self.train_op),
                             {self.s: states, self.r: rewards, self.v_: following},
                             options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if run_metadata is not None else None,
                             run_metadata=run_metadata)
        self.cost_his.append(.5*np.average(td**2))
        return td


class High_Level_Network(_Actor):
    def __init__(self, session, n_features, n_actions, lr=1e-4):
        super().__init__(session, n_features, n_actions, lr, True)


class Actor(_Actor):
    def __init__(self, session, n_features, n_actions, lr=1e-4):
        super().__init__(session, n_features, n_actions, lr, False)


class High_Level_Critic(_Critic):
    def __init__(self, session, n_features, lr=1e-4):
        super().__init__(session, n_features, lr, True)


class Critic(_Critic):
    def __init__(self, session, n_features, lr=1e-4):
        super().__init__(session, n_features, lr, False)
