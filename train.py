import tensorflow as tf
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_rnn_ops
import time
import os
import tensorflow_addons as tfa
import numpy as np
import tensorflow_probability as tfp

tfd = tfp.distributions

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
NUM_GPU = 1

checkpoint_directory = "training_checkpoints_distinguish_x16"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
max_num_visit = 200
num_code = 506
batchsize = 192


class Config(object):
    def __init__(self):
        self.vocab_dim = num_code
        self.lstm_dim = 384
        self.n_layer = 3


def locked_drop(inputs, is_training):
    if is_training:
        dropout_rate = 0.2
    else:
        dropout_rate = 0.0
    mask = tf.nn.dropout(tf.ones([inputs.shape[0], 1, inputs.shape[2]], dtype=tf.float32), dropout_rate)
    mask = tf.tile(mask, [1, inputs.shape[1], 1])
    return inputs * mask
    # b*t*u


class LSTM(tf.keras.Model):
    def __init__(self):
        super(LSTM, self).__init__()
        lstm = DropConnectLSTM
        self.layer = [lstm(config.lstm_dim, return_sequences=True) for _ in range(config.n_layer)]
        self.layer_norm = [tf.keras.layers.LayerNormalization(epsilon=1e-5) for _ in range(config.n_layer)]

    def call(self, x, length, is_training):
        for layer in self.layer:
            layer.set_mask(is_training)

        for i in range(config.n_layer):
            x = locked_drop(x, is_training)
            x = self.layer[i](x)
            x = self.layer_norm[i](x)
        return x


class DropConnectLSTM(tf.compat.v1.keras.layers.CuDNNLSTM):
    def __init__(self, unit, return_sequences):
        super(DropConnectLSTM, self).__init__(units=unit, return_sequences=return_sequences)
        self.mask = None

    def set_mask(self, is_training):
        if is_training:
            self.mask = tf.nn.dropout(tf.ones([self.units, self.units * 4]), 0.2)
        else:
            self.mask = tf.ones([self.units, self.units * 4])

    def _process_batch(self, inputs, initial_state):
        if not self.time_major:
            inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
        input_h = initial_state[0]
        input_c = initial_state[1]
        input_h = array_ops.expand_dims(input_h, axis=0)
        input_c = array_ops.expand_dims(input_c, axis=0)

        params = recurrent_v2._canonical_to_params(  # pylint: disable=protected-access
            weights=[
                self.kernel[:, :self.units],
                self.kernel[:, self.units:self.units * 2],
                self.kernel[:, self.units * 2:self.units * 3],
                self.kernel[:, self.units * 3:],
                self.recurrent_kernel[:, :self.units] * self.mask[:, :self.units],
                self.recurrent_kernel[:, self.units:self.units * 2] * self.mask[:, self.units:self.units * 2],
                self.recurrent_kernel[:, self.units * 2:self.units * 3] * self.mask[:, self.units * 2:self.units * 3],
                self.recurrent_kernel[:, self.units * 3:] * self.mask[:, self.units * 3:],
            ],
            biases=[
                self.bias[:self.units],
                self.bias[self.units:self.units * 2],
                self.bias[self.units * 2:self.units * 3],
                self.bias[self.units * 3:self.units * 4],
                self.bias[self.units * 4:self.units * 5],
                self.bias[self.units * 5:self.units * 6],
                self.bias[self.units * 6:self.units * 7],
                self.bias[self.units * 7:],
            ],
            shape=self._vector_shape)

        outputs, h, c, _ = gen_cudnn_rnn_ops.cudnn_rnn(
            inputs,
            input_h=input_h,
            input_c=input_c,
            params=params,
            is_training=True)

        if self.stateful or self.return_state:
            h = h[0]
            c = c[0]
        if self.return_sequences:
            if self.time_major:
                output = outputs
            else:
                output = array_ops.transpose(outputs, perm=(1, 0, 2))
        else:
            output = outputs[-1]
        return output, [h, c]


class Embedding(tf.keras.Model):
    def __init__(self):
        super(Embedding, self).__init__()
        self.linear_forward = tf.keras.layers.Dense(config.lstm_dim)

    def call(self, code, others, length):
        code = tf.cast(tf.reduce_sum(tf.one_hot(code, depth=num_code, dtype=tf.int8), axis=-2), tf.float32)
        feature = tf.concat((code, others), axis=-1)
        x_forward = self.linear_forward(feature)
        return x_forward


class FeatureNet(tf.keras.Model):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.embeddings = Embedding()
        self.lstm = LSTM()
        self.dense = tf.keras.layers.Dense(256)
        self.embed = tf.keras.layers.Embedding(1, 256)
        self.mlp0 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.mlp1 = tf.keras.layers.Dense(256)

    def call(self, code, others, length, is_training=False):
        length = tf.squeeze(length)
        x = self.embeddings(code, others, length)
        x = self.lstm(x, length, is_training)
        x = tf.gather_nd(x, tf.concat(
            (tf.expand_dims(tf.range(batchsize, dtype=tf.int32), -1), tf.expand_dims(length - 1, -1)),
            axis=-1))
        feature_vec = self.mlp1(self.mlp0(x))
        feature_vec = feature_vec / tf.math.sqrt(tf.reduce_sum(feature_vec ** 2, axis=-1, keepdims=True))
        return feature_vec


def train():
    beta1 = tfd.Beta(4, 2)
    beta2 = tfd.Beta(2, 4)

    def myfunc(feature, summary, length):
        length = tf.cast(length, tf.float32)
        length1 = tf.cast(beta1.sample() * (length - 5) + 5, tf.int32)
        length2 = tf.cast(beta2.sample() * (length - 5) + 5, tf.int32)
        length = tf.cast(length, tf.int32)
        pos1 = tf.random.uniform(shape=(), minval=0, maxval=length - length1 + 1, dtype=tf.int32)
        pos2 = tf.random.uniform(shape=(), minval=0, maxval=length - length2 + 1, dtype=tf.int32)
        feature1 = tf.concat(
            (feature[pos1:pos1 + length1], -tf.ones((200 - length1, 101), dtype=tf.int32)), axis=0)
        feature2 = tf.concat(
            (feature[pos2:pos2 + length2], -tf.ones((200 - length2, 101), dtype=tf.int32)), axis=0)
        summary1 = tf.concat((summary[pos1:pos1 + length1], tf.zeros((200 - length1, 39), dtype=tf.float32)), axis=0)
        summary2 = tf.concat((summary[pos2:pos2 + length2], tf.zeros((200 - length2, 39), dtype=tf.float32)), axis=0)
        return (feature1, summary1, length1), (-tf.ones_like(feature2), summary2, length2), \
               (tf.where(feature2 < 244, feature2, -tf.ones_like(feature2)), summary2[:, 19:], length2), \
               (tf.where(feature2 >= 244, feature2, -tf.ones_like(feature2)), summary2[:, 19:], length2)

    lengths = np.load('length.npy')
    features = np.load('code.npy').astype('int32')
    summaries = np.load('others.npy').astype('float32')

    train_idx, val_idx = np.load('train_idx.npy'), np.load('val_idx.npy')
    dataset_train = tf.data.Dataset.from_tensor_slices((features[train_idx], summaries[train_idx],
                                                        lengths[train_idx])).shuffle(4096 * 8,
                                                                                     reshuffle_each_iteration=True)
    parsed_dataset_train = dataset_train.map(myfunc, num_parallel_calls=2).batch(
        batchsize * NUM_GPU, drop_remainder=True).prefetch(2)

    dataset_val = tf.data.Dataset.from_tensor_slices((features[val_idx], summaries[val_idx],
                                                      lengths[val_idx])).map(myfunc,
                                                                             num_parallel_calls=2)
    parsed_dataset_val = dataset_val.batch(batchsize * NUM_GPU, drop_remainder=True).prefetch(2)

    del features, summaries
    optimizer = tfa.optimizers.AdamW(learning_rate=5e-5, weight_decay=0)
    feature_net = FeatureNet()
    local_1 = FeatureNet()
    local_2 = FeatureNet()
    local_3 = FeatureNet()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, feature_net=feature_net)
    # checkpoint.restore(checkpoint_prefix + '-40').expect_partial()
    print('start')

    @tf.function
    def one_step(batch, batch_1, batch_2, batch_3, is_training):
        with tf.GradientTape() as tape:
            feature_vec = feature_net(*batch, is_training)
            feature_vec_1 = local_1(*batch_1, is_training)
            feature_vec_2 = local_2(*batch_2, is_training)
            feature_vec_3 = local_3(*batch_3, is_training)
            pair_wise_1 = tf.matmul(feature_vec, feature_vec_1, transpose_b=True) * 10
            loss1 = tf.linalg.diag_part(tf.nn.log_softmax(pair_wise_1))
            loss1 = -tf.reduce_mean(loss1 * (1 - tf.math.exp(loss1)) ** 2)
            pair_wise_2 = tf.matmul(feature_vec, feature_vec_2, transpose_b=True) * 10
            loss2 = tf.linalg.diag_part(tf.nn.log_softmax(pair_wise_2))
            loss2 = -tf.reduce_mean(loss2 * (1 - tf.math.exp(loss2)) ** 2)
            pair_wise_3 = tf.matmul(feature_vec, feature_vec_3, transpose_b=True) * 10
            loss3 = tf.linalg.diag_part(tf.nn.log_softmax(pair_wise_3))
            loss3 = -tf.reduce_mean(loss3 * (1 - tf.math.exp(loss3)) ** 2)
            loss = loss1 + loss2 + loss3
        if is_training:
            grads = tape.gradient(loss,
                                  feature_net.trainable_variables + local_1.trainable_variables + local_2.trainable_variables + local_3.trainable_variables)
            optimizer.apply_gradients(zip(grads, feature_net.trainable_variables + local_1.trainable_variables + local_2.trainable_variables + local_3.trainable_variables))
        return loss

    @tf.function
    def output_step(batch):
        feature_vec = feature_net(*batch)
        return feature_vec

    print('training start')
    for epoch in range(500):
        step_val = 0
        step_train = 0

        start_time = time.time()
        loss_val = 0
        loss_train = 0

        for batch_sample in parsed_dataset_train:
            aug1, aug2, aug3, aug4 = batch_sample
            step_loss = one_step(aug1, aug2, aug3, aug4, True).numpy()
            loss_train += step_loss
            step_train += 1

        for batch_sample in parsed_dataset_val:
            aug1, aug2, aug3, aug4 = batch_sample
            step_loss = one_step(aug1, aug2, aug3, aug4, False).numpy()
            loss_val += step_loss
            step_val += 1

        duration_epoch = int(time.time() - start_time)
        format_str = 'epoch: %d, train_loss = %f, val_loss = %f (%d)'
        print(format_str % (epoch, loss_train / step_train, loss_val / step_val,
                            duration_epoch))
        if epoch % 20 == 19:
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':
    config = Config()
    train()
