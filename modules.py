# encoding: utf-8

import tensorflow as tf
import numpy as np
from transformer.load_data import load_data
from transformer import parameters as params



def multi_head_attention(
    queries,
    keys,
    values,
    num_units,
    num_heads,
    dropout_rate,
    is_training=True,
    future_mask=False,
    reuse=None,
    scope="multi_head_attention"
):
    with tf.variable_scope(scope, reuse=reuse):
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) # [n, L_q, d]
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu) # [n, L_k, d]
        V = tf.layers.dense(values, num_units, activation=tf.nn.relu) # [n, L_v, d]
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # [h * n, L_q, d / h]
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # [h * n, L_k, d / h]
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # [h * n, L_v, d / h]
        outputs = tf.matmul(Q_, tf.transpose(K_, perm=[0, 2, 1])) * (num_units / num_heads) ** (-0.5) # [h * n, L_q, L_k]

        key_mask = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
        key_mask = tf.tile(tf.expand_dims(key_mask, axis=-1), [num_heads, 1, tf.shape(queries)[1]])
        outputs = tf.where(tf.equal(key_mask, 0), tf.ones_like(outputs) * (-1e9), outputs)  # [h * n, L_q, L_k]

        if future_mask: # future_mask
            future_mask = tf.ones_like(outputs[0, :, :])
            future_mask = tf.contrib.linalg.LinearOperatorTriL(future_mask).to_dense()
            future_mask = tf.expand_dims(future_mask, axis=0)
            future_mask = tf.tile(future_mask, [tf.shape(outputs)[0], 1, 1])
            outputs = tf.where(tf.equal(future_mask, 0), tf.ones_like(outputs) * (-1e9), outputs)  # [h * n, L_q, L_k]

        outputs = tf.nn.softmax(outputs)  # [h * n, L_q, L_k]

        query_mask = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))# [n, L_q, 1]
        query_mask = tf.tile(tf.expand_dims(query_mask, axis=-1), [num_heads, 1, tf.shape(keys)[1]])
        outputs = tf.where(tf.equal(query_mask, 0), tf.zeros_like(outputs), outputs)  # [h * n, L_q, L_k]

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs = tf.matmul(outputs, V_) # [h * n, L_q, d / h]
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        outputs += queries
        outputs = normalize(outputs)
        return outputs



def label_smoothing(inputs, epsilon):
    return ((1 - epsilon) * inputs) + (epsilon / (inputs.get_shape().as_list()[-1]))


def save_model(sess, name):
    saver = tf.train.Saver()
    saver.save(sess, name)


def load_model(sess, name):
    saver = tf.train.Saver()
    saver.restore(sess, name)


def get_loss(model_outputs, target_inputs, target_vocab_size):
    target_inputs_smoothed = label_smoothing(tf.one_hot(target_inputs, depth=target_vocab_size), epsilon=0.1)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=model_outputs,
        labels=target_inputs_smoothed
    )
    flag = tf.to_float(tf.not_equal(target_inputs, 0))
    return tf.reduce_sum(loss * flag) / (tf.reduce_sum(flag))


def get_accuracy(preds, target_inputs):
    flag = tf.to_float(tf.not_equal(target_inputs, 0))
    return tf.reduce_sum(tf.to_float(tf.equal(preds, target_inputs)) * flag) / (tf.reduce_sum(flag))


def get_train_op(loss, global_step, learning_rate):
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9).minimize(loss, global_step=global_step)
    return train_op



# source_inputs, \
# target_inputs, \
# source_vocab_size, \
# target_vocab_size, \
# source_idx2word, \
# target_idx2word = load_data()


def embedding(
    inputs,
    vocab_size,
    num_units,
    zero_pad=True,
    reuse=None,
    scope="embedding"
):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable(
            name='lookup_table',
            dtype=tf.float32,
            shape=[vocab_size, num_units],
            initializer=tf.truncated_normal_initializer(stddev=0.1, mean=0.001, seed=10)
        )
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)
        return outputs


def position_encoding(
    batch_size,
    max_len,
    num_units,
    zero_pad=True,
    reuse=None,
    scope="position_encoding"
):
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = np.array(
            [[pos / (10000.0 ** (2.0 * i / num_units)) for i in range(num_units)] for pos in range(max_len)]
        )
        lookup_table[:, 0::2] = np.sin(lookup_table[:, 0::2])
        lookup_table[:, 1::2] = np.cos(lookup_table[:, 1::2])
        lookup_table = tf.convert_to_tensor(lookup_table, dtype=tf.float32)
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), lookup_table[1:, :]), 0)
        positions = tf.tile(tf.expand_dims(tf.range(max_len), 0), [batch_size, 1])
        outputs = tf.nn.embedding_lookup(lookup_table, positions)
        return outputs


def self_attention(
    queries,
    keys,
    values,
    num_heads,
    reuse=None,
    scope="self-attention"
):
    with tf.variable_scope(scope, reuse=reuse):
        num_units = queries.get_shape().as_list()[-1]
        new_num_units = num_units
        Wq = tf.get_variable(
            name="query_weights",
            dtype=tf.float32,
            shape=[num_units, new_num_units],
            initializer=tf.truncated_normal_initializer(mean=0.001, stddev=0.1)
        )
        Wk = tf.get_variable(
            name="key_weights",
            dtype=tf.float32,
            shape=[num_units, new_num_units],
            initializer=tf.truncated_normal_initializer(mean=0.001, stddev=0.1)
        )
        Wv = tf.get_variable(
            name="value_weights",
            dtype=tf.float32,
            shape=[num_units, new_num_units],
            initializer=tf.truncated_normal_initializer(mean=0.001, stddev=0.1)
        )
        Q = tf.nn.relu(tf.matmul(queries, tf.tile(tf.expand_dims(Wq, 0), [queries.get_shape().as_list()[0], 1, 1]))) # [batch_size, max_len, new_num_units]
        K = tf.nn.relu(tf.matmul(keys, tf.tile(tf.expand_dims(Wk, 0), [keys.get_shape().as_list()[0], 1, 1]))) # [batch_size, max_len, new_num_units]
        V = tf.nn.relu(tf.matmul(values, tf.tile(tf.expand_dims(Wv, 0), [values.get_shape().as_list()[0], 1, 1]))) # [batch_size, max_len, new_num_units]
        Q = tf.concat(tf.split(Q, num_heads, axis=2), axis=0) # [batch_size * num_heads, max_len, new_num_units / num_heads]
        K = tf.concat(tf.split(K, num_heads, axis=2), axis=0) # [batch_size * num_heads, max_len, new_num_units / num_heads]
        V = tf.concat(tf.split(V, num_heads, axis=2), axis=0) # [batch_size * num_heads, max_len, new_num_units / num_heads]
        outputs = tf.matmul(tf.nn.softmax(tf.div(tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])), tf.sqrt(tf.to_float(tf.shape(Q)[-1]))), dim=-1), V)
        # outputs = tf.matmul(tf.nn.softmax(tf.div(tf.matmul(Q, tf.transpose(K, perm=[0, 2, 1])), tf.sqrt(tf.to_float(tf.shape(Q)[-1]))), dim=-1), V)
        # outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
        # if zero_pad_mask:
        #     queries_mask = tf.tile(tf.expand_dims(tf.equal(tf.reduce_sum(queries, axis=-1), 0), axis=-1), [1, 1, queries.get_shape().as_list()[-1]])
        #     outputs = tf.where(queries_mask, tf.zeros_like(outputs), outputs)
        return outputs




def feed_forward(
    inputs,
    reuse=None,
    scope="feed_forward"
):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape().as_list()
        W = tf.get_variable(
            name="weights",
            dtype=tf.float32,
            shape=[inputs_shape[-1], inputs_shape[-1]],
            initializer=tf.truncated_normal_initializer(mean=0.001, stddev=0.1)
        )
        outputs = tf.matmul(inputs, tf.tile(tf.expand_dims(W, 0), [inputs_shape[0], 1, 1]))
        outputs = tf.nn.relu(outputs)
        return outputs


def layer_normalize(
    inputs,
    reuse=None,
    scope="layer_normalize"
):
    with tf.variable_scope(scope, reuse=reuse):
        gamma = tf.Variable(1.0)
        beta = tf.Variable(0.0)
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        normalized = (inputs - mean) / (variance + 1e-8) ** 0.5
        outputs = tf.add(tf.multiply(gamma, normalized), beta)
        return outputs


def encoder(
    inputs,
    num_heads,
    reuse=None,
    scope="encoder"
):
    with tf.variable_scope(scope, reuse=reuse):
        atten_outputs = self_attention(inputs, inputs, inputs, num_heads)
        atten_outputs = inputs + atten_outputs
        atten_outputs = layer_normalize(atten_outputs)
        forward_outputs = feed_forward(atten_outputs)
        forward_outputs = atten_outputs + forward_outputs
        forward_outputs = layer_normalize(forward_outputs)
        return forward_outputs

##################################test########################
source_inputs = np.array([[2, 1, 3, 0, 0]])
target_inputs = np.array([[2, 8, 4, 5, 6, 0]])
source_output = embedding(
    source_inputs,
    100,
    params.num_units,
    zero_pad=True,
    scope="source"
)
target_output = embedding(
    target_inputs,
    100,
    params.num_units,
    zero_pad=True,
    scope="target"
)

output = self_attention(
    target_output,
    source_output,
    source_output,
    params.num_heads
)

# output = encoder(
#     output
# )

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(output)
    print(source_inputs)
    print(target_inputs)
    print(sess.run(output))
    print(sess.run(output).shape)



