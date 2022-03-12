# encoding: utf-8



import tensorflow as tf
import numpy as np


def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).
  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (labels-batch, inputs-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".
  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.
  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logit)
      else:
        crossent = softmax_loss_function(target, logit)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps






batch_size = 32
num_decoder_symbols = 66




logits = tf.nn.softmax(tf.reshape(
    tf.range(batch_size * num_decoder_symbols, dtype=tf.float32),
    shape=[batch_size, num_decoder_symbols]
))

targets = tf.reshape(tf.range(batch_size, dtype=tf.int32), [-1])
weights = tf.ones(shape=[batch_size], dtype=tf.float32)

output = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
    logits=logits,
    targets=targets,
    weights=weights,
    average_across_timesteps=True,
    softmax_loss_function=None,
    name=None
)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(targets))
    print(sess.run(output))


