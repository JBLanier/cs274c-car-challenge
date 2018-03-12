from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def rnn_fn(features, labels, mode, params):
    sequence_length = params.get("sequence_length", 20)
    rnn_size = params.get("hidden_units", 200)
    stateful = params.get("stateful", False)
    batch_size = params.get("batch_size", 16)

    if isinstance(features, dict):
        features = features['x']

    features = tf.reshape(features, [-1, features.shape[-3], features.shape[-2], features.shape[-1]])

    conv1 = tf.layers.conv2d(features, 24, 5, strides=(2, 2), padding='valid', activation=tf.nn.relu, name='conv1')
    conv2 = tf.layers.conv2d(conv1, 36, 5, strides=(2, 2), padding='valid', activation=tf.nn.relu, name='conv2')
    conv3 = tf.layers.conv2d(conv2, 48, 5, strides=(2, 2), padding='valid', activation=tf.nn.relu, name='conv3')
    conv4 = tf.layers.conv2d(conv3, 64, 3, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv4')
    conv5 = tf.layers.conv2d(conv4, 64, 3, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv5')

    flattened = tf.layers.flatten(conv5)
    sequenced = tf.reshape(flattened, [-1, sequence_length, flattened.shape[1]])
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)

    if stateful is True:
        initial_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
        initial_state = tf.identity(initial_state, name="RNN/init_states")
        rnn_outputs, rnn_out_state = tf.nn.dynamic_rnn(rnn_cell,
                                                       sequenced,
                                                       dtype=tf.float32,
                                                       initial_state=initial_state)
    else:
        rnn_outputs, rnn_out_state = tf.nn.dynamic_rnn(rnn_cell,
                                                       sequenced,
                                                       dtype=tf.float32)

    if stateful is True:
        rnn_out_state = tf.identity(rnn_out_state, name="RNN/output_states")

    rnn_outputs = tf.reshape(rnn_outputs, [-1, rnn_size])

    predictions = tf.squeeze(tf.layers.dense(rnn_outputs, units=1, activation=None), 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # In `PREDICT` mode we only need to return predictions.
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'angle': predictions})

    # Calculate loss using mean squared error
    mean_squared_error = tf.losses.mean_squared_error(labels=tf.reshape(labels, [-1]), predictions=predictions)
    root_mean_squared_error = tf.sqrt(mean_squared_error)

    # # Pre-made estimators use the total_loss instead of the average,
    # # so report total_loss for compatibility.
    # batch_size = tf.shape(labels)[0]
    # total_loss = tf.to_float(batch_size) * mean_squared_error

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params.get("optimizer", tf.train.AdamOptimizer)
        optimizer = optimizer(params.get("learning_rate", None))
        train_op = optimizer.minimize(loss=mean_squared_error, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=root_mean_squared_error, train_op=train_op)

    # In evaluation mode we will calculate evaluation metrics.
    assert mode == tf.estimator.ModeKeys.EVAL

    eval_metrics = {}

    # TODO: There's nothing to do. I just want to highlight what total bullshit 'tf.metrics.root_mean_squared_error' is.
    # TODO: It's not RMSE for the predictions of just this batch, it's a rolling RMSE over the whole session.
    # TODO: Meaning it's never what you expect it to be.
    # TODO: tf.metrics.root_mean_squared_error is a lie!
    # TODO: It used to be called 'streaming_mean_squared_error' but then someone decided "No, let's confuse everyone."

    # rmse = tf.metrics.root_mean_squared_error(labels, predictions)
    # eval_metrics["wish_i_knew_what_this_did_2_hours_ago_rmse"] = rmse

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=root_mean_squared_error,
        eval_metric_ops=eval_metrics)

    # TODO: Furthermore, see that loss parameter in the estimator spec right there? Yeah it's a lie too,
    # TODO: but it's a lie for the good of society. If you pass an RMSE tensor like that in training mode,
    # TODO: it'll report what you gave, the RMSE for the batch, but when you're evaluating, you don't want that.
    # TODO: When evaluating, you want the RMSE over ALL the batches, and guess what?
    # TODO: In the Estimator evaluate() function,
    # TODO: they average that loss over all the batches before reporting it back to you,
    # TODO: so you don't have to worry about it, that is unless you question the meaning of your own existence and
    # TODO: realize that none of the loss functions are doing explicitly what you'd think they should do and then time itself starts to become an incorrect implementation of RMSE and you can see the RMSE between your past and future selves

    # TODO: I promise, 'loss' on tensorboard is RMSE. It works, I swear. - JB <3
