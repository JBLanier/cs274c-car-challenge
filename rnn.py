from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def rnn_fn(features, labels, mode, params):
    sequence_length = 16
    if mode == tf.estimator.ModeKeys.TRAIN:
        sequence_length = params.get("train_sequence_length", 20)
    rnn_size = params.get("rnn_size", 50)

    if isinstance(features, dict):
        features = features['x']

    orig_features = features

    print("FEATURES SHAPE: {}".format(features.shape.as_list()))
    if features.shape.as_list() != [None,66,200,3]:
        features = tf.reshape(features, [-1, features.shape[-3], features.shape[-2], features.shape[-1]])
        print("had to reshape to {}".format(features.shape))
    else:
        print("did not have to reshape")

    conv1 = tf.layers.conv2d(features, 24, 5, strides=(2, 2), padding='valid', activation=tf.nn.relu, name='conv1')
    conv2 = tf.layers.conv2d(conv1, 36, 5, strides=(2, 2), padding='valid', activation=tf.nn.relu, name='conv2')
    conv3 = tf.layers.conv2d(conv2, 48, 5, strides=(2, 2), padding='valid', activation=tf.nn.relu, name='conv3')
    conv4 = tf.layers.conv2d(conv3, 64, 3, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv4')
    conv5 = tf.layers.conv2d(conv4, 64, 3, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv5')

    flattened = tf.layers.flatten(conv5)

    # Add fully-connected layers
    fc1 = tf.layers.dense(flattened, units=1164, activation=tf.nn.relu, name="fc_1")
    fc2 = tf.layers.dense(fc1, units=100, activation=tf.nn.relu, name="fc_2")
    fc3 = tf.layers.dense(fc2, units=50, activation=tf.nn.relu, name="fc_3")

    print("fc3 shape: {}".format(fc3.shape))

    sequenced = tf.reshape(fc3, [-1, sequence_length, fc3.shape[1]])
    print("Sequenced size: {}".format(sequenced.shape))
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)

    # if mode == tf.estimator.ModeKeys.TRAIN:
    rnn_outputs, rnn_out_state = tf.nn.dynamic_rnn(rnn_cell,
                                                       sequenced,
                                                       dtype=tf.float32)
    # else:
    #     initial_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
    #     initial_state = tf.identity(initial_state, name="RNN/init_states")
    #     rnn_outputs, rnn_out_state = tf.nn.dynamic_rnn(rnn_cell,
    #                                                    sequenced,
    #                                                    dtype=tf.float32,
    #                                                    initial_state=initial_state)
    #
    #     rnn_out_state = tf.identity(rnn_out_state, name="RNN/output_states")

    rnn_outputs = tf.reshape(rnn_outputs, [-1, rnn_size])

    predictions = tf.layers.dense(rnn_outputs, units=1, activation=None)

    print("predictions shape {}".format(predictions.shape))
    if mode == tf.estimator.ModeKeys.PREDICT:
        # In `PREDICT` mode we only need to return predictions.
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'angle': predictions})

    # Calculate loss using mean squared error
    mean_squared_error = tf.losses.mean_squared_error(labels=tf.reshape(labels, [-1]), predictions=tf.squeeze(predictions, 1))

    # # Pre-made estimators use the total_loss instead of the average,
    # # so report total_loss for compatibility.
    # batch_size = tf.shape(labels)[0]
    # total_loss = tf.to_float(batch_size) * mean_squared_error

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params.get("optimizer", tf.train.AdamOptimizer)
        optimizer = optimizer(params.get("learning_rate", None))
        train_op = optimizer.minimize(
            loss=mean_squared_error, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=mean_squared_error, train_op=train_op)

        # In evaluation mode we will calculate evaluation metrics.
    assert mode == tf.estimator.ModeKeys.EVAL

    eval_metrics = {}

    rmse = tf.metrics.root_mean_squared_error(tf.reshape(labels, [-1]), tf.squeeze(predictions,1))
    eval_metrics["validation_rmse"] = rmse

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=mean_squared_error,
        eval_metric_ops=eval_metrics)
