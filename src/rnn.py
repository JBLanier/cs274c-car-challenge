from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def rnn_fn(features, labels, mode, params):
    sequence_length = params["sequence_length"]
    rnn_size = params.get("rnn_size", 50)

    if isinstance(features, dict):
        features = features['x']


    features = tf.placeholder_with_default(features,[None,sequence_length,66,200,3])

    print("features shape: {}".format(features.shape))


    conv1 = tf.layers.conv3d(features, 24, (3, 5, 5), strides=(1, 2, 2), padding='valid', activation=tf.nn.relu, name='conv1')
    conv2 = tf.layers.conv3d(conv1, 36, (3, 5, 5), strides=(1, 2, 2), padding='valid', activation=tf.nn.relu, name='conv2')
    conv3 = tf.layers.conv3d(conv2, 48, (3, 5, 5), strides=(1, 2, 2), padding='valid', activation=tf.nn.relu, name='conv3')
    conv4 = tf.layers.conv3d(conv3, 64, 3, strides=(1, 1, 1), padding='valid', activation=tf.nn.relu, name='conv4')
    conv5 = tf.layers.conv3d(conv4, 64, (1, 3, 3), strides=(1, 1, 1), padding='valid', activation=tf.nn.relu, name='conv5')

    print("conv shape: {}".format(conv5.shape))

    conv_shape = conv5.get_shape().as_list()
    flattened = tf.reshape(conv5, [-1, conv_shape[1], conv_shape[2] * conv_shape[3] * conv_shape[4]])

    print("flattened shape: {}".format(flattened.shape))

    rnn_cell1 = tf.nn.rnn_cell.BasicLSTMCell(1050)
    rnn_cell2 = tf.nn.rnn_cell.BasicLSTMCell(100)
    rnn_cell3 = tf.nn.rnn_cell.BasicLSTMCell(50)
    rnn_cell4 = tf.nn.rnn_cell.BasicLSTMCell(10)

    multi_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell1, rnn_cell2, rnn_cell3, rnn_cell4])

    # if mode == tf.estimator.ModeKeys.TRAIN:
    rnn_outputs, rnn_out_state = tf.nn.dynamic_rnn(multi_cell,flattened, dtype=tf.float32)
    # else:
    #     initial_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)
    #     initial_state = tf.identity(initial_state, name="RNN/init_states")
    #     rnn_outputs, rnn_out_state = tf.nn.dynamic_rnn(rnn_cell,
    #                                                    sequenced,
    #                                                    dtype=tf.float32,
    #                                                    initial_state=initial_state)
    #
    #     rnn_out_state = tf.identity(rnn_out_state, name="RNN/output_states")

    rnn_outputs = rnn_outputs[:,-1,:]  # We only care about the last outputs in each sequence
    print ("run outputs shape: {}".format(rnn_outputs.shape))

    predictions = tf.squeeze(tf.layers.dense(rnn_outputs, units=1, activation=None),1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # In `PREDICT` mode we only need to return predictions.
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'angle': predictions})

    labels = labels[:,-1]  # We only care about the last label in each sequence

    # Calculate loss using mean squared error
    mean_squared_error = tf.losses.mean_squared_error(labels=labels, predictions=predictions)

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

    rmse = tf.metrics.root_mean_squared_error(labels, predictions)
    eval_metrics["validation_rmse"] = rmse

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=mean_squared_error,
        eval_metric_ops=eval_metrics)
