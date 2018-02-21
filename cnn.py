from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def cnn_fn(features, labels, mode, params):

    if isinstance(features, dict):
        features = features['x']

    conv1 = tf.layers.conv2d(features, 16, 5, strides=(4, 4), padding='same', activation=tf.nn.elu, name='conv1')
    conv2 = tf.layers.conv2d(conv1, 32, 5, strides=(2, 2), padding='same', activation=tf.nn.elu, name='conv1')
    conv3 = tf.layers.conv2d(conv2, 64, 5, strides=(2, 2), padding='same', activation=tf.nn.elu, name='conv3')

    flattened = tf.layers.flatten(conv3)

    # Need to modify "training" param based on training or not
    drp1 = tf.layers.dropout(flattened, rate = .2, training = True, name="drp1")
    fc1 = tf.layers.dense(drp1, units=512, activation=tf.nn.elu, name="fc1")
    drp2 = tf.layers.dropout(fc1, rate=.5, training=True, name="drp2")
    output_layer = tf.layers.dense(drp2, units=1, activation=None, name="output")

    # conv1 = tf.layers.conv2d(features, 24, 5, strides=(2, 2), padding='valid', activation=tf.nn.relu, name='conv1')
    # conv2 = tf.layers.conv2d(conv1, 36, 5, strides=(2, 2), padding='valid', activation=tf.nn.relu, name='conv2')
    # conv3 = tf.layers.conv2d(conv2, 48, 5, strides=(2, 2), padding='valid', activation=tf.nn.relu, name='conv3')
    # conv4 = tf.layers.conv2d(conv3, 64, 3, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv4')
    # conv5 = tf.layers.conv2d(conv4, 64, 3, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv5')
    #
    # flattened = tf.layers.flatten(conv5)
    #
    # # Add our own fully-connected layers
    # fc1 = tf.layers.dense(flattened, units=1164, activation=tf.nn.relu, name="fc_1")
    # fc2 = tf.layers.dense(fc1, units=100, activation=tf.nn.relu, name="fc_2")
    # fc3 = tf.layers.dense(fc2, units=50, activation=tf.nn.relu, name="fc_3")
    # fc4 = tf.layers.dense(fc3, units=10, activation=tf.nn.relu, name="fc_4")
    #
    # output_layer = tf.layers.dense(fc4, units=1, activation=None, name="output")


    # Reshape the output layer to a 1-dim Tensor to return predictions
    predictions = tf.squeeze(output_layer, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # In `PREDICT` mode we only need to return predictions.
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions={'angle': predictions})

    # Calculate loss using mean squared error
    mean_squared_error = tf.losses.mean_squared_error(labels=labels, predictions=predictions)
    root_mean_squared_error = tf.sqrt(mean_squared_error)

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
            mode=mode, loss=root_mean_squared_error, train_op=train_op)

    # In evaluation mode we will calculate evaluation metrics.
    assert mode == tf.estimator.ModeKeys.EVAL

    # Calculate root mean squared error
    print(labels)
    print(predictions)
    rmse = tf.metrics.root_mean_squared_error(labels, predictions)

    # Add the rmse to the collection of evaluation metrics.
    eval_metrics = {"rmse": rmse}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        # Report sum of error for compatibility with pre-made estimators
        loss=mean_squared_error,
        eval_metric_ops=eval_metrics)
