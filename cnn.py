from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def cnn_fn(features, labels, mode, params):
    #print(len(features))
    training = False
    if mode == tf.estimator.ModeKeys.TRAIN:
        training = True


    if isinstance(features, dict):
        features = features['x']

    #Normal Model
    conv1 = tf.layers.conv2d(features, 8, 7, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv1')
    conv2 = tf.layers.conv2d(conv1, 10, 7, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv2')
    conv3 = tf.layers.conv2d(conv2, 11, 5, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv3')
    conv4 = tf.layers.conv2d(conv3, 11, 5, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv4')
    conv5 = tf.layers.conv2d(conv4, 11, 3, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv5')

    # # Max Pooling Model
    # conv1 = tf.layers.conv2d(features, 8, 7, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv1')
    # maxp1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='valid', name='maxp1')
    # conv2 = tf.layers.conv2d(maxp1, 10, 7, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv2')
    # maxp2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='valid', name='maxp2')
    # conv3 = tf.layers.conv2d(maxp2, 11, 5, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv3')
    # #maxp3 = tf.layers.max_pooling2d(conv3, 2, 2, padding='same', name='maxp3')
    # conv4 = tf.layers.conv2d(conv3, 11, 5, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv4')
    # #maxp4 = tf.layers.max_pooling2d(conv4, 2, 2, padding='valid', name='maxp4')
    # conv5 = tf.layers.conv2d(conv4, 11, 3, strides=(1, 1), padding='valid', activation=tf.nn.relu, name='conv5')
    # #maxp5 = tf.layers.max_pooling2d(conv1, 2, 2)

    flattened = tf.layers.flatten(conv5)

    # #Normal Model
    # # Add fully-connected layers
    # fc1 = tf.layers.dense(flattened, units=1307, activation=tf.nn.relu, name="fc1")
    # fc2 = tf.layers.dense(fc1, units=522, activation=tf.nn.relu, name="fc2")
    # fc3 = tf.layers.dense(fc2, units=208, activation=tf.nn.relu, name="fc3")


    # Dropout Model
    fc1 = tf.layers.dense(flattened, units=1307, activation=tf.nn.relu, name="fc1")
    drp1 = tf.layers.dropout(fc1, rate=.2, training=training, name="drp1")
    fc2 = tf.layers.dense(drp1, units=522, activation=tf.nn.relu, name="fc2")
    drp2 = tf.layers.dropout(fc2, rate=.2, training=training, name="drp2")
    fc3 = tf.layers.dense(drp2, units=208, activation=tf.nn.relu, name="fc3")
    #drp3 = tf.layers.dropout(fc3, rate=.2, training=training, name="drp3")

    output_layer = tf.layers.dense(fc3, units=1, activation=None, name="output")

    # Reshape the output layer to a 1-dim Tensor to return predictions
    predictions = tf.squeeze(output_layer, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # In `PREDICT` mode we only need to return predictions.
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions={'angle': predictions})

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

