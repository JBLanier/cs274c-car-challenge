from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_model_fn(conv_layers, dense_units, learning_rate):

    def cnn_fn(features, labels, mode, params):

        if isinstance(features, dict):
            features = features['x']

        net = features

        for conv in conv_layers:
            net = tf.layers.conv2d(net, filters=conv[1], kernel_size=conv[0], strides=(conv[2], conv[2]),
                                   padding='valid', activation=tf.nn.relu)

        net = tf.layers.flatten(net)

        # Add fully-connected layers
        for units in dense_units:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

        output_layer = tf.layers.dense(net, units=1, activation=None, name="output")

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

    return cnn_fn
