from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import multiprocessing


def _image_preprocess_fn(image_buffer, input_height, input_width, input_mean, input_std, return_full_size_image=False):
    """Adds operations that perform JPEG decoding and resizing to the graph..

        Args:
          image_buffer: 1-D string Tensor representing the raw JPEG image buffer.
          input_width: Desired width of the image fed into the recognizer graph.
          input_height: Desired width of the image fed into the recognizer graph.
          input_mean: Pixel value that should be zero in the image for the graph.
          input_std: How much to divide the pixel values by before recognition.

        Returns:
          Tensors for the node to feed JPEG data into, and the output of the
          prepossessing steps.
    """

    # image_buffer 1-D string Tensor representing the raw JPEG image buffer.

    # Extract image shape from raw JPEG image buffer.
    image_shape = tf.image.extract_jpeg_shape(image_buffer)

    # Decode and crop image.
    offset_x = 0
    offset_y = image_shape[0] // 3  # We want to crop off the top fifth of the image
    crop_width = image_shape[1]
    crop_height = 2 * image_shape[0] // 3
    crop_window = tf.stack([offset_y, offset_x, crop_height, crop_width])
    cropped_image = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=3)

    # Resize image.
    # decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(cropped_image, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)

    # Normalize image
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)

    if return_full_size_image:
        return tf.squeeze(mul_image, axis=0), cropped_image

    return tf.squeeze(mul_image, axis=0)


def get_input_fn(input_file_names, batch_size=1, num_epochs=None, shuffle=False, shard_size=3000,
                 return_full_size_image=False, offset=0):
    """Creates input_fn according to parameters

        Args:
            input_file_names: (Ordered) List of TFRecord file names to parse data from.
            batch_size: Number of entries to be returned by a single call to the produced input_fn
            num_epochs: Max number of iterations allowed over the entire dataset (Use 'None' for no limit)
            shuffle: 'True' to shuffle the data or 'False' to keep order.
            shard_size: Average number of data points per shard (needs to be accurate for shuffle to work well)
            return_full_size_image: (For debugging) 'True' causes input_fn to return 3 tensors:
                (preprocessed_image, target_label, full_size_image) rather than the normal behavior of
                just returning the first two.

        Returns:
            input_fn that returns a batch of features and labels upon every call to it.
    """

    def parse_fn(example):
        """Parse TFExample records and perform simple data augmentation."""

        if offset != 0:
            example_fmt = {
                "image": tf.FixedLenFeature((), tf.string),
                "target": tf.FixedLenFeature((), tf.float32, -1),
                "camera_index": tf.FixedLenFeature((), tf.int64, -1)
            }
        else:
            example_fmt = {
                "image": tf.FixedLenFeature((), tf.string),
                "target": tf.FixedLenFeature((), tf.float32, -1)
            }

        parsed = tf.parse_single_example(example, example_fmt)

        target = parsed["target"]

        if offset != 0:
            print("OFFSET WILL BE APPLIED")

            camera_index = parsed["camera_index"]

            target = tf.cond(tf.equal(camera_index, 0), lambda: target + offset, lambda: target)
            target = tf.cond(tf.equal(camera_index, 2), lambda: target - offset, lambda: target)
            # This should never happen, cause an error if it does
            target = tf.cond(tf.equal(camera_index, -1), lambda: tf.constant([-1,-1,-1,-1],tf.float32), lambda: target)

        if return_full_size_image:
            preprocessed_image, full_size_image = _image_preprocess_fn(
                image_buffer=parsed["image"], input_height=66, input_width=200, input_mean=128,
                input_std=128, return_full_size_image=True)
            return preprocessed_image, target, full_size_image

        preprocessed_image = _image_preprocess_fn(image_buffer=parsed["image"], input_height=66, input_width=200,
                                                  input_mean=128, input_std=128)

        return preprocessed_image, target

    def input_fn():
        file_names = tf.constant(input_file_names, dtype=tf.string, name='input_file_names')

        if shuffle:
            num_shards = len(input_file_names)

            files = tf.data.Dataset.from_tensor_slices(file_names).shuffle(num_shards)
            dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=num_shards)
            dataset = dataset.shuffle(buffer_size=shard_size * 2)

        else:
            dataset = tf.data.TFRecordDataset(file_names)

        dataset = dataset.map(map_func=parse_fn, num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.repeat(num_epochs)  # the input is repeated indefinitely if num_epochs is None
        dataset = dataset.prefetch(buffer_size=64)

        # print("Dataset ouput types: {}".format(dataset.output_types))
        # print("Dataset ouput shapes: {}".format(dataset.output_shapes))

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    return input_fn


def get_model_fn():

    def cnn_fn(features, labels, mode, params):

        if isinstance(features, dict):
            features = features['x']

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
        fc4 = tf.layers.dense(fc3, units=10, activation=tf.nn.relu, name="fc_4")

        output_layer = tf.layers.dense(fc4, units=1, activation=None, name="output")

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
            optimizer = optimizer(params.get("learning_rate", 0.0001))
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
