
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys

from transfer_learning import transfer_learning_cnn_fn
import player

FLAGS = None
tf.set_random_seed(42)


def _image_preprocess_fn(image_buffer, input_height, input_width, input_mean, input_std):
    """Adds operations that perform JPEG decoding and resizing to the graph..

        Args:
          image_buffer: 1-D string Tensor representing the raw JPEG image buffer.
          input_width: Desired width of the image fed into the recognizer graph.
          input_height: Desired width of the image fed into the recognizer graph.
          input_depth: Desired channels of the image fed into the recognizer graph.
          input_mean: Pixel value that should be zero in the image for the graph.
          input_std: How much to divide the pixel values by before recognition.

        Returns:
          Tensors for the node to feed JPEG data into, and the output of the
            preprocessing steps.
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
    resized_image = tf.image.resize_bilinear(decoded_image_4d,resize_shape_as_int)

    # Normalize image
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)
    return tf.squeeze(resized_image, axis=0)



def parse_fn(example):
  "Parse TFExample records and perform simple data augmentation."
  example_fmt = {
    "image": tf.FixedLenFeature((), tf.string),
    "target": tf.FixedLenFeature((), tf.float32, -1)
  }
  parsed = tf.parse_single_example(example, example_fmt)

  image = _image_preprocess_fn(image_buffer=parsed["image"], input_height=299, input_width=299, input_mean=128, input_std=128)
  return image, parsed["target"]


def get_input_fn(input_file_names, batch_size=1, num_epochs=None, shuffle=False, num_shards=None, shard_size=None):

    def input_fn():
        dataset = None
        file_names = tf.constant(input_file_names, dtype=tf.string, name='input_file_names')

        if shuffle:
            if num_shards is None:
                tf.logging.error("num_shards must be specified to use shuffling in the input_fn")
                return None
            if shard_size is None:
                tf.logging.error("shard_size must be specified to use shuffling in the input_fn")
                return None

            files = tf.data.Dataset.from_tensor_slices(file_names).shuffle(num_shards)
            dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=3)
            dataset = dataset.shuffle(buffer_size=shard_size*2)

        else:
            dataset = tf.data.TFRecordDataset(file_names)

        dataset = dataset.map(map_func=parse_fn, num_parallel_calls=FLAGS.num_parallel_img_parsers)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.repeat(num_epochs)  # the input is repeated indefinitely if num_epochs is None
        dataset = dataset.prefetch(buffer_size=2)

        # print("Dataset ouput types: {}".format(dataset.output_types))
        # print("Dataset ouput shapes: {}".format(dataset.output_shapes))

        iterator = dataset.make_one_shot_iterator()
        features, targets = iterator.get_next()
        return features, targets

    return input_fn

# def main(argv):
#     args = parser.parse_args(argv[1:])
#
#     # Fetch the data
#     (train_x, train_y), (test_x, test_y) = iris_data.load_data()
#
#     # Feature columns describe how to use the input.
#     my_feature_columns = []
#     for key in train_x.keys():
#         my_feature_columns.append(tf.feature_column.numeric_column(key=key))
#
#     # Build 2 hidden layer DNN with 10, 10 units respectively.
#     classifier = tf.estimator.Estimator(
#         model_fn=my_model,
#         params={
#             'feature_columns': my_feature_columns,
#             # Two hidden layers of 10 nodes each.
#             'hidden_units': [10, 10],
#             # The model must choose between 3 classes.
#             'n_classes': 3,
#         })
#
#     # Train the Model.
#     classifier.train(
#         input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
#         steps=args.train_steps)
#
#     # Evaluate the model.
#     eval_result = classifier.evaluate(
#         input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))
#
#     print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
#
#     # Generate predictions from the model
#     expected = ['Setosa', 'Versicolor', 'Virginica']
#     predict_x = {
#         'SepalLength': [5.1, 5.9, 6.9],
#         'SepalWidth': [3.3, 3.0, 3.1],
#         'PetalLength': [1.7, 4.2, 5.4],
#         'PetalWidth': [0.5, 1.5, 2.1],
#     }
#
#     predictions = classifier.predict(
#         input_fn=lambda:iris_data.eval_input_fn(predict_x,
#                                                 labels=None,
#                                                 batch_size=args.batch_size))
#
#     for pred_dict, expec in zip(predictions, expected):
#         template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
#
#         class_id = pred_dict['class_ids'][0]
#         probability = pred_dict['probabilities'][class_id]
#
#         print(template.format(iris_data.SPECIES[class_id],
#                               100 * probability, expec))



def main(argv):
    print("yo")
    # features, targets = input_fn(batch_size=64, num_epochs=None, shuffle=True, shard_size=3000, num_shards=2)

    # print(features.shape)

    # Get all file names in input_dir with leading numbers before the first '_' and sort them by those numbers
    dir_files = os.listdir(FLAGS.input_dir)
    dir_files = sorted(filter(lambda f: str.isdigit(f.split('_')[0]), dir_files), key=lambda f: int(f.split('_')[0]))
    dir_files = list(map(lambda f: os.path.join(FLAGS.input_dir, f), dir_files))

    model = tf.estimator.Estimator(
        model_fn=transfer_learning_cnn_fn,
        params={
            "architecture": "inception_v3",
            "learning_rate": 0.001,
            "optimizer": tf.train.AdamOptimizer,
            "hidden_units": [-1, -1],
            "model_dir": "tf_files/downloaded_models"

        })

    # Train the model.
    model.train(input_fn=get_input_fn(input_file_names= dir_files, batch_size=1, num_epochs=None, shuffle=True, shard_size=3000, num_shards=2),
                steps=100)


    # with tf.Session() as sess:
    #     sess.run(iterator.initializer, feed_dict={'input_file_names:0': dir_files})
    #     for i in range(423400):
    #
    #             try:
    #                 out = sess.run(next_element)
    #                 # print(out[1])
    #
    #                 for j in range(50):
    #                     if player.display_frame(np.squeeze(out[0][j]).astype(np.uint8),debug_info=str(out[1][j]),milliseconds_time_to_wait=1):
    #                         break
    #                 # plt.figure()
    #                 # plt.imshow(out[0][0])
    #                 # plt.show()  # display it
    #             except tf.errors.OutOfRangeError:
    #                 break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Path to tf records dir'
    )

    FLAGS, _ = parser.parse_known_args()

    if FLAGS.input_dir is None:
        print("--input_dir must be specified.")
        exit(1)



    tf.logging.set_verbosity(tf.logging.INFO)

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.num_parallel_img_parsers = 4
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)