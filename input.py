
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import cv2

from transfer_learning import transfer_learning_cnn_fn
from cnn import cnn_fn
import player
from fast_predict import FastPredict

FLAGS = None
tf.set_random_seed(42)


def _image_preprocess_fn(image_buffer, input_height, input_width, input_mean, input_std, return_full_size_image=False):
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
    resized_image = tf.image.resize_bilinear(decoded_image_4d, resize_shape_as_int)

    # Normalize image
    offset_image = tf.subtract(resized_image, input_mean)
    mul_image = tf.multiply(offset_image, 1.0 / input_std)

    if return_full_size_image:
        return tf.squeeze(mul_image, axis=0), cropped_image

    return tf.squeeze(mul_image, axis=0)


def get_input_fn(input_file_names, batch_size=1, num_epochs=None, shuffle=False, num_shards=None, shard_size=None,
                 return_full_size_image=False):

    def parse_fn(example):
        "Parse TFExample records and perform simple data augmentation."
        example_fmt = {
            "image": tf.FixedLenFeature((), tf.string),
            "target": tf.FixedLenFeature((), tf.float32, -1)
        }
        parsed = tf.parse_single_example(example, example_fmt)

        if return_full_size_image:
            preprocessed_image, full_size_image = _image_preprocess_fn(
                image_buffer=parsed["image"], input_height=66, input_width=200, input_mean=128,
                input_std=128, return_full_size_image=True)
            return preprocessed_image, parsed["target"], full_size_image

        preprocessed_image = _image_preprocess_fn(image_buffer=parsed["image"], input_height=66, input_width=200,
                                                    input_mean=128, input_std=128)
        return preprocessed_image, parsed["target"]

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
        dataset = dataset.prefetch(buffer_size=64)

        # print("Dataset ouput types: {}".format(dataset.output_types))
        # print("Dataset ouput shapes: {}".format(dataset.output_shapes))

        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    return input_fn


def main(argv):
    print("yo")

    # Get all file names in input_dir with leading numbers before the first '_' and sort them by those numbers
    dir_files = os.listdir(FLAGS.input_dir)
    dir_files = sorted(filter(lambda f: str.isdigit(f.split('_')[0]), dir_files), key=lambda f: int(f.split('_')[0]))
    dir_files = list(map(lambda f: os.path.join(FLAGS.input_dir, f), dir_files))

    model_dir = "tf_files/cnn/"

    model = tf.estimator.Estimator(
        model_fn=cnn_fn,
        params={
            "architecture": "inception_v3",
            "learning_rate": 0.0001,
            "optimizer": tf.train.AdamOptimizer,
            "hidden_units": [-1, -1],
            "model_dir": model_dir},
        model_dir=model_dir)

    """-------------------------------------------------

    Train the model.
    
    """

    # model.train(input_fn=get_input_fn(input_file_names=dir_files, batch_size=16,
    #                                   num_epochs=None, shuffle=True, shard_size=3000, num_shards=12),
    #             steps=100000)

    """-------------------------------------------------
    
    Debug Code to run the net on images and visualize real time the predictions on video. 
    The video is also saved to 'predictions.mp4'
    
    """

    fast_predict = FastPredict(model)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_writer = None

    predict_sess = tf.Session()
    batch_size = 1
    input_fn = get_input_fn(input_file_names= dir_files, batch_size=batch_size, num_epochs=None, shuffle=False,
                            shard_size=3000, num_shards=2, return_full_size_image=True)
    next_element = input_fn()
    for i in range(10000):

            try:
                out = predict_sess.run(next_element)
                #
                # predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                #     x={"x": out[0]},
                #     num_epochs=1,
                #     shuffle=False)

                predictions = list(fast_predict.predict(out[0]))

                # print((out[2].shape[2],out[2].shape[1]))

                if video_writer is None:
                    video_writer = cv2.VideoWriter("predictions.mp4", fourcc, 40.0, (out[2].shape[2], out[2].shape[1]))

                # play frames in batch
                for j, frame in enumerate(out[2]):
                    # It's assumed that the pixel values are decimals between -1 and 1.
                    # We put them back to between 0 and 255 before playing.
                    if player.display_frame(img=(np.squeeze(frame)).astype(np.uint8),
                                            debug_info=str(out[1][j]),
                                            milliseconds_time_to_wait=1,
                                            predicted_angle=predictions[j]['angle'],
                                            true_angle=out[1][j],
                                            video_writer=video_writer):
                        break

            except tf.errors.OutOfRangeError:
                break

    print("release")
    video_writer.release()

    """-------------------------------------------------

    Debug code to visualize what's coming out of the input_fn
    
    """

    # with tf.Session() as sess:
    #     batch_size = 16
    #     input_fn = get_input_fn(input_file_names= dir_files, batch_size=batch_size,
    #                             num_epochs=None, shuffle=False, shard_size=3000, num_shards=2)
    #     next_element = input_fn()
    #     for i in range(40000):
    #
    #             try:
    #                 out = sess.run(next_element)
    #                 print(out[1]) # prints the label(s) in the batch
    #
    #                 # play frames in batch
    #                 for j, frame in enumerate(out[0]):
    #                     # It's assumed that the pixel values are decimals between -1 and 1.
    #                     # We put them back to between 0 and 255 before playing.
    #                     if player.display_frame(img=((np.squeeze(frame)*128) + 128).astype(np.uint8),
    #                                             debug_info=str(out[1][j]),
    #                                             milliseconds_time_to_wait=1):
    #                         break
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
    FLAGS.num_parallel_img_parsers = 8
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)