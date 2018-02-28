
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import gmtime, strftime

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import multiprocessing
import cv2

from transfer_learning import transfer_learning_cnn_fn
from cnn import cnn_fn
import player
from fast_predict import FastPredict

from hyperopt import fmin, tpe, hp, Trials

FLAGS = None
tf.set_random_seed(42)


def get_tfrecord_file_names_from_directory(dir):
    dir_files = os.listdir(dir)
    dir_files = sorted(filter(lambda f: str.isdigit(f.split('_')[0]), dir_files), key=lambda f: int(f.split('_')[0]))
    return list(map(lambda f: os.path.join(dir, f), dir_files))


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
                 return_full_size_image=False):
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
        file_names = tf.constant(input_file_names, dtype=tf.string, name='input_file_names')

        if shuffle:
            num_shards = len(input_file_names)

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

    train_file_names = get_tfrecord_file_names_from_directory(FLAGS.train_dir)
    val_file_names = get_tfrecord_file_names_from_directory(FLAGS.val_dir)

    train_input_fn = get_input_fn(input_file_names=train_file_names,
                                  batch_size=32,
                                  num_epochs=None,
                                  shuffle=True)

    val_input_fn = get_input_fn(input_file_names=val_file_names,
                                batch_size=1,
                                num_epochs=1,
                                shuffle=False)

    # Arbitrarily sticking a timestamp on the model_dirs to make each run different
    #   - probably want this to be hyperparameter specs later
    model_dir = 'tf_files/models/cnn-' + strftime("%Y-%m-%d-%H:%M:%S", gmtime()) + '/'


    #Objective function for hyperopt
    def objective(args):
        estimator_config = tf.estimator.RunConfig(
            save_summary_steps=100,        # Log a training summary (training loss by default) to tensorboard every n steps
            save_checkpoints_steps=10000,  # Stop and save a checkpoint every n steps
            keep_checkpoint_max=50         # How many checkpoints we save for this model before we start deleting old ones
        )

        model = tf.estimator.Estimator(
            model_fn=cnn_fn,
            params={
                "learning_rate": 0.0001,
                "optimizer": tf.train.AdamOptimizer,
                "hidden_units": ["Cool story bro"],
                'c1_size_filter': args['c1_size_filter'],
                'c2_size_filter': args['c2_size_filter'],
                'c3_size_filter': args['c3_size_filter'],
                'c4_size_filter': args['c4_size_filter'],
                'c5_size_filter': args['c5_size_filter'],
                'c1_size_kernel': args['c1_size_kernel'],
                'c2_size_kernel': args['c2_size_kernel'],
                'c3_size_kernel': args['c3_size_kernel'],
                'c4_size_kernel': args['c4_size_kernel'],
                'c5_size_kernel': args['c5_size_kernel'],
                'c1_stride_height': args['c1_stride_height'],
                'c2_stride_height': args['c2_stride_height'],
                'c3_stride_height': args['c3_stride_height'],
                'c4_stride_height': args['c4_stride_height'],
                'c5_stride_height': args['c5_stride_height'],
                'c1_stride_width': args['c1_stride_width'],
                'c2_stride_width': args['c2_stride_width'],
                'c3_stride_width': args['c3_stride_width'],
                'c4_stride_width': args['c4_stride_width'],
                'c5_stride_width': args['c5_stride_width'],
                'c1_activation': args['c1_activation'],
                'c2_activation': args['c2_activation'],
                'c3_activation': args['c3_activation'],
                'c4_activation': args['c4_activation'],
                'c5_activation': args['c5_activation'],
            },
            model_dir=model_dir,
            config=estimator_config
        )

        experiment = tf.contrib.learn.Experiment(estimator=model,
                                                 train_input_fn=train_input_fn,
                                                 train_steps=1000,
                                                 eval_input_fn=val_input_fn,
                                                 eval_steps=None,
                                                 checkpoint_and_export=False)

        # Setting 'checkpoint_and_export' to 'True' will cause checkpoints to be exported every n steps according to
        #   'save_checkpoints_steps' in the estimator's config. It will also cause experiment.train_and_evaluate() to
        #   run it's evaluation step (for us that's validation) whenever said checkpoints are exported.

        results = experiment.train_and_evaluate()
        return (float(results[0]['loss']))


    #TODO: Currently there is a bug with the space because the dimensions for the following layer needs to have dimensions
    #      compatibly with the previous layer. Not sure yet how to force this with hyperopt
    space = {
        # 'c1_size_filter': hp.choice('c1_size_filter', np.arange(1,80+1, dtype=int)),
        # 'c2_size_filter': hp.choice('c2_size_filter', np.arange(1,80+1, dtype=int)),
        # 'c3_size_filter': hp.choice('c3_size_filter', np.arange(1,80+1, dtype=int)),
        # 'c4_size_filter': hp.choice('c4_size_filter', np.arange(1,80+1, dtype=int)),
        # 'c5_size_filter': hp.choice('c5_size_filter', np.arange(1,80+1, dtype=int)),

        'c1_size_filter': hp.choice('c1_size_filter', [12, 24]),
        'c2_size_filter': hp.choice('c2_size_filter', [24, 36]),
        'c3_size_filter': hp.choice('c3_size_filter', [36, 48]),
        'c4_size_filter': hp.choice('c4_size_filter', [48, 64]),
        'c5_size_filter': hp.choice('c5_size_filter', [64, 76]),

        'c1_size_kernel': hp.choice('c1_size_kernel', [3, 5, 7]),
        'c2_size_kernel': hp.choice('c2_size_kernel', [3, 5, 7]),
        'c3_size_kernel': hp.choice('c3_size_kernel', [3, 5, 7]),
        'c4_size_kernel': hp.choice('c4_size_kernel', [3, 5, 7]),
        'c5_size_kernel': hp.choice('c5_size_kernel', [3, 5, 7]),
        'c1_stride_height': hp.choice('c1_stride_height', [1, 2, 3]),
        'c2_stride_height': hp.choice('c2_stride_height', [1, 2, 3]),
        'c3_stride_height': hp.choice('c3_stride_height', [1, 2, 3]),
        'c4_stride_height': hp.choice('c4_stride_height', [1, 2, 3]),
        'c5_stride_height': hp.choice('c5_stride_height', [1, 2, 3]),
        'c1_stride_width': hp.choice('c1_stride_width', [1, 2, 3]),
        'c2_stride_width': hp.choice('c2_stride_width', [1, 2, 3]),
        'c3_stride_width': hp.choice('c3_stride_width', [1, 2, 3]),
        'c4_stride_width': hp.choice('c4_stride_width', [1, 2, 3]),
        'c5_stride_width': hp.choice('c5_stride_width', [1, 2, 3]),
        'c1_activation': hp.choice('c1_activation', [tf.nn.relu, tf.nn.elu]),
        'c2_activation': hp.choice('c2_activation', [tf.nn.relu, tf.nn.elu]),
        'c3_activation': hp.choice('c3_activation', [tf.nn.relu, tf.nn.elu]),
        'c4_activation': hp.choice('c4_activation', [tf.nn.relu, tf.nn.elu]),
        'c5_activation': hp.choice('c5_activation', [tf.nn.relu, tf.nn.elu]),
    }

    # #IGNORE. For quick copy and paste
    # 'c1_size_filter':
    # 'c2_size_filter':
    # 'c3_size_filter':
    # 'c4_size_filter':
    # 'c5_size_filter':
    # 'c1_size_kernel':
    # 'c2_size_kernel':
    # 'c3_size_kernel':
    # 'c4_size_kernel':
    # 'c5_size_kernel':
    # 'c1_stride_height':
    # 'c2_stride_height':
    # 'c3_stride_height':
    # 'c4_stride_height':
    # 'c5_stride_height':
    # 'c1_stride_width':
    # 'c2_stride_width':
    # 'c3_stride_width':
    # 'c4_stride_width':
    # 'c5_stride_width':
    # 'c1_activation':
    # 'c2_activation':
    # 'c3_activation':
    # 'c4_activation':
    # 'c5_activation':

    max_evals = 30
    trials = Trials()
    #Hyperopt for optimizing NN parameters
    best = fmin(fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals, # how many parameter combinations we want to try
        trials=trials,
        )

    print("\n\n\nBest parameters found: {}\n\n\n".format(best))






    """-------------------------------------------------
    
    Debug Code to run the net on images and visualize real time the predictions on video. 
    The video is also saved to 'predictions.mp4'
    
    """

    # # TODO: YOUR GOING TO WANT TO COMMENT THIS VIDEO PLAYER STUFF OUT WHEN RUNNING IN THE CLOUD
    #
    # fast_predict = FastPredict(model)
    #
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # video_writer = None
    #
    # predict_sess = tf.Session()
    # input_fn = get_input_fn(input_file_names=val_file_names, batch_size=1, num_epochs=None,
    #                         shuffle=False, return_full_size_image=True)
    # next_element = input_fn()
    # for i in range(10000):
    #
    #         try:
    #             out = predict_sess.run(next_element)
    #
    #             predictions = list(fast_predict.predict(out[0]))
    #
    #             if video_writer is None:
    #                 video_writer = cv2.VideoWriter("predictions.mp4", fourcc, 40.0, (out[2].shape[2], out[2].shape[1]))
    #
    #             # play frames in batch
    #             for j, frame in enumerate(out[2]):
    #                 # It's assumed that the pixel values are decimals between -1 and 1.
    #                 # We put them back to between 0 and 255 before playing.
    #                 if player.display_frame(img=(np.squeeze(frame)).astype(np.uint8),
    #                                         debug_info=str(out[1][j]),
    #                                         milliseconds_time_to_wait=1,
    #                                         predicted_angle=predictions[j]['angle'],
    #                                         true_angle=out[1][j],
    #                                         video_writer=video_writer):
    #                     break
    #
    #         except tf.errors.OutOfRangeError:
    #             break
    #
    # print("release")
    # video_writer.release()

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
        '--train_dir',
        '-t',
        type=str,
        help='Path to training data tfrecords directory'
    )

    parser.add_argument(
        '--val_dir',
        '-v',
        type=str,
        help='Path to validation data tfrecords directory'
    )

    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.train_dir is None:
        print("\033[93m--train_dir wasn't specified.\033[0m")

    if FLAGS.val_dir is None:
        print("\033[93m--val_dir wasn't specified.\033[0m")

    tf.logging.set_verbosity(tf.logging.INFO)

    # You may want to adjust this to be just cpu_count if you're using a server with few other processes
    FLAGS.num_parallel_img_parsers = multiprocessing.cpu_count() - 2

    # Any arguments not explicitly parsed by the parser code above are sent in as part of argv to main
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
