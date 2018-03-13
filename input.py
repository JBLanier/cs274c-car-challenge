
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
from rnn import rnn_fn
import player
import data
from fast_predict import FastPredict
from rnn_hook import RNNStateHook

FLAGS = None
tf.set_random_seed(42)


def get_tfrecord_file_names_from_directory(dir):
    dir_files = os.listdir(dir)
    dir_files = sorted(filter(lambda f: str.isdigit(f.split('_')[0]), dir_files), key=lambda f: int(f.split('_')[0]))
    return list(map(lambda f: os.path.join(dir, f), dir_files))

def main(argv):
    train_file_names = get_tfrecord_file_names_from_directory(FLAGS.train_dir)
    val_file_names = get_tfrecord_file_names_from_directory(FLAGS.val_dir)

    train_sequence_length = 16
    train_batch_size = 16
    rnn_size = 50

    train_input_fn = data.get_input_fn(input_file_names=train_file_names,
                                       batch_size=train_batch_size,
                                       num_epochs=50,
                                       shuffle=True,
                                       window_size=train_sequence_length,
                                       stride=1,
                                       apply_distortions=True)

    val_input_fn = data.get_input_fn(input_file_names=val_file_names,
                                     batch_size=train_batch_size,
                                     num_epochs=1,
                                     window_size=train_sequence_length,
                                     stride=1,
                                     shuffle=False)

    # Arbitrarily sticking a timestamp on the model_dirs to make each run different
    # - probably want this to be hyperparameter specs later
    model_dir = 'tf_files/models/cnn-' + strftime("%Y-%m-%d-%H:%M:%S", gmtime()) + '/'

    estimator_config = tf.estimator.RunConfig(
        save_summary_steps=100,        # Log a training summary (training loss by default) to tensorboard every n steps
        save_checkpoints_steps=10000,  # Stop and save a checkpoint every n steps
        keep_checkpoint_max=50,        # How many checkpoints we save for this model before we start deleting old ones
        save_checkpoints_secs=None
    )

    model = tf.estimator.Estimator(model_fn=rnn_fn,
                                   config=estimator_config,
                                   params={"learning_rate": 0.0001,
                                           "optimizer": tf.train.AdamOptimizer,
                                           "train_sequence_length": train_sequence_length,
                                           "rnn_size": rnn_size},
                                   model_dir=model_dir)

    state_hook = RNNStateHook(state_size=rnn_size)

    experiment = tf.contrib.learn.Experiment(estimator=model,
                                             train_input_fn=train_input_fn,
                                             train_steps=None,
                                             eval_input_fn=val_input_fn,
                                             eval_steps=None,
                                             eval_hooks=[],
                                             checkpoint_and_export=True,
                                             )
    # Setting 'checkpoint_and_export' to 'True' will cause checkpoints to be exported every n steps according to
    # 'save_checkpoints_steps' in the estimator's config. It will also cause experiment.train_and_evaluate() to
    # run it's evaluation step (for us that's validation) whenever said checkpoints are exported.

    experiment.train_and_evaluate()  #-0.38883182

    """-------------------------------------------------
    Debug Code to run the net on images and visualize real time the predictions on video. 
    The video is also saved to 'predictions.mp4'
    """

    # TODO: YOUR GOING TO WANT TO COMMENT THIS VIDEO PLAYER STUFF OUT WHEN RUNNING IN THE CLOUD

    model = tf.estimator.Estimator(model_fn=rnn_fn,
                                   config=estimator_config,
                                   params={"learning_rate": 0.0001,
                                           "optimizer": tf.train.AdamOptimizer,
                                           "rnn_size": rnn_size},
                                   model_dir=model_dir)   #"tf_files/models/cnn-2018-03-13-06:32:20/"

    fast_predict = FastPredict(model, hooks=[])

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    video_writer = None

    predict_sess = tf.Session()
    print("call visualization input fn")
    input_fn = data.get_input_fn(input_file_names=val_file_names,
                                       batch_size=1,
                                       num_epochs=1,
                                       window_size=train_sequence_length,
                                       stride=1,
                                       shuffle=False,
                                       return_full_size_image=True,
                                    )
    next_element = input_fn()
    print("done")
    for i in range(100000):
            try:
                out = predict_sess.run(next_element)
                predictions = list(fast_predict.sequence_predict(out[0]))
                print("PREDICTION: {}".format(predictions))
                if video_writer is None:
                    video_writer = cv2.VideoWriter("predictions.mp4", fourcc, 40.0, (out[2][0,-1].shape[1], out[2][0,-1].shape[0]))

                frame = out[2][0,-1]
                # It's assumed that the pixel values are decimals between -1 and 1.
                # We put them back to between 0 and 255 before playing.
                if player.display_frame(img=(np.squeeze(frame)).astype(np.uint8),
                                        debug_info=str(out[1][0,-1]),
                                        milliseconds_time_to_wait=1,
                                        predicted_angle=predictions[-1]['angle'],
                                        true_angle=out[1][0,-1],
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
