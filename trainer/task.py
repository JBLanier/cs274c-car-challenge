import argparse
import os
import math
import multiprocessing
import glob
import shutil

import tensorflow as tf
from trainer.low_performance_stop_hook import LowPerformanceStopHook
from tensorflow.contrib.training.python.training import hparam
from tensorflow.python.client import device_lib
import numpy as np

import player

import trainer.model as model


def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def get_tfrecord_file_names_from_directory(dir):
    dir_files = os.listdir(dir)
    dir_files = sorted(filter(lambda f: str.isdigit(f.split('_')[0]), dir_files), key=lambda f: int(f.split('_')[0]))
    return list(map(lambda f: os.path.join(dir, f), dir_files))


def calculate_conv_output_size(input_size, kernel_size, stride):
        out_height = math.ceil(float(input_size[0] - kernel_size + 1) / float(stride))
        out_width = math.ceil(float(input_size[1] - kernel_size + 1) / float(stride))
        return out_height, out_width


def run_experiment(hparams):
    """Run the training and evaluate using the high level API"""

    # print("OFFSET IS {}".format(hparams.offset))
    #
    # with tf.Session() as sess:
    #     batch_size = 16
    #     input_fn = model.get_input_fn(input_file_names=hparams.train_files,
    #                           batch_size=hparams.eval_batch_size,
    #                           num_epochs=1,
    #                           shuffle=True,
    #                           return_full_size_image=True, offset=hparams.offset)
    #     next_element = input_fn()
    #     for i in range(40000):
    #
    #             try:
    #                 out = sess.run(next_element)
    #                 print(out[1]) # prints the label(s) in the batch
    #
    #                 # play frames in batch
    #                 for j, frame in enumerate(out[2]):
    #                     # It's assumed that the pixel values are decimals between -1 and 1.
    #                     # We put them back to between 0 and 255 before playing.
    #                     if player.display_frame(img=frame,
    #                                             debug_info=str(out[1][j]),
    #                                             true_angle=out[1],
    #                                             milliseconds_time_to_wait=1):
    #                         break
    #             except tf.errors.OutOfRangeError:
    #                 break

    train_input = model.get_input_fn(input_file_names=hparams.train_files,
                               batch_size=hparams.train_batch_size,
                               num_epochs=hparams.num_epochs,
                               shuffle=True,
                               offset=hparams.offset)

    eval_input = model.get_input_fn(input_file_names=hparams.eval_files,
                              batch_size=hparams.eval_batch_size,
                              num_epochs=1,
                              shuffle=False)

    model_fn = model.get_model_fn()

    keep_checkpoints_max = 1
    if hparams.save_checkpoints:
        keep_checkpoints_max = 10

    estimator_config = tf.estimator.RunConfig(
        save_summary_steps=100,  # Log a training summary (training loss by default) to tensorboard every n steps
        save_checkpoints_steps=10000,  # Stop and save a checkpoint every n steps
        keep_checkpoint_max=keep_checkpoints_max,  # How many checkpoints we save for this model before we start deleting old ones
        save_checkpoints_secs=None  # Don't save any checkpoints based on how long it's been
    )

    # kills the process if it runs too slow
    low_perf_hook = LowPerformanceStopHook(min_steps_per_sec=640/hparams.train_batch_size)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=hparams.job_dir,
                                       params={'model_dir': hparams.job_dir},
                                       config=estimator_config)

    experiment = tf.contrib.learn.Experiment(estimator=estimator,
                                             train_input_fn=train_input,
                                             train_steps=hparams.train_steps,
                                             eval_input_fn=eval_input,
                                             eval_steps=hparams.eval_steps,
                                             train_monitors=[low_perf_hook],
                                             checkpoint_and_export=True)

    # Setting 'checkpoint_and_export' to 'True' will cause checkpoints to be exported every n steps according to
    #   'save_checkpoints_steps' in the estimator's config. It will also cause experiment.train_and_evaluate() to
    #   run it's evaluation step (for us that's validation) whenever said checkpoints are exported.

    experiment.train_and_evaluate()

    #
    # estimator_config = tf.estimator.RunConfig(
    #     save_summary_steps=100,  # Log a training summary (training loss by default) to tensorboard every n steps
    #     log_step_count_steps=100,
    #     save_checkpoints_steps=50000,  # Stop and save a checkpoint every n steps
    #     keep_checkpoint_max=1,  # How many checkpoints we save for this model before we start deleting old ones
    #     save_checkpoints_secs=None  # Don't save any checkpoints based on how long it's been
    # )
    #
    # estimator = tf.estimator.Estimator(model_fn=model_fn,
    #                                    model_dir=hparams.job_dir,
    #                                    params={'model_dir': hparams.job_dir},
    #                                    config=estimator_config)
    #
    # estimator.train(input_fn=train_input, steps=hparams.train_steps)
    # estimator.evaluate(input_fn=eval_input,
    #                    steps=hparams.eval_steps,
    #                    checkpoint_path=None,
    #                    name='intermediate_export')

    if not hparams.save_checkpoints:
        print("Removing Checkpoints")
        if hparams.job_dir is None or len(hparams.job_dir) == 0:
            print("\n\n\nERROR, JOB_DIR is empty or None")
            exit(1)

        for filename in glob.glob("{}/model*".format(hparams.job_dir)):
            remove(filename)
        for filename in glob.glob("{}/checkpoint".format(hparams.job_dir)):
            remove(filename)

    print("Done training and evaluating.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--train-files',
        help='GCS or local paths to training data',
        nargs='+',
        required=True
    )

    parser.add_argument(
        '--save-checkpoints',
        help='',
        type=bool,
    )

    parser.add_argument(
        '--num-epochs',
        help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
        type=int,
    )
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=32
    )
    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=1
    )
    parser.add_argument(
        '--eval-files',
        help='GCS or local paths to evaluation data',
        nargs='+',
        required=True
    )
    # Training arguments

    parser.add_argument(
        '--offset',
        help='label offset for left and right camera images',
        type=float,
        default=0
    )

    # end training arguments

    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
        help='Set logging verbosity'
    )
    # Experiment arguments
    parser.add_argument(
        '--train-steps',
        help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.\
      """,
        type=int
    )
    parser.add_argument(
        '--eval-steps',
        help="""\
      Number of steps to run evalution for at each checkpoint.
      If unspecified will run until the input from --eval-files is exhausted
      """,
        default=None,
        type=int
    )

    args = parser.parse_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    print("\nWorking with {} cores.".format(multiprocessing.cpu_count()))
    print("GPUS: {}\n".format(get_available_gpus()))

    # Run the training job
    hparams = hparam.HParams(**args.__dict__)
    run_experiment(hparams)
