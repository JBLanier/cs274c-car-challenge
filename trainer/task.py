import argparse
import os
import math
import multiprocessing
import glob

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
from tensorflow.python.client import device_lib


import trainer.model as model


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

    allowed_kernel_sizes = [3, 5, 7, 12]
    allowed_filter_nums = [8, 16, 24, 32]
    allowed_strides = [1, 2, 4]

    conv_layers = []
    filter_num_scale_factor = hparams.filter_num_scale_factor
    kernel_size = allowed_kernel_sizes[hparams.conv1_kernel_size]
    filter_num = allowed_filter_nums[hparams.conv1_filter_num]
    stride = allowed_strides[hparams.conv1_stride]

    kernel_decrement_layers = []
    for i in range(hparams.num_kernal_size_decrements):
        decrement_layer = int((hparams.last_decrement_layer / (i + 1)) * hparams.num_conv_layers)
        kernel_decrement_layers.append(decrement_layer)

    print("{} Convolutional Layers Requested".format(hparams.num_conv_layers))
    print("\n-----Kernal Decrements will be performed at layers: {}".format(kernel_decrement_layers))

    input_size = (66, 200)

    print("\n-----CONV LAYERS: ")
    for i in range(hparams.num_conv_layers):
        if input_size[0] < kernel_size or input_size[1] < kernel_size:
            print("\nNo more conv layers will be made because the next input size {} is too small for kernel size {}"
                  .format(input_size, kernel_size))
            break

        conv_layers.append((kernel_size, filter_num, stride))

        print("\n--Layer {}\nInput Size: {} Kernel Size: {} Num Filters: {} Stride: {}"
              .format(i+1, input_size, kernel_size, filter_num, stride))

        input_size = calculate_conv_output_size(input_size, kernel_size, stride)
        print("  Output Size {}".format(input_size))

        if i+1 in kernel_decrement_layers:
            print("  (decrementing kernel size/stride after this layer)")
            kernel_size = allowed_kernel_sizes[max(0, allowed_kernel_sizes.index(kernel_size)-1)]
            stride = allowed_strides[max(0, allowed_strides.index(stride)-1)]

        filter_num = int(filter_num * max(1, filter_num_scale_factor))
        filter_num_scale_factor = filter_num_scale_factor * hparams.filter_num_scale_decay

    print("{} Dense Layers Requested".format(hparams.num_dense_layers))
    print("\n-----Dense LAYERS: ")
    dense_units = []
    for i in range(hparams.num_dense_layers):
        units = max(1, int(hparams.first_dense_layer_size * hparams.dense_scale_factor ** i))
        if units == 1:
            print("\nNo more dense layers will be made because the next hidden layer would only have one neuron.")
            break
        dense_units.append(units)

    print(dense_units)
    print()

    train_input = model.get_input_fn(input_file_names=hparams.train_files,
                               batch_size=hparams.train_batch_size,
                               num_epochs=hparams.num_epochs,
                               shuffle=True)

    eval_input = model.get_input_fn(input_file_names=hparams.eval_files,
                              batch_size=hparams.eval_batch_size,
                              num_epochs=1,
                              shuffle=False)

    model_fn = model.get_model_fn(
        conv_layers=conv_layers,
        # Construct dense layer sizes with exponential decay
        dense_units=dense_units,
        learning_rate=hparams.learning_rate)

    if hparams.save_checkpoints:

        estimator_config = tf.estimator.RunConfig(
            save_summary_steps=100,  # Log a training summary (training loss by default) to tensorboard every n steps
            save_checkpoints_steps=25000,  # Stop and save a checkpoint every n steps
            keep_checkpoint_max=1,  # How many checkpoints we save for this model before we start deleting old ones
            save_checkpoints_secs=None  # Don't save any checkpoints based on how long it's been
        )

        estimator = tf.estimator.Estimator(model_fn=model_fn,
                                           model_dir=hparams.job_dir,
                                           params={'model_dir': hparams.job_dir},
                                           config=estimator_config)

        experiment = tf.contrib.learn.Experiment(estimator=estimator,
                                                 train_input_fn=train_input,
                                                 train_steps=hparams.train_steps,
                                                 eval_input_fn=eval_input,
                                                 eval_steps=hparams.eval_steps,
                                                 checkpoint_and_export=True)

        # Setting 'checkpoint_and_export' to 'True' will cause checkpoints to be exported every n steps according to
        #   'save_checkpoints_steps' in the estimator's config. It will also cause experiment.train_and_evaluate() to
        #   run it's evaluation step (for us that's validation) whenever said checkpoints are exported.

        experiment.train_and_evaluate()

    else:

        estimator_config = tf.estimator.RunConfig(
            save_summary_steps=100,  # Log a training summary (training loss by default) to tensorboard every n steps
            log_step_count_steps=100,
            save_checkpoints_steps=50000,  # Stop and save a checkpoint every n steps
            keep_checkpoint_max=1,  # How many checkpoints we save for this model before we start deleting old ones
            save_checkpoints_secs=None  # Don't save any checkpoints based on how long it's been
        )

        estimator = tf.estimator.Estimator(model_fn=model_fn,
                                           model_dir=hparams.job_dir,
                                           params={'model_dir': hparams.job_dir},
                                           config=estimator_config)

        estimator.train(input_fn=train_input, steps=hparams.train_steps)
        estimator.evaluate(input_fn=eval_input,
                           steps=hparams.eval_steps,
                           checkpoint_path=None,
                           name='intermediate_export')

        print("Removing Checkpoints")
        if hparams.job_dir is None or len(hparams.job_dir) == 0:
            print("\n\n\nERROR, JOB_DIR is empty or None")
            exit(1)

        for filename in glob.glob("{}/model*".format(hparams.job_dir)):
            os.remove(filename)
        os.remove("{}/checkpoint".format(hparams.job_dir))

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
        '--learning-rate',
        help='Learning rate for the optimizer',
        default=0.0001,
        type=float
    )
    parser.add_argument(
        '--num-conv-layers',
        help='',
        default=5,
        type=int
    )
    parser.add_argument(
        '--conv1-kernel-size',
        help='',
        default=1,
        type=int
    )

    parser.add_argument(
        '--conv1-filter-num',
        help='',
        default=2,
        type=int
    )

    parser.add_argument(
        '--conv1-stride',
        help='',
        default=1,
        type=int
    )

    parser.add_argument(
        '--filter-num-scale-factor',
        help='',
        default=1.5,
        type=float
    )

    parser.add_argument(
        '--filter-num-scale-decay',
        help='',
        default=0.8,
        type=float
    )

    parser.add_argument(
        '--num-kernal-size-decrements',
        help='',
        default=1,
        type=int
    )

    parser.add_argument(
        '--last-decrement-layer',
        help='',
        default=0.6,
        type=float
    )

    parser.add_argument(
        '--first-dense-layer-size',
        help='',
        default=1200,
        type=int
    )

    parser.add_argument(
        '--num-dense-layers',
        help='',
        default=4,
        type=int
    )

    parser.add_argument(
        '--dense-scale-factor',
        help='How quickly should the size of the layers in the DNN decay',
        default=0.7,
        type=float
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
