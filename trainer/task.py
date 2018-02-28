import argparse
import multiprocessing
import os

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

import trainer.model as model


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


def run_experiment(hparams):
    """Run the training and evaluate using the high level API"""

    # train_file_names = get_tfrecord_file_names_from_directory(hparams.train_files[0])
    # val_file_names = get_tfrecord_file_names_from_directory(hparams.eval_files[0])

    train_input = get_input_fn(input_file_names=hparams.train_files,
                               batch_size=hparams.train_batch_size,
                               num_epochs=hparams.num_epochs,
                               shuffle=True)

    eval_input = get_input_fn(input_file_names=hparams.eval_files,
                              batch_size=hparams.eval_batch_size,
                              num_epochs=1,
                              shuffle=False)

    train_spec = tf.estimator.TrainSpec(train_input,
                                        max_steps=hparams.train_steps
                                        )

    # exporter = tf.estimator.FinalExporter('car',
    #         model.SERVING_FUNCTIONS[hparams.export_format])

    eval_spec = tf.estimator.EvalSpec(eval_input,
                                      steps=hparams.eval_steps,
                                      exporters=[],
                                      name='core'
                                      )

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
    print("\n-----KD LAYERS")
    print(kernel_decrement_layers)

    for i in range(hparams.num_conv_layers):
        conv_layers.append((kernel_size, filter_num, stride))

        if i+1 in kernel_decrement_layers:
            print("decrementing kernel size")
            kernel_size = allowed_kernel_sizes[max(0, allowed_kernel_sizes.index(kernel_size)-1)]
            stride = allowed_strides[max(0, allowed_strides.index(stride)-1)]

        filter_num = int(filter_num * max(1, filter_num_scale_factor))
        filter_num_scale_factor = filter_num_scale_factor * hparams.filter_num_scale_decay

    print("\n-----CONV LAYERS: ")
    for layer in conv_layers:
        print(layer)

    model_fn = model.get_model_fn(
        conv_layers=conv_layers,
        # Construct dense layer sizes with exponential decay
        dense_units=[max(1, int(hparams.first_dense_layer_size * hparams.dense_scale_factor ** i))
                     for i in range(hparams.num_dense_layers)],
        learning_rate=hparams.learning_rate)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=hparams.job_dir,
                                       params={'model_dir': hparams.job_dir})

    tf.estimator.train_and_evaluate(estimator,
                                    train_spec,
                                    eval_spec)

    print("Done (: ")

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
        '--num_kernal_size_decrements',
        help='',
        default=1,
        type=int
    )

    parser.add_argument(
        '--last_decrement_layer',
        help='',
        default=0.6,
        type=float
    )

    parser.add_argument(
        '--first_dense_layer_size',
        help='',
        default=1200,
        type=int
    )

    parser.add_argument(
        '--num_dense_layers',
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
    parser.add_argument(
        '--export-format',
        help='The input format of the exported SavedModel binary',
        choices=['JSON', 'CSV', 'EXAMPLE'],
        default='JSON'
    )

    args = parser.parse_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    # Run the training job
    hparams = hparam.HParams(**args.__dict__)
    run_experiment(hparams)
