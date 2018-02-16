from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
from tensorflow.python.platform import gfile


def maybe_download_and_extract(data_url, model_dir):
    """Download and extract model tar file.
    If the pretrained model we're using doesn't already exist, this function
    downloads it from the TensorFlow.org website and unpacks it into a directory.
    Args:
      data_url: Web location of the tar file containing the pretrained model.
      model_dir: Dictionary where the model might be/will be saved.
    """
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        tf.logging.info('Successfully downloaded %s, %d', filename, statinfo.st_size,
                        'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def create_model_info(architecture):
    """Given the name of a model architecture, returns information about it.
    There are different base image recognition pretrained models that can be
    retrained using transfer learning, and this function translates from the name
    of a model to the attributes that are needed to download and train with it.
    Args:
      architecture: Name of a model architecture.
    Returns:
      Dictionary of information about the model, or None if the name isn't
      recognized
    Raises:
      ValueError: If architecture name is unknown.
    """
    architecture = architecture.lower()
    if architecture == 'inception_v3':
        # pylint: disable=line-too-long
        data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        # pylint: enable=line-too-long
        bottleneck_tensor_name = 'pool_3:0'
        # bottleneck_tensor_name = 'pool_3/(pool_3):0'
        bottleneck_tensor_size = 2048
        input_width = 299
        input_height = 299
        input_depth = 3
        resized_input_tensor_name = 'Mul:0'
        model_file_name = 'classify_image_graph_def.pb'
        input_mean = 128
        input_std = 128
    elif architecture.startswith('mobilenet_'):
        parts = architecture.split('_')
        if len(parts) != 3 and len(parts) != 4:
            tf.logging.error("Couldn't understand architecture name '%s'",
                             architecture)
            return None
        version_string = parts[1]
        if (version_string != '1.0' and version_string != '0.75' and
                    version_string != '0.50' and version_string != '0.25'):
            tf.logging.error(
                """"The Mobilenet version should be '1.0', '0.75', '0.50', or '0.25',
        but found '%s' for architecture '%s'""",
                version_string, architecture)
            return None
        size_string = parts[2]
        if (size_string != '224' and size_string != '192' and
                    size_string != '160' and size_string != '128'):
            tf.logging.error(
                """The Mobilenet input size should be '224', '192', '160', or '128',
       but found '%s' for architecture '%s'""",
                size_string, architecture)
            return None
        if len(parts) == 3:
            is_quantized = False
        else:
            if parts[3] != 'quantized':
                tf.logging.error(
                    "Couldn't understand architecture suffix '%s' for '%s'", parts[3],
                    architecture)
                return None
            is_quantized = True
        data_url = 'http://download.tensorflow.org/models/mobilenet_v1_'
        data_url += version_string + '_' + size_string + '_frozen.tgz'
        bottleneck_tensor_name = 'MobilenetV1/Predictions/Reshape:0'
        bottleneck_tensor_size = 1001
        input_width = int(size_string)
        input_height = int(size_string)
        input_depth = 3
        resized_input_tensor_name = 'input:0'
        if is_quantized:
            model_base_name = 'quantized_graph.pb'
        else:
            model_base_name = 'frozen_graph.pb'
        model_dir_name = 'mobilenet_v1_' + version_string + '_' + size_string
        model_file_name = os.path.join(model_dir_name, model_base_name)
        input_mean = 127.5
        input_std = 127.5
    else:
        tf.logging.error("Couldn't understand architecture name '%s'", architecture)
        raise ValueError('Unknown architecture', architecture)

    return {
        'data_url': data_url,
        'bottleneck_tensor_name': bottleneck_tensor_name,
        'bottleneck_tensor_size': bottleneck_tensor_size,
        'input_width': input_width,
        'input_height': input_height,
        'input_depth': input_depth,
        'resized_input_tensor_name': resized_input_tensor_name,
        'model_file_name': model_file_name,
        'input_mean': input_mean,
        'input_std': input_std,
    }


def transfer_learning_cnn_fn(features, labels, mode, params):

    # Build the pre-trained cnn portion of the net
    model_info = create_model_info(params['architecture'])
    maybe_download_and_extract(model_info['data_url'], params['model_dir'])

    model_path = os.path.join(params['model_dir'], model_info['model_file_name'])

    bottleneck_tensor=None

    with tf.name_scope(params['architecture']):
        with gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            bottleneck_tensor = tf.squeeze(tf.import_graph_def(
                graph_def,
                name='imported_model',
                return_elements=[
                    model_info['bottleneck_tensor_name'],
                ],
                input_map={
                    model_info['resized_input_tensor_name']: features
                }
            )[0], [1, 2])


    # Add our own fully-connected layers
    fc1 = tf.layers.dense(bottleneck_tensor, units=200, activation=tf.nn.relu, name="fc_1")
    fc2 = tf.layers.dense(fc1, units=100, activation=tf.nn.relu, name="fc_2")
    output_layer = tf.layers.dense(fc2, units=1, activation=None, name="output")

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

    # Calculate root mean squared error
    print(labels)
    print(predictions)
    rmse = tf.metrics.root_mean_squared_error(labels, predictions)

    # Add the rmse to the collection of evaluation metrics.
    eval_metrics = {"rmse": rmse}

    return tf.estimator.EstimatorSpec(
        mode=mode,
        # Report sum of error for compatibility with pre-made estimators
        loss=total_loss,
        eval_metric_ops=eval_metrics)


    # tf.summary.scalar('RMSE', root_mean_squared_error)
    # # tf.summary.scalar('Ground_truth', tf.reshape(ground_truth_input,[]))
    # # tf.summary.scalar('Prediction', tf.reshape(predictions,[]))
    #
    # with tf.name_scope('train'):
    #     optimizer = tf.train.GradientDescentOptimizer(params['learning_rate'])
    #     train_step = optimizer.minimize(root_mean_squared_error)




def get_model_fn(num_gpus, variable_strategy, num_workers):
    """Returns a function that will build the resnet model."""

    def _resnet_model_fn(features, labels, mode, params):
        """Resnet model body.
        Support single host, one or more GPU training. Parameter distribution can
        be either one of the following scheme.
        1. CPU is the parameter server and manages gradient updates.
        2. Parameters are distributed evenly across all GPUs, and the first GPU
           manages gradient updates.
        Args:
          features: a list of tensors, one for each tower
          labels: a list of tensors, one for each tower
          mode: ModeKeys.TRAIN or EVAL
          params: Hyperparameters suitable for tuning
        Returns:
          A EstimatorSpec object.
        """
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        weight_decay = params.weight_decay
        momentum = params.momentum

        tower_features = features
        tower_labels = labels
        tower_losses = []
        tower_gradvars = []
        tower_preds = []

        # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
        # on CPU. The exception is Intel MKL on CPU which is optimal with
        # channels_last.
        data_format = params.data_format
        if not data_format:
            if num_gpus == 0:
                data_format = 'channels_last'
            else:
                data_format = 'channels_first'

        if num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = num_gpus
            device_type = 'gpu'

        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if variable_strategy == 'CPU':
                device_setter = cifar10_utils.local_device_setter(
                    worker_device=worker_device)
            elif variable_strategy == 'GPU':
                device_setter = cifar10_utils.local_device_setter(
                    ps_device_type='gpu',
                    worker_device=worker_device,
                    ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                        num_gpus, tf.contrib.training.byte_size_load_fn))
            with tf.variable_scope('resnet', reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):
                        loss, gradvars, preds = _tower_fn(
                            is_training, weight_decay, tower_features[i], tower_labels[i],
                            data_format, params.num_layers, params.batch_norm_decay,
                            params.batch_norm_epsilon)
                        tower_losses.append(loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                            # Only trigger batch_norm moving mean and variance update from
                            # the 1st tower. Ideally, we should grab the updates from all
                            # towers but these stats accumulate extremely fast so we can
                            # ignore the other stats from the other towers without
                            # significant detriment.
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                           name_scope)

        # Now compute global loss and gradients.
        gradvars = []
        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in six.iteritems(all_grads):
                # Average gradients on the same device as the variables
                # to which they apply.
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                gradvars.append((avg_grad, var))

        # Device that runs the ops to apply global gradient updates.
        consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
        with tf.device(consolidation_device):
            # Suggested learning rate scheduling from
            # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py#L155
            num_batches_per_epoch = cifar10.Cifar10DataSet.num_examples_per_epoch(
                'train') // (params.train_batch_size * num_workers)
            boundaries = [
                num_batches_per_epoch * x
                for x in np.array([82, 123, 300], dtype=np.int64)
            ]
            staged_lr = [params.learning_rate * x for x in [1, 0.1, 0.01, 0.002]]

            learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
                                                        boundaries, staged_lr)

            loss = tf.reduce_mean(tower_losses, name='loss')

            examples_sec_hook = cifar10_utils.ExamplesPerSecondHook(
                params.train_batch_size, every_n_steps=10)

            tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}

            logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=100)

            train_hooks = [logging_hook, examples_sec_hook]

            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=momentum)

            if params.sync:
                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer, replicas_to_aggregate=num_workers)
                sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
                train_hooks.append(sync_replicas_hook)

            # Create single grouped train op
            train_op = [
                optimizer.apply_gradients(
                    gradvars, global_step=tf.train.get_global_step())
            ]
            train_op.extend(update_ops)
            train_op = tf.group(*train_op)

            predictions = {
                'classes':
                    tf.concat([p['classes'] for p in tower_preds], axis=0),
                'probabilities':
                    tf.concat([p['probabilities'] for p in tower_preds], axis=0)
            }
            stacked_labels = tf.concat(labels, axis=0)
            metrics = {
                'accuracy':
                    tf.metrics.accuracy(stacked_labels, predictions['classes'])
            }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            training_hooks=train_hooks,
            eval_metric_ops=metrics)

    return _resnet_model_fn
