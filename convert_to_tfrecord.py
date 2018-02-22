import numpy as np
import tensorflow as tf
import os
import sys
import imageio
import argparse


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total:
        print()


def ensure_dir_exists(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(csv_path, frames_dir, out_dir, batch_size):
    csv = np.genfromtxt(csv_path, dtype=None, delimiter=',', names=True)
    csv_index = 0

    if len(csv) == 0:
        print('csv file %s is empty.' % csv_path)
        exit(1)

    ensure_dir_exists(out_dir)

    print('\nConverting from %s\nand from %s\nto %s' %
          tuple(map(lambda x: BColors.OKBLUE + x + BColors.ENDC, (csv_path, frames_dir, out_dir))))

    print('(%d frames to write)\n' % len(csv))

    num_shards = 0

    while csv_index < len(csv):
        start_index = csv_index
        end_index = min(csv_index + batch_size - 1, len(csv) - 1)
        num_to_write = end_index-start_index

        output_file = os.path.join(
            out_dir,
            '%02d_frames_%sto%s.tfrecords' % (num_shards, start_index, end_index)
        )
        with tf.python_io.TFRecordWriter(output_file) as record_writer:
            while csv_index <= end_index:
                    row = csv[csv_index]
                    filename = "{}/{}.jpg".format(frames_dir, row[0])
                    image_bytes = open(filename, "rb").read()
                    target = row[1]
                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'image': _bytes_feature(image_bytes),
                            'target': _float_feature(target)
                        }))
                    record_writer.write(example.SerializeToString())

                    print_progress_bar(csv_index - start_index, num_to_write, 'Generating %s' % output_file)
                    csv_index += 1

        rough_size_in_megabytes = os.path.getsize(output_file) / (1024 * 1024.0)
        print('%d frames written (%.2f MB) %stotal %d/%d%s' %
              (csv_index - start_index, rough_size_in_megabytes, BColors.OKGREEN, csv_index, len(csv), BColors.ENDC))
        num_shards += 1

    print(BColors.OKGREEN + 'Done' + BColors.ENDC)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--csv_path',
        type=str,
        help='Path to csv file with target estimator outputs.'
    )
    parser.add_argument(
        '--frames_dir',
        type=str,
        help='Path to directory of input images.'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='tfrecord_out',
        help='Directory to output TFRecords to'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=3000,
        help='Max number of images per tfrecord (making each record ~100MB is preferable)'
    )
    args, _ = parser.parse_known_args()

    if args.csv_path is None:
        print("--csv_path must be specified.")
        exit(1)
    if args.frames_dir is None:
        print("--frame_dir must be specified.")
        exit(1)

    main(args.csv_path, args.frames_dir, args.out_dir, args.batch_size)
