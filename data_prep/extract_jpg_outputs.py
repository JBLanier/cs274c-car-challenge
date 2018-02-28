# Extracts clean separate episodes with center frames and corresponding steering values to separate folders.
# Each episode may end up having one or two duplicate frames at the beginning, which are from the episode before.

# 'Quick' messy script that may or may not ever need to be run again

import numpy as np
import os
from shutil import copyfile
import csv
from os.path import basename

images_dir_path = "/media/jb/m2_linux/additional data/UdacitySDC_ElCamino/2/center/"
interpolated_csv = np.genfromtxt("/media/jb/m2_linux/additional data/UdacitySDC_ElCamino/2/interpolated.csv", dtype=None, delimiter=',', names=True)

set_start_times = [0]  # [1479424214743813120, 1479424438022508032, 1479425728713969920, 1479425832947890944, 1479426201119651072]
set_end_times = [1579424435955945984]  # [1479424435955945984, 1479425229044132096, 1479425828505740800, 1479426045822525952, 1479426572360982784]
set_names = ["el_camino_back"]

sets = zip(set_start_times,set_end_times,set_names)

for (set_start_time, set_end_time, set_name) in sets:

    print("Working on {}".format(set_name))

    output_directory_path = "/media/jb/m2_linux/additional data/{}_prepared".format(set_name)

    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)

    if not os.path.exists(output_directory_path + "/frames"):
        os.makedirs(output_directory_path + "/frames")

    input_directory = os.fsencode(images_dir_path)
    csv_index = 0  # 'center' images are every third image in interpolated.csv
    count = 0

    missed = 0
    prev = ""
    with open(output_directory_path + "/data.csv", 'w') as fp:
        writer = csv.writer(fp, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE)

        writer.writerow(["'timestamp'", "'angle'", "'torque'"])  # write header

        dir_files = os.listdir(images_dir_path)
        dir_files = sorted(filter(lambda f: str.isdigit(f.split('.')[0]), dir_files),
                           key=lambda f: int(f.split('.')[0]))
        dir_files = list(map(lambda f: os.path.join(images_dir_path, f), dir_files))

        for file in dir_files:

            filename = os.fsdecode(file).split('/')[-1]
            if filename.endswith(".jpg"):

                file_timestamp = int(os.path.splitext(filename)[0])

                tries = 0
                while interpolated_csv['timestamp'][csv_index] != file_timestamp:
                    tries += 1
                    if tries >= 1:
                        print("Can't find csv entry for {}!".format(file_timestamp))
                        print("Prev: {}".format(prev))
                        exit(0)
                    csv_index += 1


                timestamp = interpolated_csv['timestamp'][csv_index]
                angle = interpolated_csv['angle'][csv_index]
                torque = interpolated_csv['torque'][csv_index]

                if timestamp >= set_start_time:
                    if timestamp <= set_end_time:
                        copyfile(images_dir_path + "/" + filename, output_directory_path + "/frames/" + filename)
                        writer.writerow(("%d" % timestamp, angle, torque))
                        count += 1
                    else:
                        break

                csv_index += 1

                prev = filename
            else:
                print("{} is not a jpeg image!!!".format(filename))

    print("Done, {} frames".format(count))


