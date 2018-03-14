# example.py

import os

import imageio
import numpy as np

from src import player


def play_jpg_frames_from_dir(frames_dir):
    input_directory = os.fsencode(frames_dir)

    for file in os.listdir(input_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            frame = imageio.imread(frames_dir + "/" + filename)

            # Make color distributions more even for each RGB channel
            # for i in range(3):
            #     frame[:,:,i] = cv2.equalizeHist(frame[:,:,i])

            if player.display_frame(frame):
                break


def play_frames_with_csv_data(csv_path,frames_dir):
    csv = np.genfromtxt(csv_path, dtype=None, delimiter=',', names=True)
    for row in csv:
        frame = imageio.imread("{}/{}.jpg".format(frames_dir, row[0]))
        angle = row[1]
        if player.display_frame(frame, true_angle=angle, debug_info=str(row[0])):
            break


if __name__ == '__main__':
    # play_jpg_frames_from_dir("CH_02_Prepared/HMB_1_prepared/frames")
    play_frames_with_csv_data("CH_02_Prepared/HMB_1_prepared/data.csv","CH_02_Prepared/HMB_1_prepared/frames")





