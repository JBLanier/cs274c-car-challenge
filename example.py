# example.py

import os
import imageio
import player


def play_jpg_frames_from_dir(path):
    input_directory = os.fsencode(path)

    for file in os.listdir(input_directory):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg"):
            frame = imageio.imread(path + "/" + filename)

            # Make color distributions more even for each RGB channel
            # for i in range(3):
            #     frame[:,:,i] = cv2.equalizeHist(frame[:,:,i])

            if player.display_frame(frame):
                break


if __name__ == '__main__':
    play_jpg_frames_from_dir("HMB_1_prepared/frames")




