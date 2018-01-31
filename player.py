import cv2
import numpy as np
import os
import math


def draw_steering_line(img, angle, color, guide=False):

    length = img.shape[0]//4
    center_x = img.shape[1]//2
    center_y = img.shape[0]//7*6

    x1 = int(center_x - length * math.cos(angle))
    y1 = int(center_y + length * math.sin(angle))
    x2 = int(center_x + length * math.cos(angle))
    y2 = int(center_y - length * math.sin(angle))

    # Draw a diagonal line with thickness of 3 px
    cv2.line(frame, (x1, y1), (x2, y2), color, 3, lineType=cv2.LINE_AA)

    if guide:

        cv2.line(frame, (center_x + length + 4, center_y), (center_x + length + 25, center_y), (255, 255, 255), 1)
        cv2.line(frame, (center_x - length - 4, center_y), (center_x - length - 25, center_y), (255, 255, 255), 1)


# Todo: Replace hard coded file paths
images_dir_path = "/data/jpgs_output/center/"
ground_truth = np.genfromtxt("/data/jpgs_output/interpolated.csv", dtype='float', delimiter=',', names=True)

directory = os.fsencode(images_dir_path)

csv_index = 2
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        frame = cv2.imread(images_dir_path + "/" + filename)
        angle = ground_truth['angle'][csv_index]

        overlay = frame.copy()
        alpha = 0.5
        draw_steering_line(overlay, 0, (255, 0, 0), guide=True)
        draw_steering_line(overlay, angle, (0, 255, 0))
        cv2.putText(overlay,filename,
                    (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        # apply the overlay
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha,
                        0, frame)

        # Display the resulting frame
        cv2.imshow('Frame', frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(23) & 0xFF == ord('q'):
            break

        csv_index += 3

    else:
        print("{} is not a jpeg image".format(filename))



# Closes all the frames
cv2.destroyAllWindows()