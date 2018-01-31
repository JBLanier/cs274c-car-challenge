import cv2
import numpy as np
import os
import math


# Internal function to overlay steering wheel line on an image
def draw_steering_line(img, steering_angle, color, guide=False):

    length = img.shape[0]//4
    center_x = img.shape[1]//2
    center_y = img.shape[0]//7*6

    x1 = int(center_x - length * math.cos(steering_angle))
    y1 = int(center_y + length * math.sin(steering_angle))
    x2 = int(center_x + length * math.cos(steering_angle))
    y2 = int(center_y - length * math.sin(steering_angle))

    # Draw a diagonal line with thickness of 3 px
    cv2.line(frame, (x1, y1), (x2, y2), color, 3, lineType=cv2.LINE_AA)

    if guide:

        cv2.line(frame, (center_x + length + 4, center_y), (center_x + length + 25, center_y), (255, 255, 255), 1)
        cv2.line(frame, (center_x - length - 4, center_y), (center_x - length - 25, center_y), (255, 255, 255), 1)


# Internal function, hold frame for <time_to_wait> and intercept/return True for quit ('q') key command
def intercept_quit_key_command(milliseconds_time_to_wait):
    # Wait <time_to_wait> milliseconds and close preview windows if 'q' is pressed.
    # <time_to_wait> essentially controls our video frame rate.
    if cv2.waitKey(milliseconds_time_to_wait) & 0xFF == ord('q'):
        # Closes all the frames
        cv2.destroyAllWindows()
        return True
    return False


# Display a single frame in a gui window, debug_info is displayed as text im corner of image.
# Steering angles are graphically displayed.
#
# Closes all windows and returns true if user pressed key command to quit.
# <milliseconds_time_to_wait> is how long you want the thread to hang and continue displaying the frame, determines fps
# Ideally make <milliseconds_time_to_wait> = (milliseconds for desired fps - measured calculation time)
def display_frame(img, true_angle=None, predicted_angle=None, debug_info=None, milliseconds_time_to_wait=23):
    img_to_modify = img.copy()
    if true_angle or predicted_angle or debug_info:
        alpha = 0.5
        if predicted_angle is not None:
            draw_steering_line(img_to_modify, predicted_angle, (255, 0, 0), guide=True)
        if true_angle is not None:
            draw_steering_line(img_to_modify, true_angle, (0, 255, 0))
        if debug_info:
            cv2.putText(img_to_modify, debug_info,
                        (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        # apply the overlay
        cv2.addWeighted(img_to_modify, alpha, img, 1 - alpha,
                        0, img_to_modify)

    # Display the resulting frame
    cv2.imshow('Preview', img_to_modify)

    return intercept_quit_key_command(milliseconds_time_to_wait)


if __name__ == '__main__':
    # Play a video with no predictions
    # Currently just a debug script, you should probably be running something else as main and be using display_frame().

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

            if display_frame(frame, true_angle=angle, predicted_angle=0, debug_info=filename):
                break

            csv_index += 3

        else:
            print("{} is not a jpeg image".format(filename))



