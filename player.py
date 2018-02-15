# player.py

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
    cv2.line(img, (x1, y1), (x2, y2), color, 3, lineType=cv2.LINE_AA)

    if guide:
        # draw simple guide lines at horizontal axis
        cv2.line(img, (center_x + length + 4, center_y), (center_x + length + 25, center_y), (255, 255, 255), 1)
        cv2.line(img, (center_x - length - 4, center_y), (center_x - length - 25, center_y), (255, 255, 255), 1)


# Internal function, hold frame for <time_to_wait> and intercept/return True for quit ('q') key command
def intercept_quit_key_command(milliseconds_time_to_wait):
    # Wait <time_to_wait> milliseconds and close preview windows if 'q' is pressed.
    # <time_to_wait> essentially controls our video frame rate.
    if cv2.waitKey(milliseconds_time_to_wait) & 0xFF == ord('q'):
        # Closes all the frames
        cv2.destroyAllWindows()
        return True
    return False


#
# -- Call this externally --
#
# Display a single frame in a gui window, debug_info is displayed as text in corner of image.
# Steering angles are graphically displayed.
#
# Closes all windows and returns true, if user pressed key command to quit.
# <milliseconds_time_to_wait> is how long you want the thread to hang and continue displaying the frame, determines fps
# Ideally make <milliseconds_time_to_wait> = (milliseconds for desired fps - measured calculation time)
def display_frame(img, true_angle=None, predicted_angle=None, debug_info=None, guide=True, milliseconds_time_to_wait=23):
    img_to_modify = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    if true_angle or predicted_angle or debug_info:
        if predicted_angle is not None:
            draw_steering_line(img_to_modify, predicted_angle, (255, 0, 0), guide=guide)
        if true_angle is not None:
            draw_steering_line(img_to_modify, true_angle, (0, 255, 0), guide=guide)
        if debug_info:
            cv2.putText(img_to_modify, debug_info,
                        (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # apply the overlay
        np.clip(img_to_modify, 0, 255)

    # Display the resulting frame
    cv2.imshow('Preview', img_to_modify)

    return intercept_quit_key_command(milliseconds_time_to_wait)
