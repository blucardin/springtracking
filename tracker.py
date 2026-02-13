# NV 2026
# Sources
# https://www.youtube.com/watch?v=caNUo-bQV9c
# https://docs.opencv.org/3.4/da/d97/tutorial_threshold_inRange.html
# https://docs.opencv.org/4.x/d9/dc8/tutorial_py_trackbar.html
#

# from __future__ import print_function
import cv2 as cv
# import argparse
import numpy as np
import time 
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


def on_low_H_thresh_trackbar(val):
    global low_H
    # global high_H
    low_H = val
    # low_H = min(high_H-1, low_H)
    # cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_high_H_thresh_trackbar(val):
    # global low_H
    global high_H
    high_H = val
    # high_H = max(high_H, low_H+1)
    # cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    # global high_S
    low_S = val
    # low_S = min(high_S-1, low_S)
    # cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    # global low_S
    global high_S
    high_S = val
    # high_S = max(high_S, low_S+1)
    # cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    # global high_V
    low_V = val
    # low_V = min(high_V-1, low_V)
    # cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    # global low_V
    global high_V
    high_V = val
    # high_V = max(high_V, low_V+1)
    # cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

def on_gaussian_blur_trackbar(val):
    global gaussian_blur_radius
    gaussian_blur_radius = val

def on_gaussian_stdev(val):
    global gaussian_stdev
    gaussian_stdev = val

def makeIndicator(): 
    global indicator
    size = 100
    xx, yy = np.meshgrid(
    np.linspace(0, max_value, frame.shape[1]),
    np.linspace(0, max_value, size) / 255
    )
    indicator = np.stack( (xx, yy, np.full(xx.shape, 1)), axis = 2, dtype="float32" )

    indicator = (cv.cvtColor(indicator, cv.COLOR_HSV2BGR) * 255).astype("uint8")

# parser = argparse.ArgumentParser(description='Code for Thresholding Operations.')
# parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
# args = parser.parse_args()
cap = cv.VideoCapture("IMG_0613.mov") # 0)  # args.camera)

cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name, low_H,
                  max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name, high_H,
                  max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name, low_S,
                  max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name, high_S,
                  max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name, low_V,
                  max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name, high_V,
                  max_value, on_high_V_thresh_trackbar)


gaussian_blur_radius  = 1

cv.createTrackbar("G Radius", window_detection_name, 0,
                  100, on_gaussian_blur_trackbar)

cv.createTrackbar("G stdev", window_detection_name, 0,
                  100, on_gaussian_stdev)


ret, frame = cap.read()
makeIndicator()

mask = True
target = True

gaussian_stdev = 0
current_frame = 0
output = np.zeros((int(cap.get(cv.CAP_PROP_FRAME_COUNT)), 2))

while cap.isOpened():
    cap.set(cv.CAP_PROP_POS_FRAMES, current_frame)

    ret, frame = cap.read()
    if frame is None:
        break

    frame = cv.GaussianBlur(frame,( 1 + 2 *(gaussian_blur_radius), ) * 2 , gaussian_stdev)

    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    frame_threshold = cv.inRange(
        frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    # print("threshold shpa ", frame_threshold.shape)
    # print("frame shape", frame.shape)
    # print(frame_HSV.dtype)

    # cv.imshow(window_capture_name, frame)
    masked_image = frame

    # 255 - frame  * np.tile(np.expand_dims(frame_threshold, axis=2), (1, 1,3) )
    if mask:
        masked_image = cv.bitwise_and(frame, frame, mask=frame_threshold)

    
    # print(np.concatenate((indicator, masked_image)))
    if target: 
        M = cv.moments(frame_threshold, True)
        # Avoid division by zero
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv.circle(masked_image, (cX, cY), 10, (255, 0, 0), 3)

            output[current_frame] = np.array([cX, cY])
            
        # print(cv.moments(frame_threshold, True))
        # cv.circle(masked_image, cv.moments(frame_threshold, True), 10, color=255)

    
    cv.imshow(window_detection_name, np.concatenate((indicator, masked_image)))

    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        np.savetxt(f"positions{time.time()}.csv", output, delimiter=",", fmt="%d")
        break
    elif key == ord('m'):
        mask = not mask
    elif key == ord('t'):
        target = not target
    elif key == ord('l'): 
        current_frame += 1
    elif key == ord('j'):
        current_frame = max(0, current_frame - 1)

print(low_H, low_S, low_V, high_H, high_S, high_V, gaussian_blur_radius, gaussian_stdev)
np.savetxt(f"positions{time.time()}.csv", output, delimiter=",", fmt="%d")
