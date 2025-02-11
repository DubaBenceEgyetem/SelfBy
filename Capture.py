import cv2 
import numpy as np
from LineDetection import cannyImage, region_triangle,bluredImage,ModifyImages
import matplotlib.pyplot as plt

image = cv2.VideoCapture("example.mp4")
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 1500,700)
fps = image.get(cv2.CAP_PROP_FPS)

slow_down_factor = 2
frame_delay = 1 / (fps / slow_down_factor)
paused = False

while image.isOpened():
    if not paused:
        ret, frame = image.read()
        all_res = ModifyImages(frame)
        cv2.imshow('frame',all_res)

    key = cv2.waitKey(int(frame_delay * 200)) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused


image.release()
cv2.destroyAllWindows()


