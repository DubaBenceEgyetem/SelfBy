import cv2 
import numpy as np
from LineDetection import processImage, region_triangle, display_lines, avg_slope_intersept, coordinates, viz_colorspace
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = cv2.VideoCapture("test/videos/example.mp4")
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 1500,700)
fps = image.get(cv2.CAP_PROP_FPS)

slow_down_factor = 2
frame_delay = 1 / (fps / slow_down_factor)
paused = False



while image.isOpened():
    if not paused:
        ret, frame = image.read()
        canny = processImage(frame, gamma=1.5)
        cropped = region_triangle(canny)
        cv2.imshow('gamma', cropped)
        cv2.waitKey(1)


    key = cv2.waitKey(int(frame_delay * 200)) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused


image.release()
cv2.destroyAllWindows()


