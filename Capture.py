import cv2 
import numpy as np
from LineDetection import processImage, region_triangle, display_lines, avg_slope_intersept, coordinates
import matplotlib.pyplot as plt

image = cv2.VideoCapture("example2.mp4")
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 1500,700)
fps = image.get(cv2.CAP_PROP_FPS)

slow_down_factor = 2
frame_delay = 1 / (fps / slow_down_factor)
paused = False

while image.isOpened():
    if not paused:
        ret, frame = image.read()
        canny = processImage(frame)
        cropped = region_triangle(canny)
        lines = cv2.HoughLinesP(cropped, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=10)
        avg_lines = avg_slope_intersept(frame, lines)
        line_frame =display_lines(frame, avg_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_frame, 1, 1) 
        cv2.imshow('frame',combo_image)
        cv2.waitKey(1)

    key = cv2.waitKey(int(frame_delay * 200)) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused


image.release()
cv2.destroyAllWindows()


