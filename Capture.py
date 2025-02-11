import cv2 
import numpy as np
from LineDetection import frame_processor

image = cv2.VideoCapture("example.mp4")
fps = image.get(cv2.CAP_PROP_FPS)

slow_down_factor = 2 
frame_delay = 1 / (fps / slow_down_factor)

while image.isOpened():
    ret, frame = image.read()
    if not ret:
        break

    processed_frame = frame_processor(process_image=frame)
    cv2.imshow('frame', processed_frame)

    if cv2.waitKey(int(frame_delay * 200)) & 0xFF == ord('q'):
        break


image.release()
cv2.destroyAllWindows()


