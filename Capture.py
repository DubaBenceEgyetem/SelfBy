import cv2 
import numpy as np
from LineDetection import processImage, region_triangle, display_lines, avg_slope_intersept, coordinates,perspective_transform,resize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = cv2.VideoCapture("test/videos/example.mp4")
fps = image.get(cv2.CAP_PROP_FPS)

slow_down_factor = 2
frame_delay = 1 / (fps / slow_down_factor)
paused = False



def birdeyeView(frame):
     Warped, M, Minv = perspective_transform(frame)
     cv2.imshow('birdEyeView',Warped)
     plt.show()


while image.isOpened():
    if not paused:
        ret, frame = image.read()
        birdeyeView(frame)
        canny = processImage(frame, gamma=1.5)
        cropped = region_triangle(canny)
        lines = cv2.HoughLinesP(cropped, 2, np.pi / 180, 100, minLineLength=40, maxLineGap=5)
        frame_with_lines = display_lines(frame, lines)        
        frame, canny, cropped, frame_with_lines =resize(frame, canny, cropped, frame_with_lines)

        # Szürkeárnyalatos képek 3 dimenzióssá alakítása
        canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
        cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

        # Képek összeállítása
        top = np.hstack((canny, cropped))
        bottom = np.hstack((frame_with_lines, frame))
        combined = np.vstack((top, bottom))
         # Eredmény megjelenítése
        cv2.imshow('frames', combined)     

       
   

    
    key = cv2.waitKey(int(frame_delay * 200)) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused
  


image.release()
cv2.destroyAllWindows()


