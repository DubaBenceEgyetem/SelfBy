import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def resize(frame, canny,cropped, frame_with_lines):
    # Képek méreteinek beállítása 360x640-re
    canny = cv2.resize(canny, (640, 360))
    cropped = cv2.resize(cropped, (640, 360))
    frame_with_lines = cv2.resize(frame_with_lines, (640, 360))
    frame = cv2.resize(frame, (640, 360))
    return (frame, canny, cropped, frame_with_lines)

def processImage(frame, gamma):
    
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    gamma = cv2.LUT(frame, table)    
    adjusted = cv2.convertScaleAbs(gamma, alpha=1.0, beta=2.0)
    grayImage = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    bluredImage = cv2.medianBlur(grayImage, 5)
    cannyImage = cv2.Canny(bluredImage, 20, 150, apertureSize=3) #20 a minimum 100 az optimális, 150
    return cannyImage

def region_triangle(canny):
    polygon = np.array([[(50, 720), (1200, 720), (650, 400)]])
    #array x and y and the top endpoint
    mask = np.zeros_like(canny)
    if len(canny.shape) > 2:
        channel_count = canny.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, polygon, ignore_mask_color)
    masked_image = cv2.bitwise_and(canny, mask)    
    return masked_image


def perspective_transform(frame):
    frame_size = (frame.shape[1], frame.shape[0]) #dimensions of the frame
    src = np.float32([[192, 720], [582,457], [701, 457], [1145,720]])
    offset = [150,0]
    dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset, np.array([src[3, 0], 0]) - offset, src[3] - offset])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(frame, M, frame_size)
    return warped, M, Minv


def display_lines(frame, lines):
    if lines is not None:
        for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return frame

def avg_slope_intersept(frame, lines):
    left_fit = []    
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

            left_fit_avg = np.mean(left_fit, axis=0) 
            left_line = coordinates(frame, left_fit_avg)   
            right_fit_avg = np.mean(right_fit, axis=0) 
            right_line = coordinates(frame, right_fit_avg)
   
    return [left_line, right_line]

def coordinates(frame, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0
    y1 = frame.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1,y1,x2,y2])