import cv2 
import numpy as np
import matplotlib.pyplot as plt

def processImage(frame):
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bluredImage = cv2.GaussianBlur(grayImage, (5,5), 0)
    cannyImage = cv2.Canny(bluredImage, 80,100)
    return cannyImage

def region_triangle(frame):
    height = frame.shape[0]
    polygons = np.array([[(50, height), (1150, height), (750, 350)]]) #array x and y and the top endpoint
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons, 200)
    masked_image = cv2.bitwise_and(frame, mask)    
    return masked_image


def display_lines(frame, lines):
    line_image = np.zeros_like(frame)   
    if lines is not None:
        print(lines, type(lines))
        for line in lines:
             if line is not None and isinstance(line, np.ndarray) and line.shape == (4,):  
                x1, y1, x2, y2 = map(int, line.tolist())  # Convert NumPy array to Python integers
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

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
        if left_fit:
            left_fit_avg = np.mean(left_fit, axis=0) 
            print(left_fit_avg, 'left')
            left_line = coordinates(frame, left_fit_avg)
        else:
            left_line = None
        if right_fit:
            right_fit_avg = np.mean(right_fit, axis=0) 
            print(right_fit_avg, 'right')
            right_line = coordinates(frame, right_fit_avg)
        else:
            right_line = None
    else:
        left_line = None
        right_line = None
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