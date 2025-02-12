import cv2 
import numpy as np
import matplotlib.pyplot as plt

def cannyImage(frame):
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bluredImage = cv2.GaussianBlur(grayImage, (5,5), 0)
    cannyImage = cv2.Canny(bluredImage, 80,100)
    return cannyImage

def bluredImage(frame):
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bluredImage = cv2.GaussianBlur(grayImage, (5,5), 0)
    return bluredImage

def region_triangle(frame):
    height = frame.shape[0]
    polygons = np.array([[(50, height), (1150, height), (750, 350)]]) #array x and y and the top endpoint
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons, (255,255,255))
    masked_image = cv2.bitwise_and(frame, mask)    
    return masked_image


def display_lines(frame, lines):
    line_image = np.zeros_like(frame)   
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)   
            line = cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image