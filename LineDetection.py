import cv2 
import numpy as np

def cannyImage(frame):
    grayImage = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bluredImage = cv2.GaussianBlur(grayImage, (5,5), 0)
    cannyImage = cv2.Canny(bluredImage, 80,100)
    return cannyImage

def bluredImage(frame):
    grayImage = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bluredImage = cv2.GaussianBlur(grayImage, (5,5), 0)
    return bluredImage

def region_triangle(frame):
    height = frame.shape[0]
    polygons = np.array([[(50, height), (1150, height), (750, 350)]]) #array x and y and the top endpoint
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons, 200)
    masked_image = cv2.bitwise_and(frame, mask)
    return masked_image

def ModifyImages(frame):
    canny = cannyImage(frame)
    blured = bluredImage(frame)
    maskedRoad =  region_triangle(bluredImage(frame))
    masked =  region_triangle(canny)
    row1 = np.hstack((blured, canny))
    row2 = np.hstack((masked, maskedRoad))
    combined  = np.vstack((row1, row2))
    return combined
