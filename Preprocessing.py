import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def colorspace(frame):
    img_HLS =cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    imgHLS_L = img_HLS[:,:,1] #:,:,1 a legjobb
    img_HSV = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    imgHSV_V = img_HSV[:,:,2]
    return imgHLS_L  


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
    img_HLS = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV)
    imgHLS_S = img_HLS[:,:,2]
    bluredImage = cv2.medianBlur(imgHLS_S, 5)
    cannyImage = cv2.Canny(bluredImage, 100, 150, apertureSize=3) #20 a minimum 100 az optimális, 150
    return cannyImage


def abs_sobel_thresh(frame, orient='x', thresh=(0,255)):
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(grayImage, cv2.CV_64F, 1, 0))
    elif orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(grayImage, cv2.CV_64F, 0, 1))
    else:
        raise ValueError("Invalid orientation. Use 'x' or 'y'.")
    
    # Scale the Sobel output to 8-bit (0-255)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    
    # Create a binary output image
    binout = np.zeros_like(scaled_sobel)
    
    # Apply the threshold to create a binary image
    binout[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binout


def hls_selected_channel(frame, thresh=(200,255)):
    hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    channel = hls[:,:,2]
    binout =np.zeros_like(channel)
    binout[(channel > thresh[0]) & (channel <= thresh[1])] = 1
    return binout

