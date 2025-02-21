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
    # Convert the frame to grayscale
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel operator in the specified direction
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

'''
def hls_selected_channel(frame, thresh = (150)):
    hlv = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, binout =cv2.threshold(hlv, thresh,1,cv2.THRESH_BINARY)
    return binout
'''

def road_condition(frame):
    x,y,w,h = 0,0,250,75
    cv2.rectangle(frame, (x,x), (x+w, y+h), (255,0,0), -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50,50)
    fontSC = 1
    color = (0,0,255)
    thickness = 2 
    yuv =cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
    ychannel = np.average(yuv[:,500,0])
    #print(ychannel)
    if(ychannel > 120):
        condition = "Faded road"
    elif((ychannel <= 120) & (ychannel > 71)):
        condition = "Normal road"
    elif(ychannel <= 71):
        condition = "Shadow road"
    condition = cv2.putText(frame, condition, (x + int(w/10),y + int(h/2)), cv2.LINE_AA, fontSC, color, thickness)
    print(condition)
    return condition




def region_triangle(canny):
    polygon = np.array([[(0, 720), (1200, 720), (750, 400),(600, 400)]])
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
