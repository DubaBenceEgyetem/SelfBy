import cv2 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Preprocessing import hls_selected_channel, abs_sobel_thresh


def road_condition(frame):
   # x,y,w,h = 0,0,250,75
    #cv2.rectangle(frame, (x,x), (x+w, y+h), (255,0,0), -1)
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
    #condition = cv2.putText(frame, condition, (x + int(w/10),y + int(h/2)), cv2.LINE_AA, fontSC, color, thickness)
    #print(condition)
    return condition

def gray_threshold(frame,thresh=150):
    '''
    Applies a threshold to grayscale image.
    Used to recognize white/yellow lane lines.
    '''
    gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    _, binary_output = cv2.threshold(gray,thresh,1,cv2.THRESH_BINARY)
    return binary_output



def binarize(frame):
    condition = road_condition(frame)
    binouts = gray_threshold(frame, thresh=150)
    #print(f"Binary Output from gray_threshold - Min: {np.min(binouts)}, Max: {np.max(binouts)}")
    combined = np.zeros_like(binouts)
    
    if condition == "Faded Road":
        hls_s_binary = hls_selected_channel(frame, thresh=(200, 255)) 
        combined[(hls_s_binary == 1)] = 1         
    elif condition == "Normal Road":
        hls_s_binary = hls_selected_channel(frame, thresh=(210, 255)) 
        combined[(hls_s_binary == 1)] = 1    
    elif condition == "Road with Shadows":
        hls_s_binary = hls_selected_channel(frame, thresh=(235, 255)) 
        combined[(hls_s_binary == 1)] = 1 
   
    print(combined)
    return combined



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
