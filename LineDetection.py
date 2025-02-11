import cv2 
import numpy as np

def frame_processor(process_image):
    grayscale = cv2.cvtColor(process_image, cv2.COLOR_BGR2GRAY)
    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)
    low = 50
    high = 150

    edges = cv2.Canny(blur, low, high)
    region = region_selection(edges)
    hough = hough_transform(region)

    result = draw_line(process_image, lane_lines(process_image, hough))
    return result


def region_selection(process_image):
    mask = np.zeros_like(process_image)
    if len(process_image.shape) > 2:
        channel_count = process_image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    rows, cols = process_image.shape[:2]
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(process_image, mask)
    return masked_image

def hough_transform(process_image):
    rho = 1 
    theta = np.pi/180
    threshold = 20
    minLineLength = 20
    maxLineGap = 500
    return cv2.HoughLinesP(process_image, rho = rho, theta = theta, threshold = threshold, minLineLength = minLineLength, maxLineGap = maxLineGap)

def draw_line(process_image, lane_lines, color=[255,0,0], thickness=12):
    line_image=np.zeros_like(process_image)
    for line in lane_lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(process_image, 1.0, line_image, 1.0, 1.0)

def lane_lines(process_image, lane):
    pass
