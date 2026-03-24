import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time

def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, int(height * 0.9)),
        (width, int(height * 0.9)),
        (int(width * 0.5), int(height * 0.4))
    ]], np.int32)
    cv.fillPoly(mask, polygon, 255)
    masked = cv.bitwise_and(img, mask)
    return masked

def rho_theta_to_endpoints(line):
    rho, theta = line[0]
    dhat = np.array([[np.cos(theta)], [np.sin(theta)]])
    d = rho * dhat
    lhat = np.array([[-np.sin(theta)], [np.cos(theta)]])
    k = 3000
    p1 = d + k * lhat
    p2 = d - k * lhat
    p1 = p1.astype(int)
    p2 = p2.astype(int)
    return (p1[0][0], p1[1][0], p2[0][0], p2[1][0])

def cluster_lines(lines_mb):
    positive_slope_lines = []
    negative_slope_lines = []
    for line in lines_mb:
        if line[0] > 0: positive_slope_lines.append(line)
        else: negative_slope_lines.append(line)

    lines = []
    if len(positive_slope_lines) > 0:
        lines.append(np.mean(positive_slope_lines, axis = 0))
    if len(negative_slope_lines) > 0:
        lines.append(np.mean(negative_slope_lines, axis = 0))
    return lines

def endpoints_to_slope_intercept(lines):
    slope_intercepts = []
    for line in lines:
        endpoints = rho_theta_to_endpoints(line)
        p1x, p1y, p2x, p2y = endpoints[0], endpoints[1], endpoints[2], endpoints[3]
        slope = float(p2y - p1y) / float(p2x - p1x)
        y_intercept = float(p1y - slope * p1x)
        slope_intercepts.append((slope, y_intercept))
    return slope_intercepts

def slope_intercept_to_endpoints(lines_mb, width, height):
    lines = []
    for line in lines_mb:
        slope = line[0]
        y_intercept = line[1]
        p1x = int(0); p1y = int(y_intercept); p2x = int(width); p2y = int(p1y + width * slope)
        lines.append((p1x, p1y, p2x, p2y))
    return lines

def remove_duplicate_lines(lines, width, height):
    slope_intercepts = endpoints_to_slope_intercept(lines)
    clusters = cluster_lines(slope_intercepts)
    endpoints = slope_intercept_to_endpoints(clusters, width, height)
    return (clusters, endpoints)

def compute_triangle(slope_intercepts_list):
    (m1, b1), (m2, b2) = slope_intercepts_list
    x_int = (b2 - b1) / (m1 - m2)
    y_int = m1 * x_int + b1
    intersection = (x_int, y_int)
    x_axis_points = []
    for m, b in slope_intercepts_list:
        x = -b / m
        x_axis_points.append((x, 0))
    return (intersection, x_axis_points)

def draw_lane_triangle(img, intersection, x_axis_points):
    height = img.shape[0]; width = img.shape[1]
    (x_top, y_top) = intersection
    (x_left, _), (x_right, _) = x_axis_points
    pt1 = (width - int(x_left), height)
    pt2 = (width - int(x_right), height)
    pt3 = (int(x_top), int(y_top))
    triangle = np.array([[pt1, pt2, pt3]], dtype=np.int32)
    overlay = img.copy()
    cv.fillPoly(overlay, triangle, (0, 255, 0))
    alpha = 0.3
    output = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return output

def hough_transform(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (7, 7), 3)
    canny = cv.Canny(blur, 20, 60)
    masked = region_of_interest(canny)
    lines = cv.HoughLines(masked, 1, np.pi / 180.0, 150)
    if lines is None:
        lines = []
    slope_intercepts_list, endpoints_list = remove_duplicate_lines(lines, img.shape[1], img.shape[0])
    output_image = img
    if len(endpoints_list) == 1:
        endpoints = endpoints_list[0]
        output_image = cv.line(img, (endpoints[0], endpoints[1]), (endpoints[2], endpoints[3]), (0, 255, 0), 10)
    elif len(endpoints_list) == 2:
        triangle = compute_triangle(slope_intercepts_list)
        output_image = draw_lane_triangle(img, triangle[0], triangle[1])
    else:
        print(slope_intercepts_list, endpoints_list)
    return output_image