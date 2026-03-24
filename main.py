import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import utils

# Start Time (in seconds)
start_time_sec = 0

# Footage
footage = cv.VideoCapture('Driving Footage.mp4')

fps = footage.get(cv.CAP_PROP_FPS)
frame_number = int(start_time_sec * fps)
frame_count = 0
footage.set(cv.CAP_PROP_POS_FRAMES, frame_number)

while cv.waitKey(1) & 0xFF != ord('d'):
    isTrue, frame = footage.read()
    hough_transformed_frame = utils.hough_transform(frame)
    cv.imshow('Lane Detection Output', hough_transformed_frame)

footage.release()
cv.destroyAllWindows()