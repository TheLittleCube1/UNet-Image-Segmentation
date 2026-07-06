# Lane Detection using OpenCV

This project implements a real-time lane detection system using classical computer vision techniques including Canny Edge Detection, Gaussian Blur, and Hough Transform.

The goal is to detect road lanes from dashcam footage and visualize them in real-time.

---

## Features

- Canny Edge Detection for lane extraction
- Gaussian Blur for noise reduction
- Hough Line Transform for lane detection
- Region of Interest (ROI) masking using triangular region
- Lane visualization overlay on video frames

---

## Tech Stack

- Python
- OpenCV
- NumPy
- Matplotlib (for debugging visualization)

---

## How to Run

1. Install dependencies:

```bash
pip install opencv-python numpy matplotlib
python main.py
