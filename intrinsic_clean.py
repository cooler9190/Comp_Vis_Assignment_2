import numpy as np
import cv2 as cv
import glob
import os

# Clean manual images; given a pre-exported image sequence (from the intrinsic video, using premiere pro).

# Using Code from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html [source_1]
# and https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html [source_2]

# What camera?
# calibration_images = glob.glob('data/cam1/intrinsic/*.png') # [!] Clean Camera 1 image sequence to use for intrinsic calibration.
# calibration_images = glob.glob('data/cam2/intrinsic/*.png') # [!] Clean Camera 2 image sequence to use for intrinsic calibration.
# calibration_images = glob.glob('data/cam3/intrinsic/*.png') # [!] Clean Camera 3 image sequence to use for intrinsic calibration.
calibration_images = glob.glob('data/cam4/intrinsic/*.png') # [!] Clean Camera 4 image sequence to use for intrinsic calibration.

# Settings
resize_windows = False
chessboard_size = (8, 6) # From checkerboard.xml

# Creates window (of given name) to show a given image.
def show_image(image, window):
    # Create window.
    cv.namedWindow(window, cv.WINDOW_NORMAL)
    # Resize window.
    if resize_windows:
        h, w = image.shape[:2]
        ratio = 800 / w
        cv.resizeWindow(window, 800, int(h * ratio))
    # Draw the image with cube and axis.
    cv.imshow(window, image)

def clean_manual():
    # Update arrays with corner points from the images.
    for filename in calibration_images:
        # Read image and turn into gray scale.
        image = cv.imread(filename)
        processed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        has_found_corners, corners = cv.findChessboardCorners(processed_image, chessboard_size, None)

        # To skip automatically detected images (including pre-processed ones).
        found_auto = has_found_corners

        # Delete manual images.
        if not found_auto:
            print("Not auto! - Removing: ", filename)
            os.remove(filename)
            continue
        print("Auto Found: ", filename)

def main():
    clean_manual()
main()