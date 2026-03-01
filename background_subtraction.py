import numpy as np
import cv2 as cv

# Define a structuring element for morphological operations
# An elliptical kernel is often used for morphological operations to better preserve the shape of objects in the foreground mask
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

# Create the background subtractor object using MOG2 algorithm and Gaussian Mixture-based Background/Foreground Segmentatio
fgbg = cv.createBackgroundSubtractorMOG2(varThreshold=100, detectShadows=True)

# Phase 1: Background Learning
# Read and process frames to learn the background model

print("Learning background model. Please wait...")
# Capture video from the specified path
cap_bg = cv.VideoCapture('data/cam4/background.avi')

while True:
    ret, frame = cap_bg.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    
   # Apply the background subtractor to learn the background model
    fgbg.apply(frame)

cap_bg.release()
print("Background model learned.")

# Phase 2: Apply the learned background model to the test video
print("Applying background model to test video...")

cap_test = cv.VideoCapture('data/cam4/video.avi')

while True:
    ret, frame = cap_test.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    
    # Pass frame through the background subtractor to get the foreground mask
    # Stopping the background model from updating ensures the person in the test video is not absorbed into the background model if they remain stationary
    fg_mask = fgbg.apply(frame, learningRate=0)  # Set learningRate to 0 to stop updating the background model

    # Morphological operations to clean up the foreground mask
    # OPENING: Erosion followed by dilation to remove noise and small objects from the foreground mask
    erosion = cv.erode(fg_mask, kernel, iterations=1)
    opening = cv.dilate(erosion, kernel, iterations=1)

    # CLOSING: Dilation followed by erosion to fill small holes in the foreground mask
    dilation = cv.dilate(opening, kernel, iterations=5)
    closing = cv.erode(dilation, kernel, iterations=5)

    cv.imshow('Foreground Mask', closing)
    cv.imshow('Original Frame', frame)

    k = cv.waitKey(30) & 0xff
    if k == 27:  # Press 'ESC' to exit
        break

cap_test.release()
cv.destroyAllWindows()
