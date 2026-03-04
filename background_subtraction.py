import os
import numpy as np
import cv2 as cv

# Select camera
camera_path = "data/cam1/" # Cam 1.
# camera_path = "data/cam2/" # Cam 2.
# camera_path = "data/cam3/" # Cam 3.
# camera_path = "data/cam4/" # Cam 4.
show_model = False
background_path = os.path.join(camera_path, "background.avi")
video_path = os.path.join(camera_path, "video.avi")

# Subtraction thresholds
hue_threshold = 30
saturation_threshold = 40
value_threshold = 40
thresholds = (hue_threshold, saturation_threshold, value_threshold)

def show_image(image, window):
    # Create window.
    cv.namedWindow(window, cv.WINDOW_NORMAL)
    # Draw the image with cube and axis.
    cv.imshow(window, image)

def create_background_model():
    # Open capture.
    capture = cv.VideoCapture(background_path)

    # Check if the video was opened successfully
    if not capture.isOpened():
        Exception("Error: Could not open video file.")
    else:
        Exception("Video file opened successfully!")

    # Obtain number of frames.
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    # Arrays to save totals for HSV components.
    accumulated_hues = [] # Circular technically, but ignoring this...
    accumulated_saturations = []
    accumulated_values = []

    # Iterate through all frames, and accumulate totals for HSV components.
    for frame_index in range(frame_count):
        # Obtain frame (if no frame found go to next).
        ret, frame = capture.read()
        if not ret: continue

        # Convert to HSV color space and obtain components.
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        hue_array, saturation_array, value_array = cv.split(frame_hsv)

        if frame_index <= 0:
            # Add components to total.
            accumulated_hues = np.float32(hue_array)
            accumulated_saturations = np.float32(saturation_array)
            accumulated_values = np.float32(value_array)
            continue

        # Add components to total.
        accumulated_hues += np.float32(hue_array)
        accumulated_saturations += np.float32(saturation_array)
        accumulated_values += np.float32(value_array)

    # Compute averages.
    accumulated_hues /= frame_count
    accumulated_saturations /= frame_count
    accumulated_values /= frame_count

    model = cv.merge((accumulated_hues.astype(np.uint8), accumulated_saturations.astype(np.uint8), accumulated_values.astype(np.uint8)))
    return model

def subtract_background(background_model):
    # Open capture.
    capture = cv.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not capture.isOpened():
        Exception("Error: Could not open video file.")
    else:
        Exception("Video file opened successfully!")

    # Iterate through all frames
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    for frame_index in range(frame_count):
        # Obtain frame (if no frame found go to next).
        ret, frame = capture.read()
        if not ret: continue

        # Convert to HSV, and compute differences per component
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        differences = cv.absdiff(frame_hsv, background_model)
        hue_array, saturation_array, value_array = cv.split(differences)

        # If value above threshold, turn pixel white.
        hue_mask = cv.threshold(hue_array, thresholds[0], 255, cv.THRESH_BINARY)[1]
        saturation_mask = cv.threshold(saturation_array, thresholds[1], 255, cv.THRESH_BINARY)[1]
        value_mask = cv.threshold(value_array, thresholds[2], 255, cv.THRESH_BINARY)[1]

        # Foreground mask.
        foreground_mask = cv.bitwise_or(cv.bitwise_or(hue_mask, saturation_mask), value_mask)



        show_image(foreground_mask, "Oops")
        cv.waitKey(0)
        cv.destroyAllWindows()

def main():
    background_model = create_background_model()
    if show_model:
        model = cv.cvtColor(background_model, cv.COLOR_HSV2BGR)
        show_image(model, "Oops")
        cv.waitKey(0)
        cv.destroyAllWindows()
    subtract_background(background_model)
main()