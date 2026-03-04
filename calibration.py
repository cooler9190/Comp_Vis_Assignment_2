import numpy as np
import cv2 as cv
import glob

# Modified from Assignment 1 (some code from there has been omitted)

# Using Code from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html [source_1]
# and https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html [source_2]

# Select camera
camera_path = "data/cam1/" # Cam 1.
# camera_path = "data/cam2/" # Cam 2.
# camera_path = "data/cam3/" # Cam 3.
# camera_path = "data/cam4/" # Cam 4.

# Other filepaths.
calibration_images = glob.glob(camera_path + "intrinsic/*.png") # [!] Intrinsic folder.
config_path = camera_path + "config.xml" # Where to save camera config.
test_images = glob.glob(camera_path + "checkerboard.avi") # [!] Is ignored during live feed.
axis_image = camera_path + "axes.png"

# General settings
use_preprocessing = False # A1 - Toggle Choice 5
use_warping = False # A1 - Toggle Choice 3
camera_index = 0

skip_auto_images = True # Don't show automatically found chessboard corners.

# Drawing settings
should_draw_axes = True
thickness_axes = 2

should_draw_cube = True
thickness_cube = 1

should_draw_text = True
scale_text = 1
thickness_text = 2
x_text = 20
height_interval_text = 30

# Window Settings
resize_windows = True
window_auto_corners = 'Chessboard Corners'
window_online = 'Axis and Cube'

# Chessboard settings.
chessboard_size = (8, 6)
chessboard_width = chessboard_size[0]
chessboard_height = chessboard_size[1]
chessboard_cells = chessboard_size[0] * chessboard_size[1] # Width x Height
cell_size = 0.115 # meters

# CornerSubPix termination criteria.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # [!] Values copied from [source_1]

# Prepare object points, like (0,0,0), (cell_size,0,0), (2*cs,0,0) ...., (h*cs,w*cs,0). Required for both phases.
object_points_world = np.zeros((chessboard_cells, 3), np.float32) # 3 floats (xyz) allocated for each board cell each set to zero.
object_points_world[:, :2] = np.mgrid[0:chessboard_width, 0:chessboard_height].T.reshape(-1, 2) # 2D Grid coordinates flattened into pair list, and then assigned to x's and y's of objp.
object_points_world *= cell_size # Multiply by cell size to get accurate world coordinates for each corner point.

# Creates window (of given name) to show a given image.
def show_image(image, window):
    # Create window.
    cv.namedWindow(window, cv.WINDOW_NORMAL)
    # Resize window.
    if resize_windows:
        h, w = image.shape[:2]
        cv.resizeWindow(window, 2 * w, 2 * h)
    # Draw the image with cube and axis.
    cv.imshow(window, image)


# Ran during manual corner selection, handles clicking on the screen.
def manual_click_event(event, x, y, flags, params):
    # Manual corner selection if chessboard corners can't be found automatically.
    img_display, points, window_name = params

    if event == cv.EVENT_LBUTTONDOWN:
        if len(points) < 4:

            # Store the point.
            points.append((x, y))
            print(f"Captured point {len(points)}: ({x}, {y})")

            # Visual Feedback: Draw a circle and order number
            cv.circle(img_display, (x, y), 2, (0, 255, 0), -1) # Green circle
            cv.putText(img_display, str(len(points)), (x + 10, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv.imshow(window_name, img_display)

            if len(points) == 4:
                print("Captured all 4 points. You can now close the window.")

# Ran during manual corner selection.
# Choice 3. Calculate corners in warped space (through homography), and return them in original perspective.
def get_inner_corners_warping(image, corners, grid_size):
    # Use Homography to warp the image to a flat view, refine corners and map them back

    # Convert clicked corners to numpy array.
    src_corners = np.array(corners, dtype=np.float32)
    cols, rows = grid_size

    # Arbitrary size that is large enough to preserve detail
    square_size_pixels = 50

    # Calculate width/height of the "flat" virtual image
    w_virtual = cols * square_size_pixels
    h_virtual = rows * square_size_pixels

    # Destination corners in the virtual flat image (0,0), (w,0), (w,h), (0,h)
    dst_corners = np.array([[0, 0], [w_virtual, 0], [w_virtual, h_virtual], [0, h_virtual]], dtype=np.float32)

    # Compute Homography from clicked corners to virtual corners.
    H, _ = cv.findHomography(src_corners, dst_corners)

    # Warp the image (forward)
    warped_image = cv.warpPerspective(image, H, (w_virtual, h_virtual))

    # Find corners in the warped image (should be easier since it's flat).
    flat_points = []
    for r in range(rows):
        for c in range(cols):
            x = c * square_size_pixels
            y = r * square_size_pixels
            flat_points.append([[x, y]]) # Center of each square
    
    flat_points = np.array(flat_points, dtype=np.float32)

    # Refine using image content
    # The gradients are now orthogonal, so we can use a search window of half the square size.
    win_size = (int(square_size_pixels // 2), int(square_size_pixels // 2))

    refined_flat_points = cv.cornerSubPix(warped_image, flat_points, win_size, (-1, -1), criteria)

    # Warp back
    # Invert the Homography matrix to map from virtual flat image back to original image.
    H_inv = np.linalg.inv(H)

    refined_original_points = cv.perspectiveTransform(refined_flat_points, H_inv)

    return refined_original_points

# Ran during manual corner selection, linearly interpolates between the 4 selected corners,
# to obtain all the chessboard cell corners.
def get_inner_corners_manual(corners, grid_size):
    # Simple linear interpolation to get inner corners from the 4 outer corners provided by the user.
    tl, tr, br, bl = corners
    cols, rows = grid_size

    # Generate left and right edges
    left_edge = np.linspace(tl, bl, rows)
    right_edge = np.linspace(tr, br, rows)

    grid_points = []

    # For each row, interpolate between left and right edge
    for i in range(rows):
        row_points = np.linspace(left_edge[i], right_edge[i], cols)
        grid_points.append(row_points)
    
    return np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2) # Reshape to match OpenCV format (N, 1, 2)

# Perform pre-processing pipeline (noise reduction -> gray -> sharpening)
def pre_processing(image):
    if not use_preprocessing:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Noise Reduction on color, with bilateral filtering (blur)
    blur = cv.bilateralFilter(image,9,25,13)

    # Grayscale
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

    # Sharpening via unsharp (Gaussian) mask (edge strengthening).
    blur = cv.GaussianBlur(gray, (9,9), 0)
    mask_weight = -0.4 # Determined through trial and error.
    sharp = cv.addWeighted(gray, 1.0, blur, mask_weight, 0)

    return sharp

def offline():
    # Set placeholder image size, will be set to last image size from calibration.
    # Assumes all images used in calibration are the same size (and in general the function expects equal camera (settings)).
    image_size = (0, 0)

    # Arrays to store object points and image points from all the images.
    object_points = [] # 3d point in real world space.
    image_points = [] # 2d points in image plane.

    size = len(calibration_images)

    # Update arrays with corner points from the images.
    for filename in calibration_images:
        # Read image and turn into gray scale.
        image = cv.imread(filename)
        processed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_size = processed_image.shape[::-1] # Update image size.

        # Find the chess board corners
        has_found_corners, corners = cv.findChessboardCorners(processed_image, chessboard_size, cv.CALIB_CB_NORMALIZE_IMAGE)

        # Apply pre-processing (if allowed) when automatic detection fails, and try again.
        if use_preprocessing and not has_found_corners:
            processed_image = pre_processing(image)
            has_found_corners, corners = cv.findChessboardCorners(processed_image, chessboard_size, cv.CALIB_CB_NORMALIZE_IMAGE)

        # To skip automatically detected images (including pre-processed ones).
        found_auto = has_found_corners

        # Detection failed (even with pre-processing - if allowed)
        if not has_found_corners:
            print(f"Corners not found for image {filename}. Switching to manual mode.")
            print("Please click the 4 outer corners of the INNER grid in this order:")
            print("1. Top-Left -> 2. Top-Right --> 3. Bottom-Right --> 4. Bottom-Left")
            print("Remember that this is meant vertically!")

            # Create a copy of the image for display and clicking.
            manual_display = image.copy()
            manual_corners = []

            # Set up window and callback
            show_image(manual_display, window_auto_corners)
            cv.setMouseCallback(window_auto_corners, manual_click_event, [manual_display, manual_corners, window_auto_corners])

            # Wait until 4 points are clicked
            while len(manual_corners) < 4:
                cv.waitKey(10)

            if use_warping:
                # Use warping method to get corners.
                # We pass gray_image to the warping function so that it can refine corners in the warped space using image content
                corners = get_inner_corners_warping(processed_image, manual_corners, chessboard_size)
            else:
                # Get all corners from the 4 clicked points.
                corners = get_inner_corners_manual(manual_corners, chessboard_size)

            has_found_corners = True

            # Disable callback.
            cv.setMouseCallback(window_auto_corners, lambda *args : None)

        #If corners found, add object points, and image points (after refining them).
        if has_found_corners:
            print("to go: ", str(size))
            size -= 1
            # Add object points (static; only dependent on chessboard size)
            object_points.append(object_points_world)

            # Add refined image points.
            refined_corners = cv.cornerSubPix(processed_image, corners, (11, 11), (-1, -1), criteria)
            image_points.append(refined_corners)

            # Draw and display the corners
            if skip_auto_images and found_auto: continue
            cv.drawChessboardCorners(image, chessboard_size, refined_corners, has_found_corners)

            # Show Image
            show_image(image, window_auto_corners)
            # Wait for user input to continue to next image.
            cv.waitKey(0)
            cv.destroyAllWindows()

    # Use corners to calibrate camera.
    if image_size == (0, 0):
        raise Exception("Last image either 0x0, or no valid images provided.")
    print("Calculating intrinsics...")
    return cv.calibrateCamera(object_points, image_points, image_size, None, None)

# Draws 3D axes gizmo on the origin top-left cell of the chessboard.
def draw_axes(axis_points, rvecs, tvecs, intrinsic, distortion, refined_corners, image):
    # Convert axis points: world space (3d) -> image space (2d).
    image_points, _ = cv.projectPoints(axis_points, rvecs, tvecs, intrinsic, distortion) # Project 3D axis points to image plane (2D).
    image_points = image_points.astype("int32") # Drawing requires (2D) int coordinates (this does truncate btw).

    # Get bottom corner (already in image space).
    corner = tuple(refined_corners[0].ravel().astype("int32"))

    # Draw each axis.
    updated_image = cv.line(image, corner, tuple(image_points[0].ravel()), (0, 0, 255), thickness_axes) # Width (x) - red
    updated_image = cv.line(updated_image, corner, tuple(image_points[1].ravel()), (0, 255, 0), thickness_axes) # Height (y) - green
    updated_image = cv.line(updated_image, corner, tuple(image_points[2].ravel()), (255, 0, 0), thickness_axes) # Depth (z) - blue

    # Text
    if should_draw_text:
        updated_image = cv.putText(updated_image, "Showing Axes", (x_text, height_interval_text), cv.FONT_HERSHEY_SIMPLEX, scale_text, (0, 255, 0), thickness_text)

    return updated_image

# Online phase of the program, only on first frame of video.
# (Since the checkerboard is .avi file for some reason instead of just a picture).
def online(intrinsic, distortion):
    # Prepare projection points in 3D world space (constant through images obv.).
    axis_size = 4 * cell_size
    axis_points = np.float32([[axis_size, 0, 0], [0, axis_size, 0], [0, 0, -axis_size]]).reshape(-1, 3)

    # Open video file
    cap = cv.VideoCapture(test_images[0]) # Assuming test_images contains the path to the video file.

    if not cap.isOpened():
        print("Cannot open video file")
        return

    ret, frame = cap.read()
    cap.release() # [!] We only need the first frame

    if not ret:
        print("Can't read video file. Exiting ...")
        return


    # Find the chess board corners
    processed_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    print("Find corners manually")
    manual_display = frame.copy()
    manual_corners = []

    show_image(manual_display, window_online)
    cv.setMouseCallback(window_online, manual_click_event, [manual_display, manual_corners, window_online])

    while len(manual_corners) < 4:
        cv.waitKey(10)

    if use_warping:
        corners = get_inner_corners_warping(processed_image, manual_corners, chessboard_size)
    else:
        corners = get_inner_corners_manual(manual_corners, chessboard_size)

    cv.setMouseCallback(window_online, lambda *args: None)

    # Refine corners
    refined_corners = cv.cornerSubPix(processed_image, corners, (11, 11), (-1, -1), criteria)

    # Solve PnP to get pose.
    has_pose, rvecs, tvecs = cv.solvePnP(object_points_world, refined_corners, intrinsic, distortion)
    if has_pose:
        # Draw axes
        if should_draw_axes:
            frame = draw_axes(axis_points, rvecs, tvecs, intrinsic, distortion, refined_corners, frame)

    # Show the frame
    show_image(frame, window_online)

    print("Press any key to close and finish...")
    cv.waitKey(0)
    cv.imwrite(axis_image, frame)
    cv.destroyAllWindows()

    # Return the extrinsics
    return rvecs, tvecs

# Save camera data to an XML file
def save_camera_config(filepath, reprojection_error, intrinsic, distortion, rotation, translation):
    print(f"Saving camera configuration to {filepath}")

    # Open the file in WRITE mode
    fs = cv.FileStorage(filepath, cv.FILE_STORAGE_WRITE)

    if not fs.isOpened():
        print(f"Failed to open {filepath} for writing.")
        return
    
    # Write data to the file
    fs.write("CameraMatrix", intrinsic)
    fs.write("DistortionCoefficients", distortion)

    # We convert the tuples of arrays into a single numpy array so OpenCV can format them in the XML
    fs.write("RotationVectors", np.asarray(rotation))
    fs.write("TranslationVectors", np.asarray(translation))

    fs.write("ReprojectionError", reprojection_error)

    # Release the file
    fs.release()
    print("Camera configuration saved successfully.")

def main():
    reprojection_error, intrinsic, distortion, _, _ = offline()
    print("Reprojection Error (px):", reprojection_error)
    print("intrinsic / Camera Matrix:")
    print(intrinsic)

    rotation, translation = online(intrinsic, distortion)
    print("rotation:")
    print(rotation)
    print("translation:")
    print(translation)

    save_camera_config(config_path, reprojection_error, intrinsic, distortion, rotation, translation)
main()