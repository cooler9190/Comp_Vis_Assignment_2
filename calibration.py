import numpy as np
import cv2 as cv
import glob

# Using Code from https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html [source_1]
# and https://docs.opencv.org/4.x/d7/d53/tutorial_py_pose.html [source_2]

# Run selection.
calibration_images = glob.glob('data/cam4/calibration photos/*.png') # [!] RUN 1: GIVE FULL DIRECTORY
# calibration_images = glob.glob('run_2/*.png') # [!] RUN 2: GIVE FULL DIRECTORY
# calibration_images = glob.glob('run_3/*.png') # [!] RUN 3: GIVE FULL DIRECTORY
test_images = glob.glob('data/cam4/checkerboard.avi') # [!] PATH SHOULD BE OUTSIDE CALIBRATION/ FOLDER. Is ignored during live feed.

# General settings
use_preprocessing = True # Toggle Choice 5
use_warping = True # Toggle Choice 3
use_live_feed = False # Toggle Choice 1
use_video_file = True
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
resize_windows = False
window_auto_corners = 'Chessboard Corners'
window_online = 'Axis and Cube'

# Chessboard settings.
chessboard_size = (8, 6) # Note that the board is 9 (width) by  7 (height) cells. Therefore, the inner size (w-1, h-1) is 8x6.
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
        ratio = 800 / w
        cv.resizeWindow(window, 800, int(h * ratio))
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
            cv.circle(img_display, (x, y), 5, (0, 255, 0), -1) # Green circle
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

    # Update arrays with corner points from the images.
    for filename in calibration_images:
        # Read image and turn into gray scale.
        image = cv.imread(filename)
        processed_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image_size = processed_image.shape[::-1] # Update image size.

        # Find the chess board corners
        has_found_corners, corners = cv.findChessboardCorners(processed_image, chessboard_size, None)

        # Apply pre-processing (if allowed) when automatic detection fails, and try again.
        if use_preprocessing and not has_found_corners:
            processed_image = pre_processing(image)
            has_found_corners, corners = cv.findChessboardCorners(processed_image, chessboard_size, None)

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
            print("Corners found for image:", filename)
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

# Calculates distance from center point of cube's top face, to the camera.
def calculate_top_center_distance(rotation_matrix, tvecs, top_center):
    # Obtain top center in camera space.
    top_center_camera = rotation_matrix @ top_center.reshape(3,1) + tvecs # x_cam = R * x_world + t

    # Compute distance from camera (origin in camera space) to top center (in camera space)
    distance = np.linalg.norm(top_center_camera) # Just pythagoras (length or normalization factor)
    if distance < 0: raise Exception("Distance between camera and top center is negative?! UH OH") # Distance can't be negative!
    return distance

# HSV, Value component based on distance between Center Point on cube's top face and Camera.
# A greater distance means a lower value: [0, 4] meters -> [255, 0] value.
def value_from_distance(distance):
    # Clamp distance and modulate value.
    clamped_distance = np.clip(distance, 0, 4) # Clip ensures d > 4 -> d = 4.
    return int(np.round(255 * (1 - clamped_distance / 4))) # [0, 4] -> [255, 0].

# Given a hue, saturation and value, computes a tuple in Blue-Green-Red color space.
def BGR_from_HSV(hue, saturation, value):
    hsv_image = np.uint8([[[hue, saturation, value]]]) # Create 1x1 pixel fake image with hsv color
    bgr_image = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)[0,0] # Then convert color space of the image, and extract pixel color at (0, 0).
    return tuple(int(c) for c in bgr_image) # Turn array into tuple.

# HSV, Hue component based on angle between cube's top face and the Camera.
# A greater angle means a lower hue: [0, 45] -> [179, 0] hue.
def calculate_hue(rotation_matrix):
    # cos θ = board_normal ⋅ cam_normal / ||b_n|| x ||c_n||. c_n is forward column (when used camera space).
    forward_column = np.array([0, 0, 1]).reshape(3,1) # Forward column vector (space-independent)
    board_normal = rotation_matrix @ forward_column # Obtain normal in camera space.
    board_normal_length = np.linalg.norm(board_normal) # Just pythagoras (length or normalization factor)
    cos_theta = np.dot(board_normal.ravel(), forward_column.ravel()) / board_normal_length # Note that c_n has length 1, so just divide by b_n length.

    # Clamp angle and modulate hue.
    theta_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0)) # Clips for safety, due to floating point errors.
    theta_degrees = np.degrees(theta_radians) # Convert to degrees.
    clamped_angle = np.clip(theta_degrees, 0, 45) # Clip ensures a > 45 -> a = 45 AND a < 0 -> a = 0
    hue = int(np.round(179 * (1 - clamped_angle / 45)))
    return hue # [0, 45] -> [179, 0]

# Draws a 3D cube on the top-left (inner) cell of the chessboard.
def draw_cube(cube_points, rvecs, tvecs, intrinsic, distortion, image):
    # Convert cube point: world space (3d) -> image space (2d).
    image_points, _ = cv.projectPoints(cube_points, rvecs, tvecs, intrinsic, distortion) # Project 3D cube points to image plane (2d).
    image_points = np.int32(image_points).reshape(-1, 2) # Drawing requires (2D) int coordinates (this does truncate btw).

    # Draw bottom face.
    cube_color = (250, 250, 0)
    updated_image = cv.drawContours(image, [image_points[:4]],-1, cube_color, thickness_cube)
    # Draw pillars.
    for i,j in zip(range(4),range(4,8)):
        updated_image = cv.line(updated_image, tuple(image_points[i]), tuple(image_points[j]), cube_color, thickness_cube)

    # Obtain top points and center (world space)
    top_points = cube_points[4:]
    top_center = top_points.mean(axis=0)

    # Calculate distance from center to camera (in meters).
    rotation_matrix, _ = cv.Rodrigues(rvecs) # Obtain rotation matrix from rvecs.
    distance = calculate_top_center_distance(rotation_matrix, tvecs, top_center)

    # Calculate value for top face color using distance.
    value = value_from_distance(distance)

    # Calculate hue for top face color using angle.
    hue = calculate_hue(rotation_matrix)

    # Draw top face using hue and value for the color.
    top_face_color = BGR_from_HSV(hue, 255, value)
    updated_image = cv.fillConvexPoly(updated_image, image_points[4:], top_face_color) # so it can be used when drawing top face.
    # [!] Alt. method? image = cv.drawContours(image, [image_points[4:]],-1,cube_color, -1)

    # Draw top face center point.
    top_center_projected, _ = cv.projectPoints(top_center, rvecs, tvecs, intrinsic, distortion)
    top_center_projected = top_center_projected[0][0].astype("int32")
    updated_image = cv.circle(updated_image, top_center_projected, 3 * thickness_cube, (255, 255, 255), -1)

    # Draw distance text.
    if should_draw_text:
        dist_text = "Distance (m): " + str(distance)
        dist_text_short = "Distance (m): " + str(round(distance, 2))
        pos =  top_center_projected + (20, -15) # Little to the top right.
        updated_image = cv.putText(updated_image, dist_text_short, pos, cv.FONT_HERSHEY_SIMPLEX, scale_text, top_face_color, thickness_text)
        # Draw the same text at the top of the screen, just in case it goes off-screen.
        updated_image = cv.putText(updated_image, dist_text, (x_text, height_interval_text * 3), cv.FONT_HERSHEY_SIMPLEX, scale_text, top_face_color, thickness_text)

    # Text
    if should_draw_text:
        updated_image = cv.putText(updated_image, "Showing Cube", (x_text, height_interval_text * 2), cv.FONT_HERSHEY_SIMPLEX, scale_text, cube_color, thickness_text)

    return updated_image

# Online (live detection) phase of the program.
def online(intrinsic, distortion):
    # Prepare projection points in 3D world space (constant through images obv.).
    axis_size = 4 * cell_size
    axis_points = np.float32([[axis_size, 0, 0], [0, axis_size, 0], [0, 0, -axis_size]]).reshape(-1, 3)
    cube_points = np.float32([[0,0, 0], [0,cell_size, 0], [cell_size,cell_size, 0], [cell_size,0, 0], [0,0,-cell_size],
                              [0,cell_size,-cell_size], [cell_size,cell_size,-cell_size], [cell_size,0,-cell_size]])

    # Use webcam
    if use_live_feed:
        # Initialize video capture (webcam).
        cap = cv.VideoCapture(camera_index)

        if not cap.isOpened():
            print("Cannot open camera")
            return

        print("Starting video stream.")

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()

            # If frame is read correctly ret is True
            if not ret:
                print("Can't receive frame. Exiting ...")
                break

            gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Find the chess board corners
            # flags=cv.CALIB_CB_FAST_CHECK helps speed up the detection on frames where no board is present.
            has_found_corners, corners = cv.findChessboardCorners(gray_image, chessboard_size, flags=cv.CALIB_CB_FAST_CHECK)

            if has_found_corners:
                # Refine corners
                refined_corners = cv.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

                # Solve PnP to get pose.
                # We use the intrinsic matrix and distortion coefficients obtained from the offline phase to get the pose of the chessboard in the current frame.
                has_pose, rvecs, tvecs = cv.solvePnP(object_points_world, refined_corners, intrinsic, distortion)

                if has_pose:
                    # Draw axes
                    if should_draw_axes:
                        frame = draw_axes(axis_points, rvecs, tvecs, intrinsic, distortion, refined_corners, frame)

                    # Draw cube
                    if should_draw_cube:
                        frame = draw_cube(cube_points, rvecs, tvecs, intrinsic, distortion, frame)

            # Show the frame
            cv.imshow(window_online, frame)

            # Exit on 'q' key press
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Release everything when done
        cap.release()
        cv.destroyAllWindows()
    
    # Using video file
    if use_video_file:
        # Open video file
        cap = cv.VideoCapture(test_images[0]) # Assuming test_images contains the path to the video file.

        if not cap.isOpened():
            print("Cannot open video file")
            return

        ret, frame = cap.read()
        cap.release() # We only need the first frame

        if not ret:
            print("Can't read video file. Exiting ...")
            return

        gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        print("Find corners manually")
        manual_display = frame.copy()
        manual_corners = []

        show_image(manual_display, window_online)
        cv.setMouseCallback(window_online, manual_click_event, [manual_display, manual_corners, window_online])

        while len(manual_corners) < 4:
            cv.waitKey(10)

        if use_warping:
            corners = get_inner_corners_warping(gray_image, manual_corners, chessboard_size)
        else:
            corners = get_inner_corners_manual(manual_corners, chessboard_size)

        cv.setMouseCallback(window_online, lambda *args: None)
        
        # Refine corners
        refined_corners = cv.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

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
        cv.destroyAllWindows()

        # Return the extrinsics
        return rvecs, tvecs
    
    # Using test_images.
    else:
        # Draw axes and cube on test images
        for filename in test_images:
            # Read image and Convert to Gray Scale.
            image = cv.imread(filename)
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # Add flags to make detection more robust against lighting/compression
            detection_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

            # Find the chess board corners
            has_found_corners, corners = cv.findChessboardCorners(gray_image, chessboard_size, flags=detection_flags)

            if use_preprocessing and not has_found_corners:
                # If not found, apply pre-processing and try again.
                processed_image = pre_processing(image)
                has_found_corners, corners = cv.findChessboardCorners(processed_image, chessboard_size, flags=detection_flags)

            # If corners found, add object points, and image points (after refining them).
            if has_found_corners:
                # Get refined image points.
                refined_corners = cv.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)

                # Find the rotation and translation vectors for the test image using intrinsics.
                has_calibrated, rvecs, tvecs = cv.solvePnP(object_points_world, refined_corners, intrinsic, distortion)

                # Draw axes on bottom left corner of chessboard.
                if should_draw_axes:
                    image = draw_axes(axis_points, rvecs, tvecs, intrinsic, distortion, refined_corners, image)

                # Draw Cube
                if should_draw_cube:
                    image = draw_cube(cube_points, rvecs, tvecs, intrinsic, distortion, image)

                # Show Image
                show_image(image, window_online)
                # Wait for user input to continue to next image.
                cv.waitKey(0)
                cv.destroyAllWindows()
            else:
                print(f"Corners not found for test image {filename}. Cannot draw axes or cube.")

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
    # print("Reprojection Error (px):", reprojection_error)
    # print("intrinsic / Camera Matrix:")
    # print(intrinsic)
    # print("rotation:")
    # print(rotation)
    # print("translation:")
    # print(translation)

    rotation, translation = online(intrinsic, distortion)

    config_path = "data/cam4/cam4_config.xml" # [!] PATH SHOULD BE OUTSIDE CALIBRATION/ FOLDER.
    save_camera_config(config_path, reprojection_error, intrinsic, distortion, rotation, translation)


main()