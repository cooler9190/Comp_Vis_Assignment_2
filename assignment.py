import glm
import random
import numpy as np
import cv2 as cv
import os
from voxel_reconstruction import VoxelReconstructor

block_size = 0.035


# Iniitialization & Setup
print("Initializing system for 4 cameras...")

# Extract camera extrinsics from XML
cameras = []
for i in range(1, 5):
    filepath = f'data/cam{i}/cam{i}_config.xml'
    cv_file = cv.FileStorage(filepath, cv.FILE_STORAGE_READ)
    rvec = cv_file.getNode('RotationVectors').mat()
    tvec = cv_file.getNode('TranslationVectors').mat()
    cv_file.release()

    # Convert rotation vector to 3x3 rotation matrix
    R, _ = cv.Rodrigues(rvec)
    cameras.append({'R': R, 't': tvec})

# Initialize Voxel Reconstructor
reconstructor = VoxelReconstructor(config_file='config.json', block_size=block_size)

# Iniitialize Background Subtractor and Video Streams
fgbg_models = []
video_caps = []

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
print("Training background models for each camera. Please wait...")
for i in range(1, 5):
    # Setup subtractor for each camera
    fgbg = cv.createBackgroundSubtractorMOG2(varThreshold=32, detectShadows=True)

    # Train on background video
    cap_bg = cv.VideoCapture(f'data/cam{i}/background.avi')
    while True:
        ret, frame = cap_bg.read()
        if not ret:
            print(f"Camera {i}: Can't receive frame. Exiting background training...")
            break
        fgbg.apply(frame)
    cap_bg.release()
    fgbg_models.append(fgbg)

    # Open test video streams
    cap_test = cv.VideoCapture(f'data/cam{i}/video.avi')
    video_caps.append(cap_test)

print("Initialization complete. Press 'G' to advance frames")

# Assignment Functions
def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([(x - width/2) * block_size, -block_size, (z - depth/2) * block_size])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.

    # Reads 1 frame from all 4 cameras, applies background subtraction, 
    # and uses the resulting masks to determine which voxels are ON/OFF in the first frame.

    current_masks = []

    # Process all 4 cameras
    for i in range(4):
        ret, frame = video_caps[i].read()
        if not ret:
            print("End of video reached!")
            return [], [] # Stop drawing for input if video ends
        
        # Get foreground mask for this frame and camera
        fg_mask = fgbg_models[i].apply(frame, learningRate=0)

        # Morphological operations to clean up the foreground mask
        erosion = cv.erode(fg_mask, kernel, iterations=1)
        opening = cv.dilate(erosion, kernel, iterations=1)

        dilation = cv.dilate(opening, kernel, iterations=5)
        closing = cv.erode(dilation, kernel, iterations=5)

        # Ensurepure binary mask
        _, binary_mask = cv.threshold(closing, 254, 255, cv.THRESH_BINARY)
        current_masks.append(binary_mask)

    # Reconstruct the 3D volume using XOR optimization
    raw_positions, _ = reconstructor.process_frame(current_masks)

    # Convert OpenCV to OpenGL coordinates and assign colors based on height
    opengl_positions, opengl_colors = [], []
    for pos in raw_positions:
        x, y, z = pos

        # Adjust these flips if person is rendering in the wrong orientation
        opengl_x = x
        opengl_y = -z # in OpenCV is often height if calibrated on the floor
        opengl_z = y # in OpenCV is often depth

        opengl_positions.append([opengl_x, opengl_y, opengl_z])

        # Color based on height (y value)
        color_val = (opengl_y / height) if height > 0 else 1.0
        opengl_colors.append([color_val, 0.5, 1.0 - color_val]) # Gradient from green to red based on height

    print(f"Frame processed: {len(opengl_positions)} voxels ON")
    return opengl_positions, opengl_colors

    # data, colors = [], []
    # for x in range(width):
    #     for y in range(height):
    #         for z in range(depth):
    #             if random.randint(0, 1000) < 5:
    #                 data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    #                 colors.append([x / width, z / depth, y / height])
    # return data, colors


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.

    # Calculates the real world camera positions from OpnCV extrinsics
    # Camera position C = -R^T * t

    cam_positions = []
    cam_colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

    for cam in cameras:
        R = cam['R']
        t = cam['t']

        # Calcualte camera center in OpenCV world coordinates
        C = -np.matrix(R).T * np.matrix(t)

        # Extract X, Y, Z
        x, y, z = C[0], C[1], C[2]

        # Match transformations to the voxels
        opengl_x = x
        opengl_y = -z # in OpenCV is often height if calibrated on the floor
        opengl_z = y # in OpenCV is often depth

        scale = 1.0

        cam_positions.append([opengl_x * scale, opengl_y * scale, opengl_z * scale])

    return cam_positions, cam_colors

    # return [[-64 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, -64 * block_size],
    #         [-64 * block_size, 64 * block_size, -64 * block_size]], \
    #     [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.

    # Converts the OpenCV 3x3 roation matrices into 4x4 OpenGL model matrices
    cam_rotations = []

    for cam in cameras:
        # R transforms World -> Camera. We need Camera -> World, which is R^T
        R = cam['R']
        R_c2w = R.T

        # Create a 4x4 transformation matrix
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = R_c2w

        # Convert to GLM matrix
        glm_mat = glm.mat4(*mat.flatten())

        # Rotate 180 degrees around the X-axis because
        # OpenCV cameras look down +Z, but OpenGL cameras look down -Z
        glm_mat = glm.rotate(glm_mat, np.pi, glm.vec3(1, 0, 0))

        cam_rotations.append(glm_mat)
    
    return cam_rotations
    # cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    # cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    # for c in range(len(cam_rotations)):
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    # return cam_rotations
