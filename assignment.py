import os
import glm
import random
import cv2 as cv
import numpy as np

block_size = 1

current_frame = 0
frame_max = 100
def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
            colors.append([1.0, 1.0, 1.0] if (x+z) % 2 == 0 else [0, 0, 0])
    return data, colors

def create_cube_grid(width, height, depth):
    # Create ranges
    x_range = (-width / 2 * block_size, width / 2 * block_size)
    y_range = (0, height * block_size)
    z_range = (-depth / 2 * block_size, depth / 2 * block_size)
    # Create axis points
    x_points = np.arange(x_range[0], x_range[1], block_size)
    y_points = np.arange(y_range[0], y_range[1], block_size)
    z_points = np.arange(z_range[0], z_range[1], block_size)
    
    # Return 3d grid.
    X, Y, Z = np.meshgrid(x_points, y_points, z_points, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

def load_camera_parameters():
    cameras = []
    for c in range(1, 5):
        filepath = f'data/cam{c}/config.xml'
        cv_file = cv.FileStorage(filepath, cv.FILE_STORAGE_READ)
        cameras.append({
            'cam_matrix': cv_file.getNode('CameraMatrix').mat(),
            'distortion': cv_file.getNode('DistortionCoefficients').mat(),
            'rvec': cv_file.getNode('RotationVectors').mat(),
            'tvec': cv_file.getNode('TranslationVectors').mat()
        })
        cv_file.release()
    return cameras

def create_lookup_table(width, height, depth):
    cube_grid = create_cube_grid(width, height, depth)



def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.

    # Reads 1 frame from all 4 cameras, applies background subtraction,
    # and uses the resulting masks to determine which voxels are ON/OFF in the first frame.
    if current_frame > (frame_max - 1): return Exception("Frame limit Reached")

    grid = create_cube_grid(width, height, depth)


    # Obtain the 4 foreground for this frame.
    foregrounds = []
    for cam in range(1, 5):
        camera_path = os.path.join(f"data/cam{cam}/")
        foreground_path = camera_path + f"foreground/{current_frame}.png"
        foreground = cv.imread(foreground_path, cv.COLOR_BGR2GRAY)
        foregrounds.append(foreground)

    data, colors = [], []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
                colors.append([x / width, z / depth, y / height])
    return data, colors


def get_cam_positions(cameras):
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.

    cam_positions = []
    cam_colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

    for cam in cameras:
        R = cam['rvec']
        R, _ = cv.Rodrigues(R)
        t = cam['tvec']

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


def get_cam_rotation_matrices(cameras):
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.

    # Converts the OpenCV 3x3 roation matrices into 4x4 OpenGL model matrices
    cam_rotations = []

    for cam in cameras:
        # R transforms World -> Camera. We need Camera -> World, which is R^T
        R = cam['rvec']
        R, _ = cv.Rodrigues(R)
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

    cam_angles = [[0, 45, -45], [0, 135, -45], [0, 225, -45], [0, 315, -45]]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
