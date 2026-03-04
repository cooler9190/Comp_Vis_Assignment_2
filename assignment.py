import os
import glm
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
    x, y, z = np.meshgrid(x_points, y_points, z_points, indexing='ij')
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=-1)

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

def create_lookup_table(cube_grid, cameras):
    cube_size = cube_grid.shape[0]
    # Convert cube to correct coordinates (opengl)
    world_cube = cube_grid.astype(np.float64) * 0.05 # Scale to fit.
    world_cube = np.stack([world_cube[:, 0], world_cube[:, 2], -world_cube[:, 1]], axis=1) # openGL format.

    # For each camera, create array with image points.
    # These image points, are the voxel cube's points projected onto the current camera.
    # The order is therefore consistent.
    table = np.empty((4, cube_size, 2), dtype=int)
    c = 0
    for camera in cameras:
        r = camera['rvec'].astype(np.float64)
        t = camera['tvec'].astype(np.float64)
        k = camera['cam_matrix'].astype(np.float64)
        d = camera['distortion'].astype(np.float64)
        # Can give points outside of image, since the cube contains points not visible with any camera (possibly).
        image_points, _ = cv.projectPoints(world_cube, r, t, k, d)
        rounded_points = np.rint(image_points[:, 0, :]).astype(int)
        table[c] = rounded_points
        c += 1

    return table

def is_voxel_foreground(lookup_table, voxel_index, foregrounds):
    for camera_index in range(0, 4):
        foreground = foregrounds[camera_index] # Current foreground.
        x, y = lookup_table[camera_index, voxel_index] # Image coordinate

        # Boundary checking (since project points can give coordinates outside of image).
        x_outside = x < 0 or x >= foreground.shape[1]
        y_outside = y < 0 or y >= foreground.shape[0]
        if x_outside or y_outside: return False # Outside of camera views.

        # Voxel not visible by current camera, so return false.
        if foreground[y, x] <= 0: return False # Flipped x y due to format.

    # Voxel visible in all cameras.
    return True

def set_voxel_positions(lookup_table, cube_grid):
    # Reads 1 frame from all 4 cameras, applies background subtraction,
    # and uses the resulting masks to determine which voxels are ON/OFF in the first frame.
    if current_frame > (frame_max - 1): return Exception("Frame limit Reached")

    # Obtain the 4 foreground for this frame.
    foregrounds = []
    for cam in range(1, 5):
        camera_path = os.path.join(f"data/cam{cam}/")
        foreground_path = camera_path + f"foreground/{current_frame}.png"
        foreground = cv.imread(foreground_path, cv.COLOR_BGR2GRAY)
        foregrounds.append(foreground)

   # Create Truth/False mask for the cube, based on visible voxels.
    voxel_count = cube_grid.shape[0]
    cube_mask = np.full(voxel_count, False, dtype=bool)
    for v in range(0, voxel_count):
        cube_mask[v] = is_voxel_foreground(lookup_table, v, foregrounds)
    # Create cube where all visible voxels are marked.
    visible_voxels = cube_grid[cube_mask] # This turns off all non-visible voxels.

    # Default color is white.
    colors = np.ones((visible_voxels.shape[0], 3), dtype=np.float32)
    return visible_voxels, colors

def get_cam_positions(cameras):
    # Empty init
    cam_positions = []
    cam_colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]

    # For every camera apply extrinsic variables and obtain position in openGL coordinate system.
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


def get_cam_rotation_matrices(cameras):

    # Converts the OpenCV 3x3 rotation matrices into 4x4 OpenGL model matrices
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
