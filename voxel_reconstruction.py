import cv2 as cv
import numpy as np
import os
import json
from collections import defaultdict

class VoxelReconstructor:
    def __init__(self, data_dir='data', config_file='config.json', block_size=1.0):
        self.data_dir = data_dir
        self.block_size = block_size

        # Load scene dimensions from json
        with open(config_file, 'r') as f:
            self.config = json.load(f)

        # Extract values from config
        self.width = self.config.get('world_width', 0)
        self.height = self.config.get('world_height', 0)
        self.depth = self.config.get('world_depth', 0)

        self.num_voxels = self.width * self.height * self.depth

        # State tracking
        self.voxel_coords = None # 3D coordinates of each voxel
        self.voxel_states = np.zeros(self.num_voxels, dtype=bool) # True = ON, False = OFF
        self.previous_masks = None # Masks from the previous frame for XOR

        # Look-Up Tables
        # lookup_table[cam_index][voxel_index] = (x, y)
        self.lookup_table = [[] for _ in range(4)] # 4 cameras, each with a list of voxel indices that project onto it
        # reverse_lookup_table[cam_index][(x, y)] = [list of voxel_indices]
        self.reverse_lookup = [defaultdict(list) for _ in range(4)] # 4 cameras, each with a dict mapping (x, y) to list of voxel indices

        self._initialize_lookup_tables()

    def _load_camera_params(self):
        # Load camera metrics from the XML

        cameras = []
        for i in range(1, 5):
            filepath = os.path.join(self.data_dir, f'cam{i}', f'cam{i}_config.xml')
            cv_file = cv.FileStorage(filepath, cv.FILE_STORAGE_READ)

            cameras.append({
                'camera_matrix': cv_file.getNode('CameraMatrix').mat(),
                'dist_coeffs': cv_file.getNode('DistortionCoefficients').mat(),
                'rvec': cv_file.getNode('RotationVectors').mat(),
                'tvec': cv_file.getNode('TranslationVectors').mat()
            })
            cv_file.release()
        return cameras

    def _initialize_lookup_tables(self):
        # Build the forward and reverse lookup tables once at startup

        print("Initializing lookup tables...")

        # Generate 3D grid
        voxel_coords = []
        for x in range(self.width):
            for y in range(self.height):
                for z in range(self.depth):
                    pos_x = (x - self.width / 2) * self.block_size
                    pos_y = (y - self.height / 2) * self.block_size
                    pos_z = (z - self.depth / 2) * self.block_size
                    voxel_coords.append([pos_x, pos_y, pos_z])
        self.voxel_coords = np.array(voxel_coords, dtype=np.float32)

        # Load cameras and project
        cameras = self._load_camera_params()

        for cam_index, cam in enumerate(cameras):
            img_points, _ = cv.projectPoints(
                self.voxel_coords, cam['rvec'], cam['tvec'], 
                cam['camera_matrix'], cam['dist_coeffs']
            )
            
            img_points = np.round(img_points.reshape(-1, 2).astype(int))

            # Popilate Forward and Reverse Lookup Tables
            for voxel_index, (x, y) in enumerate(img_points):
                self.lookup_table[cam_index].append((x, y))
                # Add this voxel to the list affected by this pixel in the reverse lookup
                self.reverse_lookup[cam_index][(x, y)].append(voxel_index)

        print("Lookup tables initialized.")
    
    def _check_voxel(self, voxel_index, masks):
        # Check if a single voxel projects to foreground pixels in all 4 cameras
        for cam_index in range(4):
            x, y = self.lookup_table[cam_index][voxel_index]
            mask = masks[cam_index]

            # Check bounds
            h, w = mask.shape
            if not (0 <= x < w and 0 <= y < h):
                return False # Out of bounds, treat as OFF
            
            # If the pixel is bakground the voxel is OFF
            if mask[y, x] == 0:
                return False
        
        # If it survived all 4 camera checks, voxel is ON
        return True
    
    def process_frame(self, current_masks):
        # Takes 4 binary masks for current frame and returns the 3D coordinates of ON voxels
        # usng XOR optimization
        if self.previous_masks is None:
            # First frame, check all voxels
            for voxel_index in range(self.num_voxels):
                self.voxel_states[voxel_index] = self._check_voxel(voxel_index, current_masks)
        else:
            # Frame n > 1: Use XOR optimization
            voxels_to_check = set()

            for cam_index in range(4):
                # Binary XOR to find changed pixels
                diff = cv.bitwise_xor(current_masks[cam_index], self.previous_masks[cam_index])

                # Get the coordinates of changed pixels
                ys, xs = np.where(diff > 0)

                # Use reverse lookup to find voxels for re-evaluation
                for y, x in zip(ys, xs):
                    affected_voxels = self.reverse_lookup[cam_index].get((x, y), [])
                    for v_index in affected_voxels:
                        voxels_to_check.add(v_index)
                
            # Re-evaluate only the affected voxels
            for voxel_index in voxels_to_check:
                self.voxel_states[voxel_index] = self._check_voxel(voxel_index, current_masks)
                

        # Update previous masks for the next frame
        self.previous_masks = [m.copy() for m in current_masks]

        # Return the 3D coordinates of ON voxels
        activate_voxel_indices = np.where(self.voxel_states)
        activate_positions = self.voxel_coords[activate_voxel_indices].tolist()

        # Generate solid white color
        colors = [[1.0, 1.0, 1.0] for _ in activate_positions]

        return activate_positions, colors