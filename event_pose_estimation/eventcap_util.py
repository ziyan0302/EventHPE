import joblib
import pdb
import pickle
from scipy.spatial import KDTree
import torch
import cv2
import numpy as np
from event_pose_estimation.geometry import projection_torch
import os

def findClosestPointTorch(source, target):
    """
    Find the closest points from 'source' to 'target' in PyTorch.

    Args:
    - source (torch.Tensor): Tensor of shape (N, 2), representing source points (e.g., vertices).
    - target (torch.Tensor): Tensor of shape (M, 2), representing target points (e.g., features).

    Returns:
    - distances (torch.Tensor): Tensor of shape (M,) containing the distances to the closest source points.
    - closest_source_indices (torch.Tensor): Tensor of shape (M,) containing the indices of the closest source points.
    """
    # Step 1: Compute pairwise distances between source and target
    # source is of shape (N, 2), target is of shape (M, 2)
    
    # Expand dimensions to compute pairwise differences
    # source.unsqueeze(1) makes it (N, 1, 2), target.unsqueeze(0) makes it (1, M, 2)
    # Broadcasting will automatically compute differences for all point pairs
    distances = torch.norm(source.unsqueeze(1) - target.unsqueeze(0), dim=2)  # Resulting shape (N, M)

    # Step 2: Find the minimum distance and corresponding source index for each target point
    min_distances, closest_source_indices = torch.min(distances, dim=0)  # Output shape (M,)

    return min_distances, closest_source_indices

def findCloestPoint(source, target):
    # Step 1: Build the KDTree with vertices
    tree = KDTree(source)
    # Step 2: Query the tree for the closest vertex for each feature
    # This returns the distances and the indices of the closest vertices
    distances, closest_source_indices = tree.query(target)
    return distances, closest_source_indices

def find_closest_events(boundary_pixels, event_u, event_t, t_f, t_0, t_N, lambda_val):
    
    # Define the time window [t_f - 10000, t_f + 10000]
    time_range = 10000
    start_time = max(t_f - time_range, event_t[0])
    end_time = min(t_f + time_range, event_t[-1])
    # Find the indices of the events that fall within the time range
    valid_indices = np.where((event_t >= start_time) & (event_t <= end_time))[0]

    # Use these indices to filter the corresponding events and timestamps
    filtered_events_u = event_u[valid_indices]
    filtered_event_ts = event_t[valid_indices]
    # duplicate
    # Expand the dimensions of boundary_pixels and events_xy for broadcasting
    boundary_pixels_expanded = boundary_pixels[:, np.newaxis, :]  # Shape (m, 1, 2)
    events_xy_expanded = filtered_events_u[np.newaxis, :, :]  # Shape (1, n, 2)
    events_xy_expanded[0,0]
    boundary_pixels_expanded[0,0]

    # Now subtract the two arrays (broadcasted subtraction)
    spatial_distances = np.sum((boundary_pixels_expanded - events_xy_expanded)**2, axis=-1)  # Shape: (n_boundary_pixels, n_events)
    
    # Compute temporal distances: normalized temporal differences
    temporal_distances = (t_f - filtered_event_ts) / (t_N - t_0)  # Shape: (1, n_events)
    
    # Compute total distance D(s_b, e) = λ * (temporal_dist)^2 + spatial_dist
    total_distances = lambda_val * (temporal_distances**2) + spatial_distances  # Shape: (n_boundary_pixels, n_events)
    
    # Find the event index that minimizes the distance for each boundary pixel
    closest_event_indices = np.argmin(total_distances, axis=1)
    
    # Get the closest events by indexing into event arrays
    closest_events_u = event_u[closest_event_indices]
    closest_events_t = event_t[closest_event_indices]
    return closest_events_u, closest_events_t

def findBoundaryPixels(vertPixels, H=256):
    boundaryPixels = []
    vertPixels = vertPixels.astype(np.int8)
    for iImg in range(vertPixels.shape[0]):
        mask = np.zeros((H, H), dtype=np.uint8)
        mask[vertPixels[iImg,:,1], vertPixels[iImg,:,0]] = 255
        # Dilate the mask to create circles
        kernel7x7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask7x7 = cv2.dilate(mask, kernel7x7)
        kernel5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask5x5 = cv2.dilate(mask, kernel5x5)
        image7x7 = np.full((H, H), 0, dtype=np.uint8)  # Grey background
        image5x5 = np.full((H, H), 0, dtype=np.uint8)  # Grey background
        # Apply the mask to the image
        # image7x7[mask7x7 == 255] += np.array([0, 127, 127]).astype(np.uint8)  # White dots
        # image5x5[mask5x5 == 255] += np.array([127, 127, 127]).astype(np.uint8)  # White dots
        image7x7[mask7x7 == 255] += np.array([255]).astype(np.uint8)
        image5x5[mask5x5 == 255] += np.array([255]).astype(np.uint8)
        edge = image7x7 - image5x5
        non_zero_locations = np.nonzero(edge != 0)
        non_zeros = np.array(list(zip(non_zero_locations[0], non_zero_locations[1])))
        boundaryPixels.append(non_zeros)
    return boundaryPixels

class ICPPoseEstimation:
    def __init__(self, projected_mesh, events):
        """
        Args:
            projected_mesh (ndarray): 3D points of the projected mesh (Nx3).
            events (ndarray): 2D array of event points (Mx2) in the image plane.
        """
        self.projected_mesh = projected_mesh
        self.events = events
        self.kdtree = KDTree(events)  # Build a KD-tree for fast nearest neighbor search.

    def icp_iteration(self, pose):
        """
        Perform one ICP iteration.
        
        Args:
            pose (ndarray): Initial pose matrix (4x4).

        Returns:
            pose (ndarray): Refined pose matrix (4x4).
        """
        # For each boundary pixel of the projected mesh, find the closest event.
        closest_events = self.find_closest_events(self.projected_mesh)

        # Define the cost function for non-linear least squares optimization.
        def cost_function(pose_params):
            refined_pose = self.params_to_pose(pose_params)
            refined_mesh = self.apply_pose(self.projected_mesh, refined_pose)
            return (refined_mesh - closest_events).ravel()

        # Solve the non-linear least squares optimization.
        initial_params = self.pose_to_params(pose)
        result = least_squares(cost_function, initial_params)

        # Update the pose with the optimized parameters.
        refined_pose = self.params_to_pose(result.x)
        return refined_pose

    def find_closest_events(self, transformed_mesh):
        """
        Find the closest events for each boundary pixel of the projected mesh.

        Args:
            transformed_mesh (ndarray): Transformed 3D points (Nx3).

        Returns:
            closest_events (ndarray): Closest events (Nx2).
        """
        _, indices = self.kdtree.query(transformed_mesh)  # Query KD-tree for closest event.
        return self.events[indices]

    def pose_to_params(self, pose):
        """
        Convert the pose matrix to a parameter vector (for optimization).

        Args:
            pose (ndarray): Pose matrix (4x4).

        Returns:
            params (ndarray): Parameter vector (6,) for optimization (e.g., rotation and translation).
        """
        # Extract translation and rotation components from the pose.
        translation = pose[:3, 3]
        rotation = pose[:3, :3]
        # Convert rotation matrix to a vector (for optimization purposes, e.g., Rodrigues' rotation formula).
        rotation_vector, _ = cv2.Rodrigues(rotation)
        return np.hstack((translation, rotation_vector.ravel()))

    def params_to_pose(self, params):
        """
        Convert the parameter vector back to a pose matrix.

        Args:
            params (ndarray): Parameter vector (6,).

        Returns:
            pose (ndarray): Pose matrix (4x4).
        """
        translation = params[:3]
        rotation_vector = params[3:]
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose = np.eye(4)
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = translation
        return pose



def joints2dOnImage(joints2d, lenOfSeq, args, action, saved_folder = "/home/ziyan/02_research/EventHPE/event_pose_estimation/tmp_folder"):
    iTest = 0
    for iTest in range(lenOfSeq):
        jointsForVisual = joints2d[iTest].detach().cpu().numpy() * 256
        jointsForVisual = jointsForVisual.astype(np.uint8)
        gray_image = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, iTest), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        imageCurr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        mask = np.zeros_like(imageCurr, dtype=np.uint8)
        mask[jointsForVisual[:,1], jointsForVisual[:,0]] = [255, 0, 255]
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        imageCurr = cv2.add(imageCurr, dilated_mask)

        # Specify the text (number) you want to write
        text = str(iTest)

        # Define font, scale, and color
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2  # Size of the text
        color = (255, 255, 255)  # White color (in BGR format)
        if iTest % 8 == 0:
            color = (0, 255, 255)
        thickness = 2  # Thickness of the text

        # Specify position (top-left corner of the text)
        position = (50, 200)  # (x, y)

        # Write the text on the image
        cv2.putText(imageCurr, text, position, font, font_scale, color, thickness)

        cv2.imwrite(os.path.join(str(saved_folder),f'eventsOnImg{iTest:04}.jpg'), imageCurr)
    

def joints2dandFeatOnImage(joints2d, feats, lenOfSeq, args, action, saved_folder = "/home/ziyan/02_research/EventHPE/event_pose_estimation/tmp_folder"):
    iTest = 0
    for iTest in range(lenOfSeq):
        jointsForVisual = joints2d[iTest].detach().cpu().numpy() * 256
        jointsForVisual = jointsForVisual.astype(np.uint8)
        gray_image = cv2.imread('%s/full_pic_256/%s/fullpic%04i.jpg' % (args.data_dir, action, iTest), cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        imageCurr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        mask = np.zeros_like(imageCurr, dtype=np.uint8)
        mask[jointsForVisual[:,1], jointsForVisual[:,0]] = [255, 0, 255]
        mask[feats[iTest][:,1].astype(np.int8), feats[iTest][:,0].astype(np.int8)] = [0, 0, 255]
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        imageCurr = cv2.add(imageCurr, dilated_mask)

        # Specify the text (number) you want to write
        text = str(iTest)

        # Define font, scale, and color
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2  # Size of the text
        color = (255, 255, 255)  # White color (in BGR format)
        if iTest % 8 == 0:
            color = (0, 255, 255)
        thickness = 2  # Thickness of the text

        # Specify position (top-left corner of the text)
        position = (50, 200)  # (x, y)

        # Write the text on the image
        cv2.putText(imageCurr, text, position, font, font_scale, color, thickness)

        cv2.imwrite(os.path.join(str(saved_folder),f'eventsOnImg{iTest:04}.jpg'), imageCurr)
