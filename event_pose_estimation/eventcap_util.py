import joblib
import pdb
import pickle
from scipy.spatial import KDTree
import torch
import cv2
import numpy as np
from event_pose_estimation.geometry import projection_torch, rot6d_to_rotmat, delta_rotmat_to_rotmat
import os
from event_pose_estimation.loss_funcs import compute_mpjpe, compute_pa_mpjpe, compute_pa_mpjpe_eventcap
from event_pose_estimation.SMPL import SMPL, batch_rodrigues
import torch.nn.functional as F



# Function to create the silhouette from vertices
def vertices_to_silhouette(vertices, H=256, W=256, device='cpu'):
    """
    Create a silhouette image from vertices while retaining gradients.
    Args:
    - vertices (torch.Tensor): Tensor of shape (N, 2) representing x, y coordinates.
    - H (int): Height of the image.
    - W (int): Width of the image.
    - device (str): Device for computation, 'cpu' or 'cuda'.
    Returns:
    - silhouette (torch.Tensor): Tensor of shape (1, 1, H, W) representing the silhouette.
    """
    # vertices[:,0], vertices[:,1] = vertices[:,0].clamp(0,W), vertices[:,0].clamp(0,H)
    vertPixels = vertices.to(torch.int64)  # Convert to int64 for indexing compatibility in PyTorch
    
    # Define structuring elements (kernels) for dilation
    kernel5x5 = torch.tensor([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0]
    ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 5, 5)

    kernel3x3 = torch.tensor([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 3, 3)
    # Initialize a mask of zeros
    mask = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)
    mask[0, 0, vertPixels[:, 1], vertPixels[:, 0]] = 1.0

    # Dilate the mask with 5x5 and 3x3 kernels
    mask5x5 = F.conv2d(mask, kernel5x5, padding=2).clamp(0, 1)
    mask3x3 = F.conv2d(mask, kernel3x3, padding=1).clamp(0, 1)
    
    # Create images with "dilated" boundaries
    image5x5 = (mask5x5 * 255).squeeze().to(torch.uint8)
    image3x3 = (mask3x3 * 255).squeeze().to(torch.uint8)
    
    # Find boundary by subtracting the two images
    edge = image5x5 - image3x3
    # non_zero_locations = torch.nonzero(edge != 0, as_tuple=False)
    x_coords, y_coords = vertices[:,0], vertices[:,1]
    
    # Create a mask for vertex indices
    # mask_with_indices = -torch.ones((H, W), dtype=torch.int32, device=device)  # Initialize with -1
    # valid_mask = (0 <= x_coords) & (x_coords < W) & (0 <= y_coords) & (y_coords < H)
    
    # collect vertices if one of its neighbors is a effective boundary edge
    
    ## right neighbor
    right_neighbor = vertices.clone()
    right_neighbor[:,0] = torch.clamp(x_coords + 2, 0, W - 1)
    right_edge_mask = edge[right_neighbor[:,1].to(torch.long), right_neighbor[:,0].to(torch.long)] != 0
    
    ## left neighbor
    left_neighbor = vertices.clone()
    left_neighbor[:,0] = torch.clamp(x_coords - 2, 0, W - 1)
    left_edge_mask = edge[left_neighbor[:,1].to(torch.long), left_neighbor[:,0].to(torch.long)] != 0

    ## above neighbor
    above_neighbor = vertices.clone()
    above_neighbor[:,1] = torch.clamp(y_coords - 2, 0, H - 1)
    above_edge_mask = edge[above_neighbor[:,1].to(torch.long), above_neighbor[:,0].to(torch.long)] != 0

    ## below neighbor
    below_neighbor = vertices.clone()
    below_neighbor[:,1] = torch.clamp(y_coords + 2, 0, H - 1)
    below_edge_mask = edge[below_neighbor[:,1].to(torch.long), below_neighbor[:,0].to(torch.long)] != 0
    
    valid_edge_mask = right_edge_mask | left_edge_mask | above_edge_mask | below_edge_mask    
    # if torch.any(valid_edge_mask) == False:
    #     pdb.set_trace()
    #     valid_edge_mask.shape
    #     torch.count_nonzero(valid_edge_mask[:100])
    #     torch.count_nonzero((right_edge_mask | left_edge_mask| above_edge_mask | below_edge_mask) == True)
    #     valid_edge_mask[3]
    #     return vertices

    if (0):
        vertices[valid_edge_mask].shape
        tmpV = vertices[valid_edge_mask].clone().detach().cpu().numpy().astype(np.uint8)
        tmpV = vertices.clone().detach().cpu().numpy().astype(np.uint8)
        npmask = np.zeros((256,256))
        edgetmp = edge.clone()
        edgetmp1 = edge.clone().detach().cpu().numpy()/2
        # tmp2 = vertices[valid_edge_mask].detach().cpu().numpy().astype(np.uint8)
        tmp2 = vertices[valid_edge_mask].detach().cpu().numpy().astype(np.uint8)
        edgetmp1[tmpV[:,1], tmpV[:,0]] = [255]
        cv2.imwrite("tmp.jpg", edgetmp1)


    return vertices[valid_edge_mask]


# Function to create the silhouette from vertices
def vertices_to_silhouette_Extreme(vertices, H=256, W=256, device='cpu'):
    """
    Create a silhouette image from vertices while retaining gradients.
    Args:
    - vertices (torch.Tensor): Tensor of shape (N, 2) representing x, y coordinates.
    - H (int): Height of the image.
    - W (int): Width of the image.
    - device (str): Device for computation, 'cpu' or 'cuda'.
    Returns:
    - silhouette (torch.Tensor): Tensor of shape (1, 1, H, W) representing the silhouette.
    """
    vertPixels = vertices.to(torch.int64)  # Convert to int64 for indexing compatibility in PyTorch
    
    # Define structuring elements (kernels) for dilation
    kernel7x7 = torch.ones((7,7), dtype=torch.float32, device=device)
    kernel7x7[0,0], kernel7x7[0,-1], kernel7x7[-1,0], kernel7x7[-1,-1] = 0,0,0,0 
    kernel7x7 = kernel7x7.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 7, 7)

    kernel5x5 = torch.ones((5,5), dtype=torch.float32, device=device)
    kernel5x5[0,0], kernel5x5[0,-1], kernel5x5[-1,0], kernel5x5[-1,-1] = 0,0,0,0 
    kernel5x5 = kernel5x5.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 5, 5)

    # Initialize a mask of zeros
    mask = torch.zeros((1, 1, H, W), dtype=torch.float32, device=device)
    mask[0, 0, vertPixels[:, 1], vertPixels[:, 0]] = 1.0

    # Dilate the mask with 5x5 and 3x3 kernels
    mask7x7 = F.conv2d(mask, kernel7x7, padding=3).clamp(0, 1)
    mask5x5 = F.conv2d(mask, kernel5x5, padding=2).clamp(0, 1)
    
    # Create images with "dilated" boundaries
    image7x7 = (mask7x7 * 255).squeeze().to(torch.uint8)
    image5x5 = (mask5x5 * 255).squeeze().to(torch.uint8)
    
    # Find boundary by subtracting the two images
    edge = image7x7 - image5x5
    # non_zero_locations = torch.nonzero(edge != 0, as_tuple=False)
    x_coords, y_coords = vertices[:,0], vertices[:,1]
    
    # Create a mask for vertex indices
    # mask_with_indices = -torch.ones((H, W), dtype=torch.int32, device=device)  # Initialize with -1
    # valid_mask = (0 <= x_coords) & (x_coords < W) & (0 <= y_coords) & (y_coords < H)
    
    # collect vertices if one of its neighbors is a effective boundary edge
    
    ## right neighbor
    right_neighbor = vertices.clone()
    right_neighbor[:,0] = torch.clamp(x_coords + 3, 0, W - 1)
    right_edge_mask = edge[right_neighbor[:,1].to(torch.long), right_neighbor[:,0].to(torch.long)] != 0

    ## left neighbor
    left_neighbor = vertices.clone()
    left_neighbor[:,0] = torch.clamp(x_coords - 3, 0, W - 1)
    left_edge_mask = edge[left_neighbor[:,1].to(torch.long), left_neighbor[:,0].to(torch.long)] != 0

    ## above neighbor
    above_neighbor = vertices.clone()
    above_neighbor[:,1] = torch.clamp(y_coords - 3, 0, H - 1)
    above_edge_mask = edge[above_neighbor[:,1].to(torch.long), above_neighbor[:,0].to(torch.long)] != 0

    ## below neighbor
    below_neighbor = vertices.clone()
    below_neighbor[:,1] = torch.clamp(y_coords + 3, 0, H - 1)
    below_edge_mask = edge[below_neighbor[:,1].to(torch.long), below_neighbor[:,0].to(torch.long)] != 0
    
    valid_edge_mask = right_edge_mask | left_edge_mask | above_edge_mask | below_edge_mask    
    if not torch.any(valid_edge_mask):
        return  vertices
    vertices = vertices[valid_edge_mask]

    if (0):
        torch.all(valid_edge_mask)
        tmpV = vertices[valid_edge_mask].clone().detach().cpu().numpy().astype(np.uint16)
        tmpV = vertices.clone().detach().cpu().numpy().astype(np.uint16)
        npmask = np.zeros((H,W))
        edgetmp = edge.clone()
        edgetmp1 = edge.clone().detach().cpu().numpy()/2
        rightTmp = right_neighbor.clone().detach().cpu().numpy().astype(np.uint16)
        # tmp2 = vertices[valid_edge_mask].detach().cpu().numpy().astype(np.uint8)
        tmp2 = vertices[valid_edge_mask].detach().cpu().numpy().astype(np.uint8)
        edgetmp1[tmpV[:,1], tmpV[:,0]] = [255]
        edgetmp1[rightTmp[:,1], rightTmp[:,0]] = [255]
        
        npmask[tmpV[:,1], tmpV[:,0]] = [255]
        cv2.imwrite("tmp.jpg", edgetmp1)


    return vertices



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
    # boundary_pixels = boundary_pixels[:,[1,0]]
    # Expand the dimensions of boundary_pixels and events_xy for broadcasting
    boundary_pixels_expanded = boundary_pixels[:, np.newaxis, :].astype(np.int64)  # Shape (m, 1, 2)
    events_xy_expanded = filtered_events_u[np.newaxis, :, :].astype(np.int64)  # Shape (1, n, 2)

    # Now subtract the two arrays (broadcasted subtraction)
    spatial_distances = np.sum((boundary_pixels_expanded - events_xy_expanded)**2, axis=-1)  # Shape: (n_boundary_pixels, n_events)
    
    # Compute temporal distances: normalized temporal differences
    temporal_distances = (t_f - filtered_event_ts) / (t_N - t_0)  # Shape: (1, n_events)
    
    # Compute total distance D(s_b, e) = λ * (temporal_dist)^2 + spatial_dist
    # total_distances = lambda_val * (temporal_distances**2) + spatial_distances  # Shape: (n_boundary_pixels, n_events)
    total_distances = spatial_distances  # Shape: (n_boundary_pixels, n_events)
    
    # Find the event index that minimizes the distance for each boundary pixel
    closest_event_indices = np.argmin(total_distances, axis=1)
    
    # Get the closest events by indexing into event arrays
    closest_events_u = filtered_events_u[closest_event_indices]
    closest_events_t = filtered_event_ts[closest_event_indices]

    return closest_events_u, closest_events_t

def find_closest_events_torch(boundary_pixels, event_u, event_t, t_f, t_0, t_N, lambda_val, H, W, device='cpu'):
    # Define the time window [t_f - 10000, t_f + 10000]
    time_range = 10000
    start_time = max(t_f - time_range, event_t[0].item())
    end_time = min(t_f + time_range, event_t[-1].item())

    # Find the indices of the events that fall within the time range
    valid_indices = (event_t >= start_time) & (event_t <= end_time)
    
    # Filter the events and timestamps based on valid_indices
    filtered_events_u = event_u[valid_indices].astype(np.int32)
    filtered_event_ts = event_t[valid_indices].astype(np.int32)

    if (0): # spend too much time
        # Expand the dimensions for broadcasting
        boundary_pixels_expanded = boundary_pixels[:, None, :].to(torch.int64).to(device)  # Shape (m, 1, 2)
        events_xy_expanded = torch.from_numpy(filtered_events_u)[None, :, :].to(torch.int64).to(device)      # Shape (1, n, 2)
        
        # Compute spatial distances (squared Euclidean distance)
        spatial_distances_1 = torch.sum((boundary_pixels_expanded - events_xy_expanded) ** 2, dim=-1)  # Shape: (m, n)

    tmp_events_xy = (torch.from_numpy(filtered_events_u) * torch.tensor([1/W, 1/H])).to(device)
    tmp_boundary_pixels = (boundary_pixels * torch.tensor([1/W, 1/H]).to(device))
    boundary_pixels_norm = (tmp_boundary_pixels ** 2).sum(dim=1, keepdim=True)  # Shape (m, 1)
    events_xy_norm = (tmp_events_xy ** 2).sum(dim=1, keepdim=True).T            # Shape (1, n)
    # (x-y)^2 = x^2 + y^2 -2xy
    # add a small regularization term
    # spatial_distances = boundary_pixels_norm + events_xy_norm - 2 * torch.mm(tmp_boundary_pixels.to(torch.float16), tmp_events_xy.to(torch.float16).T)
    spatial_distances = boundary_pixels_norm + events_xy_norm \
        - 2 * torch.mm(tmp_boundary_pixels, tmp_events_xy.T) + 0.0001
    
    # Compute temporal distances: normalized temporal differences
    temporal_distances = (t_f - filtered_event_ts) / (t_N - t_0 + 0.0001)  # Shape: (n,)
    
    # Reshape temporal_distances for broadcasting
    temporal_distances = torch.from_numpy(temporal_distances).unsqueeze(0).to(device)  # Shape: (1, n)
    
    # Compute total distance D(s_b, e) = λ * (temporal_dist)^2 + spatial_dist
    total_distances = lambda_val * (temporal_distances ** 2) + spatial_distances  # Shape: (m, n)
    # Find the event index that minimizes the distance for each boundary pixel
    closest_event_indices = torch.argmin(total_distances, dim=1).cpu()
    
    # Get the closest events by indexing into event arrays
    closest_events_u = filtered_events_u[closest_event_indices]
    closest_events_t = filtered_event_ts[closest_event_indices]
    if (0):
        tmpB = boundary_pixels.clone().detach().cpu().numpy().astype(np.uint16)
        tmpE = closest_events_u.astype(np.uint16)
        filterE = np.clip(filtered_events_u, 0 ,[W-1, H-1])
        npmask = np.zeros((H,W))
        npmask[tmpB[:,1], tmpB[:,0]] = [100]
        npmask[tmpE[:,1], tmpE[:,0]] = [255]
        npmask[event_u[:,1], event_u[:,0]] = [255]
        npmask[filterE[:,1], filterE[:,0]] = [255]
        
        cv2.imwrite("tmp.jpg", npmask)
    
    return torch.from_numpy(closest_events_u).to(torch.int64).to(device), torch.from_numpy(closest_events_t).to(torch.int64).to(device)

def find_closest_events_torch_v2(boundary_pixels, event_u, event_t, t_f, t_0, t_N, lambda_val, H, W, device='cpu'):
    # Define the time window [t_f - 10000, t_f + 10000]
    time_range = 10000
    start_time = max(t_f - time_range, event_t[0].item())
    end_time = min(t_f + time_range, event_t[-1].item())

    # Find the indices of the events that fall within the time range
    valid_indices = (event_t >= start_time) & (event_t <= end_time)
    
    # Filter the events and timestamps based on valid_indices
    filtered_events_u = event_u[valid_indices].astype(np.int32)
    filtered_event_ts = event_t[valid_indices].astype(np.int32)

    if (0): # spend too much time
        # Expand the dimensions for broadcasting
        boundary_pixels_expanded = boundary_pixels[:, None, :].to(torch.int64).to(device)  # Shape (m, 1, 2)
        events_xy_expanded = torch.from_numpy(filtered_events_u)[None, :, :].to(torch.int64).to(device)      # Shape (1, n, 2)
        
        # Compute spatial distances (squared Euclidean distance)
        spatial_distances_1 = torch.sum((boundary_pixels_expanded - events_xy_expanded) ** 2, dim=-1)  # Shape: (m, n)

    # Set the neighborhood range (8x8)
    neighborhood_size = 8
    half_neighborhood = neighborhood_size // 2
    
    
    tmp_events_xy = (torch.from_numpy(filtered_events_u) * torch.tensor([1/W, 1/H])).to(device)
    tmp_boundary_pixels = (boundary_pixels * torch.tensor([1/W, 1/H]).to(device))
    boundary_pixels_norm = (tmp_boundary_pixels ** 2).sum(dim=1, keepdim=True)  # Shape (m, 1)
    events_xy_norm = (tmp_events_xy ** 2).sum(dim=1, keepdim=True).T            # Shape (1, n)

    if torch.isnan(tmp_boundary_pixels).any() or torch.isinf(tmp_boundary_pixels).any():
        print("Invalid values found in tmp_boundary_pixels")
        pdb.set_trace()
    if torch.isnan(tmp_events_xy).any() or torch.isinf(tmp_events_xy).any():
        print("Invalid values found in tmp_events_xy")
        pdb.set_trace()

    # Calculate boundary pixels' neighborhood bounds
    x_min = (tmp_boundary_pixels[:, 0:1] - half_neighborhood / W).clamp(0,1)  # Shape: (m, 1)
    x_max = (tmp_boundary_pixels[:, 0:1] + half_neighborhood / W).clamp(0,1)  # Shape: (m, 1)
    y_min = (tmp_boundary_pixels[:, 1:2] - half_neighborhood / H).clamp(0,1)  # Shape: (m, 1)
    y_max = (tmp_boundary_pixels[:, 1:2] + half_neighborhood / H).clamp(0,1)  # Shape: (m, 1)

    # Generate masks for each event to check if within 8x8 neighborhood of each boundary pixel
    within_neighborhood_mask = (
        (tmp_events_xy[:, 0].unsqueeze(0) >= x_min) & (tmp_events_xy[:, 0].unsqueeze(0) <= x_max) &
        (tmp_events_xy[:, 1].unsqueeze(0) >= y_min) & (tmp_events_xy[:, 1].unsqueeze(0) <= y_max)
    )  # Shape: (m, n)

    # Calculate spatial distances using masking for neighborhood
    # (x-y)^2 = x^2 + y^2 -2xy
    # add a small regularization term
    
    spatial_distances = (
        boundary_pixels_norm + events_xy_norm -
        2 * torch.mm(tmp_boundary_pixels, tmp_events_xy.T) + 0.0001
    )  # Shape: (m, n)
    spatial_distances = spatial_distances.masked_fill(~within_neighborhood_mask, float('inf'))  # Exclude non-neighborhood events
    
    # Compute temporal distances: normalized temporal differences
    temporal_distances = (t_f - filtered_event_ts) / (t_N - t_0 + 0.0001)  # Shape: (n,)
    
    # Reshape temporal_distances for broadcasting
    temporal_distances = torch.from_numpy(temporal_distances).unsqueeze(0).to(device)  # Shape: (1, n)
    
    # Compute total distance D(s_b, e) = λ * (temporal_dist)^2 + spatial_dist
    total_distances = lambda_val * (temporal_distances ** 2) + spatial_distances  # Shape: (m, n)
    # Find the event index that minimizes the distance for each boundary pixel
    closest_event_indices = torch.argmin(total_distances, dim=1).cpu()
    
    # Get the closest events by indexing into event arrays
    closest_events_u = filtered_events_u[closest_event_indices]
    closest_events_t = filtered_event_ts[closest_event_indices]
    if (0):
        tmpB = boundary_pixels.clone().detach().cpu().numpy().astype(np.uint16)
        tmpE = closest_events_u.astype(np.uint16)
        filterE = np.clip(filtered_events_u, 0 ,[W-1, H-1])
        npmask = np.zeros((H,W))
        npmask[tmpB[:,1], tmpB[:,0]] = [100]
        npmask[tmpE[:,1], tmpE[:,0]] = [255]
        npmask[event_u[:,1], event_u[:,0]] = [255]
        npmask[filterE[:,1], filterE[:,0]] = [255]
        
        cv2.imwrite("tmp.jpg", npmask)
    
    return torch.from_numpy(closest_events_u).to(torch.int64).to(device), torch.from_numpy(closest_events_t).to(torch.int64).to(device)



def findBoundaryPixels(vertPixels, H=256):
    boundaryPixels = []
    vertPixels = vertPixels.astype(np.int8)
    for iImg in range(vertPixels.shape[0]):
        mask = np.zeros((H, H), dtype=np.uint8)
        mask[vertPixels[iImg,:,0], vertPixels[iImg,:,1]] = 255
        # Dilate the mask to create circles
        if (0): # more strict boundary check
            kernel7x7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask7x7 = cv2.dilate(mask, kernel7x7)
            kernel5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask5x5 = cv2.dilate(mask, kernel5x5)
            image7x7 = np.full((H, H), 0, dtype=np.uint8)  # Grey background
            image5x5 = np.full((H, H), 0, dtype=np.uint8)  # Grey background
            # Apply the mask to the image
            image7x7[mask7x7 == 255] += np.array([255]).astype(np.uint8)
            image5x5[mask5x5 == 255] += np.array([255]).astype(np.uint8)
            edge = image7x7 - image5x5
        if (1): # looser boundary check
            kernel5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask5x5 = cv2.dilate(mask, kernel5x5)
            kernel3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask3x3 = cv2.dilate(mask, kernel3x3)
            image5x5 = np.full((H, H), 0, dtype=np.uint8)  # Grey background
            image3x3 = np.full((H, H), 0, dtype=np.uint8)  # Grey background
            # Apply the mask to the image
            image5x5[mask5x5 == 255] += np.array([255]).astype(np.uint8)
            image3x3[mask3x3 == 255] += np.array([255]).astype(np.uint8)
            edge = image5x5 - image3x3

        non_zero_locations = np.nonzero(edge != 0)
        non_zeros = np.array(list(zip(non_zero_locations[0], non_zero_locations[1])))
        boundaryPixels.append(non_zeros)
    return boundaryPixels

def findBoundaryPixels_torch(vertPixels, H=256, device='cpu'):
    boundaryPixels = []
    vertPixels = vertPixels.to(torch.int64)  # Convert to int64 for indexing compatibility in PyTorch
    
    # Define structuring elements (kernels) for dilation
    kernel5x5 = torch.tensor([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0]
    ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 5, 5)

    kernel3x3 = torch.tensor([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 3, 3)
    
    for iImg in range(vertPixels.shape[0]):
        # Initialize a mask of zeros
        mask = torch.zeros((1, 1, H, H), dtype=torch.float32, device=device)
        mask[0, 0, vertPixels[iImg, :, 1], vertPixels[iImg, :, 0]] = 1.0

        # Dilate the mask with 5x5 and 3x3 kernels
        mask5x5 = F.conv2d(mask, kernel5x5, padding=2).clamp(0, 1)
        mask3x3 = F.conv2d(mask, kernel3x3, padding=1).clamp(0, 1)
        
        # Create images with "dilated" boundaries
        image5x5 = (mask5x5 * 255).squeeze().to(torch.uint8)
        image3x3 = (mask3x3 * 255).squeeze().to(torch.uint8)
        
        # Find boundary by subtracting the two images
        edge = image5x5 - image3x3
        non_zero_locations = torch.nonzero(edge != 0, as_tuple=False)
        if (0):
            pdb.set_trace()
            npmask = np.zeros((256,256))
            x = non_zero_locations[:,1].detach().cpu().numpy()
            y = non_zero_locations[:,0].detach().cpu().numpy()
            npmask[y, x] = [255]
            cv2.imwrite("tmp.jpg", npmask)
        boundaryPixels.append(non_zero_locations)  # Convert to numpy if needed
        
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


def draw_skeleton(input_image, joints, draw_edges=True, vis=None, radius=None):
    """
    joints is 3 x 19. but if not will transpose it.
    0: Right ankle
    1: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    5: Left ankle
    6: Right wrist
    7: Right elbow
    8: Right shoulder
    9: Left shoulder
    10: Left elbow
    11: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear
    """

    """
    joints is 3 x 19. but if not will transpose it.
    11: Right feet
    8: Right ankle
    5: Right knee
    2: Right hip
    3: Left hip
    4: Left knee
    7: Left ankle
    10: Left feet
    21: Right wrist
    23: Right hand
    7: Right elbow
    17: Right shoulder
    16: Left shoulder
    10: Left elbow
    20: Left wrist
    22: Left wrist
    12: Neck
    13: Head top
    14: nose
    15: left_eye
    16: right_eye
    17: left_ear
    18: right_ear

    """
    import numpy as np
    import cv2

    if radius is None:
        radius = max(4, (np.mean(input_image.shape[:2]) * 0.01).astype(int))

    colors = {
        'pink': [197, 27, 125],  # L lower leg
        'light_pink': [233, 163, 201],  # L upper leg
        'light_green': [161, 215, 106],  # L lower arm
        'green': [77, 146, 33],  # L upper arm
        'red': [215, 48, 39],  # head
        'light_red': [252, 146, 114],  # head
        'light_orange': [252, 141, 89],  # chest
        'purple': [118, 42, 131],  # R lower leg
        'light_purple': [175, 141, 195],  # R upper
        'light_blue': [145, 191, 219],  # R lower arm
        'blue': [69, 117, 180],  # R upper arm
        'gray': [130, 130, 130],  #
        'white': [255, 255, 255],  #        
    }

    if len(input_image.shape) == 3:
        image = input_image.copy()
    else:
        image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)

    input_is_float = False

    if np.issubdtype(image.dtype, np.float):
        input_is_float = True
        max_val = image.max()
        if max_val <= 2.:  # should be 1 but sometimes it's slightly above 1
            image = (image * 255).astype(np.uint8)
        else:
            image = (image).astype(np.uint8)

    if joints.shape[0] != 2:
        joints = joints.T
    joints = np.round(joints).astype(int)

    jcolors = [
        'light_pink', 'light_pink', 'light_pink', 'pink', 'pink', 'pink',
        'light_blue', 'light_blue', 'light_blue', 'blue', 'blue', 'blue',
        'purple', 'purple', 'red', 'green', 'green', 'white', 'white', 
        'light_red', 'light_green', 'light_red', 'light_green', 'white'

    ]

    if joints.shape[1] == 24:
        # parent indices -1 means no parents
        parents = np.array([
            -1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13,
       14, 16, 17, 18, 19, 20, 21
        ])
        # Left is light and right is dark
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            8: 'light_blue',
            9: 'blue',
            10: 'blue',
            11: 'blue',
            12: 'purple',
            13: 'light_green',
            14: 'light_green',
            15: 'purple',
            16: 'blue',
            17: 'blue',
            18: 'purple',
            19: 'light_green',
            20: 'light_green',
            21: 'purple',
            22: 'blue',
            23: 'blue',
        }
    elif joints.shape[1] == 19:
        parents = np.array([
            1,
            2,
            8,
            9,
            3,
            4,
            7,
            8,
            -1,
            -1,
            9,
            10,
            13,
            -1,
        ])
        ecolors = {
            0: 'light_pink',
            1: 'light_pink',
            2: 'light_pink',
            3: 'pink',
            4: 'pink',
            5: 'pink',
            6: 'light_blue',
            7: 'light_blue',
            10: 'light_blue',
            11: 'blue',
            12: 'purple'
        }
    else:
        print('Unknown skeleton!!')

    for child in range(len(parents)):
        point = joints[:, child]
        # If invisible skip
        if vis is not None and vis[child] == 0:
            continue
        if draw_edges:
            cv2.circle(image, (point[0], point[1]), radius, colors['white'], -1)
            cv2.circle(image, (point[0], point[1]), radius - 1,colors[jcolors[child]], -1)
        else:
            # cv2.circle(image, (point[0], point[1]), 5, colors['white'], 1)
            cv2.circle(image, (point[0], point[1]), radius - 1,
                       colors[jcolors[child]], 1)
            # cv2.circle(image, (point[0], point[1]), 5, colors['gray'], -1)
        pa_id = parents[child]
        if draw_edges and pa_id >= 0:
            if vis is not None and vis[pa_id] == 0:
                continue
            point_pa = joints[:, pa_id]
            cv2.circle(image, (point_pa[0], point_pa[1]), radius - 1,
                       colors[jcolors[pa_id]], -1)
            if child not in ecolors.keys():
                print('bad')
                pdb.set_trace()
            cv2.line(image, (point[0], point[1]), (point_pa[0], point_pa[1]),
                     colors[ecolors[child]], radius - 2)

    # Convert back in original dtype
    if input_is_float:
        if max_val <= 1.:
            image = image.astype(np.float32) / 255.
        else:
            image = image.astype(np.float32)

    return image

def draw_feature_dots(image, feat_array, vert_array, closest_vert_indices, colors):
    # Convert grayscale image to BGR format for color drawing
    # feat: simple color; vert: colorful
    if len(image.shape) == 2:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output_image = image.copy()

    # Draw dots for each feature in the data array
    iV = 0
    for row in feat_array:
        x = int(row[0])                # x coordinate
        y = int(row[1])                # y coordinate
        vert_idx = closest_vert_indices[iV]
        xV = int(vert_array[vert_idx,0])
        yV = int(vert_array[vert_idx,1])
        color = colors[iV]
        iV+=1
        # Draw a dot on the output image
        cv2.circle(output_image, (xV, yV), radius=2, color=color, thickness=-1)
        cv2.circle(output_image, (x, y), radius=1, color=[255,255,0], thickness=-1)
        # cv2.imwrite('tmp.jpg', output_image)
    return output_image


def draw_feature_dots_lines(image, feat_array, vert_array, closest_vert_indices, H, W):
    # Convert grayscale image to BGR format for color drawing
    # feat: simple color; vert: colorful
    if len(image.shape) == 3:
        output_image = image.copy()
    else:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw dots for each feature in the data array
    iV = 0
    for row in feat_array:
        x = int(row[0])                # x coordinate
        y = int(row[1])                # y coordinate
        vert_idx = closest_vert_indices[iV]
        xV = np.clip(vert_array[vert_idx,0], 0, W).astype(np.uint16)
        yV = np.clip(vert_array[vert_idx,1], 0, H).astype(np.uint16)
        iV+=1
        # Draw a dot on the output image
        cv2.line(output_image, (xV, yV), (x, y), [0,255,255], 1)
        cv2.circle(output_image, (xV, yV), radius=2, color=[0,255,255], thickness=-1)
        cv2.circle(output_image, (x, y), radius=1, color=[255,255,0], thickness=-1)
        # cv2.imwrite('tmp.jpg', output_image)
    return output_image

def findImgFeat(gray_img_sequence):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 500,
                        qualityLevel = 0.05,
                        minDistance = 2,
                        blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    old_gray = gray_img_sequence[0]
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


    # features on image using cv2 tracker
    featImgLocs = [p0.reshape(-1,2)]
    for iImg in range(1, gray_img_sequence.shape[0]):
        frame_gray = gray_img_sequence[iImg]
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
        featImgLocs.append(p0.reshape(-1,2))
    return featImgLocs


def evaluation(args, action, learnable_pose_and_shape, model, cam_intr, device):
    mpjpe_list = []
    pampjpe_list = []
    joints3DGTForEveryFrame = []
    joints3DPredForEveryFrame = []
    for iSplit in range(100):
        joints2dlist = []
        joints3dlist = []
        joints3dHMRlist = []
        tran_list = []
        tranHMR_list = []
        for iImg in range(8):
            beta, theta, tran, joints3dGT, joints2dGT = \
            joblib.load('%s/pose_events/%s/pose%04i.pkl' % (args.data_dir, action, iSplit*8 + iImg))
            joints2DHMR, joints3DHMR, _, transHMR, _ = \
                joblib.load('%s/hmr_results/%s/fullpic%04i_hmr.pkl' % (args.data_dir, action, iSplit*8 + iImg))

            joints2dlist.append(joints2dGT)
            joints3dlist.append(joints3dGT)
            tran_list.append(tran)
            tranHMR_list.append(transHMR)
            joints3dHMRlist.append(joints3DHMR)
        joints2dGTInSplit = torch.from_numpy(np.array(joints2dlist)).unsqueeze(0)
        joints3dGTInSplit = torch.from_numpy(np.array(joints3dlist)).unsqueeze(0).to(device)
        trans3dGTInSplit = torch.from_numpy(np.array(tran_list)).to(device)
        trans3dHMRInSplit = torch.from_numpy(np.array(tranHMR_list)).to(device)
        joints3dHMRInSplit = torch.from_numpy(np.array(joints3dHMRlist)).to(device)
        learnable_pose_and_shape[iSplit*8:(iSplit+1)*8]
        interpolated_rotmats = batch_rodrigues(learnable_pose_and_shape[iSplit*8:(iSplit+1)*8, 3:75].reshape(-1, 3)).view(-1, 24, 3, 3)  # [B, 24, 3, 3]
        interpolated_rotmats.shape
        verts, joints3d, _ = model(beta=learnable_pose_and_shape[iSplit*8:(iSplit+1)*8,75:85].view(-1, 10),
                            theta=None,
                            get_skin=True,
                            rotmats=interpolated_rotmats.view(-1, 24, 3, 3))
        verts = verts + trans3dHMRInSplit.repeat(1, verts.shape[1], 1)
        H, W = args.img_size, args.img_size
        joints3d_trans = joints3d + trans3dHMRInSplit.repeat(1, joints3d.shape[1], 1)
        joints2d = projection_torch(joints3d_trans, cam_intr, H, W)
        mpjpe = torch.mean(compute_mpjpe(joints3d_trans, joints3dGTInSplit),dim=2)  # [1, T, 24]
        pa_mpjpe = torch.mean(compute_pa_mpjpe_eventcap(joints3d_trans, joints3dGTInSplit), dim=1)  # [T, 24]
        mpjpe_list.append(mpjpe[0])
        pampjpe_list.append(pa_mpjpe)
        joints3DGTForEveryFrame.append(joints3dGTInSplit[0])
        joints3DPredForEveryFrame.append(joints3d)

    return torch.hstack(mpjpe_list), torch.hstack(pampjpe_list), \
        torch.stack(joints3DGTForEveryFrame), torch.stack(joints3DPredForEveryFrame)



def txt_to_np(file_path):
    data_list = []

    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into individual elements
            parts = line.strip().split()

            # Convert them to appropriate types
            feature_id = int(parts[0])
            timestamp = float(parts[1])
            x = float(parts[2])
            y = float(parts[3])

            # Store in dictionary with feature_id as key and the rest as values
            data_list.append([feature_id, timestamp, int(x), int(y)])
    data = np.array(data_list)

    return data