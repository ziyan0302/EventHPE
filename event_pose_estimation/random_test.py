import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pdb
import numpy as np
import cv2
import time

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
    mask = torch.zeros((1, 1, H, H), dtype=torch.float32, device=device)
    mask[0, 0, vertPixels[:, 1], vertPixels[:, 0]] = 1.0

    # Dilate the mask with 5x5 and 3x3 kernels
    mask5x5 = F.conv2d(mask, kernel5x5, padding=2).clamp(0, 1)
    mask3x3 = F.conv2d(mask, kernel3x3, padding=1).clamp(0, 1)
    
    # Create images with "dilated" boundaries
    image5x5 = (mask5x5 * 255).squeeze().to(torch.uint8)
    image3x3 = (mask3x3 * 255).squeeze().to(torch.uint8)
    
    # Find boundary by subtracting the two images
    edge = image5x5 - image3x3
    non_zero_locations = torch.nonzero(edge != 0, as_tuple=False)
    x_coords, y_coords = vertices[:,0], vertices[:,1]
    
    # Create a mask for vertex indices
    mask_with_indices = -torch.ones((H, W), dtype=torch.int32, device=device)  # Initialize with -1
    valid_mask = (0 <= x_coords) & (x_coords < W) & (0 <= y_coords) & (y_coords < H)
    
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
    
    
    vertices = vertices[valid_edge_mask]

    if (0):
        npmask = np.zeros((256,256))
        edgetmp = edge.clone()
        edgetmp1 = edge.clone().detach().cpu().numpy()/2
        # tmp2 = vertices[valid_edge_mask].detach().cpu().numpy().astype(np.uint8)
        tmp2 = vertices[valid_edge_mask].detach().cpu().numpy().astype(np.uint8)
        edgetmp1[tmp2[:,1], tmp2[:,0]] = [255]
        cv2.imwrite("random_tmp.jpg", edgetmp1)


    return vertices

# Define parameters
H, W = 256, 256
num_epochs = 1000
learning_rate = 1


# Target silhouette: create a dummy target mask (for example, a diagonal line)
target_silhouette = torch.zeros((1, 1, H, W), dtype=torch.float32)
for i in range(H):
    target_silhouette[0, 0, i, i] = 1.0
target_silhouette = target_silhouette.to('cuda' if torch.cuda.is_available() else 'cpu')

# Move initial vertices to appropriate device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Initialize vertices with random positions
vertices = torch.tensor([[50.5, 50.5], [51.5, 50.5], [52.5, 50.5], [53.5, 50.5],
                         [50.5, 51.5], [51.5, 51.5], [52.5, 51.5], [53.5, 51.5],
                         [50.5, 52.5], [51.5, 52.5], [52.5, 52.5], [53.5, 52.5]],\
                         requires_grad=True,device=device)

# vertices = initial_vertices.to(device)

# Define optimizer
optimizer = torch.optim.SGD([vertices], lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Generate the silhouette from vertices
    generated_silhouette = vertices_to_silhouette(vertices, H=H, W=W, device=device)

    # set up target
    target_silhouette = generated_silhouette.clone().detach() + 10

    # Calculate loss (MSE loss between generated silhouette and target)
    loss = F.mse_loss(generated_silhouette, target_silhouette)

    # Backpropagate
    loss.backward()
    
    # Update vertices
    optimizer.step()

    vertices.grad
    npmask = np.zeros((256,256,3))
    vert_proj = vertices.clone().detach().cpu().numpy().astype(np.uint8)
    npmask[vert_proj[:,1], vert_proj[:,0]] = [255, 255, 255]
    target = target_silhouette.cpu().numpy().astype(np.uint8)
    npmask[target[:,1], target[:,0]] = [255, 255, 0]
    cv2.imwrite("random_tmp.jpg", npmask)
    pdb.set_trace()
    
    # Print loss and vertices occasionally
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        print(f'Updated vertices:\n {vertices}')

# Plot the final generated silhouette
generated_silhouette_np = generated_silhouette.detach().cpu().numpy().squeeze()
plt.imshow(generated_silhouette_np, cmap='gray')
plt.title('Final Generated Silhouette')
plt.show()
