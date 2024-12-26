import torch
import numpy as np

def custom_collate_fn(batch):
    """
    Custom collate function to batch keypoints and camera matrices.
    """
    keypoints = np.stack([item[0] for item in batch])  # Stack into a single numpy array
    camera_matrix = np.stack([item[1] for item in batch])
    return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(camera_matrix, dtype=torch.float32)