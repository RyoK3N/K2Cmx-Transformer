from __future__ import absolute_import
import torch
import numpy as np
import scipy.sparse as sp
from dataset.skeleton import Skeleton

def normalize(mx):
    """
    Row-normalize sparse matrix.

    Args:
        mx (sp.coo_matrix): Sparse matrix in COO format.

    Returns:
        sp.coo_matrix: Row-normalized sparse matrix.
    """
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv = np.power(rowsum, -1, where=rowsum != 0)  # Avoid division by zero
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a PyTorch sparse tensor.

    Args:
        sparse_mx (sp.coo_matrix): Sparse matrix in COO format.

    Returns:
        torch.sparse.FloatTensor: PyTorch sparse tensor representation.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adj_mx_from_edges(num_pts, edges, sparse=True):
    """
    Construct an adjacency matrix from a list of edges.

    Args:
        num_pts (int): Number of points/joints.
        edges (list or np.ndarray): Edges as (child, parent) pairs.
        sparse (bool): Whether to return a sparse torch tensor.

    Returns:
        torch.Tensor or torch.sparse.FloatTensor: The adjacency matrix.
    """
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # Symmetrize adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)

    # Normalize adjacency
    adj_mx = normalize(adj_mx)

    # Add self-connections
    adj_mx = adj_mx + sp.eye(adj_mx.shape[0])

    # Convert to PyTorch representation
    if sparse:
        return sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        return torch.tensor(adj_mx.todense(), dtype=torch.float32)

def adj_mx_from_skeleton(skeleton):
    """
    Build adjacency matrix from a skeleton structure.

    Args:
        skeleton (Skeleton): Instance of the Skeleton class, providing parents() and num_joints().

    Returns:
        torch.Tensor: Adjacency matrix as a dense torch tensor.
    """
    num_joints = skeleton.num_joints()
    edges = [(child, parent) for child, parent in enumerate(skeleton.parents()) if parent >= 0]
    return adj_mx_from_edges(num_joints, edges, sparse=False)

def adj_mx_from_skeleton_2dhp(num_joints, parents):
    """
    Create adjacency matrix given num_joints and a parents array for 2DHP or similar datasets.

    Args:
        num_joints (int): Number of joints.
        parents (array-like): Parent indices for each joint.

    Returns:
        torch.Tensor: Adjacency matrix as a dense torch tensor.
    """
    edges = [(child, parent) for child, parent in enumerate(parents) if parent >= 0]
    return adj_mx_from_edges(num_joints, edges, sparse=False)

def adj_mx_from_skeleton_temporal(num_frames, parents):
    """
    Create adjacency matrix from temporal skeleton frames.

    Args:
        num_frames (int): Number of frames (treated as points).
        parents (array-like): Parent indices.

    Returns:
        torch.Tensor: Adjacency matrix as a dense torch tensor.
    """
    edges = [(child, parent) for child, parent in enumerate(parents) if parent >= 0]
    return adj_mx_from_edges(num_frames, edges, sparse=False)

def adj_mx_from_skeleton_temporal_extra(num_frames, parents, extra_parents):
    """
    Create adjacency matrix from temporal skeleton with additional edges (extra_parents).

    Args:
        num_frames (int): Number of frames (points).
        parents (array-like): Primary parent indices.
        extra_parents (array-like): Additional parent indices for extra edges.

    Returns:
        torch.Tensor: Adjacency matrix as a dense torch tensor.
    """
    edges = [(child, parent) for child, parent in enumerate(parents) if parent >= 0]
    extra_edges = [(child, parent) for child, parent in enumerate(extra_parents) if parent >= 0]
    all_edges = np.concatenate((edges, extra_edges), axis=0)
    return adj_mx_from_edges(num_frames, all_edges, sparse=False)


if __name__ == "__main__":
    # Test - 001
    connections = [
        ('Head', 'Neck'), ('Neck', 'Chest'), ('Chest', 'Hips'),
        ('Neck', 'LeftShoulder'), ('LeftShoulder', 'LeftArm'),
        ('LeftArm', 'LeftForearm'), ('LeftForearm', 'LeftHand'),
        ('Chest', 'RightShoulder'), ('RightShoulder', 'RightArm'),
        ('RightArm', 'RightForearm'), ('RightForearm', 'RightHand'),
        ('Hips', 'LeftThigh'), ('LeftThigh', 'LeftLeg'),
        ('LeftLeg', 'LeftFoot'), ('Hips', 'RightThigh'),
        ('RightThigh', 'RightLeg'), ('RightLeg', 'RightFoot'),
        ('RightHand', 'RightFinger'), ('RightFinger', 'RightFingerEnd'),
        ('LeftHand', 'LeftFinger'), ('LeftFinger', 'LeftFingerEnd'),
        ('Head', 'HeadEnd'), ('RightFoot', 'RightHeel'),
        ('RightHeel', 'RightToe'), ('RightToe', 'RightToeEnd'),
        ('LeftFoot', 'LeftHeel'), ('LeftHeel', 'LeftToe'),
        ('LeftToe', 'LeftToeEnd'),
        ('SpineLow', 'Hips'),
        ('SpineMid', 'SpineLow'),
        ('Chest', 'SpineMid')
    ]
    joints_left = [
            'LeftShoulder', 'LeftArm', 'LeftForearm', 'LeftHand', 'LeftFinger', 'LeftFingerEnd',
            'LeftThigh', 'LeftLeg', 'LeftFoot', 'LeftHeel', 'LeftToe', 'LeftToeEnd'
        ]
    joints_right = [
            'RightShoulder', 'RightArm', 'RightForearm', 'RightHand', 'RightFinger', 'RightFingerEnd',
            'RightThigh', 'RightLeg', 'RightFoot', 'RightHeel', 'RightToe', 'RightToeEnd'
        ]
    skeleton = Skeleton(connections, joints_left, joints_right)
    adj = adj_mx_from_skeleton(skeleton)
    print("Adjacency matrix:\n", adj)