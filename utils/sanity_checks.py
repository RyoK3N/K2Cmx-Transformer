import numpy as np


def sanity_check(dataset, num_samples=50):
    num_samples = min(num_samples, len(dataset))
    for idx in range(num_samples):
        keypoints = dataset[idx][0]
        camera_matrix = dataset[idx][1]
        labels = dataset[idx][2]
        label_idx = dataset[idx][3]
        assert keypoints.shape[0] == dataset.num_joints * 2, f"Sample {idx} keypoints shape mismatch"
        assert camera_matrix.shape[0] == 16, f"Sample {idx} camera matrix shape mismatch"
    print(f"Sanity check passed for {num_samples} samples.")