import matplotlib.pyplot as plt



def visualize_predictions(outputs, targets, num_samples=2):
    num_samples = min(num_samples, outputs.shape[0])
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.plot(outputs[i], label='Prediction', marker='o', linestyle='--')
        plt.plot(targets[i], label='Target', marker='x', linestyle=':')
        plt.title(f"Sample {i + 1}")
        plt.legend()
        plt.grid()
    plt.tight_layout()
    plt.show()

def visualize_keypoint_skeleton(keypoints, skeleton):


    reshaped_keypoints = keypoints.reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(8, 8))

    for child_idx, parent_idx in skeleton.get_connection_indices():
        if parent_idx != -1:
            x_vals = [reshaped_keypoints[child_idx, 0], reshaped_keypoints[parent_idx, 0]]
            y_vals = [reshaped_keypoints[child_idx, 1], reshaped_keypoints[parent_idx, 1]]
            print(x_vals)
            ax.plot(x_vals, y_vals, c='green', linewidth=2)
            # Debug: Print each connection
            print(f"Line: {skeleton._joint_names[child_idx]} -> {skeleton._joint_names[parent_idx]} | "
                  f"Points: {x_vals}, {y_vals}")

    ax.scatter(reshaped_keypoints[:, 0], reshaped_keypoints[:, 1], c='blue', marker='x', label='Joints')
    for idx, (x, y) in enumerate(reshaped_keypoints):
        ax.text(x, y, skeleton._joint_names[idx], size=8, color='k')

    ax.set_title("2D Skeleton Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()  
    ax.axis('equal')
    plt.tight_layout()
    plt.show()