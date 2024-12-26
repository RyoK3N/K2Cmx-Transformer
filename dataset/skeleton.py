import numpy as np

class Skeleton:
    def __init__(self, connections, joints_left, joints_right, ordered_joint_names):
        self._connections = connections
        self._joint_names = ordered_joint_names  
        self._joint_indices = {joint: idx for idx, joint in enumerate(self._joint_names)}
        self._parents = self._compute_parents()
        self._joints_left = [self._joint_indices[joint] for joint in joints_left if joint in self._joint_indices]
        self._joints_right = [self._joint_indices[joint] for joint in joints_right if joint in self._joint_indices]
        self._compute_metadata()

    def _compute_parents(self):

        parents = [-1] * len(self._joint_names)
        for child, parent in self._connections:
            if child in self._joint_indices and parent in self._joint_indices:
                child_idx = self._joint_indices[child]
                parent_idx = self._joint_indices[parent]
                parents[child_idx] = parent_idx
        return np.array(parents)

    def _compute_metadata(self):

        self._has_children = np.zeros(len(self._parents), dtype=bool)
        self._children = [[] for _ in range(len(self._parents))]
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True
                self._children[parent].append(i)

    def get_connection_indices(self):

        connections_idx = []
        for child, parent in self._connections:
            if child in self._joint_indices and parent in self._joint_indices:
                child_idx = self._joint_indices[child]
                parent_idx = self._joint_indices[parent]
                connections_idx.append((child_idx, parent_idx))
        return connections_idx

    def num_joints(self):

        return len(self._parents)

    def parents(self):

        return self._parents

    def joints_left(self):

        return self._joints_left

    def joints_right(self):

        return self._joints_right

    def print_structure(self):
        print(f"Skeleton Structure:\nTotal joints: {self.num_joints()}")
        print("Connections:")
        for child, parent in self._connections:
            print(f"  {child} -> {parent}")
        print("Joint Indices:")
        for joint, idx in self._joint_indices.items():
            print(f"  {joint}: {idx}")

    def get_list_coords_from_graph(self, keypoints):
        keypoints_reshaped = keypoints.reshape(-1, 2)
        return [(np.float32(x), np.float32(y)) for x, y in keypoints_reshaped]

    def get_labels_from_graph(self):
        return self._joint_names

    def get_idx_from_graph(self):
        return list(range(len(self._joint_names)))