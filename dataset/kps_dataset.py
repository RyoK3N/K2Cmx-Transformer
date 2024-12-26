import numpy as np
import ast
from .skeleton import Skeleton

class MocapDataset:
    def __init__(self, df_X, df_Y, skeleton):
        """
        Initialize the MocapDataset class.

        Args:
            df_X (pd.DataFrame): DataFrame containing 2D keypoints.
            df_Y (pd.DataFrame): DataFrame containing the camera matrix.
            skeleton (Skeleton): An instance of the Skeleton class.
        """
        self._skeleton = skeleton
        self.joint_names = list(df_X.columns)
        self._data = df_X
        self._camera_matrix = df_Y

    def parse_keypoints(self, row):
        """
        Parse keypoints for a single row of data.

        Args:
            row (pd.Series): A row from the DataFrame containing keypoints.

        Returns:
            dict: Parsed keypoints with joint names as keys and (x, y) tuples as values.
        """
        parsed_keypoints = {}
        for part, coord in row.items():
            try:
                if isinstance(coord, str):
                    coord = ast.literal_eval(coord)
                if isinstance(coord, (list, tuple)) and len(coord) == 2:
                    x, y = map(float, coord)
                    parsed_keypoints[part] = (x, y)
                else:
                    parsed_keypoints[part] = (np.nan, np.nan)  # Use NaN for invalid keypoints
            except Exception:
                parsed_keypoints[part] = (np.nan, np.nan)
        return parsed_keypoints

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self._data)

    def __getitem__(self, idx):
        """
        Retrieve the 2D keypoints and camera matrix for a given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (keypoints, camera_matrix)
        """
        keypoints_row = self._data.iloc[idx]
        keypoints = self.parse_keypoints(keypoints_row)
        camera_matrix = self._camera_matrix.iloc[idx].values.astype(np.float32)
        keypoints_flattened = np.array(
            [coord for joint in self.joint_names for coord in keypoints[joint]], dtype=np.float32
        ).flatten()
        return keypoints_flattened, camera_matrix

    def skeleton(self):
        """
        Retrieve the skeleton associated with the dataset.

        Returns:
            Skeleton: The Skeleton instance.
        """
        return self._skeleton

    def cameras(self):
        """
        Retrieve the camera matrix DataFrame.

        Returns:
            pd.DataFrame: The camera matrix.
        """
        return self._camera_matrix