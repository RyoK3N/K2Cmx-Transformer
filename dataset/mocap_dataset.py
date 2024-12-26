import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pymongo import MongoClient
from dotenv import load_dotenv


class MocapDataset(Dataset):
    def __init__(self, uri , db_name, collection_name, skeleton):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name
        self._skeleton = skeleton

        self.client = None
        self.collection = None

        self.joint_names = []
        self.num_joints = 0
        self._ids = []
        self.total = 0

        self._initialize_ids_and_metadata()

    def _initialize_ids_and_metadata(self):
        load_dotenv()
        self._connect()
        sample_doc = self.collection.find_one()
        if not sample_doc:
            raise ValueError("The collection is empty.")
        if 'kps_2d' not in sample_doc:
            raise ValueError("Documents must contain 'kps_2d' field.")

        all_joints = list(sample_doc['kps_2d'].keys())
        self.joint_names = [joint for joint in all_joints if joint.strip().lower() not in ['date', 'body']]
        self.num_joints = len(self.joint_names)

        self._ids = list(self.collection.find({}, {'_id': 1}))
        self._ids = [doc['_id'] for doc in self._ids]
        self.total = len(self._ids)

    def _connect(self):
        if self.client is None:
            self.client = MongoClient(self.uri, 27017)
            db = self.client[self.db_name]
            self.collection = db[self.collection_name]

    def parse_keypoints(self, keypoints_dict):
        keypoints_flat = []
        for joint in self.joint_names:
            coords = keypoints_dict.get(joint, [0.0, 0.0])
            if len(coords) < 2:
                coords = [0.0, 0.0]
            keypoints_flat.extend(coords[:2])
        return np.array(keypoints_flat, dtype=np.float32)

    def parse_camera_matrix(self, camera_matrix_list):
        if len(camera_matrix_list) != 16:
            raise ValueError("Camera matrix must have 16 elements.")
        return np.array(camera_matrix_list, dtype=np.float32)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total:
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {self.total}.")

        self._connect()  
        _id = self._ids[idx]
        document = self.collection.find_one({'_id': _id})

        if not document:
            raise ValueError(f"No document found with _id: {_id}")

        keypoints_dict = document.get('kps_2d', {})
        keypoints_flattened = self.parse_keypoints(keypoints_dict)

        camera_matrix_list = document.get('camera_matrix', [])
        camera_matrix = self.parse_camera_matrix(camera_matrix_list)

        expected_length = self.num_joints * 2
        actual_length = keypoints_flattened.shape[0]
        if actual_length != expected_length:
            missing_joints = [joint for joint in self.joint_names if joint not in keypoints_dict]
            if missing_joints:
                print(f"Warning: Missing joint coordinates for: {missing_joints}")
            raise ValueError(f"Sample index {idx} has {actual_length} keypoints, expected {expected_length}.")

        labels = self.joint_names
        label_idx = list(range(len(self.joint_names)))

        return keypoints_flattened, camera_matrix, labels, label_idx

    def close_connection(self):
        if self.client:
            self.client.close()
            self.client = None