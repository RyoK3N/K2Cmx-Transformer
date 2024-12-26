import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from dotenv import load_dotenv
import argparse

# Local imports
from model.model import KTPFormer
from dataset.mocap_dataset import MocapDataset
from dataset.skeleton import Skeleton
from utils.graph_utils import adj_mx_from_skeleton
from model.loss import weighted_frobenius_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Testing script for KTPFormer')
    
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for testing')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for testing')
    parser.add_argument('--model_path', type=str, 
                        default='ktpformer_best_model.pth',
                        help='Path to the trained model')
    parser.add_argument('--data_fraction', type=float, default=0.001,
                        help='Fraction of data to use for testing')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    
    return parser.parse_args()

def test(args):
    # Load environment variables
    load_dotenv()
    uri = os.getenv('URI')
    if not uri:
        raise EnvironmentError("Please set the 'URI' environment variable in your .env file.")

    # Define skeleton structure
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
        ('SpineLow', 'Hips'), ('SpineMid', 'SpineLow'), ('Chest', 'SpineMid')
    ]

    joints_left = [
        'LeftShoulder', 'LeftArm', 'LeftForearm', 'LeftHand', 'LeftFinger', 'LeftFingerEnd',
        'LeftThigh', 'LeftLeg', 'LeftFoot', 'LeftHeel', 'LeftToe', 'LeftToeEnd'
    ]

    joints_right = [
        'RightShoulder', 'RightArm', 'RightForearm', 'RightHand', 'RightFinger', 'RightFingerEnd',
        'RightThigh', 'RightLeg', 'RightFoot', 'RightHeel', 'RightToe', 'RightToeEnd'
    ]

    # Initialize dataset
    dataset = MocapDataset(uri=uri, db_name='ai', collection_name='cameraPoses', skeleton=None)
    
    # Setup skeleton
    skeleton = Skeleton(
        connections=connections,
        joints_left=joints_left,
        joints_right=joints_right,
        ordered_joint_names=dataset.joint_names
    )
    dataset.skeleton = skeleton

    # Apply data fraction
    total_samples = len(dataset)
    samples_to_use = int(total_samples * args.data_fraction)
    dataset._ids = dataset._ids[:samples_to_use]
    dataset.total = samples_to_use

    print(f"Using {samples_to_use} samples out of {total_samples}")

    # Create data loader
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Initialize model
    adj_matrix = adj_mx_from_skeleton(skeleton)
    model = KTPFormer(
        input_dim=dataset.num_joints * 2,
        embed_dim=256,
        adj=adj_matrix,
        depth=2,
        disable_tpa=True
    ).to(args.device)

    # Load trained model
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # Testing loop
    print("Starting testing...")
    test_loss = 0.0
    with torch.no_grad():
        for keypoints, camera_matrix, _, _ in test_loader:
            keypoints = keypoints.to(args.device)
            camera_matrix = camera_matrix.to(args.device)
            
            if args.device == 'cuda':
                with autocast('cuda'):
                    outputs = model(keypoints)
                    loss = weighted_frobenius_loss(outputs, camera_matrix)
            else:
                outputs = model(keypoints)
                loss = weighted_frobenius_loss(outputs, camera_matrix)
                
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Results:")
    print(f"Average Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    args = parse_args()
    test(args)