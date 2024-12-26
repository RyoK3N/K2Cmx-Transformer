import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from dotenv import load_dotenv
import argparse
from utils.sanity_checks import sanity_check
# Local imports
from model.model import KTPFormer
from dataset.mocap_dataset import MocapDataset
from dataset.skeleton import Skeleton
from utils.graph_utils import adj_mx_from_skeleton
from model.loss import weighted_frobenius_loss
from model.weights import initialize_weights
from utils.viz_kps import visualize_predictions, visualize_keypoint_skeleton
def parse_args():
    parser = argparse.ArgumentParser(description='Training script for KTPFormer')
    
    # Training hyperparameters
    parser.add_argument('--random_seed', type=int, default=100,
                        help='Random seed for reproducibility')
    parser.add_argument('--data_fraction', type=float, default=0.001,
                        help='Fraction of data to use for training')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    
    # Model saving and data splits
    parser.add_argument('--model_save_path', type=str, 
                        default='./weights/ktpformer_best_model.pth',
                        help='Path to save the model')
    parser.add_argument('--train_size', type=float, default=0.7,
                        help='Fraction of data to use for training')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Fraction of data to use for validation')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    
    # Training configuration
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Number of epochs to wait before early stopping')
    parser.add_argument('--visualize_every', type=int, default=1,
                        help='Visualize every N epochs')

    args = parser.parse_args()
    return args

def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train(args):
    # Set seeds for reproducibility
    set_seeds(args.random_seed)

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
    print(f"Number of joints: {dataset.num_joints}")
    print(f"Joint names: {dataset.joint_names}")

    sanity_check(dataset)
    
    split_generator = torch.Generator().manual_seed(args.random_seed)

    train_length = int(args.train_size * len(dataset))
    val_length = int(args.val_size * len(dataset))
    test_length = len(dataset) - train_length - val_length

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_length, val_length, test_length],
        generator=split_generator
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    loader_generator = torch.Generator().manual_seed(args.random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  
        pin_memory=True,
        generator=loader_generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  
        pin_memory=True
    )

    adj_matrix = adj_mx_from_skeleton(skeleton)
    model = KTPFormer(
        input_dim=dataset.num_joints * 2,
        embed_dim=256,
        adj=adj_matrix,
        depth=2,
        disable_tpa=True
    ).to(args.device)

    model.apply(initialize_weights)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    scaler = GradScaler() if args.device == 'cuda' else None

    def validate(show_visualization=True,skeleton = skeleton):
        model.eval()
        val_loss = 0.0
        first_batch_outputs = None
        first_batch_targets = None
        first_batch_inputs = None

        with torch.no_grad():
            for i, (keypoints, camera_matrix,_,_) in enumerate(val_loader):
                keypoints, camera_matrix  = keypoints.to(args.device), camera_matrix.to(args.device) 
                with torch.amp.autocast(device_type='cuda', enabled=(args.device == 'cuda')):
                    outputs = model(keypoints)
                    loss = weighted_frobenius_loss(outputs, camera_matrix)
                val_loss += loss.item()

                if i == 0 and show_visualization:
                    first_batch_inputs = keypoints.detach().cpu().numpy()
                    first_batch_outputs = outputs.detach().cpu().numpy()
                    first_batch_targets = camera_matrix.detach().cpu().numpy()

        val_loss /= len(val_loader)

        if show_visualization and first_batch_outputs is not None and first_batch_targets is not None:
            visualize_predictions(first_batch_outputs, first_batch_targets)
            #random = random.random()
            visualize_keypoint_skeleton(first_batch_inputs[5],skeleton=skeleton)

        return val_loss

    best_val_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)

        for batch_idx, (keypoints, camera_matrix , _ , _) in enumerate(progress_bar):
            keypoints, camera_matrix = keypoints.to(args.device), camera_matrix.to(args.device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=(args.device == 'cuda')):
                outputs = model(keypoints)
                loss = weighted_frobenius_loss(outputs, camera_matrix)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        scheduler.step()

        show_viz = (epoch % args.visualize_every == 0)
        val_loss = validate(show_visualization=show_viz,skeleton=skeleton)

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            torch.save(model.state_dict(), args.model_save_path)
            print(f"  Model saved with validation loss: {best_val_loss:.4f}")
        else:
            no_improvement_count += 1
            if no_improvement_count >= args.early_stop_patience:
                print(f"No improvement for {args.early_stop_patience} validation checks. Early stopping.")
                break

if __name__ == "__main__":
    args = parse_args()
    train(args)



