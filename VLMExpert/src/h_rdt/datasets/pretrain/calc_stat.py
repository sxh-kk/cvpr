import os
import h5py
import json
import numpy as np
from tqdm import tqdm
from glob import glob
from pathlib import Path
import argparse
import random
from scipy.spatial.transform import Rotation as R

def convert_to_camera_frame(tfs, cam_ext):
    '''
    convert transforms from the world frame to the camera frame
    tfs: a set of transforms in the world frame, shape N x 4 x 4
    cam_ext: camera extrinsics in the world frame, shape 4 x 4
    '''
    return np.linalg.inv(cam_ext)[None] @ tfs

def d9_to_mat44(nine_d):
    """
    Convert 9D representation back to 4x4 transformation matrix
    
    Args:
        nine_d: 9D vector [position(3) + rotation_6d(6)]
               rotation_6d is first two columns of rotation matrix
    
    Returns:
        mat44: 4x4 transformation matrix
    """
    position = nine_d[:3]
    rot_col0 = nine_d[3:6]
    rot_col1 = nine_d[6:9]
    
    # Gram-Schmidt orthogonalization to ensure valid rotation matrix
    col0 = rot_col0 / (np.linalg.norm(rot_col0) + 1e-8)
    col1 = rot_col1 - np.dot(rot_col1, col0) * col0
    col1 = col1 / (np.linalg.norm(col1) + 1e-8)
    col2 = np.cross(col0, col1)
    
    mat44 = np.eye(4, dtype=nine_d.dtype)
    mat44[:3, 0] = col0
    mat44[:3, 1] = col1
    mat44[:3, 2] = col2
    mat44[:3, 3] = position
    return mat44

def delta_rpy_from_tfs(tfs):
    """
    Compute delta roll, pitch, yaw between each consecutive transformation matrix.

    Args:
        tfs: np.ndarray of shape (N, 4, 4)

    Returns:
        delta_rpy: np.ndarray of shape (N-1, 3), where each row is (d_roll, d_pitch, d_yaw)
    """
    N = tfs.shape[0]
    delta_rpy = np.zeros((N-1, 3), dtype=np.float32)
    for i in range(N-1):
        # Get rotation matrices
        R1 = tfs[i][:3, :3]
        R2 = tfs[i+1][:3, :3]
        # Relative rotation: R_rel = R2 * R1.T
        R_rel = R2 @ R1.T
        # Convert to euler angles (roll, pitch, yaw)
        rpy = R.from_matrix(R_rel).as_euler('xyz', degrees=False)
        delta_rpy[i] = rpy
    return delta_rpy

def points_to_camera(points_3d, cam_ext):
    # Convert to homogeneous coordinates
    points_3d_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
    # Transform to camera coordinates using extrinsics
    points_cam = (np.linalg.inv(cam_ext) @ points_3d_homo.T).T
    # Get 3D point sin camera frame
    points_cam_3d = points_cam[:, :3]
    return points_cam_3d

def convert_to_delta_actions(actions, chunk_size, cam_ext):
    left_wrist = np.empty((chunk_size,4,4), dtype=np.float32)
    right_wrist = np.empty((chunk_size,4,4), dtype=np.float32)

    left_hand_finger_tips = {
        "leftThumbTip": np.empty((chunk_size, 3), dtype=np.float32),
        "leftIndexFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
        "leftMiddleFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
        "leftRingFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
        "leftLittleFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
    }
    right_hand_finger_tips = {
        "rightThumbTip": np.empty((chunk_size, 3), dtype=np.float32),
        "rightIndexFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
        "rightMiddleFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
        "rightRingFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
        "rightLittleFingerTip": np.empty((chunk_size, 3), dtype=np.float32),
    }

    for i in range(actions.shape[0]):
        # action = actions[i]
        left_wrist[i] = d9_to_mat44(actions[i, :9])
        left_hand_finger_tips["leftThumbTip"][i] = actions[i, 9:12]
        left_hand_finger_tips["leftIndexFingerTip"][i] = actions[i, 12:15]
        left_hand_finger_tips["leftMiddleFingerTip"][i] = actions[i, 15:18]
        left_hand_finger_tips["leftRingFingerTip"][i] = actions[i, 18:21]
        left_hand_finger_tips["leftLittleFingerTip"][i] = actions[i, 21:24]

        right_wrist[i] = d9_to_mat44(actions[i, 24:33])
        right_hand_finger_tips["rightThumbTip"][i] = actions[i, 33:36]
        right_hand_finger_tips["rightIndexFingerTip"][i] = actions[i, 36:39]
        right_hand_finger_tips["rightMiddleFingerTip"][i] = actions[i, 39:42]
        right_hand_finger_tips["rightRingFingerTip"][i] = actions[i, 42:45]
        right_hand_finger_tips["rightLittleFingerTip"][i] = actions[i, 45:48]

    left_wrist_tfs_in_cam = convert_to_camera_frame(left_wrist, cam_ext) # (N,4,4)
    right_wrist_tfs_in_cam = convert_to_camera_frame(right_wrist, cam_ext)

    # convert to deleta action representations
    delta_left_wrist_xyz = left_wrist_tfs_in_cam[1:, :3, 3] - left_wrist_tfs_in_cam[:-1, :3, 3]
    delta_left_wrist_rpy = delta_rpy_from_tfs(left_wrist_tfs_in_cam)

    delta_right_wrist_xyz = right_wrist_tfs_in_cam[1:, :3, 3] - right_wrist_tfs_in_cam[:-1, :3, 3]
    delta_right_wrist_rpy = delta_rpy_from_tfs(right_wrist_tfs_in_cam)

    leftThumbTip = points_to_camera(left_hand_finger_tips["leftThumbTip"], cam_ext)
    delta_left_thumbtip = leftThumbTip[1:] - leftThumbTip[:-1]
    leftIndexFingerTip = points_to_camera(left_hand_finger_tips["leftIndexFingerTip"], cam_ext)
    delta_left_indextip = leftIndexFingerTip[1:] - leftIndexFingerTip[:-1]
    leftMiddleFingerTip = points_to_camera(left_hand_finger_tips["leftMiddleFingerTip"], cam_ext)
    delta_left_middletip = leftMiddleFingerTip[1:] - leftMiddleFingerTip[:-1]
    leftRingFingerTip = points_to_camera(left_hand_finger_tips["leftRingFingerTip"], cam_ext)
    delta_left_ringtip = leftRingFingerTip[1:] - leftRingFingerTip[:-1]
    leftLittleFingerTip = points_to_camera(left_hand_finger_tips["leftLittleFingerTip"], cam_ext)
    delta_left_littletip = leftLittleFingerTip[1:] - leftLittleFingerTip[:-1]

    rightThumbTip = points_to_camera(right_hand_finger_tips["rightThumbTip"], cam_ext)
    delta_right_thumbtip = rightThumbTip[1:] - rightThumbTip[:-1]   
    rightIndexFingerTip = points_to_camera(right_hand_finger_tips["rightIndexFingerTip"], cam_ext)
    delta_right_indextip = rightIndexFingerTip[1:] - rightIndexFingerTip[:-1]
    rightMiddleFingerTip = points_to_camera(right_hand_finger_tips["rightMiddleFingerTip"], cam_ext)
    delta_right_middletip = rightMiddleFingerTip[1:] - rightMiddleFingerTip[:-1]
    rightRingFingerTip = points_to_camera(right_hand_finger_tips["rightRingFingerTip"], cam_ext)
    delta_right_ringtip = rightRingFingerTip[1:] - rightRingFingerTip[:-1]
    rightLittleFingerTip = points_to_camera(right_hand_finger_tips["rightLittleFingerTip"], cam_ext)
    delta_right_littletip = rightLittleFingerTip[1:] - rightLittleFingerTip[:-1]

    # re-construct the delta actions
    current_action = np.concatenate([
        delta_left_wrist_xyz,
        delta_left_wrist_rpy,
        np.zeros_like(delta_left_wrist_rpy),
        delta_left_thumbtip,
        delta_left_indextip,
        delta_left_middletip,
        delta_left_ringtip,
        delta_left_littletip,
        delta_right_wrist_xyz,
        delta_right_wrist_rpy,
        np.zeros_like(delta_right_wrist_rpy),
        delta_right_thumbtip,
        delta_right_indextip,
        delta_right_middletip,
        delta_right_ringtip,
        delta_right_littletip,
    ], axis=1)  # (N-1, 48)
    return current_action

def collect_egodex_action_stats(root_dir, output_path, large_values_log="large_values.txt", use_delta_actions=False, upsample_rate=6):
    """
    Collect statistics for pre-computed 48-dimensional action data in EgoDex dataset
    Directly read actions_48d data from HDF5 files instead of real-time computation
    
    Args:
        root_dir: EgoDex dataset root directory
        output_path: Output JSON file path
        large_values_log: Log file for recording outliers
    """
    # Initialize statistics containers
    global_min = None
    global_max = None
    global_sum = None
    global_sum_sq = None
    global_count = 0
    all_actions = []  # For quantile calculation
    file_count = 0
    error_files = []
    
    # Record files with absolute values exceeding threshold
    large_values_files = []
    threshold = 5 # Set threshold to 5

    # Find all HDF5 files
    root_path = Path(root_dir)
    hdf5_files = []
    
    # Traverse all part directories
    for part in ['part1', 'part2', 'part3', 'part4', 'part5', 'extra', 'test']:
        part_dir = root_path / part
        if part_dir.exists():
            for task_dir in part_dir.iterdir():
                if task_dir.is_dir():
                    task_hdf5_files = list(task_dir.glob('*.hdf5'))
                    hdf5_files.extend(task_hdf5_files)

    # Randomize file order
    random.shuffle(hdf5_files)
    
    print(f"Found {len(hdf5_files)} HDF5 files in EgoDex dataset")

    # Process each file
    for file_path in tqdm(hdf5_files, desc="Processing EgoDex files"):
        try:
            with h5py.File(file_path, 'r') as f:
                # Directly read pre-computed 48-dimensional action data
                if "actions_48d" in f:
                    action_data = f['actions_48d'][:][::upsample_rate]
                    
                    # Check if data shape is correct
                    if action_data.shape[1] != 48:
                        error_files.append((str(file_path), f"Wrong action dimension: {action_data.shape[1]}, expected 48"))
                        continue
                    
                    # Check if there are dimensions with absolute values exceeding threshold
                    if np.any(np.abs(action_data) > threshold):
                        large_values_files.append(str(file_path))
                        
                        # Get specific frame and dimension information
                        abs_data = np.abs(action_data)
                        frames, dims = np.where(abs_data > threshold)
                        max_val = abs_data.max()
                        print(f"Large values found in {file_path}: max={max_val:.3f}")
                    
                    if use_delta_actions:
                        cam_ext = np.array(f['/transforms/camera'][0]) # type: ignore , extrinsics
                        cam_int = f['/camera/intrinsic'][:] # # type: ignore , intrinsics
                        action_data = convert_to_delta_actions(action_data, action_data.shape[0], cam_ext)

                    # For quantile, mean, std calculation
                    all_actions.append(action_data)
                    # For mean/std: running sum and sum of squares
                    if global_sum is None:
                        global_sum = np.sum(action_data, axis=0)
                        global_sum_sq = np.sum(action_data ** 2, axis=0)
                        global_count = action_data.shape[0]
                    else:
                        global_sum += np.sum(action_data, axis=0)
                        global_sum_sq += np.sum(action_data ** 2, axis=0)
                        global_count += action_data.shape[0]

                    # Initialize or update global extremes
                    if global_min is None:
                        global_min = np.min(action_data, axis=0)
                        global_max = np.max(action_data, axis=0)
                    else:
                        global_min = np.minimum(global_min, np.min(action_data, axis=0))
                        global_max = np.maximum(global_max, np.max(action_data, axis=0))
                    
                    file_count += 1

                    # if file_count > 10_000:
                    #     print("early break for testing")
                    #     break
                else:
                    error_files.append((str(file_path), "Missing actions_48d data (run precompute_48d_actions.py first)"))
        except Exception as e:
            error_files.append((str(file_path), str(e)))

    # Compute quantiles, mean, std
    if all_actions:
        all_actions_concat = np.concatenate(all_actions, axis=0)
        quantile_01 = np.quantile(all_actions_concat, 0.01, axis=0)
        quantile_99 = np.quantile(all_actions_concat, 0.99, axis=0)
        mean = global_sum / global_count
        std = np.sqrt(global_sum_sq / global_count - mean ** 2)
        # Dump all_actions_concat as a numpy file
        suffix = "_delta" if use_delta_actions else ""
        np.save(os.path.join(os.path.dirname(output_path), f"all_actions_{suffix}.npy"), all_actions_concat)
    else:
        quantile_01 = []
        quantile_99 = []
        mean = []
        std = []

    # Generate statistics results
    stat_dict = {
        "egodex": {
            "min": global_min.tolist() if global_min is not None else [],
            "max": global_max.tolist() if global_max is not None else [],
            "q01": quantile_01.tolist() if len(quantile_01) else [],
            "q99": quantile_99.tolist() if len(quantile_99) else [],
            "mean": mean.tolist() if len(mean) else [],
            "std": std.tolist() if len(std) else [],
        }
    }

    # Save results
    with open(output_path, 'w') as f:
        json.dump(stat_dict, f, indent=4)
    
    # Save list of files with outliers
    with open(large_values_log, 'w') as f:
        f.write(f"Found {len(large_values_files)} files containing 48-dimensional action data with absolute values > {threshold}:\n\n")
        for file_path in large_values_files:
            f.write(f"{file_path}\n")

    # Print statistics information
    print(f"\nEgoDx 48-dimensional action statistics completed! Successfully processed {file_count} files")
    print(f"Action dimensions: {len(global_min) if global_min is not None else 'N/A'}")
    if global_min is not None:
        print(f"Min values (first 10): {global_min[:10]}")
        print(f"Max values (first 10): {global_max[:10]}")
        print(f"Quantile 1% (first 10): {quantile_01[:10]}")
        print(f"Quantile 99% (first 10): {quantile_99[:10]}")
        print(f"Mean (first 10): {mean[:10]}")
        print(f"Std (first 10): {std[:10]}")
        print(f"Overall range: [{global_min.min():.6f}, {global_max.max():.6f}]")
    print(f"Found {len(large_values_files)} files containing data with absolute values > {threshold}, saved to {large_values_log}")
    
    if error_files:
        print(f"Failed to process files ({len(error_files)}):")
        for i, (path, err) in enumerate(error_files):
            if i < 10:  # Only show first 10 errors
                print(f"- {path}: {err}")
            else:
                print(f"...and {len(error_files)-10} more errors")
                break

def collect_action_stats(root_dir, output_path, large_values_log="large_values.txt"):
    """
    Calculate the extreme values of joint action data from all RobotWin HDF5 files
    Args:
        root_dir: Root directory of HDF5 files (recursively searched)
        output_path: Output JSON file path
        large_values_log: Log file for recording data files with absolute values > 10
    """
    # Initialize statistics containers
    global_min = None
    global_max = None
    file_count = 0
    error_files = []
    
    # Record files with absolute values > 10
    large_values_files = []

    # Recursively find all hdf5 files
    hdf5_files = glob(os.path.join(root_dir, "**/*.hdf5"), recursive=True)
    print(f"Found {len(hdf5_files)} HDF5 files")

    # Process each file
    for file_path in tqdm(hdf5_files, desc="Processing files"):
        try:
            with h5py.File(file_path, 'r') as f:
                if "joint_states" in f:
                    action_data = f['joint_states']['positions'][:]
                    
                    # Check if there are dimensions with absolute values > 10
                    if np.any(np.abs(action_data) > 5):
                        large_values_files.append(file_path)
                        
                        # Get specific frame and dimension information (optional)
                        abs_data = np.abs(action_data)
                        frames, dims = np.where(abs_data > 5)
                        for frame, dim in zip(frames, dims):
                            # Can record more detailed information like frame and dimension indices
                            pass
                    
                    # Initialize or update global extremes
                    if global_min is None:
                        global_min = np.min(action_data, axis=0)
                        global_max = np.max(action_data, axis=0)
                    else:
                        global_min = np.minimum(global_min, np.min(action_data, axis=0))
                        global_max = np.maximum(global_max, np.max(action_data, axis=0))
                    
                    file_count += 1
                else:
                    error_files.append((file_path, "Missing joint_states/positions data"))
        except Exception as e:
            error_files.append((file_path, str(e)))

    # Generate statistics results
    stat_dict = {
        "cvpr_real": {
            "min": global_min.tolist() if global_min is not None else [],
            "max": global_max.tolist() if global_max is not None else [],
        }
    }

    # Save results
    with open(output_path, 'w') as f:
        json.dump(stat_dict, f, indent=4)
    
    # Save list of files with absolute values > 10
    with open(large_values_log, 'w') as f:
        f.write(f"Found {len(large_values_files)} files containing joint position data with absolute values > 10:\n\n")
        for file_path in large_values_files:
            f.write(f"{file_path}\n")

    # Print statistics information
    print(f"\nStatistics completed! Successfully processed {file_count} files")
    print(f"Action dimensions: {len(global_min) if global_min is not None else 'N/A'}")
    print(f"Min values: {global_min}")
    print(f"Max values: {global_max}")
    print(f"Found {len(large_values_files)} files with absolute values > 10, saved to {large_values_log}")
    
    if error_files:
        print(f"Failed files ({len(error_files)}):")
        for i, (path, err) in enumerate(error_files):
            if i < 10:  # Only show first 10 errors
                print(f"- {path}: {err}")
            else:
                print(f"...and {len(error_files)-10} more errors")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate EgoDex dataset statistics')
    parser.add_argument('--data_root', type=str,
                       default=os.environ.get('EGODEX_DATA_ROOT', '/share/hongzhe/datasets/egodex'),
                       help='EgoDex dataset root directory')
    parser.add_argument('--output_path', type=str,
                       help='Output JSON file path for statistics')
    parser.add_argument('--large_values_log', type=str,
                       help='Output text file path for files with large values')
    
    parser.add_argument('--use_delta_actions', action='store_true',
                       help='Whether to convert actions to delta action representation before calculating statistics')
    parser.add_argument("--upsample_rate", type=int, default=3,
                        help="Upsample rate for action sequence") # default 30 hz
    
    args = parser.parse_args()
    
    # Set default paths if not provided
    if args.output_path is None:
        project_root = os.environ.get('HRDT_PROJECT_ROOT', os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        output_dir = os.environ.get('HRDT_OUTPUT_DIR', os.path.join(project_root, 'datasets/pretrain'))
        args.output_path = os.path.join(output_dir, "egodex_stat.json")
    
    if args.large_values_log is None:
        output_dir = os.path.dirname(args.output_path)
        args.large_values_log = os.path.join(output_dir, "egodex_large_values.txt")
    
    print(f"Configuration:")
    print(f"- Data root: {args.data_root}")
    print(f"- Output file: {args.output_path}")
    print(f"- Large values log: {args.large_values_log}")
    print(f"- Use delta actions: {args.use_delta_actions}")
    print(f"- Upsample rate: {args.upsample_rate}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Usage example for EgoDex dataset
    collect_egodex_action_stats(
        root_dir=args.data_root,  # EgoDx dataset root directory
        output_path=args.output_path,  # Output file path
        large_values_log=args.large_values_log,  # Outlier log file
        use_delta_actions=args.use_delta_actions,
        upsample_rate=args.upsample_rate,
    )
