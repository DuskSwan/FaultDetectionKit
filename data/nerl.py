# encoding: utf-8
from pathlib import Path
from loguru import logger

import numpy as np
import pandas as pd
from scipy.io import loadmat

from torch.utils.data import DataLoader

def load_signal_from_mat(file_path) -> pd.DataFrame:
    ignore_keys = ['__header__', '__version__', '__globals__','Speed','Torque']
    data = loadmat(file_path)  # 读取.mat文件
    cols = [f'AN{str(i)}' for i in range(3, 11)] # AN3-AN10
    df = pd.DataFrame([], columns=cols)

    for k, v in data.items():
        if k in ignore_keys:
            continue
        assert isinstance(v, np.ndarray), f"Key: {k} does not contain a numpy array."
        v = v.flatten()
        df[k] = pd.Series(v)
    return df

def build_dataset(sample_n, window_size, is_train=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Build dataset for NREL data
    Args:
        sample_n (int): Number of samples to generate.
        window_size (int): Size of the window for each sample.
    Returns:
        tuple: Tuple containing the samples and labels.
    """

    if is_train: # Use only healthy data for training
        data_dirs = [r"datasets/NREL/Healthy"]
    else:
        data_dirs = [r"datasets/NREL/Healthy",r"datasets/NREL/Damaged"]

    mat_paths = []
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        mat_paths += list(data_dir.glob("*.mat"))
    files_n = len(mat_paths)

    assert files_n < sample_n, f"Not enough files in {data_dirs} to get {sample_n} samples."

    per_file_samples = sample_n // files_n
    if sample_n % files_n != 0:
        per_file_samples += 1

    logger.info(f"Total files: {per_file_samples * files_n}, samples per file: {per_file_samples}")

    X = [] # (n_samples, window_size, n_channels)
    y = [] # (n_samples, )

    for mat_path in mat_paths:
        df = load_signal_from_mat(mat_path)
        for i in range(per_file_samples):
            start = i * window_size
            end = start + window_size
            if end > len(df):
                break
            X.append(df.iloc[start:end].values)
            y.append(0 if "Healthy" in str(mat_path) else 1)

    X = np.array(X)
    y = np.array(y)
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y


def build_dataloader(sample_n, window_size, batch_size, is_train=True) -> DataLoader:
    """
    Build dataloader for NREL data
    Args:
        sample_n (int): Number of samples to generate.
        window_size (int): Size of the window for each sample.
        batch_size (int): Batch size for the dataloader.
        is_train (bool): Whether to use training data or not.
    Returns:
        DataLoader: Dataloader for the dataset.
    """
    X,y = build_dataset(sample_n, window_size, is_train)
    data_loader = DataLoader(
        dataset=list(zip(X, y)),
        batch_size=batch_size,
        shuffle=is_train,
    )

    return data_loader
