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

def get_samples_from_signal(signal:np.ndarray, sample_n, window_size) -> np.array:
    """
    Get samples from signal
    Args:
        signal (np.ndarray): Signal data.
        sample_n (int): Number of samples to generate.
        window_size (int): Size of the window for each sample.
    Returns:
        np.ndarray: Samples extracted from the signal.
    """
    n_samples = len(signal) // window_size
    assert n_samples >= sample_n, f"Not enough samples in {signal} to get {sample_n} samples."

    X = [] # (n_samples, window_size, n_channels)

    for i in range(sample_n):
        start = i * window_size
        end = start + window_size
        if end > len(signal):
            break
        X.append(signal[start:end])

    return np.array(X)

def get_samples_from_mat(file_path, sample_n, window_size) -> tuple[np.ndarray, np.ndarray]:
    """
    Get samples from .mat file
    Args:
        file_path (str): Path to the .mat file.
        sample_n (int): Number of samples to generate.
        window_size (int): Size of the window for each sample.
    Returns:
        tuple: Tuple containing the samples and labels.
    """
    df = load_signal_from_mat(file_path)
    n_samples = len(df) // window_size
    assert n_samples >= sample_n, f"Not enough samples in {file_path} to get {sample_n} samples."

    X = [] # (n_samples, window_size, n_channels)
    y = [] # (n_samples, )

    for i in range(sample_n):
        start = i * window_size
        end = start + window_size
        if end > len(df):
            break
        X.append(df.iloc[start:end].values)
        y.append(0 if "Healthy" in str(file_path) else 1)

    return np.array(X), np.array(y)

def build_dataset(sample_n, window_size, is_train=True) -> tuple[np.ndarray, np.ndarray]:
    """
    Build dataset for NREL data
    Args:
        sample_n (int): Number of samples to generate.
        window_size (int): Size of the window for each sample.
        is_train (bool): Whether to use training data or not. If True, use only healthy data.
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

    # assert files_n < sample_n, f"Not enough files in {data_dirs} to get {sample_n} samples."

    sample_per_file = sample_n // files_n
    sample_per_file_list = [sample_per_file] * files_n
    sample_per_file_list[:sample_n % files_n] = [sample_per_file + 1] * (sample_n % files_n)

    logger.info(f"Total files: {sample_n}, samples per file: {sample_per_file_list}")

    X = [] # (n_samples, window_size, n_channels)
    y = [] # (n_samples, )

    for i,mat_path in enumerate(mat_paths):
        per_file_samples = sample_per_file_list[i]
        df = load_signal_from_mat(mat_path)

        if is_train:
            start_index = np.random.choice(range(len(df) - window_size), per_file_samples, replace=False)
        else:
            start_index = np.arange(len(df) - window_size, step=window_size)[:per_file_samples]
        assert len(start_index) == per_file_samples, f"Not enough samples in {mat_path} to get {per_file_samples} samples."
        
        for start in start_index:
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
