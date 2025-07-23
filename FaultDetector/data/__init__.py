'''
该模块实现有关数据预处理的方法。
'''

from typing import Optional
from loguru import logger

import numpy as np

def sliding_window(
    signal: np.ndarray,
    win_size: int,
    stride: Optional[int] = None,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    从长信号中提取滑动窗口样本。

    参数:
        signal (np.ndarray): 输入信号，形状为 (length,) 或 (length, channel)。
        win_size (int): 滑窗长度。
        stride (int): 滑窗步长，默认为 win_size（即不重叠）。
        max_samples (int): 最多返回的窗口数量，默认不限制。
        shuffle (bool): 是否打乱窗口顺序。
        seed (int): 随机种子，用于打乱时的复现。

    返回:
        np.ndarray: 提取的窗口（样本），形状为 (num_samples, win_size) 或 (num_samples, win_size, channel)。
    """    
    signal = np.asarray(signal)
    length = signal.shape[0]

    if stride is None:
        stride = win_size

    assert win_size <= length, f"Window size {win_size} must be less than or equal to signal length {length}."

    starts = list(range(0, length - win_size + 1, stride))
    
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(starts)

    if max_samples is not None:
        assert max_samples > 0, f"max_samples {max_samples} must be greater than 0."
        assert max_samples <= len(starts), f"max_samples {max_samples} must be less than or equal to the number of possible windows {len(starts)}."
        starts = starts[:max_samples]

    # logger.debug(f"Going to extract {len(starts)} windows from signal of length {length} with window size {win_size} and stride {stride}")
    windows = [signal[start:start + win_size] for start in starts]
    windows = np.stack(windows)
    # logger.debug(f"Extracted samples of shape: {windows.shape}")

    return windows