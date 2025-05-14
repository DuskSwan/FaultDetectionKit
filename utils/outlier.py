import numpy as np

def is_outlier_zscore(ref_array: np.ndarray, value: float, threshold: float = 3.0) -> bool:
    """
    基于 Z-Score 方法判断离群值。
    参数:
        ref_array (np.ndarray): 参考数组。
        value (float): 待检测值。
        threshold (float): Z-Score 阈值，默认 3.0。
    返回:
        bool: True 表示为离群值，False 表示正常值。
    """
    mean = np.mean(ref_array)
    std = np.std(ref_array)
    if std == 0:
        return False
    z_score = abs((value - mean) / std)
    return z_score > threshold

def is_outlier_iqr(ref_array: np.ndarray, value: float, multiplier: float = 1.5) -> bool:
    """
    基于 IQR（四分位距）方法判断离群值。
    参数:
        ref_array (np.ndarray): 参考数组。
        value (float): 待检测值。
        multiplier (float): IQR 放大倍数，默认 1.5。
    返回:
        bool: True 表示为离群值，False 表示正常值。
    """
    q1, q3 = np.percentile(ref_array, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    return (value < lower_bound) or (value > upper_bound)

def is_outlier_mad(ref_array: np.ndarray, value: float, threshold: float = 3.5) -> bool:
    """
    基于 MAD（中位数绝对偏差）方法判断离群值。
    参数:
        ref_array (np.ndarray): 参考数组。
        value (float): 待检测值。
        threshold (float): 修改 Z-Score 阈值，默认 3.5。
    返回:
        bool: True 表示为离群值，False 表示正常值。
    """
    median = np.median(ref_array)
    mad = np.median(np.abs(ref_array - median))
    if mad == 0:
        return False
    modified_z = 0.6745 * (value - median) / mad
    return abs(modified_z) > threshold

def is_outlier(ref_array: np.ndarray, value: float, method: str = 'zscore') -> bool:
    """
    判断离群值。
    参数:
        ref_array (np.ndarray): 参考数组。
        value (float): 待检测值。
        method (str): 离群值检测方法，支持 'zscore'、'iqr' 和 'mad'。
    返回:
        bool: True 表示为离群值，False 表示正常值。
    """
    if method == 'zscore':
        return is_outlier_zscore(ref_array, value)
    elif method == 'iqr':
        return is_outlier_iqr(ref_array, value)
    elif method == 'mad':
        return is_outlier_mad(ref_array, value)
    else:
        raise ValueError("Unsupported method. Choose from 'zscore', 'iqr', or 'mad'.")