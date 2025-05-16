from typing import List
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
    return bool(z_score > threshold)

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
    return bool(abs(modified_z) > threshold)

def is_outlier(ref_array: np.ndarray, values: List[float], method: str = 'zscore') -> List[bool]:
    """
    判断离群值。
    参数:
        ref_array (np.ndarray): 参考数组。
        values (float): 待检测值列表。
        method (str): 离群值检测方法，支持 'zscore'、'iqr' 和 'mad'。
    返回:
        bool: True 表示为离群值，False 表示正常值。
    """
    assert method in ['zscore', 'iqr', 'mad'], f"不支持方法 {method}。请选择 'zscore'、'iqr' 或 'mad'。"
    assert len(ref_array) > 0, "参考数组不能为空"
    assert len(values) > 0, "待检测值列表不能为空"

    if method == 'zscore':
        is_outlier_func = is_outlier_zscore
    elif method == 'iqr':
        is_outlier_func = is_outlier_iqr
    elif method == 'mad':
        is_outlier_func = is_outlier_mad
    
    return [is_outlier_func(ref_array, value) for value in values]
    