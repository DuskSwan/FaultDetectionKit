import numpy as np
from scipy.stats import pearsonr

def euclidean_distance(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    计算两个一维信号（NumPy数组）之间的欧氏距离。

    参数:
        signal1 (np.ndarray): 第一个信号。
        signal2 (np.ndarray): 第二个信号。

    返回:
        float: 两个信号之间的欧氏距离。
    """
    if signal1.shape != signal2.shape:
        raise ValueError("输入信号的形状必须相同。")
    return np.linalg.norm(signal1 - signal2)

def cosine_similarity(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    计算两个一维信号（NumPy数组）之间的余弦相似度。

    参数:
        signal1 (np.ndarray): 第一个信号。
        signal2 (np.ndarray): 第二个信号。

    返回:
        float: 两个信号之间的余弦相似度。
    """
    if signal1.shape != signal2.shape:
        raise ValueError("输入信号的形状必须相同。")
    
    dot_product = np.dot(signal1, signal2)
    norm_signal1 = np.linalg.norm(signal1)
    norm_signal2 = np.linalg.norm(signal2)
    
    if norm_signal1 == 0 or norm_signal2 == 0:
        # 如果任一向量为零向量，则余弦相似度未定义或可视为0
        # 具体取决于应用场景，这里返回0
        return 0.0 
    
    return dot_product / (norm_signal1 * norm_signal2)

def manhattan_distance(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    计算两个一维信号（NumPy数组）之间的曼哈顿距离。

    参数:
        signal1 (np.ndarray): 第一个信号。
        signal2 (np.ndarray): 第二个信号。

    返回:
        float: 两个信号之间的曼哈顿距离。
    """
    if signal1.shape != signal2.shape:
        raise ValueError("输入信号的形状必须相同。")
    return np.sum(np.abs(signal1 - signal2))

def chebyshev_distance(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    计算两个一维信号（NumPy数组）之间的切比雪夫距离。

    参数:
        signal1 (np.ndarray): 第一个信号。
        signal2 (np.ndarray): 第二个信号。

    返回:
        float: 两个信号之间的切比雪夫距离。
    """
    if signal1.shape != signal2.shape:
        raise ValueError("输入信号的形状必须相同。")
    return np.max(np.abs(signal1 - signal2))

def pearson_correlation_coefficient(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    计算两个一维信号（NumPy数组）之间的皮尔逊相关系数。

    参数:
        signal1 (np.ndarray): 第一个信号。
        signal2 (np.ndarray): 第二个信号。

    返回:
        float: 两个信号之间的皮尔逊相关系数。
               返回值的范围是 [-1, 1]。1 表示完全正相关，-1 表示完全负相关，0 表示没有线性相关性。
    """
    if signal1.shape != signal2.shape:
        raise ValueError("输入信号的形状必须相同。")
    if signal1.ndim != 1 or signal2.ndim != 1:
        raise ValueError("输入信号必须是一维数组。")
    if len(signal1) < 2: # 皮尔逊相关系数至少需要两个点
        return np.nan # 或者根据需要返回0.0或抛出错误

    corr, _ = pearsonr(signal1, signal2)
    return corr

