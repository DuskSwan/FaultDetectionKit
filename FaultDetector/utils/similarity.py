import numpy as np
from scipy.stats import pearsonr
from dtaidistance import dtw

# from loguru import logger

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
    return float(np.linalg.norm(signal1 - signal2))

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

    pearson_res = pearsonr(signal1, signal2)
    return float(pearson_res.correlation)

def dtw_distance(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """
    计算两个一维信号（NumPy数组）之间的 DTW（动态时间规整）距离。

    参数:
        signal1 (np.ndarray): 第一个信号，形状 (n,)。
        signal2 (np.ndarray): 第二个信号，形状 (m,)。

    返回:
        float: 两个信号之间的 DTW 距离。
    """
    n, m = len(signal1), len(signal2)
    # 初始化 DTW 矩阵，边界填充为 +inf
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(signal1[i-1] - signal2[j-1])
            # 三种路径：插入、删除、匹配
            dtw[i, j] = cost + min(
                dtw[i-1, j],    # insertion
                dtw[i, j-1],    # deletion
                dtw[i-1, j-1],  # match
            )
    return float(dtw[n, m])

def dtw_distance_fast(signal1: np.ndarray, signal2: np.ndarray, dtw_radius = 1) -> float:
    """
    计算两个一维信号（NumPy数组）之间的 DTW（动态时间规整）距离，使用 fastdtw 库。

    参数:
        signal1 (np.ndarray): 第一个信号，形状 (n,)。
        signal2 (np.ndarray): 第二个信号，形状 (m,)。

    返回:
        float: 两个信号之间的 DTW 距离。
    """
    distance = dtw.distance(signal1, signal2, window=dtw_radius)
    return float(distance)

def calc_multi_channel_signal_similarity(signal1: np.ndarray, 
                                         signal2: np.ndarray, 
                                         method: str = 'cosine', 
                                         weights='auto',
                                         **kwargs
                                        ) -> float | None:
    """
    计算两个多通道信号之间的相似度或者距离。

    参数:
        signal1 (np.ndarray): 第一个多通道信号，形状为 (length, channel)。
        signal2 (np.ndarray): 第二个多通道信号，形状为 (length, channel)。
        method (str): 相似度/距离计算方法，可以是 'euclidean', 'cosine', 'manhattan', 'chebyshev', 'pearson' 或 'dtw'。
        weights (str or list): 权重，可以是 'auto' 或一个与通道数相同的列表。默认 'auto' 表示均匀权重。
        **kwargs: 额外参数，比如 dtw 的 radius 参数。

    返回:
        float: 两个信号之间的相似度或者距离。
    """

    if signal1.shape != signal2.shape:
        raise ValueError(f"输入信号的形状必须相同，但接收到的形状分别为 {signal1.shape} 和 {signal2.shape}。")
    
    if method == 'euclidean':
        function = euclidean_distance
        use_kwargs = False
    elif method == 'cosine':
        function = cosine_similarity
        use_kwargs = False
    elif method == 'manhattan':
        function = manhattan_distance
        use_kwargs = False
    elif method == 'chebyshev':
        function = chebyshev_distance
        use_kwargs = False
    elif method == 'pearson':
        function = pearson_correlation_coefficient
        use_kwargs = False
    elif method == 'dtw':
        function = dtw_distance_fast
        use_kwargs = True
    else:
        raise ValueError(f"不支持方法{method}。请选择 'euclidean', 'cosine', 'manhattan', 'chebyshev', 'pearson' 或 'dtw'。")

    if len(signal1.shape) == 1:
        return function(signal1, signal2, **kwargs) if use_kwargs else function(signal1, signal2)

    similarities = []
    for i in range(signal1.shape[1]):
        if use_kwargs:
            similarity = function(signal1[:, i], signal2[:, i], **kwargs)
        else:
            similarity = function(signal1[:, i], signal2[:, i])
        similarities.append(similarity)
    
    if weights == 'auto':
        return float(np.mean(similarities))
    elif isinstance(weights, (list, np.ndarray)):
        if len(weights) != signal1.shape[1]:
            raise ValueError(f"权重的长度 {len(weights)} 与信号的通道数 {signal1.shape[1]} 不匹配。")
        return np.dot(similarities, weights) / np.sum(weights)

def calculate_pairwise_similarity(ref_samples: np.ndarray, method: str, **kwargs) -> np.ndarray:
    '''
    计算参考信号片段之间的相似度分布，每个片段都是一个多通道信号
    参数:
        ref_samples (np.ndarray): 参考信号片段，形状为 (n, m, c)，其中 n 是片段数量，m 是每个片段的长度，c 是通道数。
        method (str): 相似度计算方法，支持 'cosine'、'euclidean'、'dtw' 等。
        **kwargs: 其他参数，例如 dtw_radius。
    返回:
        np.ndarray: 参考信号片段之间的相似度分布，形状为 (n, n)，其中 n 是片段数量。    
    '''
    n = len(ref_samples)
    ref_similarities = []

    ij_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    n_pairs = len(ij_pairs) # O(n^2)
    n_pairs = min(2*n, len(ij_pairs)) # O(2n)
    calc_pairs = np.random.choice(len(ij_pairs), size=n_pairs, replace=False)

    for idx in calc_pairs:
        i, j = ij_pairs[idx]
        similarity = calc_multi_channel_signal_similarity(ref_samples[i], ref_samples[j], method=method, **kwargs)
        ref_similarities.append(similarity)
    return np.array(ref_similarities)