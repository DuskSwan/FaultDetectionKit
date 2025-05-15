'''
本模块采取对原始信号直接比较相似度的方法来进行异常检测
流程为
1. 读取参考信号片段，计算参考信号片段之间的相似度分布
2. 读取待检测信号片段，计算待检测信号片段与参考信号片段之间的相似度
3. 判断待检测信号片段与参考信号的相似度是否为离群值
'''

import numpy as np
from loguru import logger

from data import sliding_window
from utils.similarity import calc_multi_channel_signal_similarity
from utils.outlier import is_outlier

def calc_ref_similarity(ref_samples: np.ndarray, method: str, **kwargs) -> np.ndarray:
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
    calc_pairs = np.random.choice(len(ij_pairs), size=min(2*n, len(ij_pairs)), replace=False)
    for idx in calc_pairs:
        i, j = ij_pairs[idx]
        similarity = calc_multi_channel_signal_similarity(ref_samples[i], ref_samples[j], method=method, **kwargs)
        ref_similarities.append(similarity)
    return np.array(ref_similarities)

class SignalSimilarityDetector:
    def __init__(self, 
                 window_size: int = 1024,
                 window_stride: int = None,
                 ref_sample_n: int = 5,
                 pred_sample_n: int = 5,
                 similarity_method: str = 'dtw',
                 dtw_radius: int = 200,
                 outlier_method: str = 'zscore',
                 sample_threshold: float = 0.75,
                 signal_threshold: float = 0.75,
                ):
        # necessary parameters
        self.window_size = window_size
        self.ref_sample_n = ref_sample_n
        self.pred_sample_n = pred_sample_n
        self.similarity_method = similarity_method
        self.dtw_radius = dtw_radius
        self.outlier_method = outlier_method
        self.sample_threshold = sample_threshold
        self.signal_threshold = signal_threshold
        # optional parameters
        self.window_stride = window_stride
        # values to be calculated
        self.ref_samples = np.array([])
        self.ref_similarities = np.array([])

    def fit(self, ref_signals: np.ndarray):
        """
        根据参考信号计算相似度分布
        参数:
            ref_signals (np.ndarray): 参考信号，形状为 (n, m, c)，其中 n 是信号数量，m 是每个信号的长度，c 是通道数。
        """
        logger.debug("Calculating reference signal similarity...")
        samples = np.array([])

        assert self.ref_sample_n > 0, "参考样本数量必须大于0"
        ref_signal_n = len(ref_signals)
        sample_per_signal = self.ref_sample_n // ref_signal_n
        sample_per_signal_list = [sample_per_signal] * ref_signal_n
        sample_per_signal_list[:self.ref_sample_n % ref_signal_n] = [sample_per_signal + 1] * (self.ref_sample_n % ref_signal_n)

        for i,signal in enumerate(ref_signals):
            if sample_per_signal_list[i] > 0:
                samples0 = sliding_window(signal, self.window_size, self.window_stride, sample_per_signal_list[i])
                samples = np.concatenate((samples, samples0), axis=0) if samples.size else samples0
        
        # self.ref_samples = np.random.choice(samples, size=min(self.ref_sample_n, len(samples)), replace=False)
        indecies = np.random.choice(len(samples), size=min(self.ref_sample_n, len(samples)), replace=False)
        self.ref_samples = samples[indecies]
        self.ref_similarities = calc_ref_similarity(samples, method=self.similarity_method, dtw_radius=self.dtw_radius)
        logger.debug(f"Reference samples shape: {self.ref_samples.shape}")
        logger.debug(f"Reference similarities shape: {self.ref_similarities.shape}")

    def check_sample(self, sample: np.ndarray) -> bool:
        '''
        检查样本相对于参考样本是否为离群值
        '''
        unknown_similarites = []
        for ref_sample in self.ref_samples:
            similarity = calc_multi_channel_signal_similarity(ref_sample, sample, method=self.similarity_method, dtw_radius=self.dtw_radius)
            unknown_similarites.append(similarity)

        outliter_cnt = 0
        for i, sim in enumerate(unknown_similarites):
            is_outlier_result = is_outlier(ref_array=self.ref_similarities, value=sim, method=self.outlier_method)
            outliter_cnt += 1 if is_outlier_result else 0
        outliter_rate = outliter_cnt / len(unknown_similarites)

        # logger.debug(f"Outlier rate: {outliter_rate:.2f}")
        return outliter_rate > self.sample_threshold

    def predict_one_signal(self, signal: np.ndarray) -> bool:
        """
        检测信号是否为异常信号
        参数:
            signal (np.ndarray): 待检测信号，形状为 (m, c)，其中m是信号的长度，c 是通道数。
        返回:
            bool: 是否为异常信号
        """        
        samples = sliding_window(signal, self.window_size, self.window_stride, self.pred_sample_n)
        abnormal_cnt = 0
        for sample in samples:
            is_abnormal = self.check_sample(sample)
            abnormal_cnt += 1 if is_abnormal else 0
        abnormal_rate = abnormal_cnt / len(samples)

        logger.debug(f"Abnormal rate: {abnormal_rate:.2f}")
        return abnormal_rate > self.signal_threshold

    def predict(self, signals: np.ndarray) -> list[bool]:
        """
        检测信号是否为异常信号
        参数:
            signals (np.ndarray): 待检测信号，形状为 (n, m, c)或者(m, c)，其中 n 是信号数量，m 是每个信号的长度，c 是通道数。
        返回:
            list[bool]: 是否为异常信号
        """
        assert self.ref_samples.size > 0 and self.ref_similarities.size > 0, "请先调用 fit 方法计算参考信号相似度分布"
        
        if len(signals.shape) == 2:
            return self.predict_one_signal(signals)
        elif len(signals.shape) == 3:
            results = []
            for signal in signals:
                result = self.predict_one_signal(signal)
                results.append(result)
            return results
        else:
            raise ValueError("信号的形状不正确，请检查输入信号的形状")

        
        