# -*- coding: utf-8 -*-

'''
该模块实现了基于单分类方法的异常检测器，其思路是建立一个单分类模型，确定正常信号和异常信号的边界。
'''

from typing import Optional, Literal
from loguru import logger
import numpy as np

from sklearn.svm import OneClassSVM

from .data import sliding_window

class OneClassifyDetector:
    """
    该类是一切单分类方法检测器的基类
    该类方法的主要思路是根据正常信号建立单分类模型，单分类模型会判别未知信号是否正常（与训练样本同类）。
    """
    def __init__(self,
        window_size: int = 1024,
        window_stride: Optional[int] = None,
        pred_sample_n: int = 20,
        train_sample_n: int = 1000,
        signal_threshold: float = 0.75,
    ):
        self.window_size = window_size
        self.window_stride = window_stride if window_stride is not None else window_size // 2
        self.pred_sample_n = pred_sample_n
        self.train_sample_n = train_sample_n
        self.signal_threshold = signal_threshold
        self.model = None
    
    def _signals_to_samples(self, signals: np.ndarray) -> np.ndarray:
        '''
        将信号转换为样本
        参数:
            signals (np.ndarray): 待检测信号，形状为 (n, m, c)，其中 n 是信号数量，m 是信号长度，c 是通道数。
        返回:
            np.ndarray: 样本，形状为 (n, m, c)，其中 n 是样本数量，m 是窗口大小，c 是通道数。
        '''
        assert len(signals.shape) == 3, "Signals must be a 3D array (samples, time steps, channels)"
        assert signals.shape[1] >= self.window_size, f"Signal length must be at least {self.window_size} for windowing"
        assert signals.shape[2] > 0, "Signals must have at least one channel"
        X_train = []
        # 对每个信号进行滑动窗口处理
        each_train_sample_n = self.train_sample_n // signals.shape[0]
        if each_train_sample_n <= 0:
            logger.warning(f"train_sample_n {self.train_sample_n} is too small for {signals.shape[0]} signals, using all samples.")
            each_train_sample_n = signals.shape[0]
        for signal in signals:
            X_train.append(
                sliding_window(signal, self.window_size, self.window_stride, each_train_sample_n, shuffle=True)
            )
        return np.concatenate(X_train, axis=0)

    def _check_samples(self, samples: np.ndarray) -> np.ndarray:
        '''
        检测样本的类别
        参数:
            samples (np.ndarray): 待检测样本，形状为 (n, m, c)，其中 n 是样本数量，m 是信号长度，c 是通道数。
        返回:
            np.ndarray: 类别编号，形状为 (n,)，其中 n 是样本数量。
        '''
        assert self.model is not None, "请先调用 fit 方法训练模型"
        assert len(samples.shape) == 3, "样本必须是三维数组 (n, m, c)"
        raise NotImplementedError("请在子类中实现 _check_samples 方法")

    def predict_one_signal(self, signal: np.ndarray) -> int:
        '''
        检测信号的类别。每个信号片段会被切分成多个样本，使用模型进行预测，最终返回出现最多的类别编号。 

        参数:
            signal (np.ndarray): 待检测信号片段，形状为 (m, c)，其中 m 是信号长度，c 是通道数。
        返回:
            int: 类别编号，1 表示正常，-1 表示异常。
        '''
        samples = sliding_window(signal, self.window_size, self.window_stride, self.pred_sample_n, shuffle=True)
        res = self._check_samples(samples)
        # 与阈值比较，判断信号是否异常
        abnormal_count = np.sum(res == -1)
        logger.debug(f"Predicted abnormal ratio: [{abnormal_count}/{len(res)}]")
        return -1 if abnormal_count / len(res) > self.signal_threshold else 1
    
    def predict(self, signals: np.ndarray) -> int | list[int]:
        '''
        检测信号的类别
        参数:
            signals (np.ndarray): 待检测信号，形状为 (n, m, c)或者(m, c)，其中 n 是信号数量，m 是每个信号的长度，c 是通道数。
        返回:
            int | list[int]: 类别编号
        '''
        assert self.model is not None , "请先调用 fit 方法训练模型"
        if len(signals.shape) == 2:
            # logger.debug("Predicting signal 1/1")
            return self.predict_one_signal(signals)
        elif len(signals.shape) == 3:
            results = []
            for i,signal in enumerate(signals):
                # logger.debug(f"Predicting signal {i+1}/{len(signals)}")
                result = self.predict_one_signal(signal)
                results.append(result)
            return results
        else:
            raise ValueError("输入信号的形状不正确")

class OCSVMDetector(OneClassifyDetector):
    """
    基于 OneClass-SVM 的分类模型，用于信号异常检测。
    """
    def __init__(
            self, 
            train_sample_n: int = 1000,
            pred_sample_n: int = 20,
            window_size: int = 256,
            window_stride: int = 128,
            signal_threshold: float = 0.75,
            kernel: Literal['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] = 'rbf',
            gamma: Literal['scale', 'auto'] = 'scale',
            nu: float = 0.5,
        ):
        OneClassifyDetector.__init__(self,
            train_sample_n = train_sample_n,
            pred_sample_n = pred_sample_n,
            window_size = window_size,
            window_stride = window_stride,
            signal_threshold = signal_threshold,
        )

        # OCSVM parameters
        self.kernel: Literal['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'] = kernel
        self.gamma: Literal['scale', 'auto'] = gamma
        self.nu: float = nu
        self.model = self._build_model()  # 初始化模型
    
    def _build_model(self) -> OneClassSVM:
        '''
        创建 OneClass SVM 模型。
        '''
        return OneClassSVM(
            kernel=self.kernel,
            gamma=self.gamma,
            nu=self.nu,
        )

    def fit(self, signals: np.ndarray) -> None:
        """
        训练模型。
        
        参数
        ----
        signals : np.ndarray
            待检测信号，形状为 (n, m, c)，其中 n 是信号数量，m 是信号长度，c 是通道数。
        """
        assert len(signals.shape) == 3, "Signals must be a 3D array (samples, time steps, channels)"
        assert signals.shape[1] >= self.window_size, f"Signal length must be at least {self.window_size} for windowing"
        assert signals.shape[2] > 0, "Signals must have at least one channel"

        X_train = self._signals_to_samples(signals)
        logger.debug(f"Training samples shape: {X_train.shape}")
        X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten the samples for OCSVM
        # 训练模型
        self.model.fit(X_train)

    def _check_samples(self, samples: np.ndarray) -> np.ndarray:
        '''
        检测样本的类别
        参数:
            samples (np.ndarray): 待检测样本，形状为 (n, m, c)，其中 n 是样本数量，m 是信号长度，c 是通道数。
        返回:
            np.ndarray: 类别编号列表，每个元素为 1 或 -1，1 表示正常，-1 表示异常。
        '''
        assert self.model is not None, "请先调用 fit 方法训练模型"
        X_samples = samples.reshape(samples.shape[0], -1)
        # 使用模型进行预测
        res = self.model.predict(X_samples)
        return res

if __name__ == "__main__":
    pass