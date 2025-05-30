# -*- coding: utf-8 -*-

'''
该模块实现了基于分类模型的信号检测器，其思路就是建立一个分类模型，将正常信号和异常信号进行分类。
'''
from loguru import logger
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader

from .data import sliding_window
from .DL.modeling import LSTMClassifier
from .DL.lightning import train_model, predict_model

class LSTMClassifyDetector:
    """
    基于 LSTM 的分类模型，用于信号异常检测。
    """
    def __init__(
            self, 
            device: str = 'cpu',
            batch_size: int = 32, 
            train_sample_n: int = 1000,
            pred_sample_n: int = 20,
            hidden_size: int = 64, 
            num_layers: int = 1, 
            max_epochs: int = 10, 
            lr: float = 0.001,
            loss_name: str = 'cross_entropy',
            optimizer: str = 'adamw',
            window_size: int = 256,
            window_stride: int = 128
        ):
        # necessary parameters
        self.device = device
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.train_sample_n = train_sample_n
        self.pred_sample_n = pred_sample_n
        self.lr = lr
        self.loss_name = loss_name
        self.optimizer = optimizer
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.window_stride = window_stride
        # parameters to be set in fit
        self.num_workers = 0  # 默认不使用多线程加载数据
        self.model = None
        self.train_loader = None
        self.train_samples = np.ndarray([])
    
    def _build_train_loader(self, ref_signals: np.ndarray, labels: np.ndarray):
        '''
        从参考信号中构建训练样本，同时获取类别数、通道数等信息。
        '''
        # logger.debug("Building train loader...")
        assert self.train_sample_n > 0, "训练用样本数量必须大于0"

        # 计算每个参考信号片段的样本数量
        ref_signal_n = len(ref_signals)
        sample_per_signal = self.train_sample_n // ref_signal_n
        sample_per_signal_list = [sample_per_signal] * ref_signal_n
        sample_per_signal_list[:self.train_sample_n % ref_signal_n] = [sample_per_signal + 1] * (self.train_sample_n % ref_signal_n)
        # logger.debug(f"Sample per signal: {sample_per_signal_list}")

        # 读取参考信号片段，构建训练样本
        X = np.array([])
        y = np.array([])
        for i,signal in enumerate(ref_signals):
            if sample_per_signal_list[i] > 0:
                X0 = sliding_window(signal, self.window_size, self.window_stride, 
                                    sample_per_signal_list[i], shuffle=True)
                y0 = labels[i] * np.ones((X0.shape[0],), dtype=np.int64)  # 标签为当前信号的标签
                X = np.concatenate((X, X0), axis=0) if X.size else X0
                y = np.concatenate((y, y0), axis=0) if y.size else y0
        
        # 获取样本数量、时间步长和通道数
        n, _, self.n_channels = X.shape
        self.n_classes = len(np.unique(y))
        # 截取特定数量的样本
        indices = np.random.choice(n, size=min(self.train_sample_n,n), replace=False)
        X_sampled = X[indices]
        y_sampled = y[indices]
        # 构建数据加载器
        X_tensor = torch.tensor(X_sampled, dtype=torch.float32)
        y_tensor = torch.tensor(y_sampled, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        logger.debug(f"Train loader size: {len(self.train_loader)}, X shape: {X_tensor.shape}, classes: {self.n_classes}, channels: {self.n_channels}")
    
    def fit(self, signals: np.ndarray, labels: np.ndarray):
        """
        训练 LSTM 分类模型。
        
        参数
        ----
        signals : np.ndarray
            输入信号，形状为 (信号数, 时间步长, 通道数)。
        labels : np.ndarray
            标签，形状为 (样本数,)。
        """
        assert len(signals.shape) == 3, "Signals must be a 3D array (samples, time steps, channels)"
        assert len(labels.shape) == 1, "Labels must be a 1D array (samples,)"
        assert signals.shape[0] == labels.shape[0], "Number of signals must match number of labels"
        assert signals.shape[1] >= self.window_size, f"Signal length must be at least {self.window_size} for windowing"
        assert signals.shape[2] > 0, "Signals must have at least one channel"

        self._build_train_loader(signals, labels)

        logger.debug(f"input size: {self.n_channels}, hidden size: {self.hidden_size}, num layers: {self.num_layers}, output size: {self.n_classes}")
        model = LSTMClassifier(
            input_size=self.n_channels, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            output_size=self.n_classes
        )
        self.model = train_model(
            model, 
            self.train_loader, 
            max_epochs=self.max_epochs, 
            lr=self.lr, 
            loss_name=self.loss_name, 
            optimizer=self.optimizer
        )

    def _check_samples(self, samples: np.ndarray) -> list[int]:
        '''
        检测样本的类别
        参数:
            samples (np.ndarray): 待检测样本，形状为 (n, m, c)，其中 n 是样本数量，m 是信号长度，c 是通道数。
        返回:
            list[int]: 类别编号列表
        '''
        assert self.model is not None, "请先调用 fit 方法训练模型"
        assert len(samples.shape) == 3, "样本必须是三维数组 (n, m, c)"
        
        # 将样本转换为 PyTorch 张量并移动到指定设备
        samples_tensor = torch.tensor(samples, dtype=torch.float32).to(self.device)
        
        # 使用模型进行预测
        preds = predict_model(self.model, samples_tensor, device=self.device)
        
        # 获取预测的类别编号
        return preds.argmax(axis=1).tolist()  
    
    def predict_one_signal(self, signal: np.ndarray) -> int:
        '''
        检测信号的类别
        参数:
            signal (np.ndarray): 待检测信号片段，形状为 (m, c)，其中 m 是信号长度，c 是通道数。
        返回:
            int: 类别编号
        '''
        samples = sliding_window(signal, self.window_size, self.window_stride, self.pred_sample_n, shuffle=True)
        res_list = self._check_samples(samples)
        res = np.array(res_list)
        # 最终结果为出现最多的编号
        unique, counts = np.unique(res, return_counts=True)
        most_common = unique[np.argmax(counts)]
        logger.debug(f"Predicted class: {most_common}, Counts: {counts}")
        return most_common
    
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