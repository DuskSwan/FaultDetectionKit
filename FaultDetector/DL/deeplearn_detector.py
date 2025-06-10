from typing import Any, Callable

from loguru import logger
import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


class DeeplearnDetector():
    """
    涉及深度学习的检测器基类，提供了基本的训练和预测接口。
    """
    def __init__(
            self, 
            device: str = 'cpu',
            batch_size: int = 32,
            max_epochs: int = 10, 
            lr: float = 0.001,
            loss_name: str = 'cross_entropy',
            optimizer_name: str = 'adamw',
        ):
        # necessary parameters
        self.device = device
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.loss_name = loss_name
        self.optimizer_name = optimizer_name
        # parameters to be set in fit
        self.num_workers = 0  # 默认不使用多线程加载数据
        self.train_loader = None
        self.train_samples = np.ndarray([])
        self.model = self._build_model()
    
    def _build_model(self) -> nn.Module:
        '''
        构建模型，子类需要实现具体的模型构建逻辑。
        '''
        raise NotImplementedError("请在子类中实现 _build_model 方法")

    def _build_train_loader(self, *args, **kwargs) -> None:
        '''
        从参考信号中构建训练样本，同时获取类别数、通道数等信息。
        '''

        raise NotImplementedError("请在子类中实现 _build_train_loader 方法")

    def _train_model(self, *args, **kwargs) -> None:
        """
        训练模型。
        
        参数
        ----
        samples : np.ndarray
            输入信号，形状为 (样本数, 时间步长, 通道数)。
        labels : np.ndarray
            标签，形状为 (样本数,)。
        """
        self._build_model()
        self._build_train_loader()
        # self.model = train_model()
        raise NotImplementedError("请在子类中实现 train_model 方法")
    
    def _predict(self, samples: np.ndarray) -> Any:
        '''
        预测信号的结果，可能是分类结果也可能是别的。这里注释的例子是预测类别编号。
        参数:
            samples (np.ndarray): 一组待检测样本，形状为 (n, m, c)，其中 n 是样本数， m 是信号长度，c 是通道数。
        返回:
            Any: 预测结果，可能是类别编号或其他形式的结果。
        '''
        raise NotImplementedError("请在子类中实现 predict_one_signal 方法")