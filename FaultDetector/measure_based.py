# -*- coding: utf-8 -*-

'''
该模块实现了基于信号相似度的故障检测器，整体思路是：
1. 读取参考信号片段，计算参考信号片段之间的相似度分布
2. 读取待检测信号片段，计算待检测信号片段与参考信号片段之间的相似度
3. 判断待检测信号片段与参考信号的相似度是否为离群值，是则认为该信号片段为异常信号
'''

import numpy as np
from loguru import logger
from typing import Optional, List, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from .data import sliding_window
from .DL.modeling import TimeSeriesConvAE
from .DL.lightning import train_model, loss_fn_resolve
from .utils.similarity import calc_multi_channel_signal_similarity, calculate_pairwise_similarity
from .utils.outlier import is_outlier

class MeasureDetector:
    '''
    该类是一切基于信号相似度的检测器的基类
    该类方法的主要思路是计算参考信号片段之间的相似度分布，并判断待检测信号片段与参考信号片段之间的相似度是否为离群值
    '''
    def __init__(self, 
        window_size: int = 1024,
        window_stride: Optional[int] = None,
        ref_sample_n: int = 5,
        pred_sample_n: int = 5,
        outlier_method: str = 'zscore',
        outlier_threshold: Optional[float] = None,
        signal_threshold: float = 0.75,
    ):
        # necessary parameters
        self.window_size = window_size
        self.ref_sample_n = ref_sample_n
        self.pred_sample_n = pred_sample_n
        self.outlier_method = outlier_method
        self.signal_threshold = signal_threshold
        # optional parameters
        self.window_stride = window_stride
        self.outlier_threshold = outlier_threshold
        # parameters to be set in fit
        self.ref_samples = np.ndarray([])
        self.ref_measure = np.ndarray([])
    
    def _build_ref_samples(self, ref_signals: np.ndarray):
        '''
        从参考信号中构建参考样本
        ref_signals是一个多通道信号，形状为 (n, m, c)，其中 n 是信号数量，m 是每个信号的长度，c 是通道数。
        该方法会将参考信号片段划分为多个窗口，并从中随机选择一定数量的样本作为参考样本。
        '''
        # logger.debug("Calculating reference signal similarity...")
        samples = np.array([])
        assert len(ref_signals.shape) == 3, "参考信号应该以(n, m, c)的形式输入，实际为" + str(ref_signals.shape)

        assert self.ref_sample_n > 0, "参考样本数量必须大于0"
        ref_signal_n = len(ref_signals)
        sample_per_signal = self.ref_sample_n // ref_signal_n
        sample_per_signal_list = [sample_per_signal] * ref_signal_n
        sample_per_signal_list[:self.ref_sample_n % ref_signal_n] = [sample_per_signal + 1] * (self.ref_sample_n % ref_signal_n)

        for i,signal in enumerate(ref_signals):
            if sample_per_signal_list[i] > 0:
                samples0 = sliding_window(signal, self.window_size, self.window_stride, 
                                          sample_per_signal_list[i], shuffle=True)
                samples = np.concatenate((samples, samples0), axis=0) if samples.size else samples0
        
        indecies = np.random.choice(len(samples), size=min(self.ref_sample_n, len(samples)), replace=False)
        self.ref_samples = samples[indecies]
        # logger.debug(f"Build reference samples of {self.ref_samples.shape}")
    
    def fit(self, ref_signals: np.ndarray):
        '''
        根据参考信号片段计算归为正常信号的参考度量（如相似度）self.ref_measure
        ref_signals是一个多通道信号，形状为 (n, m, c)，其中 n 是信号数量，m 是每个信号的长度，c 是通道数。
        '''
        assert len(ref_signals.shape) == 3, "参考信号应该以(n, m, c)的形式输入，实际为" + str(ref_signals.shape)
        self._build_ref_samples(ref_signals)
        # self.ref_measure = ... 必须要计算出self.ref_measure
        raise NotImplementedError("请在子类中实现该方法")
    
    def _check_samples(self, samples: np.ndarray) -> List[bool]:
        '''
        检查样本相对于参考样本是否为离群值
        '''
        unknown_measures = self._calc_unknown_measures(samples)
        logger.debug(f"Unknown measures: {np.mean(unknown_measures):.2f}")
        dectect_result = self._is_outlier(unknown_measures)
        # dectect_result = self._is_outlier([np.mean(unknown_measures)])
        # logger.debug(f"Detect result: {dectect_result}")
        return dectect_result
    
    def _calc_unknown_measures(self, samples: np.ndarray) -> List[float]:
        '''
        计算待检测信号片段的度量（如相似度），每个样本计算一个度量
        '''
        raise NotImplementedError("请在子类中实现该方法")
    
    def _is_outlier(self, unknown_measures: List[float]) -> List[bool]:
        '''
        检查待检测信号片段的度量是否为离群值，每个度量计算一次
        '''
        is_outlier_result = is_outlier(ref_array=self.ref_measure, values=unknown_measures, 
                                       method=self.outlier_method, threshold=self.outlier_threshold)
        return is_outlier_result

    def predict_one_signal(self, signal: np.ndarray) -> bool:
        '''
        检测信号是否为异常信号
        参数:
            signal (np.ndarray): 待检测信号片段，形状为 (m, c)，其中 m 是信号长度，c 是通道数。
        返回:
            bool: True 表示异常信号，False 表示正常信号
        '''
        samples = sliding_window(signal, self.window_size, self.window_stride, self.pred_sample_n, shuffle=True)
        outlier_list = self._check_samples(samples)
        # return any(outlier_list)
        outlier_rate = sum(outlier_list) / len(samples)
        logger.debug(f"Outlier rate: {outlier_rate:.2f}")
        return outlier_rate > self.signal_threshold

    def predict(self, signals: np.ndarray) -> bool | list[bool]:
        '''
        检测信号是否为异常信号
        参数:
            signals (np.ndarray): 待检测信号，形状为 (n, m, c)或者(m, c)，其中 n 是信号数量，m 是每个信号的长度，c 是通道数。
        返回:
            bool | list[bool]: 是否为异常信号. 如果输入信号为多通道信号，则返回一个布尔值列表，表示每个信号是否为异常信号。
        '''
        assert self.ref_samples.size > 0 and self.ref_measure.size > 0, "请先调用 fit 方法计算参考信号相似度分布"
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

class RawSignalSimilarityDetector(MeasureDetector):
    '''
    本模块采取对原始信号直接比较相似度的方法来进行异常检测

    训练数据仅需正常信号

    流程为
    1. 读取参考信号片段，计算参考信号片段之间的相似度分布
    2. 读取待检测信号片段，计算待检测信号片段与参考信号片段之间的相似度
    3. 判断待检测信号片段与参考信号的相似度是否为离群值
    '''
    def __init__(self, 
                 similarity_method: str = 'dtw',
                 dtw_radius: Optional[int] = 5,
                 **kwargs,
                ):
        # father class parameters
        super().__init__(**kwargs)
        # necessary parameters
        self.similarity_method = similarity_method
        self.dtw_radius = dtw_radius

    def fit(self, ref_signals: np.ndarray):
        '''
        根据参考信号片段计算归为正常信号的相似度分布
        '''
        self._build_ref_samples(ref_signals)
        self.ref_measure = calculate_pairwise_similarity(self.ref_samples, method=self.similarity_method, dtw_radius=self.dtw_radius)
        # logger.debug(f"Reference signal similarity shape: {self.ref_measure.shape}")
        logger.debug(f"Reference signal similarity mean: {np.mean(self.ref_measure):.2f}")
    
    def _calc_unknown_measures(self, samples: np.ndarray) -> List[float]:
        '''
        计算待检测信号片段的相似度（如欧氏距离）
        '''
        unknown_measures = []
        for sample in samples:
            unknown_similarites = []
            for ref_sample in self.ref_samples:
                similarity = calc_multi_channel_signal_similarity(ref_sample, sample, method=self.similarity_method, dtw_radius=self.dtw_radius)
                unknown_similarites.append(similarity)
            unknown_measures.append(np.mean(unknown_similarites))
        
        return unknown_measures

class AEDetector(MeasureDetector):
    '''
    本模块采取对原始信号进行建模的方式来实现故障检测，对正常信号建立AE模型，其在正常信号上的重建误差小
    故障信号上的误差相对于正常信号上的误差为离群值，利用离群值检测算法来判断待检测信号片段是否为故障信号。

    训练仅需正常信号

    流程为
    1. 读取参考信号片段，训练模型
    2. 读取待检测信号片段，进行预测，计算预测误差
    3. 检查预测误差是否为离群值
    '''
    def __init__(self,
                 device: str = 'cpu',
                 batch_size: int = 32,
                 max_epochs: int = 10,
                 lr: float = 1e-3,
                 train_sample_n: int = 1000,
                 loss_name: str = 'mse',
                 num_workers: int = 0,
                 optimizer: str = 'adam',
                 latent_dim: int = 32,
                 hidden_dims: List[int] = [16, 32,],
                 **kwargs,
                ):
        # father class parameters
        super().__init__(**kwargs)
        # necessary parameters
        self.device = device
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.train_sample_n = train_sample_n
        self.lr = lr
        self.loss_name = loss_name
        self.optimizer = optimizer
        # optional parameters
        self.num_workers = num_workers
        self.hidden_dims = hidden_dims
        # parameters to be set in fit
        self.model = None
        self.train_loader = None
        self.train_samples = np.ndarray([])
        self.ref_measure = np.ndarray([])
        self.ref_samples = np.ndarray([])
        
    def _build_train_loader(self, ref_signals: np.ndarray):
        '''
        从参考信号中构建训练样本，注意这不同于构造参考样本
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
        samples = np.array([])
        for i,signal in enumerate(ref_signals):
            if sample_per_signal_list[i] > 0:
                samples0 = sliding_window(signal, self.window_size, self.window_stride, 
                                          sample_per_signal_list[i], shuffle=True)
                samples = np.concatenate((samples, samples0), axis=0) if samples.size else samples0
        
        # 截取特定数量的样本
        indecies = np.random.choice(len(samples), size=min(self.train_sample_n, len(samples)), replace=False)
        self.train_samples = samples[indecies]

        # 构建数据加载器
        tensor_train_samples = torch.from_numpy(self.train_samples).float() # (n, seq_len, channels)
        dataset = TensorDataset(tensor_train_samples)
        persistent_workers = self.num_workers > 0
        self.train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, persistent_workers=persistent_workers)
        # logger.debug(f"Train loader size: {len(self.train_loader)}")
        logger.debug(f"Train samples shape: {self.train_samples.shape}")

    def fit(self, ref_signals: np.ndarray):
        self._build_ref_samples(ref_signals)
        self._build_train_loader(ref_signals)
        self.model = self._build_model()

        self.model = train_model(
            model=self.model,
            train_dataloader=self.train_loader,
            loss_name=self.loss_name,
            optimizer_name=self.optimizer,
            lr=self.lr,
            max_epochs=self.max_epochs,
            device=self.device,  # 或 "cpu"
        )

        ref_samples_batch = torch.from_numpy(self.ref_samples).float() # (n, seq_len, channels)
        preds, losses = AE_predict(
            model=self.model,
            batch=ref_samples_batch,
            loss_name=self.loss_name,
            device=self.device,
        )
        self.ref_measure = losses # (n,)
        logger.debug(f"Ref singal mean loss: {np.mean(losses):.4f} ,max: {np.max(losses):.2f}")
        logger.debug(f"Ref signal measure: {self.ref_measure}")
        # logger.debug(f"Reference signal measure shape: {self.ref_measure.shape}")
    
    def _build_model(self):
        sample = self.ref_samples[0]
        seq_len, n_channels = sample.shape
        model = TimeSeriesConvAE(
            seq_len=seq_len,
            n_channels=n_channels,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
        )
        return model

    def _calc_unknown_measures(self, samples: np.ndarray) -> List[float]:
        '''
        计算待检测信号片段的重建误差
        '''
        assert self.model is not None, "请先调用 fit 方法训练模型"
        assert len(samples.shape) == 3, "待检测信号应该以(n, m, c)的形式输入，实际为" + str(samples.shape)

        unknown_samples_batch = torch.from_numpy(samples).float() # (n, seq_len, channels)
        preds, losses = AE_predict(
            model=self.model,
            batch=unknown_samples_batch,
            loss_name=self.loss_name,
            device=self.device,
        )
        unknown_measures = [float(loss) for loss in losses] # (n,)
        
        return unknown_measures

@torch.no_grad()
def AE_predict(
    model: torch.nn.Module,
    batch: torch.Tensor,
    loss_name: str,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    AE推理函数，返回模型对一组样本(也就是一个batch)的输出以及对应的loss。

    参数:
        model: 训练好的模型
        batch: 输入数据
        loss_name: 损失函数名称，例如 "mse" 
        device: 推理设备（"cpu" | "cuda"）

    返回:
        一个元组，前者是模型输出，后者是损失值，统一以np.ndarray 形式返回
    """
    model.eval()
    model = model.to(device)
    x = y = batch.to(device)
    preds = model(x) # (batch_size, seq_len, n_channels)

    loss_fn = loss_fn_resolve(loss_name)
    loss_elementwise = loss_fn(preds, y, reduction='none') # (batch_size,)
        # 这里的损失函数需要支持 reduction='none'，也即不合并，返回每个点位的损失
    losses = loss_elementwise.mean(dim=(1, 2))  # 将损失在通道维度上求平均，得到每个样本的损失值

    # 将 preds 和 loss 移动到 CPU 上
    if device == "cuda":
        preds = preds.cpu().numpy()
        losses = losses.cpu().numpy()
    elif device == "cpu":
        preds = preds.numpy()
        losses = losses.numpy()
    # logger.debug(f"Preds shape: {preds.shape}, Losses shape: {losses.shape}")
    return preds, losses