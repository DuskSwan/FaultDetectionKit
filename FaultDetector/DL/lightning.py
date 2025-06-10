# -*- coding: utf-8 -*-
from typing import Callable,Tuple
from loguru import logger
import numpy as np

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import Callback

class EpochEndCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        loss = trainer.callback_metrics["train_loss"].item()
        print(f"[{epoch}/{trainer.max_epochs}] train_loss: {loss:.4f}")

class GeneralLightningModule(L.LightningModule):
    def __init__(self, model, loss_fn, optimizer_class, lr):
        """
        通用 LightningModule 封装

        参数:
            model: 任意 nn.Module 类型的模型
            loss_fn: 损失函数 (callable)
            optimizer_class: 优化器类，例如 torch.optim.AdamW
            lr: 学习率
        """
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.lr = lr

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        """
        自定义的训练逻辑，可根据需要扩展。
        """
        # batch 可能是 (x, y) 或 (x,)
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 1:
            x = y = batch[0]
        else:
            logger.error(f"a batch is in shape {batch.shape}")
            raise ValueError("Batch format not recognized. Expected (x, y) or (x,)")

        x = x.to(self.device)
        y = y.to(self.device)

        preds = self.forward(x)
        loss = self.loss_fn(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), lr=self.lr)


def train_model(
    model,
    train_dataloader,
    loss_name: str,
    optimizer_name: str,
    lr=1e-3,
    max_epochs=20,
    device="cpu",
    # log_dir="logs",
    # ckpt_dir="./checkpoints",
):
    """
    通用训练入口。

    参数:
        model: 你的模型（nn.Module）
        train_dataloader: 训练数据加载器
        loss_fn: 损失函数，例如 F.mse_loss、nn.CrossEntropyLoss()
        optimizer_class: 优化器类，默认 AdamW
        lr: 学习率
        max_epochs: 最大训练轮次
        device: "cuda", "cpu" 或 "auto"
        log_dir: 日志文件保存目录
        ckpt_dir: 模型检查点保存目录
        extra_args: 自定义 batch 处理函数等额外组件，例如:
                    {"preprocess_batch": fn}，或者用于 Diffusion 的组件
    """
    optimizer_class = optimizer_resolve(optimizer_name)
    loss_fn = loss_fn_resolve(loss_name)
    lightning_model = GeneralLightningModule(
        model, loss_fn, optimizer_class, lr
    )

    # csvlogger = CSVLogger(log_dir, name="training_log")

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="train_loss",
    #     dirpath=ckpt_dir,
    #     filename="model-{epoch:02d}-{train_loss:.4f}",
    #     save_top_k=1,
    #     mode="min",
    # )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=device,
        devices=1,
        callbacks=[
            EpochEndCallback(),
        ],
        # log_every_n_steps=1,
        enable_checkpointing=False,  # 关闭默认检查点保存
        logger=False,  # 关闭默认日志
        enable_progress_bar=False, # 关闭默认进度条
    )

    logger.debug(f"Training on {device} for {max_epochs} epochs with {len(train_dataloader)} batches.")
    trainer.fit(lightning_model, train_dataloader)
    return model

@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: str = "cpu",
) -> np.ndarray:
    """
    通用推理函数，返回模型对一组样本(也就是一个batch)的输出以及对应的loss。

    参数:
        model: 训练好的模型
        batch: 一组输入数据，形状为 (batch_size, seq_len, n_channels) 
        loss_name: 损失函数名称，例如 "mse" 或 "cross_entropy"
        device: 推理设备（"cpu" | "cuda"）

    返回:
        一个元组，前者是模型输出，后者是损失值，统一以np.ndarray 形式返回
    """
    model.eval()
    model = model.to(device)
    x = batch.to(device)
    preds = model(x) # (batch_size, seq_len, n_channels)

    # 将 preds 和 loss 移动到 CPU 上
    if device == "cuda":
        preds = preds.cpu().numpy()
    elif device == "cpu":
        preds = preds.numpy()
    # logger.debug(f"Preds shape: {preds.shape}, Losses shape: {losses.shape}")
    return preds

def optimizer_resolve(optimizer_class: str) -> Callable:
    """
    解析优化器类并返回实例化的优化器。

    参数:
        optimizer_class: 优化器类名称，例如 "AdamW" 或 "SGD"

    返回:
        实例化的优化器
    """
    if optimizer_class == "adamw":
        return torch.optim.AdamW
    if optimizer_class == "sgd":
        return torch.optim.SGD
    if optimizer_class == "adam":
        return torch.optim.Adam
    
    raise ValueError(f"Unsupported optimizer class: {optimizer_class}. Supported classes are 'adamw', 'sgd', 'adam'.")

def loss_fn_resolve(loss_fn: str) -> Callable:
    """
    解析损失函数名称并返回对应的函数。

    参数:
        loss_fn: 损失函数名称，例如 "mse" 或 "cross_entropy"

    返回:
        对应的损失函数
    """
    valid_loss_fn = {
        "mse": F.mse_loss,
        "mae": F.l1_loss,
        "cross_entropy": F.cross_entropy,
    }
    assert loss_fn in valid_loss_fn, f"Unsupported loss function: {loss_fn}. Supported functions are {list(valid_loss_fn.keys())}."
    return valid_loss_fn[loss_fn]