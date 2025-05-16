# -*- coding: utf-8 -*-
from typing import Callable,Tuple
from loguru import logger
import numpy as np

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint


class GeneralLightningModule(L.LightningModule):
    def __init__(self, model, loss_fn, optimizer_class, lr, extra_args=None):
        """
        通用 LightningModule 封装

        参数:
            model: 任意 nn.Module 类型的模型
            loss_fn: 损失函数 (callable)
            optimizer_class: 优化器类，例如 torch.optim.AdamW
            lr: 学习率
            extra_args: 包含任意额外组件的 dict，例如 noise_scheduler、timesteps_sampler 等
        """
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class
        self.lr = lr
        self.extra_args = extra_args or {}

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        """
        自定义的训练逻辑，可根据需要扩展。
        对于 Diffusion，可在 extra_args 中传入 noise_scheduler 等模块。
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

        # 如有自定义输入处理（例如 add_noise），通过 extra_args 使用
        # if "preprocess_batch" in self.extra_args:
        #     x, y = self.extra_args["preprocess_batch"](x, y, self.device)

        preds = self.forward(x)
        loss = self.loss_fn(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer_class(self.model.parameters(), lr=self.lr)


def train_model(
    model,
    train_dataloader,
    loss_fn,
    optimizer_class=torch.optim.AdamW,
    lr=1e-3,
    max_epochs=20,
    device="cpu",
    log_dir="logs",
    ckpt_dir="./checkpoints",
    extra_args=None,
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
    lightning_model = GeneralLightningModule(
        model, loss_fn, optimizer_class, lr, extra_args
    )

    csvlogger = CSVLogger(log_dir, name="training_log")

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=ckpt_dir,
        filename="model-{epoch:02d}-{train_loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    # trainer = L.Trainer(
    #     max_epochs=max_epochs,
    #     accelerator=device,
    #     logger=csvlogger,
    #     callbacks=[checkpoint_callback],
    # )
    logger.debug(f"Training on {device} for {max_epochs} epochs.")
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=device,
        devices=1,
        # callbacks=[
        #     checkpoint_callback,
        # ],
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
    batch: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
    loss_fn: Callable,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    通用推理函数，返回模型对一组样本(也就是一个batch)的输出以及对应的loss。

    参数:
        model: 训练好的模型
        batch: 输入数据，可能是 (x, y) 或 x
        loss_fn: 损失函数
        device: 推理设备（"cpu" | "cuda"）

    返回:
        一个元组，前者是模型输出，后者是损失值，统一以np.ndarray 形式返回
    """
    model.eval()
    model = model.to(device)

    # batch 可能是 (x, y) 或 x
    if isinstance(batch, tuple):
        x, y = batch
    elif isinstance(batch, torch.Tensor):
        x = y = batch
    else:
        logger.error(f"Can't recognize batch type: {type(batch)}")
        raise ValueError("Batch format not recognized. Expected tuple (x, y) or torch.tensor x")
    
    x = x.to(device) # (batch_size, seq_len, n_channels)
    y = y.to(device)
    preds = model(x) # (batch_size, seq_len, n_channels)
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
    logger.debug(f"Preds shape: {preds.shape}, Losses shape: {losses.shape}")
    return preds, losses