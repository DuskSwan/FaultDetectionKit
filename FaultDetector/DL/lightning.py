from loguru import logger

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

    logger = CSVLogger(log_dir, name="training_log")

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
    #     logger=logger,
    #     callbacks=[checkpoint_callback],
    # )
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

    trainer.fit(lightning_model, train_dataloader)
    return model

@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
) -> list[torch.Tensor]:
    """
    通用推理函数，返回模型对 dataloader 中所有样本的输出。

    参数:
        model: 训练好的模型
        dataloader: 推理用 DataLoader
        device: 推理设备（"cpu" | "cuda" | "auto"）

    返回:
        一个列表，每个元素是一个 batch 的预测结果 Tensor
    """
    model.eval()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    outputs = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)
        preds = model(x)
        outputs.append(preds.cpu())  # 返回 cpu 上的 tensor，方便后续处理
    

    return outputs