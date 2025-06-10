import torch
import torch.nn as nn

class AEEncoder(nn.Module):
    def __init__(self, seq_len, n_channels, latent_dim, hidden_dims):
        super().__init__()
        layers = []
        in_channels = n_channels
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.enc_out_len = seq_len
        for h in hidden_dims:
            layers.append(nn.Conv1d(in_channels, h, kernel_size=3, stride=1, padding=1))  # stride=1
            layers.append(nn.ReLU(True))
            layers.append(nn.MaxPool1d(kernel_size=2))  # 添加池化层
            in_channels = h
            self.enc_out_len = self.enc_out_len // 2  # 每次池化长度减半

        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dims[-1] * self.enc_out_len, latent_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        x = self.conv(x)
        x = x.flatten(1)
        latent = self.fc(x)
        return latent

class AEDecoder(nn.Module):
    def __init__(self, seq_len, n_channels, latent_dim, hidden_dims):
        super().__init__()

        self.enc_out_len = seq_len
        for _ in hidden_dims[::-1]:
            self.enc_out_len = self.enc_out_len // 2

        self.fc = nn.Linear(latent_dim, hidden_dims[0] * self.enc_out_len)

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))  # 对应 MaxPool
            layers.append(nn.Conv1d(hidden_dims[i], hidden_dims[i + 1],
                                    kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))

        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.append(nn.Conv1d(hidden_dims[-1], n_channels,
                                kernel_size=3, stride=1, padding=1))
        # layers.append(nn.Sigmoid()) # Sigmoid 会将输出限制在 [0, 1] 之间，可能不适合所有数据

        self.deconv = nn.Sequential(*layers)

    def forward(self, latent):
        x = self.fc(latent)
        x = x.view(x.size(0), -1, self.enc_out_len)
        x = self.deconv(x)
        x = x.permute(0, 2, 1)  # (B, C, L) -> (B, L, C)
        return x

class TimeSeriesConvAE(nn.Module):
    def __init__(self, seq_len, n_channels, latent_dim=64, hidden_dims=[16, 32]):
        super().__init__()
        self.encoder = AEEncoder(seq_len, n_channels, latent_dim, hidden_dims)
        self.decoder = AEDecoder(seq_len, n_channels, latent_dim, hidden_dims[::-1])

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


class LSTMClassifier(nn.Module):
    def __init__(self, 
        input_dim: int, 
        hidden_dim: int, 
        num_layers: int, 
        output_dim: int, 
        hidden_sizes: list[int], 
        dropout_rate: float = 0.0
    ):
        """
        初始化信号分类网络。

        Args:
            input_dim (int): 输入信号的特征维度。
            hidden_dim (int): LSTM 层的隐藏状态维度。
            num_layers (int): LSTM 层的层数。
            output_dim (int): 分类器的输出类别数量。
            hidden_sizes (list[int]): 线性层的隐藏维度列表。
                                       例如：[128, 64] 表示两个线性层，第一个输出128维，第二个输出64维。
            dropout_rate (float): LSTM 层和线性层之间的 Dropout 比率，默认为 0.0。
        """
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        
        linear_layers_input_dim = hidden_dim # 第一个线性层的输入维度是 LSTM 的 hidden_dim
        self.linear_layers = nn.ModuleList()
        for i, h_size in enumerate(hidden_sizes):
            self.linear_layers.append(nn.Linear(linear_layers_input_dim, h_size))
            self.linear_layers.append(nn.ReLU()) # 可以根据需要更换激活函数，例如 nn.LeakyReLU(), nn.Sigmoid()
            linear_layers_input_dim = h_size # 更新下一个线性层的输入维度
        self.linear_layers.pop(-1)  # 移除最后一个 ReLU 激活层

        # 如果 hidden_sizes 为空，则直接从 LSTM 输出到 output_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim) if not hidden_sizes else nn.Linear(hidden_sizes[-1], output_dim)

    def forward(self, x):
        """
        定义前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状通常为 (batch_size, sequence_length, input_dim)。

        Returns:
            torch.Tensor: 分类器的输出，形状为 (batch_size, output_dim)。
        """
        out, (hn, cn) = self.lstm(x) #, (h0, c0)
        features = hn[-1] # hn[-1] 是最后一个 LSTM 层的隐藏状态，形状为 (batch_size, hidden_dim)
        for layer in self.linear_layers:
            features = layer(features)
        output = self.output_layer(features)
        return output

# 示例用法
if __name__ == "__main__":
    B, L, C = 16, 100, 3
    model = TimeSeriesConvAE(seq_len=L, n_channels=C, latent_dim=64, hidden_dims=[16, 32, 64])
    dummy = torch.randn(B, L, C)
    recon = model(dummy)
    print("输出维度:", recon.shape)  # 应为 (16, 100, 3)
