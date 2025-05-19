import torch
import torch.nn as nn

class Encoder(nn.Module):
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


class Decoder(nn.Module):
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
        self.encoder = Encoder(seq_len, n_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(seq_len, n_channels, latent_dim, hidden_dims[::-1])

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon



# 示例用法
if __name__ == "__main__":
    B, L, C = 16, 100, 3
    model = TimeSeriesConvAE(seq_len=L, n_channels=C, latent_dim=64, hidden_dims=[16, 32, 64])
    dummy = torch.randn(B, L, C)
    recon = model(dummy)
    print("输出维度:", recon.shape)  # 应为 (16, 100, 3)
