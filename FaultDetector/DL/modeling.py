import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, seq_len, n_channels, latent_dim, hidden_dims):
        super().__init__()
        layers = []
        in_channels = n_channels
        for h in hidden_dims:
            layers.append(nn.Conv1d(in_channels, h, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(True))
            in_channels = h
        self.conv = nn.Sequential(*layers)

        # 计算下采样后的序列长度
        self.enc_out_len = seq_len
        for _ in hidden_dims:
            self.enc_out_len = (self.enc_out_len + 2 * 1 - 1 * (3 - 1) - 1) // 2 + 1  # Conv1d公式

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
        # rev_dims = list(reversed(hidden_dims))
        self.enc_out_len = seq_len
        for _ in hidden_dims[::-1]:
            self.enc_out_len = (self.enc_out_len + 2 * 1 - 1 * (3 - 1) - 1) // 2 + 1

        self.fc = nn.Linear(latent_dim, hidden_dims[0] * self.enc_out_len)

        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.ConvTranspose1d(hidden_dims[i], hidden_dims[i + 1],
                                              kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.ReLU(True))

        layers.append(nn.ConvTranspose1d(hidden_dims[-1], n_channels,
                                         kernel_size=3, stride=2, padding=1, output_padding=1))
        layers.append(nn.Sigmoid())

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
