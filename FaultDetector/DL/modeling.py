import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, seq_len, n_channels, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=3, stride=2, padding=1),  # -> (B,16, L/2)
            nn.ReLU(True),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),          # -> (B,32, L/4)
            nn.ReLU(True)
        )
        # 计算 flatten 后的长度
        self.enc_out_len = seq_len // 4 if seq_len % 4 == 0 else (seq_len // 4 + 1)
        self.fc = nn.Linear(32 * self.enc_out_len, latent_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)         # (B, L, C) -> (B, C, L)
        x = self.conv(x)              # (B, C', L')
        x = x.flatten(1)              # 展平
        latent = self.fc(x)           # (B, latent_dim)
        return latent


class Decoder(nn.Module):
    def __init__(self, seq_len, n_channels, latent_dim):
        super().__init__()
        self.enc_out_len = seq_len // 4 if seq_len % 4 == 0 else (seq_len // 4 + 1)
        self.fc = nn.Linear(latent_dim, 32 * self.enc_out_len)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, n_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, latent):
        x = self.fc(latent)           # (B, 32 * L/4)
        x = x.view(x.size(0), 32, -1) # (B, 32, L/4)
        x = self.deconv(x)           # (B, C, L)
        x = x.permute(0, 2, 1)       # (B, L, C)
        return x


class TimeSeriesConvAE(nn.Module):
    def __init__(self, seq_len, n_channels, latent_dim=64):
        super().__init__()
        self.encoder = Encoder(seq_len, n_channels, latent_dim)
        self.decoder = Decoder(seq_len, n_channels, latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon


# 示例用法
if __name__ == "__main__":
    B, L, C = 16, 100, 3
    model = TimeSeriesConvAE(seq_len=L, n_channels=C, latent_dim=64)
    dummy = torch.randn(B, L, C)
    recon = model(dummy)
    print("输出维度:", recon.shape)  # 应为 (16, 100, 3)
