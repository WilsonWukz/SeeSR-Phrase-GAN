# prompt_gan_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x)       # [B, 32, H, W]
        x2 = self.enc2(x1)      # [B, 64, H/2, W/2]
        return x1, x2


class UNetDecoder(nn.Module):
    def __init__(self, base_channels=64, out_channels=91):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        u1 = self.up1(x2)
        x = self.dec1(torch.cat([u1, x1], dim=1))
        out = self.out_conv(x)
        return out


class PromptGenerator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=91):
        super().__init__()
        self.encoder = UNetEncoder(in_channels=3, base_channels=32)
        self.decoder = UNetDecoder(base_channels=32, out_channels=output_dim)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x1, x2 = self.encoder(x)
        logits_map = self.decoder(x1, x2)
        pooled = self.pool(logits_map).squeeze(-1).squeeze(-1)
        return pooled

# ====================== Discriminator ======================
class PromptDiscriminator(nn.Module):
    def __init__(self, input_dim=91):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)