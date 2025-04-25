# Adapted from : https://github.com/dome272/Diffusion-Models-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(B, C, H, W)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256,  up_to_14=False):
        super().__init__()

        target_size = (14, 14) if up_to_14 else (7, 7)
        self.up = nn.Upsample(size=target_size, mode="bilinear", align_corners=False)

        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)

        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb



class UNet(nn.Module):
    def __init__(self, img_size=14, c_in=13, c_out=13, time_dim=256, device="cpu", channels=32):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, channels)

        self.down1 = Down(channels, channels * 2, emb_dim=time_dim)     # 14 → 7
        self.down2 = Down(channels * 2, channels * 4, emb_dim=time_dim) # 7 → 3

        self.bot1 = DoubleConv(channels * 4, channels * 4)
        self.bot2 = DoubleConv(channels * 4, channels * 4)

        self.up1 = Up(in_channels=channels * 4 + channels * 2, out_channels=channels * 2, emb_dim=time_dim, up_to_14=False)  # 3 → 7
        self.up2 = Up(in_channels=channels * 2 + channels, out_channels=channels, emb_dim=time_dim, up_to_14=True)          # 7 → 14

        self.attn = SelfAttention(channels, size=14)

        self.outc = nn.Conv2d(channels, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)           # [B, 32, 14, 14]
        x2 = self.down1(x1, t)     # [B, 64, 7, 7]
        x3 = self.down2(x2, t)     # [B, 128, 3, 3]

        x = self.bot1(x3)          # [B, 128, 3, 3]
        x = self.bot2(x)           # [B, 128, 3, 3]

        x = self.up1(x, x2, t)     # [B, 64, 7, 7]
        x = self.up2(x, x1, t)     # [B, 32, 14, 14]

        x = self.attn(x)

        return self.outc(x)        # [B, 13, 14, 14]
