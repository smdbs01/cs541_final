import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_mean(x, axis=(0,), keepdims=False):
    mask = (x != 0).float()

    numerator = torch.sum(x * mask, dim=axis, keepdim=keepdims)
    denominator = torch.sum(mask, dim=axis, keepdim=keepdims)

    return numerator / torch.clamp(denominator, min=1e-6)


def zero_std(x, center=None, axis=(0,), keepdims=False):
    if center is None:
        center = zero_mean(x, axis=axis, keepdims=True)
    d = x - center
    return torch.sqrt(zero_mean(d * d, axis=axis, keepdims=keepdims))


class PosePreprocess(nn.Module):
    def __init__(self, point_landmarks):
        super().__init__()
        self.point_landmarks = point_landmarks

    def forward(self, inputs):
        if inputs.dim() == 3:
            x = inputs.unsqueeze(0)
        else:
            x = inputs

        mean_feature = x[:, :, [17], :]  # nose
        mean = zero_mean(mean_feature, axis=(1, 2), keepdims=True)
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)

        # feature points
        x = x[:, :, self.point_landmarks, :]
        std = zero_std(x, center=mean, axis=(1, 2), keepdims=True)
        x = (x - mean) / torch.clamp(std, min=1e-6)

        seq_len = x.shape[1]

        if seq_len > 1:
            dx = torch.cat([x[:, 1:] - x[:, :-1], torch.zeros_like(x[:, -1:])], dim=1)
        else:
            dx = torch.zeros_like(x)

        if seq_len > 2:
            dx2 = torch.cat([x[:, 2:] - x[:, :-2], torch.zeros_like(x[:, -2:])], dim=1)
        else:
            dx2 = torch.zeros_like(x)

        # combine features [B,T,3*2*N_landmarks]
        x_flat = x.view(x.size(0), seq_len, -1)
        dx_flat = dx.view(dx.size(0), seq_len, -1)
        dx2_flat = dx2.view(dx2.size(0), seq_len, -1)
        x_out = torch.cat([x_flat, dx_flat, dx2_flat], dim=-1)

        return x_out


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class ECA(nn.Module):
    def __init__(self, num_channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )

    def forward(self, x, mask=None):
        # Inputs: [B, T, C]
        nn = x.mean(dim=1, keepdim=True)  # Avg Pooling [B, 1, C]
        nn = self.conv(nn)  # [B, C, 1]
        nn = torch.sigmoid(nn)  # [B, C, 1]
        return x * nn


class LateDropout(nn.Module):
    def __init__(self, rate, start_step=0):
        super().__init__()
        self.rate = rate
        self.start_step = start_step
        self.register_buffer("_train_counter", torch.tensor(0))

    def forward(self, x):
        if self.training:
            if self._train_counter >= self.start_step:
                return F.dropout(x, p=self.rate, training=True)
            self._train_counter += 1
        return x


class CausalDWConv1D(nn.Module):
    def __init__(self, channels, kernel_size=17, dilation_rate=1):
        super().__init__()
        self.padding = (dilation_rate * (kernel_size - 1), 0)
        self.dw_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=dilation_rate,
            groups=channels,
            bias=False,
        )

    def forward(self, x):
        # [B, C, T]
        x = F.pad(x, self.padding)
        return self.dw_conv(x)


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        channel_size,
        kernel_size,
        dilation_rate=1,
        drop_rate=0.0,
        expand_ratio=2,
        se_ratio=0.25,
    ):
        super().__init__()
        self.expand_conv = nn.Linear(channel_size, channel_size * expand_ratio)
        self.act = nn.SiLU()
        self.dw_conv = CausalDWConv1D(
            channel_size * expand_ratio, kernel_size, dilation_rate
        )
        self.bn = nn.BatchNorm1d(channel_size * expand_ratio, momentum=0.05)
        self.eca = ECA(channel_size * expand_ratio)
        self.project_conv = nn.Linear(channel_size * expand_ratio, channel_size)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        # [B, T, C]
        residual = x
        x = self.act(self.expand_conv(x))  # [B, T, C*expand]

        x = x.permute(0, 2, 1)  # [B, C*expand, T]
        x = self.dw_conv(x)
        x = x.permute(0, 2, 1)  # [B, T, C*expand]

        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.eca(x)
        x = self.project_conv(x)
        x = self.dropout(x)

        if x.shape[-1] == residual.shape[-1]:
            x += residual
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim=256, num_heads=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim, momentum=0.05)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_dropout)
        self.dropout1 = nn.Dropout(drop_rate)
        self.norm2 = nn.BatchNorm1d(dim, momentum=0.05)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * expand),
            nn.SiLU(),
            nn.Linear(dim * expand, dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x):
        # [B, T, C]
        x_norm = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        attn_out = self.attn(x_norm)
        attn_out = self.dropout1(attn_out)
        x = x + attn_out

        x_norm = self.norm2(x.permute(0, 2, 1)).permute(0, 2, 1)
        mlp_out = self.mlp(x_norm)
        return x + mlp_out


class GestureModel(nn.Module):
    def __init__(
        self, max_len=50, in_channels=708, dim=192, num_classes=2000, dropout_step=0
    ):
        super().__init__()
        self.dim = dim
        self.stem = nn.Sequential(
            nn.Linear(in_channels, dim, bias=False),
            Permute(0, 2, 1),
            nn.BatchNorm1d(dim, momentum=0.05),
            Permute(0, 2, 1),
        )

        self.blocks = nn.Sequential(
            *[Conv1DBlock(dim, 17) for _ in range(3)],
            TransformerBlock(dim, expand=2),
            *[Conv1DBlock(dim, 17) for _ in range(3)],
            TransformerBlock(dim, expand=2),
        )

        self.top = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.LayerNorm(2 * dim),
            LateDropout(0.8, start_step=dropout_step),
            nn.Linear(2 * dim, num_classes),
        )

    def forward(self, x):
        # [B, T, C]
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean(dim=1)
        return self.top(x)
