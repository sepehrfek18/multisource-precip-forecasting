from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x.div(keep_prob) * mask


class ConvNormAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        activation: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True) if activation else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels)
        self.conv2 = ConvNormAct(out_channels, out_channels, activation=False)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.dropout = nn.Dropout2d(float(dropout))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.conv1(x))
        h = self.dropout(self.conv2(h))
        return self.activation(h + self.skip(x))


class ResBlocks(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        blocks = []
        for idx in range(int(depth)):
            blocks.append(
                ResBlock(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1),
        )
        self.blocks = ResBlocks(out_channels, out_channels, depth=depth, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.blocks = ResBlocks(
            out_channels + skip_channels,
            out_channels,
            depth=depth,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self.blocks(torch.cat([x, skip], dim=1))


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, ratio: float = 0.25) -> None:
        super().__init__()
        hidden = max(1, int(channels * ratio))
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


def _rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def _get_relative_position_index(height: int, width: int) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(height),
            torch.arange(width),
            indexing="ij",
        )
    )
    coords_flat = torch.flatten(coords, 1)
    relative_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += height - 1
    relative_coords[:, :, 1] += width - 1
    relative_coords[:, :, 0] *= 2 * width - 1
    return relative_coords.sum(-1)


class RelativePositionalMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        head_dim: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        if channels % head_dim != 0:
            raise ValueError(
                f"channels={channels} must be divisible by head_dim={head_dim}"
            )

        size = int(math.sqrt(max_seq_len))
        if size * size != max_seq_len:
            raise ValueError("max_seq_len must be a square number.")

        self.channels = int(channels)
        self.head_dim = int(head_dim)
        self.num_heads = self.channels // self.head_dim
        self.max_seq_len = int(max_seq_len)
        self.size = size
        self.scale_factor = self.channels**-0.5

        self.to_qkv = nn.Linear(
            self.channels,
            self.num_heads * self.head_dim * 3,
            bias=False,
        )
        self.merge = nn.Linear(self.channels, self.channels, bias=False)
        self.relative_position_bias_table = nn.Parameter(
            torch.empty(
                (2 * self.size - 1) * (2 * self.size - 1),
                self.num_heads,
                dtype=torch.float32,
            )
        )
        self.register_buffer(
            "relative_position_index",
            _get_relative_position_index(self.size, self.size),
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def get_relative_positional_bias(self) -> torch.Tensor:
        bias_index = self.relative_position_index.reshape(-1)
        relative_bias = self.relative_position_bias_table[bias_index].reshape(
            self.max_seq_len,
            self.max_seq_len,
            self.num_heads,
        )
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        return relative_bias.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, p, c = x.shape
        if p != self.max_seq_len:
            raise ValueError(
                f"Expected token length {self.max_seq_len}, got {p}."
            )
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}.")

        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = _rms_norm(q)
        k = _rms_norm(k)

        q = q.reshape(n, p, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(n, p, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(n, p, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        k = k * self.scale_factor
        dot_prod = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        dot_prod = F.softmax(dot_prod + self.get_relative_positional_bias(), dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", dot_prod, v)
        out = out.permute(0, 2, 1, 3).reshape(n, p, c)
        return self.merge(out)


class MBConv(nn.Module):
    def __init__(
        self,
        channels: int,
        expansion_ratio: float = 4.0,
        squeeze_ratio: float = 0.25,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(channels * expansion_ratio)
        self.norm = nn.GroupNorm(1, channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
                groups=hidden,
                bias=False,
            ),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            SqueezeExcite(hidden, ratio=squeeze_ratio),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.net(self.norm(x)))


class PartitionAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        partition_size: int = 8,
        partition_type: str = "window",
        head_dim: int = 32,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if partition_type not in {"window", "grid"}:
            raise ValueError("partition_type must be 'window' or 'grid'")
        if channels % head_dim != 0:
            raise ValueError(
                f"channels={channels} must be divisible by head_dim={head_dim}"
            )

        self.channels = int(channels)
        self.partition_size = int(partition_size)
        self.partition_type = partition_type
        self.norm = nn.GroupNorm(1, channels)
        self.attn = RelativePositionalMultiHeadAttention(
            channels=channels,
            head_dim=head_dim,
            max_seq_len=self.partition_size * self.partition_size,
        )
        self.attn_dropout = nn.Dropout(float(dropout))
        self.drop_path = DropPath(drop_path)

    def _partition_window(
        self,
        x: torch.Tensor,
        token_size: int,
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        b, c, h, w = x.shape
        p = int(token_size)
        if h % p != 0 or w % p != 0:
            raise ValueError(
                f"Feature map {(h, w)} must be divisible by partition size {p}."
            )

        gh, gw = h // p, w // p
        x = x.reshape(b, c, gh, p, gw, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(b * gh * gw, p * p, c)
        return x, (b, gh, gw, p)

    def _departition_window(
        self,
        x: torch.Tensor,
        meta: tuple[int, int, int, int],
    ) -> torch.Tensor:
        b, gh, gw, p = meta
        c = self.channels
        x = x.reshape(b, gh, gw, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, gh * p, gw * p)
        return x

    def _partition_grid(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        b, c, h, w = x.shape
        grid_groups = self.partition_size
        if h % grid_groups != 0 or w % grid_groups != 0:
            raise ValueError(
                f"Feature map {(h, w)} must be divisible by grid groups {grid_groups}."
            )

        p_h, p_w = h // grid_groups, w // grid_groups
        x = x.reshape(b, c, grid_groups, p_h, grid_groups, p_w)
        x = x.permute(0, 3, 5, 2, 4, 1).reshape(
            b * p_h * p_w,
            grid_groups * grid_groups,
            c,
        )
        return x, (b, grid_groups, p_h, p_w)

    def _departition_grid(
        self,
        x: torch.Tensor,
        meta: tuple[int, int, int, int],
    ) -> torch.Tensor:
        b, grid_groups, p_h, p_w = meta
        c = self.channels
        x = x.reshape(b, p_h, p_w, grid_groups, grid_groups, c)
        x = x.permute(0, 5, 3, 1, 4, 2).reshape(
            b,
            c,
            grid_groups * p_h,
            grid_groups * p_w,
        )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        if self.partition_type == "window":
            tokens, meta = self._partition_window(x, self.partition_size)
            out = self.attn_dropout(self.attn(tokens))
            x = self._departition_window(out, meta)
        else:
            tokens, meta = self._partition_grid(x)
            out = self.attn_dropout(self.attn(tokens))
            x = self._departition_grid(out, meta)

        return residual + self.drop_path(x)


class MaxViTLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        partition_size: int = 8,
        head_dim: int = 32,
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.mbconv = MBConv(channels, drop_path=drop_path)
        self.window_attention = PartitionAttention(
            channels=channels,
            partition_size=partition_size,
            partition_type="window",
            head_dim=head_dim,
            dropout=dropout,
            drop_path=drop_path,
        )
        self.grid_attention = PartitionAttention(
            channels=channels,
            partition_size=partition_size,
            partition_type="grid",
            head_dim=head_dim,
            dropout=dropout,
            drop_path=drop_path,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mbconv(x)
        x = self.window_attention(x)
        x = self.grid_attention(x)
        return x


class MaxViTBlocks(nn.Module):
    def __init__(
        self,
        channels: int,
        depth: int,
        partition_size: int = 8,
        head_dim: int = 32,
        dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
    ) -> None:
        super().__init__()
        depth = int(depth)
        if depth <= 0:
            raise ValueError("MaxViT depth must be positive.")

        drop_rates = torch.linspace(0.0, float(stochastic_depth_prob), depth).tolist()
        self.layers = nn.ModuleList(
            [
                MaxViTLayer(
                    channels=channels,
                    partition_size=partition_size,
                    head_dim=head_dim,
                    dropout=dropout,
                    drop_path=float(drop_rates[idx]),
                )
                for idx in range(depth)
            ]
        )
        self.final_conv = nn.Conv2d(channels * depth, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
        return self.final_conv(torch.cat(outputs, dim=1))
