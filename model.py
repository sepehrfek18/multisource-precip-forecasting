from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import (
    ConvNormAct,
    DownBlock,
    MaxViTBlocks,
    ResBlocks,
    UpBlock,
)


# Canonical fusion order used by every source combination.
AUX_SOURCE_ORDER = ("satellite", "ifs")


def _normalize_aux_sources(aux_sources: Sequence[str] | None) -> list[str]:
    if aux_sources is None:
        return []
    requested = [str(s).strip().lower() for s in aux_sources]
    unknown = [s for s in requested if s not in AUX_SOURCE_ORDER]
    if unknown:
        raise ValueError(
            f"Unknown aux source(s): {unknown}. Supported: {list(AUX_SOURCE_ORDER)}."
        )
    # Canonical order, de-duplicated.
    return [s for s in AUX_SOURCE_ORDER if s in requested]


class MultiSourcePrecipitationModel(nn.Module):
    """
    Source-configurable probabilistic precipitation forecasting architecture.

    Radar is encoded from the target grid to quarter resolution. Each enabled
    auxiliary source is embedded independently and concatenated with the radar
    representation before the MaxViT bottleneck. The decoder uses radar and
    fusion skip connections to return to the target grid.

        ["satellite", "ifs"]  -> radar + SEVIRI + IFS
        ["satellite"]         -> radar + SEVIRI
        ["ifs"]               -> radar + IFS
        []                    -> radar only

    Input tensors:
      x:         [B, T_in, C_in, H, W]            radar (e.g. 256x256)
      satellite: [B, T_sat, C_sat, H/4, W/4]      SEVIRI (optional)
      ifs:       [B, T_ifs, C_ifs, H/4, W/4]      IFS HRES (optional)

    Output:
      logits:    [B, T_out, K, H, W]              categorical precipitation bins
    """

    def __init__(
        self,
        input_frames: int,
        aux_sources: Sequence[str] | None = None,
        input_channels: int = 1,
        output_frames: int = 12,
        output_channels: int = 8,
        satellite_input_frames: int = 12,
        satellite_input_channels: int = 8,
        ifs_input_frames: int = 12,
        ifs_input_channels: int = 10,
        channels: Sequence[int] = (64, 64, 128, 256),
        maxvit_layers: int = 8,
        partition_size: int = 8,
        head_dim: int = 32,
        resnet_depth: int = 2,
        dropout: float = 0.1,
        stochastic_depth_prob: float = 0.1,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError(
                "channels must contain four values: [c1_256, c2_128, c3_64, c4_32]."
            )
        c1, c2, c3, c4 = [int(v) for v in channels]

        self.aux_sources = _normalize_aux_sources(aux_sources)
        self.use_satellite = "satellite" in self.aux_sources
        self.use_ifs = "ifs" in self.aux_sources

        self.input_frames = int(input_frames)
        self.input_channels = int(input_channels)
        self.output_frames = int(output_frames)
        self.output_channels = int(output_channels)
        self.satellite_input_frames = int(satellite_input_frames)
        self.satellite_input_channels = int(satellite_input_channels)
        self.ifs_input_frames = int(ifs_input_frames)
        self.ifs_input_channels = int(ifs_input_channels)

        radar_in = self.input_frames * self.input_channels

        # ------------------------------------------------------------------
        # Per-source 1x1 embeddings (only the active aux sources are built).
        # ------------------------------------------------------------------
        self.embed_radar = ConvNormAct(radar_in, c1, kernel_size=1)
        if self.use_satellite:
            sat_in = self.satellite_input_frames * self.satellite_input_channels
            self.embed_satellite = ConvNormAct(sat_in, c3, kernel_size=1)
        if self.use_ifs:
            ifs_in = self.ifs_input_frames * self.ifs_input_channels
            self.embed_ifs = ConvNormAct(ifs_in, c3, kernel_size=1)

        n_aux = len(self.aux_sources)
        fusion_channels = c3 * (1 + n_aux)

        # ------------------------------------------------------------------
        # Encoder.
        # ------------------------------------------------------------------
        self.down_256_to_128 = DownBlock(c1, c2, depth=resnet_depth, dropout=dropout)
        self.down_128_to_64 = DownBlock(c2, c3, depth=resnet_depth, dropout=dropout)
        self.down_64_to_32 = DownBlock(
            fusion_channels, c4, depth=resnet_depth, dropout=dropout
        )

        # ------------------------------------------------------------------
        # Bottleneck: MaxViT at 32x32.
        # ------------------------------------------------------------------
        self.maxvit = MaxViTBlocks(
            channels=c4,
            depth=maxvit_layers,
            partition_size=partition_size,
            head_dim=head_dim,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
        )

        # ------------------------------------------------------------------
        # Decoder.
        # ------------------------------------------------------------------
        self.up_32_to_64 = UpBlock(
            in_channels=c4,
            skip_channels=fusion_channels,
            out_channels=c3,
            depth=resnet_depth,
            dropout=dropout,
        )
        self.up_64_to_128 = UpBlock(
            in_channels=c3,
            skip_channels=c2,
            out_channels=c2,
            depth=resnet_depth,
            dropout=dropout,
        )
        self.up_128_to_256_upsample = nn.ConvTranspose2d(
            c2, c1, kernel_size=2, stride=2,
        )
        self.up_128_to_256_blocks = ResBlocks(
            c1 + c1, c2, depth=resnet_depth, dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Output head.
        # ------------------------------------------------------------------
        self.to_logits = nn.Conv2d(
            c2, self.output_frames * self.output_channels, kernel_size=1,
        )

        # Public flags describing which optional inputs the model requires.
        self.supports_aux_inputs = n_aux > 0
        self.required_aux_input_keys = tuple(self.aux_sources)

    def _check_aux(
        self,
        name: str,
        tensor: torch.Tensor | None,
        exp_frames: int,
        exp_channels: int,
        b: int,
        expected_hw: tuple[int, int],
    ) -> None:
        if tensor is None:
            raise ValueError(
                f"Model is configured to use '{name}' but no {name} tensor was provided."
            )
        if tensor.ndim != 5:
            raise ValueError(
                f"Expected {name} input [B, T, C, H, W], got {tuple(tensor.shape)}"
            )
        bb, t, c, h, w = tensor.shape
        if bb != b:
            raise ValueError(f"{name} batch {bb} != radar batch {b}.")
        if t != exp_frames:
            raise ValueError(f"Expected {exp_frames} {name} frames, got {t}.")
        if c != exp_channels:
            raise ValueError(f"Expected {exp_channels} {name} channels, got {c}.")
        if (h, w) != expected_hw:
            raise ValueError(
                f"Expected {name} spatial shape {expected_hw}, got {(h, w)}."
            )

    def forward(
        self,
        x: torch.Tensor,
        satellite: torch.Tensor | None = None,
        ifs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected radar input [B, T, C, H, W], got {tuple(x.shape)}"
            )
        b, t, c, h, w = x.shape
        if t != self.input_frames:
            raise ValueError(f"Expected {self.input_frames} radar frames, got {t}.")
        if c != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} radar channels, got {c}.")
        if h % 8 != 0 or w % 8 != 0:
            raise ValueError("Radar H and W must be divisible by 8.")

        expected_aux_hw = (h // 4, w // 4)

        # Radar encoder.
        x_radar = x.reshape(b, t * c, h, w)
        radar_emb = self.embed_radar(x_radar)       # [B, c1, 256, 256]
        skip_256 = radar_emb
        d1 = self.down_256_to_128(radar_emb)        # [B, c2, 128, 128]
        skip_128 = d1
        radar_64 = self.down_128_to_64(d1)          # [B, c3, 64, 64]

        # Fusion list in canonical order: radar, satellite, ifs.
        fusion = [radar_64]
        if self.use_satellite:
            self._check_aux(
                "satellite", satellite, self.satellite_input_frames,
                self.satellite_input_channels, b, expected_aux_hw,
            )
            sat = satellite.reshape(b, -1, satellite.shape[-2], satellite.shape[-1])
            fusion.append(self.embed_satellite(sat))
        if self.use_ifs:
            self._check_aux(
                "ifs", ifs, self.ifs_input_frames,
                self.ifs_input_channels, b, expected_aux_hw,
            )
            ifs_r = ifs.reshape(b, -1, ifs.shape[-2], ifs.shape[-1])
            fusion.append(self.embed_ifs(ifs_r))

        # When radar-only, there is nothing to concatenate.
        x64 = fusion[0] if len(fusion) == 1 else torch.cat(fusion, dim=1)
        skip_64 = x64

        x = self.down_64_to_32(x64)                 # [B, c4, 32, 32]
        x = self.maxvit(x)                          # [B, c4, 32, 32]
        x = self.up_32_to_64(x, skip_64)            # [B, c3, 64, 64]
        x = self.up_64_to_128(x, skip_128)          # [B, c2, 128, 128]

        x = self.up_128_to_256_upsample(x)          # [B, c1, 256, 256]
        if x.shape[-2:] != skip_256.shape[-2:]:
            x = F.interpolate(
                x, size=skip_256.shape[-2:], mode="bilinear", align_corners=False,
            )
        x = torch.cat([x, skip_256], dim=1)         # [B, 2*c1, 256, 256]
        x = self.up_128_to_256_blocks(x)            # [B, c2, 256, 256]

        logits = self.to_logits(x)                  # [B, T_out*K, 256, 256]
        return logits.reshape(
            b, self.output_frames, self.output_channels, h, w,
        )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
