"""Synthetic forward pass for the architecture; no weather data are required."""

from __future__ import annotations

import torch

from model import MultiSourcePrecipitationModel, count_trainable_parameters


def main() -> None:
    # Small channels and spatial dimensions keep this example CPU-friendly while
    # preserving the same source-fusion and encoder/decoder structure.
    model = MultiSourcePrecipitationModel(
        input_frames=2,
        aux_sources=["satellite", "ifs"],
        output_frames=3,
        satellite_input_frames=2,
        satellite_input_channels=2,
        ifs_input_frames=3,
        ifs_input_channels=2,
        channels=(8, 8, 16, 32),
        maxvit_layers=1,
        partition_size=2,
        head_dim=8,
        resnet_depth=1,
        dropout=0.0,
        stochastic_depth_prob=0.0,
    ).eval()

    radar = torch.randn(1, 2, 1, 32, 32)
    satellite = torch.randn(1, 2, 2, 8, 8)
    ifs = torch.randn(1, 3, 2, 8, 8)

    with torch.no_grad():
        logits = model(radar, satellite=satellite, ifs=ifs)
        probabilities = torch.softmax(logits, dim=2)

    assert logits.shape == (1, 3, 8, 32, 32)
    assert torch.allclose(
        probabilities.sum(dim=2),
        torch.ones_like(probabilities[:, :, 0]),
        atol=1e-6,
    )

    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Output shape: {tuple(logits.shape)}")
    print("Synthetic forward pass: OK")


if __name__ == "__main__":
    main()
