from __future__ import annotations

import unittest

import torch

from model import MultiSourcePrecipitationModel


class ArchitectureTest(unittest.TestCase):
    def build_model(self, aux_sources: list[str]) -> MultiSourcePrecipitationModel:
        return MultiSourcePrecipitationModel(
            input_frames=2,
            aux_sources=aux_sources,
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

    def test_all_source_combinations(self) -> None:
        for aux_sources in (
            [],
            ["satellite"],
            ["ifs"],
            ["satellite", "ifs"],
        ):
            with self.subTest(aux_sources=aux_sources):
                model = self.build_model(aux_sources)
                radar = torch.randn(1, 2, 1, 32, 32)
                satellite = (
                    torch.randn(1, 2, 2, 8, 8)
                    if "satellite" in aux_sources
                    else None
                )
                ifs = (
                    torch.randn(1, 3, 2, 8, 8)
                    if "ifs" in aux_sources
                    else None
                )

                with torch.no_grad():
                    logits = model(radar, satellite=satellite, ifs=ifs)

                self.assertEqual(tuple(logits.shape), (1, 3, 8, 32, 32))
                self.assertTrue(torch.isfinite(logits).all())

    def test_required_auxiliary_input_is_checked(self) -> None:
        model = self.build_model(["ifs"])
        with self.assertRaisesRegex(ValueError, "no ifs tensor"):
            model(torch.randn(1, 2, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
