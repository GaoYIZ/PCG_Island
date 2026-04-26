import unittest

import torch

from vae_model import BetaVAE


class BetaVAESkipConnectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vae = BetaVAE(map_size=128, latent_dim=128)

    def test_encode_with_skips_returns_expected_shapes(self) -> None:
        x = torch.randn(2, 1, 128, 128)

        mu, logvar, skips = self.vae.encode_with_skips(x)

        self.assertEqual(mu.shape, (2, 128))
        self.assertEqual(logvar.shape, (2, 128))
        self.assertEqual(len(skips), 2)
        self.assertEqual(skips[0].shape, (2, 32, 64, 64))
        self.assertEqual(skips[1].shape, (2, 64, 32, 32))

    def test_decode_with_skips_restores_heightmap_shape(self) -> None:
        x = torch.randn(2, 1, 128, 128)
        mu, logvar, skips = self.vae.encode_with_skips(x)
        z = self.vae.reparameterize(mu, logvar)

        reconstruction = self.vae.decode(z, skips=skips)

        self.assertEqual(reconstruction.shape, (2, 1, 128, 128))

    def test_reconstruct_from_input_runs_end_to_end(self) -> None:
        x = torch.randn(2, 1, 128, 128)

        reconstruction, mu, logvar, structure_prediction = self.vae.reconstruct_from_input(
            x, deterministic=True
        )

        self.assertEqual(reconstruction.shape, x.shape)
        self.assertEqual(mu.shape, (2, 128))
        self.assertEqual(logvar.shape, (2, 128))
        self.assertIsNone(structure_prediction)


if __name__ == "__main__":
    unittest.main()
