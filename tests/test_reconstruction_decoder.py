import unittest

import torch

from src.reconstruction_decoder import ReconstructionDecoder, reconstruction_loss


class ReconstructionDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.features = torch.randn(2, 8, 4, 4)

    def test_baseline_forward_backward_and_single_batch_loss_reduction(self):
        decoder = ReconstructionDecoder(
            in_channels=8,
            hidden_dim=16,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            feedforward_dim=32,
            dropout=0,
            base_grid_size=(4, 4),
        )
        optimizer = torch.optim.Adam(decoder.parameters(), lr=0.01)
        with torch.no_grad():
            initial_loss = reconstruction_loss(
                self.features, decoder(self.features)
            ).item()

        decoder.train()
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            reconstruction = decoder(self.features)
            loss = reconstruction_loss(self.features, reconstruction)
            loss.backward()
            optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertLess(loss.item(), initial_loss * 0.6)
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in decoder.parameters()
            if parameter.requires_grad
        ))

    def test_jitter_is_training_only(self):
        decoder = ReconstructionDecoder(
            in_channels=8,
            hidden_dim=16,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            feedforward_dim=32,
            dropout=0,
            base_grid_size=(4, 4),
            jitter_scale=20,
            jitter_probability=1,
        )
        decoder.eval()
        with torch.no_grad():
            expected = decoder(self.features)
            repeated = decoder(self.features)
        self.assertTrue(torch.equal(expected, repeated))

        decoder.train()
        with torch.no_grad():
            jittered = decoder(self.features)
        self.assertFalse(torch.allclose(expected, jittered))

    def test_masked_attention_is_local_and_full_decoder_backpropagates(self):
        decoder = ReconstructionDecoder(
            in_channels=8,
            hidden_dim=16,
            num_heads=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            feedforward_dim=32,
            dropout=0,
            base_grid_size=(4, 4),
            neighbor_size=3,
            jitter_scale=20,
            jitter_probability=1,
        )
        mask = decoder._attention_mask(4, 4, self.features.device)
        self.assertFalse(mask[0, 0].item())
        self.assertFalse(mask[0, 1].item())
        self.assertTrue(mask[0, 10].item())
        self.assertFalse(mask[5, 10].item())

        decoder.train()
        reconstruction = decoder(self.features)
        loss = reconstruction_loss(self.features, reconstruction)
        loss.backward()
        self.assertEqual(tuple(reconstruction.shape), tuple(self.features.shape))
        self.assertTrue(torch.isfinite(loss))

    def test_rejects_mismatched_target_shape(self):
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            reconstruction_loss(torch.zeros(1, 2, 3, 3), torch.zeros(1, 2, 2, 3))


if __name__ == "__main__":
    unittest.main()
