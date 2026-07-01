import unittest

import gymnasium as gym
import torch

from alloc_env.cnn_extractor import PointerAttentionCnnExtractor
from train import build_policy_kwargs


class PointerAttentionExtractorTests(unittest.TestCase):
    def test_build_policy_kwargs_selects_pointer_attention_extractor(self):
        policy_kwargs = build_policy_kwargs(
            extractor="pointer-attn",
            features_dim=256,
            cnn_out_dim=64,
            embed_dim=64,
            num_heads=4,
        )

        self.assertIs(
            policy_kwargs["features_extractor_class"],
            PointerAttentionCnnExtractor,
        )
        self.assertEqual(
            policy_kwargs["features_extractor_kwargs"]["embed_dim"],
            64,
        )

    def test_pointer_attention_extractor_returns_feature_vector(self):
        observation_space = gym.spaces.Dict({
            "block": gym.spaces.Box(0.0, 1.0, shape=(10,), dtype=float),
            "grids": gym.spaces.Box(0.0, 1.0, shape=(3, 3, 32, 32), dtype=float),
            "ws_meta": gym.spaces.Box(0.0, 1.0, shape=(3, 2), dtype=float),
        })
        extractor = PointerAttentionCnnExtractor(
            observation_space,
            features_dim=128,
            cnn_out_dim=16,
            embed_dim=32,
            num_heads=4,
        )

        observations = {
            "block": torch.rand(2, 10),
            "grids": torch.rand(2, 3, 3, 32, 32),
            "ws_meta": torch.rand(2, 3, 2),
        }

        features = extractor(observations)

        self.assertEqual((2, 128), tuple(features.shape))
        self.assertTrue(torch.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
