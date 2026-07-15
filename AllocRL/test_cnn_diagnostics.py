import unittest

import gymnasium as gym
import numpy as np
import torch

from alloc_env.callbacks import CnnDiagnosticTracker
from alloc_env.cnn_extractor import (
    CandidateCnnExtractor,
    FixedGridExtractor,
    StructuredExtractor,
)


def space() -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "block": gym.spaces.Box(
            0.0, 1.0, shape=(10,), dtype=np.float32
        ),
        "grids": gym.spaces.Box(
            0.0, 1.0, shape=(2, 4, 16, 16), dtype=np.float32
        ),
        "ws_meta": gym.spaces.Box(
            0.0, 1.0, shape=(2, 3), dtype=np.float32
        ),
        "future_blocks": gym.spaces.Box(
            0.0, 1.0, shape=(2, 8), dtype=np.float32
        ),
        "future_mask": gym.spaces.Box(
            0.0, 1.0, shape=(2,), dtype=np.float32
        ),
    })


def observation() -> dict[str, torch.Tensor]:
    return {
        "block": torch.rand(2, 10),
        "grids": torch.rand(2, 2, 4, 16, 16),
        "ws_meta": torch.rand(2, 2, 3),
        "future_blocks": torch.rand(2, 2, 8),
        "future_mask": torch.ones(2, 2),
    }


class CnnDiagnosticTrackerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)

    def test_tracker_records_gradient_and_weight_change(self):
        extractor = CandidateCnnExtractor(space(), features_dim=32)
        tracker = CnnDiagnosticTracker(extractor)
        tracker.attach()
        optimizer = torch.optim.Adam(extractor.parameters(), lr=1e-3)

        output = extractor(observation())
        output.square().mean().backward()
        optimizer.step()
        metrics = tracker.record_update()
        tracker.close()

        self.assertGreater(metrics["cnn_gradient_norm"], 0.0)
        self.assertGreater(metrics["cnn_weight_change"], 0.0)

    def test_candidate_sensitivity_uses_candidate_channel(self):
        extractor = CandidateCnnExtractor(
            space(), features_dim=32
        ).eval()

        metrics = CnnDiagnosticTracker(extractor).measure_features(
            observation()
        )

        self.assertGreaterEqual(metrics["workspace_feature_variance"], 0.0)
        self.assertGreater(metrics["candidate_channel_sensitivity"], 0.0)

    def test_non_cnn_extractors_emit_no_cnn_metrics(self):
        for extractor_class in (StructuredExtractor, FixedGridExtractor):
            with self.subTest(extractor=extractor_class.__name__):
                tracker = CnnDiagnosticTracker(extractor_class(space()))
                tracker.attach()
                self.assertEqual({}, tracker.record_update())
                self.assertEqual({}, tracker.measure_features(observation()))
                tracker.close()


if __name__ == "__main__":
    unittest.main()
