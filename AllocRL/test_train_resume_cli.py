"""CLI regression tests for training resume support."""

import sys
import unittest
from unittest.mock import patch

import train as train_module


class TrainResumeCliTest(unittest.TestCase):
    def test_resume_from_argument_is_accepted(self):
        captured = {}

        def fake_train(args):
            captured["resume_from"] = args.resume_from
            captured["extractor"] = args.extractor
            captured["n_future_blocks"] = args.n_future_blocks
            captured["gae_lambda"] = args.gae_lambda
            captured["seed"] = args.seed

        argv = [
            "train.py",
            "--resume-from",
            ".\\output\\block_placement_ppo.zip",
            "--no-export-onnx",
        ]

        with patch.object(sys, "argv", argv), patch.object(train_module, "train", fake_train):
            train_module.main()

        self.assertEqual(captured["resume_from"], ".\\output\\block_placement_ppo.zip")
        self.assertEqual("candidate-cnn", captured["extractor"])
        self.assertEqual(4, captured["n_future_blocks"])
        self.assertEqual(0.98, captured["gae_lambda"])
        self.assertEqual(0, captured["seed"])


if __name__ == "__main__":
    unittest.main()
