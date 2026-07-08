"""BlockSetAttentionCnnExtractor + 미래 블록 관측 테스트 (torch/gym/numpy 필요).

RL 의존성이 설치된 venv에서 실행:
    python -m pytest test_block_attention_and_future_obs.py
    # 또는
    python test_block_attention_and_future_obs.py

순수 로직(upcoming_block_indices)만 검증하려면 test_future_block_lookahead.py 사용.
"""

import unittest
from datetime import date

import gymnasium as gym
import numpy as np
import torch

from alloc_env.alloc_env import BlockPlacementEnv, FUTURE_BLOCK_FEATURE_DIM
from alloc_env.block import Block
from alloc_env.workspace import Workspace
from alloc_env.strategy import BaseGridStrategy
from alloc_env.cnn_extractor import BlockSetAttentionCnnExtractor
from train import build_policy_kwargs


def make_block(name: str, in_date: date) -> Block:
    return Block(
        name=name, ship_no="T001", block_type="BUILD",
        length=10.0, breadth=10.0, height=5.0, weight=10.0,
        in_date=in_date, out_date=date(2026, 5, 30),
    )


def make_workspaces(n: int = 3):
    return [
        Workspace(
            code=f"PE00{i}", origin_x=0.0, origin_y=0.0,
            breadth=100.0, length=100.0,
            strategy=BaseGridStrategy(step=10.0),
        )
        for i in range(n)
    ]


class FutureBlockObservationTests(unittest.TestCase):
    def _make_env(self, k):
        blocks = [make_block(f"B{i}", date(2026, 4, 6 + i)) for i in range(6)]
        return BlockPlacementEnv(
            blocks, make_workspaces(3),
            use_synthetic=False, grid_size=32, n_future_blocks=k,
        )

    def test_default_env_keeps_legacy_contract(self):
        env = self._make_env(0)
        self.assertEqual(
            set(env.observation_space.spaces.keys()),
            {"block", "grids", "ws_meta"},
        )
        obs, _ = env.reset()
        self.assertEqual(set(obs.keys()), {"block", "grids", "ws_meta"})

    def test_future_keys_present_with_correct_shape(self):
        k = 4
        env = self._make_env(k)
        space = env.observation_space.spaces
        self.assertIn("future_blocks", space)
        self.assertIn("future_mask", space)
        self.assertEqual(
            space["future_blocks"].shape, (k, FUTURE_BLOCK_FEATURE_DIM)
        )
        self.assertEqual(space["future_mask"].shape, (k,))

        obs, _ = env.reset()
        self.assertEqual(
            obs["future_blocks"].shape, (k, FUTURE_BLOCK_FEATURE_DIM)
        )
        self.assertEqual(obs["future_mask"].shape, (k,))
        # 6블록 중 1개 current → 미래 유효 슬롯 존재
        self.assertGreaterEqual(float(obs["future_mask"].sum()), 1.0)

    def test_future_mask_valid_slots_are_contiguous_front(self):
        k = 4
        env = self._make_env(k)
        obs, _ = env.reset()
        mask = obs["future_mask"]
        n_valid = int(mask.sum())
        np.testing.assert_array_equal(
            mask[:n_valid], np.ones(n_valid, dtype=np.float32)
        )
        np.testing.assert_array_equal(
            mask[n_valid:], np.zeros(k - n_valid, dtype=np.float32)
        )
        # 패딩 슬롯의 피처는 0
        np.testing.assert_array_equal(
            obs["future_blocks"][n_valid:],
            np.zeros((k - n_valid, FUTURE_BLOCK_FEATURE_DIM), dtype=np.float32),
        )

    def test_obs_stays_within_space_across_steps(self):
        env = self._make_env(3)
        obs, _ = env.reset()
        self.assertTrue(env.observation_space.contains(obs))
        for _ in range(4):
            mask = env.action_masks()
            action = int(np.argmax(mask))
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                # 종료 관측도 space 안에 있어야
                self.assertTrue(env.observation_space.contains(obs))
                break
            self.assertTrue(env.observation_space.contains(obs))


class BlockSetAttentionExtractorTests(unittest.TestCase):
    def _space(self, with_future, k=4, n=3, g=32):
        d = {
            "block": gym.spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32),
            "grids": gym.spaces.Box(0.0, 1.0, shape=(n, 3, g, g), dtype=np.float32),
            "ws_meta": gym.spaces.Box(0.0, 1.0, shape=(n, 3), dtype=np.float32),
        }
        if with_future:
            d["future_blocks"] = gym.spaces.Box(
                0.0, 1.0, shape=(k, FUTURE_BLOCK_FEATURE_DIM), dtype=np.float32
            )
            d["future_mask"] = gym.spaces.Box(
                0.0, 1.0, shape=(k,), dtype=np.float32
            )
        return gym.spaces.Dict(d)

    def test_with_future_returns_finite_vector(self):
        k, n, g = 4, 3, 32
        space = self._space(True, k, n, g)
        extractor = BlockSetAttentionCnnExtractor(
            space, features_dim=128, cnn_out_dim=16, embed_dim=32, num_heads=4
        )
        B = 2
        obs = {
            "block": torch.rand(B, 10),
            "grids": torch.rand(B, n, 3, g, g),
            "ws_meta": torch.rand(B, n, 3),
            "future_blocks": torch.rand(B, k, FUTURE_BLOCK_FEATURE_DIM),
            "future_mask": torch.tensor(
                [[1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
            ),
        }
        feats = extractor(obs)
        self.assertEqual(tuple(feats.shape), (B, 128))
        self.assertTrue(torch.isfinite(feats).all())

    def test_all_future_padding_produces_no_nan(self):
        # 미래 전부 패딩(마스크 전부 0)이어도 현재 토큰은 유효 → NaN 없어야 함
        k, n, g = 3, 2, 32
        space = self._space(True, k, n, g)
        extractor = BlockSetAttentionCnnExtractor(
            space, features_dim=64, cnn_out_dim=16, embed_dim=32, num_heads=4
        )
        B = 2
        obs = {
            "block": torch.rand(B, 10),
            "grids": torch.rand(B, n, 3, g, g),
            "ws_meta": torch.rand(B, n, 3),
            "future_blocks": torch.zeros(B, k, FUTURE_BLOCK_FEATURE_DIM),
            "future_mask": torch.zeros(B, k),
        }
        feats = extractor(obs)
        self.assertTrue(torch.isfinite(feats).all())

    def test_without_future_key_degrades_gracefully(self):
        n, g = 3, 32
        space = self._space(False, n=n, g=g)
        extractor = BlockSetAttentionCnnExtractor(
            space, features_dim=64, cnn_out_dim=16, embed_dim=32, num_heads=4
        )
        B = 2
        obs = {
            "block": torch.rand(B, 10),
            "grids": torch.rand(B, n, 3, g, g),
            "ws_meta": torch.rand(B, n, 3),
        }
        feats = extractor(obs)
        self.assertEqual(tuple(feats.shape), (B, 64))
        self.assertTrue(torch.isfinite(feats).all())

    def test_build_policy_kwargs_selects_block_attn(self):
        pk = build_policy_kwargs(
            extractor="block-attn", features_dim=256,
            cnn_out_dim=64, embed_dim=64, num_heads=4,
        )
        self.assertIs(
            pk["features_extractor_class"], BlockSetAttentionCnnExtractor
        )
        self.assertEqual(pk["features_extractor_kwargs"]["embed_dim"], 64)


if __name__ == "__main__":
    unittest.main()
