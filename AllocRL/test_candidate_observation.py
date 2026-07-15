import unittest
from datetime import date

import numpy as np

from alloc_env.alloc_env import BlockPlacementEnv
from alloc_env.block import Block, PrePlacedBlock
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace


def make_env(
    *,
    block_length: float,
    block_breadth: float,
    workspace_length: float = 100.0,
    workspace_breadth: float = 100.0,
    fill_workspace_with_preplacement: bool = False,
) -> BlockPlacementEnv:
    strategy = BaseGridStrategy(step=5.0)
    workspace = Workspace(
        code="PE001",
        origin_x=0.0,
        origin_y=0.0,
        length=workspace_length,
        breadth=workspace_breadth,
        strategy=strategy,
    )
    if fill_workspace_with_preplacement:
        workspace.add_pre_placement(
            PrePlacedBlock(
                label="FULL",
                pos_x=workspace_length / 2.0,
                pos_y=workspace_breadth / 2.0,
                length=workspace_length,
                breadth=workspace_breadth,
                start_date=date(2026, 1, 1),
                end_date=date(2026, 2, 28),
            )
        )
    block = Block(
        name="A",
        ship_no="T001",
        block_type="BUILD",
        length=block_length,
        breadth=block_breadth,
        height=5.0,
        weight=10.0,
        in_date=date(2026, 1, 5),
        out_date=date(2026, 1, 30),
    )
    return BlockPlacementEnv([block], [workspace], strategy, grid_size=32)


class CandidateObservationTests(unittest.TestCase):
    def test_candidate_channel_marks_strategy_position(self):
        env = make_env(block_length=20.0, block_breadth=10.0)
        obs, _ = env.reset(seed=3)
        candidate = env._candidate_placements[0]

        self.assertTrue(candidate.placeable)
        self.assertEqual((10.0, 5.0), candidate.position)
        self.assertEqual((1, 32, 32), obs["grids"][0, 3:4].shape)
        self.assertGreater(float(obs["grids"][0, 3].sum()), 0.0)

    def test_candidate_position_matches_the_applied_placement(self):
        env = make_env(block_length=20.0, block_breadth=10.0)
        env.reset(seed=3)
        candidate_position = env._candidate_placements[0].position

        env.step(0)
        placed = env._placement_simulator.blocks[0]

        self.assertEqual(candidate_position, (placed.ref_x, placed.ref_y))

    def test_observation_does_not_mutate_current_block(self):
        env = make_env(block_length=20.0, block_breadth=10.0)
        env.reset(seed=3)
        current = env._placement_simulator.current_block
        before = (
            current.length,
            current.breadth,
            current.ref_x,
            current.ref_y,
            current.angle,
        )

        env._get_obs()

        after = (
            current.length,
            current.breadth,
            current.ref_x,
            current.ref_y,
            current.angle,
        )
        self.assertEqual(before, after)

    def test_unplaceable_candidate_channel_is_zero(self):
        env = make_env(
            block_length=10.0,
            block_breadth=10.0,
            fill_workspace_with_preplacement=True,
        )
        obs, _ = env.reset(seed=3)

        self.assertEqual(0.0, float(obs["grids"][0, 3].sum()))
        self.assertEqual(0.0, float(obs["ws_meta"][0, 2]))

    def test_rotated_candidate_uses_rotated_dimensions(self):
        env = make_env(
            block_length=20.0,
            block_breadth=30.0,
            workspace_length=30.0,
            workspace_breadth=20.0,
        )
        obs, _ = env.reset(seed=3)
        candidate = env._candidate_placements[0]
        mask = obs["grids"][0, 3]
        rows, columns = np.where(mask > 0.0)

        self.assertTrue(candidate.rotated)
        self.assertEqual((30.0, 20.0), (candidate.length, candidate.breadth))
        self.assertGreater(np.unique(columns).size, np.unique(rows).size)


if __name__ == "__main__":
    unittest.main()
