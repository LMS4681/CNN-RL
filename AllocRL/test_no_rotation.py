from datetime import date

from alloc_env.alloc_env import BlockPlacementEnv
from alloc_env.block import Block
from alloc_env.constraints import DimensionConstraint
from alloc_env.incremental_simulator import IncrementalPlacementSimulator
from alloc_env.simulator import PlacementSimulator, SimulationResult
from alloc_env.strategy import BaseGridStrategy
from alloc_env.workspace import Workspace


def make_block(length: float, breadth: float) -> Block:
    return Block(
        name="B-1",
        ship_no="S-1",
        block_type="BUILD",
        length=length,
        breadth=breadth,
        height=1.0,
        weight=1.0,
        in_date=date(2026, 1, 5),
        out_date=date(2026, 1, 20),
    )


def make_workspace(code: str, length: float, breadth: float) -> Workspace:
    return Workspace(
        code=code,
        origin_x=0.0,
        origin_y=0.0,
        length=length,
        breadth=breadth,
        strategy=BaseGridStrategy(step=1.0),
    )


def test_dimension_constraint_rejects_rotation_only_fit():
    block = make_block(length=8.0, breadth=4.0)
    workspace = make_workspace("ROTATION_ONLY", length=5.0, breadth=10.0)

    assert not DimensionConstraint().is_feasible(block, workspace)


def test_candidate_does_not_rotate_to_find_a_position():
    block = make_block(length=8.0, breadth=4.0)
    strategy = BaseGridStrategy(step=1.0)
    env = BlockPlacementEnv(
        [block],
        [
            make_workspace("ROTATION_ONLY", 5.0, 10.0),
            make_workspace("VALID", 20.0, 20.0),
        ],
        strategy,
        grid_size=64,
    )
    env.reset(seed=0)

    candidate = env.unwrapped._compute_candidate_placements(
        env.unwrapped._placement_simulator.current_block
    )[0]

    assert candidate.position is None
    assert candidate.length == 8.0
    assert candidate.breadth == 4.0
    assert not hasattr(candidate, "rotated")


def test_incremental_and_replay_keep_original_orientation():
    block = make_block(length=8.0, breadth=4.0)
    workspace = make_workspace("ROTATION_ONLY", length=5.0, breadth=10.0)

    incremental = IncrementalPlacementSimulator(
        [block], [workspace], dropout_threshold=0
    )
    incremental.assign_current(0)
    assert incremental.result().delay_days == [SimulationResult.DROPOUT]

    replay = PlacementSimulator().replay([block], [workspace], [0], 0)
    assert replay.delay_days == [SimulationResult.DROPOUT]
    assert replay.blocks[0].length == 8.0
    assert replay.blocks[0].breadth == 4.0
