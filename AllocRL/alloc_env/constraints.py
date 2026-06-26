"""
하드 제약 조건(Hard Constraint) 및 유효 작업장 필터.

C# IHardConstraint, DimensionConstraint, ValidWorkspacePicker의 Python 재구현.
"""

from __future__ import annotations

import random
from typing import List, Optional, Protocol

from .block import Block
from .workspace import Workspace


# ── 하드 제약 프로토콜 ────────────────────────────────────────────

class HardConstraint(Protocol):
    """블록-작업장 배정 시 절대적 제약 조건 인터페이스."""
    def is_feasible(self, block: Block, workspace: Workspace) -> bool: ...


# ── DimensionConstraint ──────────────────────────────────────────

class DimensionConstraint:
    """
    블록의 물리적 치수가 작업장의 허용 한계를 초과하는지 판정.
    - 크기: 90° 회전 허용
    - 폭/무게/높이: 작업장 max 대비
    """

    def is_feasible(self, block: Block, ws: Workspace) -> bool:
        # ① 작업장 레이아웃 크기 (90° 회전 허용)
        no_rot = block.length <= ws.length and block.breadth <= ws.breadth
        rot90  = block.breadth <= ws.length and block.length  <= ws.breadth
        if not no_rot and not rot90:
            return False

        # ② 폭 제약
        if block.breadth > ws.max_breadth:
            return False

        # ③ 무게 제약
        if block.weight > ws.max_weight:
            return False

        # ④ 높이 제약
        if block.height > ws.max_height:
            return False

        return True


class BlockPatternConstraint:
    """작업장별 허용 블록명 패턴을 적용합니다."""

    def is_feasible(self, block: Block, ws: Workspace) -> bool:
        return ws.is_block_allowed(block.name)


# ── ValidWorkspacePicker ─────────────────────────────────────────

class ValidWorkspacePicker:
    """
    하드 제약 기반 유효 작업장 필터링 및 선택.
    생성 시 모든 블록×작업장 조합을 사전 평가하여 캐시 구축.
    """

    def __init__(
        self,
        blocks: List[Block],
        workspaces: List[Workspace],
        constraints: List[HardConstraint],
        rng: Optional[random.Random] = None,
    ):
        self._rng = rng or random.Random()
        self._cache: List[List[int]] = []

        for block in blocks:
            valid = []
            for w_idx, ws in enumerate(workspaces):
                if all(c.is_feasible(block, ws) for c in constraints):
                    valid.append(w_idx)
            self._cache.append(valid)

    @property
    def block_count(self) -> int:
        return len(self._cache)

    def pick(self, block_index: int) -> int:
        """유효한 작업장 중 하나를 랜덤 반환."""
        valid = self._cache[block_index]
        if not valid:
            raise ValueError(f"블록 {block_index}에 대해 유효한 작업장이 없습니다.")
        return self._rng.choice(valid)

    def is_valid(self, block_index: int, workspace_index: int) -> bool:
        return workspace_index in self._cache[block_index]

    def get_valid_workspaces(self, block_index: int) -> List[int]:
        return list(self._cache[block_index])

    def get_action_mask(self, block_index: int, num_workspaces: int) -> List[bool]:
        """RL 학습용: 블록에 대한 작업장별 액션 마스크."""
        mask = [False] * num_workspaces
        for ws_idx in self._cache[block_index]:
            mask[ws_idx] = True
        return mask

    def get_infeasible_blocks(self) -> List[int]:
        return [i for i, v in enumerate(self._cache) if not v]
