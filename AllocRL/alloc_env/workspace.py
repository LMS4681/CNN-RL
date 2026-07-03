"""
작업장(Workspace) 및 지번(Lot) 데이터 모델.

C# WorkSpaceIF + LotWorkSpace의 Python 재구현.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .strategy import BaseGridStrategy

from .block import Block, PrePlacedBlock


@dataclass
class LotRegion:
    """작업장 내 격자 구획 단위(지번)."""
    lot_id: str
    origin_x: float     # 좌하단 X
    origin_y: float     # 좌하단 Y
    breadth: float      # Y 방향 (m)
    length: float       # X 방향 (m)

    @property
    def area(self) -> float:
        return self.breadth * self.length


@dataclass
class Workspace:
    """
    작업장.
    C# WorkSpaceIF + LotWorkSpace 통합 구현.
    """
    code: str
    origin_x: float       # 좌하단 X
    origin_y: float       # 좌하단 Y
    breadth: float        # Y 방향 전체 크기 (m)
    length: float         # X 방향 전체 크기 (m)
    max_weight: float = float("inf")
    max_breadth: float = float("inf")
    max_height: float = float("inf")
    name: str = ""

    # 작업장에 배치 가능한 블록명 패턴 리스트 (Glob, case-insensitive).
    # None 또는 빈 리스트 = 제약 없음 (모든 블록 허용).
    allowable_block_patterns: Optional[List[str]] = None

    # 지번 목록 (LotWorkSpace 역할)
    lots: List[LotRegion] = field(default_factory=list)

    # 배치된 블록 관리
    blocks: List[Block] = field(default_factory=list, repr=False)
    pre_placements: List[PrePlacedBlock] = field(default_factory=list, repr=False)

    # 배치 전략 (런타임에 주입)
    strategy: Optional[BaseGridStrategy] = field(default=None, repr=False)

    # ── 지번 관리 ─────────────────────────────────────────────────

    def add_lot(self, lot: LotRegion) -> None:
        self.lots.append(lot)

    @property
    def has_lots(self) -> bool:
        return len(self.lots) > 0

    def get_lot(self, lot_id: str) -> Optional[LotRegion]:
        for lot in self.lots:
            if lot.lot_id == lot_id:
                return lot
        return None

    # ── 기배치 블록 관리 ──────────────────────────────────────────

    def add_pre_placement(self, pp: PrePlacedBlock) -> None:
        self.pre_placements.append(pp)

    def get_overlapping_pre_placements(
        self, from_d: date, to_d: date
    ) -> List[PrePlacedBlock]:
        """주어진 기간과 점유 기간이 겹치는 기배치 블록 목록."""
        return [pp for pp in self.pre_placements if pp.overlaps_period(from_d, to_d)]

    def get_active_pre_placements(self, env_date: date) -> List[PrePlacedBlock]:
        """특정 날짜에 활성(점유 중)인 기배치 블록."""
        return [pp for pp in self.pre_placements if pp.is_active_on(env_date)]

    def intersects_with_any_pre_placed(
        self, block: Block, from_d: date, to_d: date
    ) -> bool:
        """기배치 블록과 시간·공간 모두에서 충돌하는지 검사."""
        for pp in self.pre_placements:
            if pp.intersects_block_in_period(block, from_d, to_d):
                return True
        return False

    # ── 블록 추가/제거 ────────────────────────────────────────────

    def add_block(self, block: Block, env_date: date) -> None:
        """블록을 배치하고 작업장 코드를 기록."""
        block.workspace_code = self.code
        self.blocks.append(block)

    def remove_block(self, block: Block) -> bool:
        try:
            self.blocks.remove(block)
            return True
        except ValueError:
            return False

    def clear_outgoing_blocks(self, env_date: date) -> None:
        """출고일이 env_date보다 빠른 블록만 제거하여 공간 해제."""
        self.blocks = [b for b in self.blocks if b.out_date >= env_date]

    # ── 배치 좌표 결정 (전략 위임) ────────────────────────────────

    def determine_placement_position(
        self, block: Block, env_date: date
    ) -> Optional[tuple[float, float]]:
        """전략에 배치 좌표 결정을 위임합니다."""
        if self.strategy is None:
            return None
        return self.strategy.determine_position(self, block, env_date)

    # ── 작업장-블록 호환성 ────────────────────────────────────────

    def set_allowable_block_patterns(self, patterns: Optional[List[str]]) -> None:
        """배치 가능 블록명 패턴 리스트를 설정 (None/빈 리스트 → 제약 없음)."""
        self.allowable_block_patterns = (
            None if patterns is None or len(patterns) == 0 else list(patterns)
        )

    def is_block_allowed(self, block_name: str) -> bool:
        """블록명이 이 작업장에 허용되는지 검사. 제약 없으면 항상 True."""
        if not self.allowable_block_patterns:
            return True
        if not block_name:
            return False
        for pat in self.allowable_block_patterns:
            if _glob_match_ci(pat, block_name):
                return True
        return False

    # ── 유틸리티 ──────────────────────────────────────────────────

    def deep_copy(self) -> Workspace:
        return copy.deepcopy(self)

    @staticmethod
    def deep_copy_list(workspaces: List[Workspace]) -> List[Workspace]:
        return [ws.deep_copy() for ws in workspaces]

    def __repr__(self) -> str:
        return (f"Workspace({self.code}, L={self.length:.0f}, B={self.breadth:.0f}, "
                f"lots={len(self.lots)}, blocks={len(self.blocks)})")


# ── Glob 매칭 (case-insensitive) ─────────────────────────────────────

def _glob_match_ci(pattern: str, text: str) -> bool:
    """'*' 와일드카드 지원 Glob 매칭, 대소문자 무시."""
    if pattern is None:
        return False
    p = t = 0
    star = -1
    match = 0
    pat = pattern
    txt = text
    pat_len = len(pat)
    txt_len = len(txt)
    while t < txt_len:
        if p < pat_len and pat[p] == '*':
            star = p
            p += 1
            match = t
        elif p < pat_len and pat[p].lower() == txt[t].lower():
            p += 1
            t += 1
        elif star != -1:
            p = star + 1
            match += 1
            t = match
        else:
            return False
    while p < pat_len and pat[p] == '*':
        p += 1
    return p == pat_len
