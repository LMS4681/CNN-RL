"""
블록(Block) 데이터 모델.

C# BlockIF + TestBlock의 Python 재구현.
- AABB 충돌 검사 (Intersects)
- 90° 회전 (Turn)
- 이동 (Move)
- 배치 지연 (DelayPlacement)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from . import calendar as cal

# C# AllocConst.Epsilon 대응
EPSILON = 1e-5
SAFETY_DISTANCE = 1.0


@dataclass
class PrePlacedBlock:
    """기배치 블록 — 특정 기간·위치에 고정 점유된 블록."""
    label: str
    pos_x: float
    pos_y: float
    length: float         # X 방향
    breadth: float        # Y 방향
    start_date: date
    end_date: date

    def is_active_on(self, env_date: date) -> bool:
        return self.start_date <= env_date <= self.end_date

    def overlaps_period(self, from_d: date, to_d: date) -> bool:
        return self.start_date <= to_d and from_d <= self.end_date

    def intersects_block(self, block: Block) -> bool:
        """AABB 충돌 검사 (경계 맞닿음은 비충돌)."""
        a_left   = self.pos_x - self.length  / 2
        a_right  = self.pos_x + self.length  / 2
        a_bottom = self.pos_y - self.breadth / 2
        a_top    = self.pos_y + self.breadth / 2

        b_left   = block.ref_x - block.length  / 2
        b_right  = block.ref_x + block.length  / 2
        b_bottom = block.ref_y - block.breadth / 2
        b_top    = block.ref_y + block.breadth / 2

        sep_x = (
            a_right + SAFETY_DISTANCE <= b_left + EPSILON
            or b_right + SAFETY_DISTANCE <= a_left + EPSILON
        )
        sep_y = (
            a_top + SAFETY_DISTANCE <= b_bottom + EPSILON
            or b_top + SAFETY_DISTANCE <= a_bottom + EPSILON
        )

        return not (sep_x or sep_y)

    def intersects_block_in_period(self, block: Block, from_d: date, to_d: date) -> bool:
        """기간 겹침이 있으면 공간 충돌 검사."""
        if not self.overlaps_period(from_d, to_d):
            return False
        return self.intersects_block(block)


@dataclass
class Block:
    """배치 대상 블록."""
    name: str
    ship_no: str
    block_type: str
    length: float         # X 방향 (m)
    breadth: float        # Y 방향 (m)
    height: float         # 높이 (m)
    weight: float         # 무게 (ton)
    in_date: date         # 입고일 (변경 가능)
    out_date: date        # 출고일 (변경 가능)

    # 내부 상태
    ref_x: float = 0.0
    ref_y: float = 0.0
    angle: int = 0        # 0 또는 90
    workspace_code: Optional[str] = None

    # 원본 (불변) — __post_init__에서 설정
    original_in_date: date = field(init=False, repr=False)
    original_out_date: date = field(init=False, repr=False)
    original_length: float = field(init=False, repr=False)
    original_breadth: float = field(init=False, repr=False)
    original_duration: int = field(init=False, repr=False)

    def __post_init__(self):
        self.original_in_date = self.in_date
        self.original_out_date = self.out_date
        self.original_length = self.length
        self.original_breadth = self.breadth
        self.original_duration = cal.get_working_days_between(self.in_date, self.out_date)

    # ── 충돌 검사 ─────────────────────────────────────────────────

    def intersects(self, other: Block) -> bool:
        """AABB 충돌 검사 (경계 맞닿음은 비충돌)."""
        a_left   = self.ref_x  - self.length  / 2
        a_right  = self.ref_x  + self.length  / 2
        a_bottom = self.ref_y  - self.breadth / 2
        a_top    = self.ref_y  + self.breadth / 2

        b_left   = other.ref_x - other.length  / 2
        b_right  = other.ref_x + other.length  / 2
        b_bottom = other.ref_y - other.breadth / 2
        b_top    = other.ref_y + other.breadth / 2

        sep_x = (
            a_right + SAFETY_DISTANCE <= b_left + EPSILON
            or b_right + SAFETY_DISTANCE <= a_left + EPSILON
        )
        sep_y = (
            a_top + SAFETY_DISTANCE <= b_bottom + EPSILON
            or b_top + SAFETY_DISTANCE <= a_bottom + EPSILON
        )

        return not (sep_x or sep_y)

    # ── 이동 / 회전 / 지연 ────────────────────────────────────────

    def move(self, dx: float, dy: float) -> None:
        self.ref_x += dx
        self.ref_y += dy

    def turn(self) -> None:
        """90° 회전 — length ↔ breadth swap."""
        self.length, self.breadth = self.breadth, self.length
        self.angle = 90 if self.angle == 0 else 0

    def delay_placement(self, days: int) -> bool:
        """
        배치를 days 근무일만큼 지연시킵니다.
        C# TestBlock.DelayPlacement와 동일한 로직.
        """
        if days <= 0:
            return False
        from datetime import timedelta
        self.in_date = cal.calculate_end_date(
            self.in_date + timedelta(days=1), days)
        self.out_date = cal.calculate_end_date(
            self.in_date, self.original_duration)
        self.out_date = cal.adjust_to_working_day(self.out_date, forward=True)
        return True

    # ── 유틸리티 ──────────────────────────────────────────────────

    @property
    def duration(self) -> int:
        return cal.get_working_days_between(self.in_date, self.out_date)

    def clone(self) -> Block:
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (f"Block({self.name}, L={self.length:.1f}, B={self.breadth:.1f}, "
                f"in={self.in_date}, out={self.out_date})")
