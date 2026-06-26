"""
작업장 점유 상태 → 3채널 정규화 그리드 렌더러.

각 작업장의 블록 배치 상태를 128×128 의 3채널 이미지로 변환.
- Ch0: 블록 점유 마스크       (0=빈 셀, 1=점유)
- Ch1: 잔여 출고 공기         (0=빈 셀, 정규화된 잔여일수)
- Ch2: 작업장 경계 마스크     (1=작업장 내부, 0=패딩)

정규화 방식: 비율 유지 리사이즈 (Aspect-Ratio-Preserving Resize)
  - 작업장의 큰 축을 GRID_SIZE에 맞추고, 작은 축은 비율에 따라 축소
  - 패딩 영역은 Ch2=0으로 구분
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

import numpy as np

from .block import Block, PrePlacedBlock
from .workspace import Workspace


# ── 상수 ──────────────────────────────────────────────────────────

GRID_SIZE = 64
NUM_CHANNELS = 3
MAX_REMAINING_DAYS = 60  # 잔여 공기 정규화 기준 (60일 이상은 1.0으로 클리핑)


class OccupancyGridRenderer:
    """
    작업장 점유 상태를 3채널 정규화 그리드로 렌더링.

    비율 유지 리사이즈를 통해 모든 작업장이 동일한 GRID_SIZE×GRID_SIZE에 수용되며,
    CNN 가중치 공유가 가능하다.
    """

    def __init__(self, grid_size: int = GRID_SIZE):
        self.grid_size = grid_size

    # ── 공개 API ──────────────────────────────────────────────────

    def render(
        self,
        ws: Workspace,
        env_date: date,
        max_remaining_days: int = MAX_REMAINING_DAYS,
    ) -> np.ndarray:
        """
        단일 작업장 → (3, grid_size, grid_size) float32 그리드.

        Args:
            ws: 렌더링할 작업장
            env_date: 현재 환경 날짜 (잔여 공기 계산 기준)
            max_remaining_days: 잔여 공기 정규화 최대값

        Returns:
            (3, H, W) float32 ndarray, 값 범위 [0, 1]
        """
        G = self.grid_size
        grid = np.zeros((NUM_CHANNELS, G, G), dtype=np.float32)

        # ── 스케일 계산 (비율 유지) ───────────────────────────────
        scale_info = self._compute_scale(ws)
        grid_l, grid_b = scale_info["grid_l"], scale_info["grid_b"]
        offset_x, offset_y = scale_info["offset_x"], scale_info["offset_y"]

        # ── Ch2: 작업장 경계 마스크 ───────────────────────────────
        grid[2, offset_y : offset_y + grid_b, offset_x : offset_x + grid_l] = 1.0

        # ── 배치된 블록 렌더링 (Ch0, Ch1) ─────────────────────────
        for blk in ws.blocks:
            self._render_block(
                grid, blk, ws, scale_info, env_date, max_remaining_days,
                pos_x=blk.ref_x, pos_y=blk.ref_y,
                length=blk.length, breadth=blk.breadth,
                out_date=blk.out_date,
            )

        # ── 기배치 블록 렌더링 (Ch0, Ch1) ─────────────────────────
        for pp in ws.get_active_pre_placements(env_date):
            self._render_block(
                grid, None, ws, scale_info, env_date, max_remaining_days,
                pos_x=pp.pos_x, pos_y=pp.pos_y,
                length=pp.length, breadth=pp.breadth,
                out_date=pp.end_date,
            )

        return grid

    def render_all(
        self,
        workspaces: List[Workspace],
        env_date: date,
        max_remaining_days: int = MAX_REMAINING_DAYS,
    ) -> np.ndarray:
        """
        전체 작업장 → (N, 3, grid_size, grid_size) float32 텐서.
        """
        N = len(workspaces)
        grids = np.zeros((N, NUM_CHANNELS, self.grid_size, self.grid_size),
                         dtype=np.float32)
        for i, ws in enumerate(workspaces):
            grids[i] = self.render(ws, env_date, max_remaining_days)
        return grids

    def compute_scale_value(self, ws: Workspace) -> float:
        """
        작업장의 1px당 미터 수 (정규화용).

        Returns:
            scale (m/px), 정규화 시 max_scale로 나눌 수 있음.
        """
        max_dim = max(ws.length, ws.breadth, 1.0)
        return max_dim / self.grid_size

    # ── 내부 헬퍼 ─────────────────────────────────────────────────

    def _compute_scale(self, ws: Workspace) -> Dict[str, int]:
        """비율 유지 리사이즈를 위한 스케일 정보 계산."""
        G = self.grid_size
        ws_l = max(ws.length, 1.0)
        ws_b = max(ws.breadth, 1.0)
        max_dim = max(ws_l, ws_b)

        scale = G / max_dim  # px/m
        grid_l = max(1, min(G, round(ws_l * scale)))
        grid_b = max(1, min(G, round(ws_b * scale)))

        # 센터링 오프셋
        offset_x = (G - grid_l) // 2
        offset_y = (G - grid_b) // 2

        return {
            "scale": scale,
            "grid_l": grid_l,
            "grid_b": grid_b,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "ws_length": ws_l,
            "ws_breadth": ws_b,
        }

    def _world_to_pixel(
        self,
        wx: float,
        wy: float,
        scale_info: Dict[str, int],
        ws: Workspace,
    ) -> Tuple[int, int]:
        """물리 좌표(m, 센터 기준) → 픽셀 좌표."""
        scale = scale_info["scale"]
        offset_x = scale_info["offset_x"]
        offset_y = scale_info["offset_y"]

        # 작업장 원점 기준 상대 좌표
        rel_x = wx - ws.origin_x
        rel_y = wy - ws.origin_y

        px = int(rel_x * scale) + offset_x
        py = int(rel_y * scale) + offset_y

        return px, py

    def _render_block(
        self,
        grid: np.ndarray,
        blk: Optional[Block],
        ws: Workspace,
        scale_info: Dict,
        env_date: date,
        max_remaining_days: int,
        *,
        pos_x: float,
        pos_y: float,
        length: float,
        breadth: float,
        out_date: date,
    ) -> None:
        """단일 블록을 그리드에 렌더링 (Ch0, Ch1)."""
        G = self.grid_size
        scale = scale_info["scale"]
        offset_x = scale_info["offset_x"]
        offset_y = scale_info["offset_y"]

        # 블록의 좌하단 좌표 (센터 → 좌하단)
        bx0 = pos_x - length / 2.0 - ws.origin_x
        by0 = pos_y - breadth / 2.0 - ws.origin_y

        # 픽셀 좌표 (정수 변환)
        px0 = max(0, int(bx0 * scale) + offset_x)
        py0 = max(0, int(by0 * scale) + offset_y)
        px1 = min(G, int((bx0 + length) * scale) + offset_x)
        py1 = min(G, int((by0 + breadth) * scale) + offset_y)

        if px1 <= px0 or py1 <= py0:
            return  # 너무 작아서 렌더링 불가

        # Ch0: 점유 마스크
        grid[0, py0:py1, px0:px1] = 1.0

        # Ch1: 잔여 출고 공기 (정규화)
        remaining = (out_date - env_date).days
        remaining = max(0, remaining)
        norm_remaining = min(remaining / max_remaining_days, 1.0)
        grid[1, py0:py1, px0:px1] = norm_remaining


class GridCache:
    """
    그리드 캐싱 — 변경된 작업장만 재렌더링.

    step() 시 action이 지정된 작업장만 무효화하고,
    나머지는 캐시된 그리드를 재사용하여 렌더링 비용 절감.
    """

    def __init__(self, renderer: OccupancyGridRenderer, n_workspaces: int):
        self._renderer = renderer
        self._n_ws = n_workspaces
        G = renderer.grid_size
        self._cache = np.zeros(
            (n_workspaces, NUM_CHANNELS, G, G), dtype=np.float32
        )
        self._dirty = [True] * n_workspaces  # 초기에는 전부 dirty

    def invalidate(self, ws_index: int) -> None:
        """특정 작업장을 dirty로 표시."""
        self._dirty[ws_index] = True

    def invalidate_all(self) -> None:
        """전체 작업장을 dirty로 표시 (reset 시)."""
        self._dirty = [True] * self._n_ws

    def get_grids(
        self,
        workspaces: List[Workspace],
        env_date: date,
    ) -> np.ndarray:
        """
        캐시 기반 전체 그리드 반환.

        dirty 상태인 작업장만 재렌더링하고 캐시 업데이트.
        """
        for i in range(self._n_ws):
            if self._dirty[i]:
                self._cache[i] = self._renderer.render(
                    workspaces[i], env_date
                )
                self._dirty[i] = False
        return self._cache.copy()
