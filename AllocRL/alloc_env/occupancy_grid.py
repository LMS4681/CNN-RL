"""Physical workspace rasterization for base and candidate state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from numbers import Integral
from typing import List, Optional, Tuple

import numpy as np

from .block import SAFETY_DISTANCE, Block
from .observation_state import working_days_until
from .strategy import BaseGridStrategy
from .workspace import Workspace


GRID_SIZE = 64
BASE_CHANNELS = 2
CANDIDATE_CONTEXT_CHANNELS = 2
MAX_REMAINING_DAYS = 60

@dataclass(frozen=True)
class CandidatePlacement:
    position: Optional[Tuple[float, float]]
    length: float
    breadth: float

    @property
    def placeable(self) -> bool:
        return self.position is not None


@dataclass(frozen=True)
class CoordinateMap:
    x_px_per_m: float
    y_px_per_m: float


class OccupancyGridRenderer:
    def __init__(self, grid_size: int = GRID_SIZE):
        if (
            isinstance(grid_size, bool)
            or not isinstance(grid_size, Integral)
            or grid_size <= 0
        ):
            raise ValueError("grid_size must be a positive integer")
        self.grid_size = int(grid_size)

    def coordinate_map(self, ws: Workspace) -> CoordinateMap:
        self._validate_workspace_geometry(ws)
        return CoordinateMap(
            x_px_per_m=self.grid_size / ws.length,
            y_px_per_m=self.grid_size / ws.breadth,
        )

    def rectangle_bounds(
        self,
        ws: Workspace,
        center_x: float,
        center_y: float,
        length: float,
        breadth: float,
    ) -> tuple[int, int, int, int]:
        mapping = self.coordinate_map(ws)
        self._validate_rectangle_geometry(
            center_x, center_y, length, breadth
        )

        x0, x1 = self._axis_bounds(
            center_x,
            length,
            ws.origin_x,
            ws.length,
            mapping.x_px_per_m,
        )
        y0, y1 = self._axis_bounds(
            center_y,
            breadth,
            ws.origin_y,
            ws.breadth,
            mapping.y_px_per_m,
        )
        return x0, y0, x1, y1

    def render_base(self, ws: Workspace, env_date: date) -> np.ndarray:
        self.coordinate_map(ws)
        grid = np.zeros(
            (BASE_CHANNELS, self.grid_size, self.grid_size),
            dtype=np.float32,
        )
        for block in ws.blocks:
            if not block.in_date <= env_date <= block.out_date:
                continue
            self._render_existing_exclusion(
                grid,
                ws,
                block.ref_x,
                block.ref_y,
                block.length,
                block.breadth,
                block.out_date,
                env_date,
            )
        for placed in ws.get_active_pre_placements(env_date):
            self._render_existing_exclusion(
                grid,
                ws,
                placed.pos_x,
                placed.pos_y,
                placed.length,
                placed.breadth,
                placed.end_date,
                env_date,
            )
        return grid

    def render_candidate_context(
        self,
        ws: Workspace,
        candidate: CandidatePlacement,
        current_block: Block,
        env_date: date,
    ) -> np.ndarray:
        self.coordinate_map(ws)
        context = np.zeros(
            (CANDIDATE_CONTEXT_CHANNELS, self.grid_size, self.grid_size),
            dtype=np.float32,
        )
        preview = ws.deep_copy()

        if candidate.position is not None:
            center_x, center_y = candidate.position
            block = current_block.clone()
            block.move(center_x - block.ref_x, center_y - block.ref_y)
            preview.add_block(block, env_date)
            self._render_rectangle(
                context[1],
                ws,
                center_x,
                center_y,
                block.length + 2 * SAFETY_DISTANCE,
                block.breadth + 2 * SAFETY_DISTANCE,
                value=1.0,
            )

        if not preview.has_lots:
            context[0].fill(0.25)
            return context

        strategy = preview.strategy or BaseGridStrategy()
        occupied = strategy.occupied_lot_ids(
            preview, current_block.in_date, current_block.out_date
        )
        for lot in preview.lots:
            self._render_rectangle(
                context[0],
                preview,
                lot.origin_x + lot.length / 2.0,
                lot.origin_y + lot.breadth / 2.0,
                lot.length,
                lot.breadth,
                value=1.0 if lot.lot_id in occupied else 0.25,
            )
        return context

    def _render_existing_exclusion(
        self,
        grid: np.ndarray,
        ws: Workspace,
        center_x: float,
        center_y: float,
        length: float,
        breadth: float,
        out_date: date,
        env_date: date,
    ) -> None:
        x0, y0, x1, y1 = self.rectangle_bounds(
            ws,
            center_x,
            center_y,
            length + 2 * SAFETY_DISTANCE,
            breadth + 2 * SAFETY_DISTANCE,
        )
        if x1 <= x0 or y1 <= y0:
            return

        grid[0, y0:y1, x0:x1] = 1.0
        lifetime = min(
            working_days_until(env_date, out_date) / MAX_REMAINING_DAYS,
            1.0,
        )
        lifetime_slice = grid[1, y0:y1, x0:x1]
        np.maximum(lifetime_slice, lifetime, out=lifetime_slice)

    def _render_rectangle(
        self,
        channel: np.ndarray,
        ws: Workspace,
        center_x: float,
        center_y: float,
        length: float,
        breadth: float,
        value: float,
    ) -> None:
        x0, y0, x1, y1 = self.rectangle_bounds(
            ws, center_x, center_y, length, breadth
        )
        if x1 <= x0 or y1 <= y0:
            return
        rectangle = channel[y0:y1, x0:x1]
        np.maximum(rectangle, value, out=rectangle)

    def _axis_bounds(
        self,
        center: float,
        rectangle_extent: float,
        origin: float,
        workspace_extent: float,
        px_per_m: float,
    ) -> tuple[int, int]:
        low = center - rectangle_extent / 2.0
        high = center + rectangle_extent / 2.0
        axis_end = origin + workspace_extent
        if high <= low:
            if center < origin:
                return 0, 0
            if center > axis_end:
                return self.grid_size, self.grid_size
            pixel = math.floor((center - origin) * px_per_m)
            pixel = min(max(pixel, 0), self.grid_size - 1)
            return pixel, pixel + 1

        clipped_low = max(low, origin)
        clipped_high = min(high, axis_end)
        if clipped_high <= clipped_low:
            boundary = 0 if high <= origin else self.grid_size
            return boundary, boundary

        lower = math.floor((clipped_low - origin) * px_per_m)
        upper = math.ceil((clipped_high - origin) * px_per_m)
        lower = min(max(lower, 0), self.grid_size)
        upper = min(max(upper, 0), self.grid_size)
        if upper <= lower:
            if lower >= self.grid_size:
                lower = self.grid_size - 1
                upper = self.grid_size
            else:
                upper = lower + 1
        return lower, upper

    @staticmethod
    def _validate_workspace_geometry(ws: Workspace) -> None:
        values = {
            "origin_x": ws.origin_x,
            "origin_y": ws.origin_y,
            "length": ws.length,
            "breadth": ws.breadth,
        }
        for name, value in values.items():
            try:
                finite = math.isfinite(value)
            except TypeError as error:
                raise ValueError(
                    f"workspace {name} must be finite"
                ) from error
            if not finite:
                raise ValueError(f"workspace {name} must be finite")
        if ws.length <= 0 or ws.breadth <= 0:
            raise ValueError("workspace length and breadth must be positive")

    @staticmethod
    def _validate_rectangle_geometry(
        center_x: float,
        center_y: float,
        length: float,
        breadth: float,
    ) -> None:
        values = {
            "center_x": center_x,
            "center_y": center_y,
            "length": length,
            "breadth": breadth,
        }
        for name, value in values.items():
            try:
                finite = math.isfinite(value)
            except TypeError as error:
                raise ValueError(
                    f"rectangle {name} must be finite"
                ) from error
            if not finite:
                raise ValueError(f"rectangle {name} must be finite")
        if length <= 0 or breadth <= 0:
            raise ValueError("rectangle length and breadth must be positive")

class BaseGridCache:
    def __init__(self, renderer: OccupancyGridRenderer, n_workspaces: int):
        if (
            isinstance(n_workspaces, bool)
            or not isinstance(n_workspaces, Integral)
            or n_workspaces < 0
        ):
            raise ValueError("n_workspaces must be a non-negative integer")
        self._renderer = renderer
        self._n_ws = int(n_workspaces)
        self._cache = np.zeros(
            (
                self._n_ws,
                BASE_CHANNELS,
                renderer.grid_size,
                renderer.grid_size,
            ),
            dtype=np.float32,
        )
        self._dirty = [True] * self._n_ws
        self._env_date: Optional[date] = None
        self._workspace_signature: Optional[tuple[Workspace, ...]] = None

    def invalidate(self, ws_index: int) -> None:
        self._dirty[ws_index] = True

    def invalidate_all(self) -> None:
        self._dirty = [True] * self._n_ws

    def get_base_grids(
        self,
        workspaces: List[Workspace],
        env_date: date,
    ) -> np.ndarray:
        self._prepare_refresh(workspaces, env_date)
        for index, workspace in enumerate(workspaces):
            if self._dirty[index]:
                self._cache[index] = self._renderer.render_base(
                    workspace, env_date
                )
                self._dirty[index] = False
        return self._cache.copy()

    def _prepare_refresh(
        self,
        workspaces: List[Workspace],
        env_date: date,
    ) -> None:
        if len(workspaces) != self._n_ws:
            raise ValueError(
                "workspace count does not match cache construction"
            )
        signature = tuple(workspaces)
        if (
            self._workspace_signature is None
            or any(
                previous is not current
                for previous, current in zip(
                    self._workspace_signature, signature
                )
            )
        ):
            self.invalidate_all()
            self._workspace_signature = signature
        if env_date != self._env_date:
            self.invalidate_all()
            self._env_date = env_date
