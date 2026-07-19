"""
날짜 기반 배치 시뮬레이터 + 단계별 파이프라인.

C# PlacementSimulator + StagedPlacementPipeline의 Python 재구현.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, List, Optional, Tuple

from . import calendar as cal
from .block import Block, PrePlacedBlock
from .workspace import Workspace


# ── 시뮬레이션 결과 ──────────────────────────────────────────────

@dataclass
class SimulationResult:
    """시뮬레이션 결과."""
    workspaces: List[Workspace]
    blocks: List[Block]
    delay_days: List[int]       # int.MaxValue = 탈락 → 여기서는 큰 정수 사용

    DROPOUT = 2_147_483_647     # C# int.MaxValue 대응


# ── PlacementSimulator ───────────────────────────────────────────

class PlacementSimulator:
    """
    날짜 기반 배치 시뮬레이터.
    C# PlacementSimulator.Replay와 동일한 로직.
    """

    def replay(
        self,
        original_blocks: List[Block],
        original_workspaces: List[Workspace],
        assignments: List[int],
        dropout_threshold: int,
    ) -> SimulationResult:
        # ── 1. Deep Copy ─────────────────────────────────────────
        sim_ws = Workspace.deep_copy_list(original_workspaces)
        sim_blocks = [b.clone() for b in original_blocks]

        # ── 2. 초기화 ───────────────────────────────────────────
        n = len(sim_blocks)
        delay_days = [0] * n
        pending = set(range(n))
        earliest = min(b.in_date for b in sim_blocks)

        # ── 3. 날짜 기반 루프 ────────────────────────────────────
        env_date = cal.adjust_to_working_day(earliest, forward=True)

        while pending:
            # 3-a. 출고 완료 블록 제거
            for ws in sim_ws:
                ws.clear_outgoing_blocks(env_date)

            # 3-b. 당일 배치 대상 수집
            today_targets = [
                idx for idx in pending
                if sim_blocks[idx].in_date <= env_date
            ]

            if not today_targets:
                env_date = cal.next_working_day(env_date)
                continue

            # 3-c. 정렬: 지연 블록 우선(지연일↓), 원래 입고일 순
            def sort_key(idx: int) -> Tuple[int, date]:
                delay = cal.get_working_days_between(
                    original_blocks[idx].in_date, sim_blocks[idx].in_date) - 1
                return (-delay, original_blocks[idx].in_date)

            today_targets.sort(key=sort_key)

            # 3-d. 각 대상 블록 배치 시도
            resolved = []
            for idx in today_targets:
                blk = sim_blocks[idx]
                ws = sim_ws[assignments[idx]]

                cur_delay = cal.get_working_days_between(
                    original_blocks[idx].in_date, blk.in_date) - 1

                # 탈락 판정
                if cur_delay > dropout_threshold:
                    delay_days[idx] = SimulationResult.DROPOUT
                    resolved.append(idx)
                    continue

                trial = blk.clone()
                pos = ws.determine_placement_position(trial, env_date)

                if pos is not None:
                    cx, cy = pos
                    blk.move(cx - blk.ref_x, cy - blk.ref_y)
                    ws.add_block(blk, env_date)
                    delay_days[idx] = cur_delay
                    resolved.append(idx)
                else:
                    blk.delay_placement(1)

            for idx in resolved:
                pending.discard(idx)

            env_date = cal.next_working_day(env_date)

        return SimulationResult(sim_ws, sim_blocks, delay_days)


# ── StagedPlacementPipeline ──────────────────────────────────────

@dataclass
class PlacementStage:
    """배치 단계 정의."""
    name: str
    filter_fn: Optional[Callable[[Block], bool]] = None


class StagedPipeline:
    """
    단계별 배치 파이프라인.
    C# StagedPlacementPipeline과 동일한 로직.
    """

    def __init__(
        self,
        simulator: PlacementSimulator,
        dropout_threshold: int,
    ):
        self._simulator = simulator
        self._dropout_threshold = dropout_threshold
        self._stages: List[PlacementStage] = []

    @property
    def dropout_threshold(self) -> int:
        return self._dropout_threshold

    def add_stage(
        self, name: str, filter_fn: Optional[Callable[[Block], bool]] = None,
    ) -> StagedPipeline:
        self._stages.append(PlacementStage(name, filter_fn))
        return self

    def execute(
        self,
        blocks: List[Block],
        workspaces: List[Workspace],
        assignments: List[int],
    ) -> SimulationResult:
        if not self._stages:
            return self._simulator.replay(
                blocks, workspaces, assignments, self._dropout_threshold)

        total = len(blocks)
        final_blocks: List[Optional[Block]] = [None] * total
        final_delay = [0] * total
        placed = [False] * total

        accumulated_pp: List[PrePlacedBlock] = []
        latest_ws = workspaces

        for stage in self._stages:
            # 1. 이 단계의 배치 대상 선별
            stage_indices = []
            for i in range(total):
                if placed[i]:
                    continue
                if stage.filter_fn is None or stage.filter_fn(blocks[i]):
                    stage_indices.append(i)

            if not stage_indices:
                continue

            # 2. 서브셋 구성
            stage_blocks = [blocks[idx] for idx in stage_indices]
            stage_assignments = [assignments[idx] for idx in stage_indices]

            # 3. 기배치 블록 주입
            stage_ws = Workspace.deep_copy_list(workspaces)
            for pp in accumulated_pp:
                ws_code = pp.label.split("|")[0]
                for ws in stage_ws:
                    if ws.code == ws_code:
                        ws.add_pre_placement(pp)
                        break

            # 4. 시뮬레이션 실행
            result = self._simulator.replay(
                stage_blocks, stage_ws, stage_assignments, self._dropout_threshold)

            # 5. 결과 수집 + 기배치 누적
            for s, orig_idx in enumerate(stage_indices):
                result_block = result.blocks[s]
                final_blocks[orig_idx] = result_block
                final_delay[orig_idx] = result.delay_days[s]
                placed[orig_idx] = True

                if (result.delay_days[s] != SimulationResult.DROPOUT
                        and result_block.workspace_code is not None):
                    pp = PrePlacedBlock(
                        label=f"{result_block.workspace_code}|{result_block.name}",
                        pos_x=result_block.ref_x,
                        pos_y=result_block.ref_y,
                        length=result_block.length,
                        breadth=result_block.breadth,
                        start_date=result_block.in_date,
                        end_date=result_block.out_date,
                    )
                    accumulated_pp.append(pp)

            latest_ws = result.workspaces

        # 미배치 블록 처리
        for i in range(total):
            if not placed[i]:
                final_blocks[i] = blocks[i].clone()
                final_delay[i] = SimulationResult.DROPOUT

        return SimulationResult(
            latest_ws,
            [b for b in final_blocks],  # type: ignore
            final_delay,
        )
