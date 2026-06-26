# -*- coding: utf-8 -*-
"""
CNN 입력용 점유 그리드 시각화 — 3채널 그리드를 이미지로 저장.

각 작업장의 3채널 점유 그리드를 컬러 이미지로 변환하여 저장합니다.
- Ch0 (점유 마스크): 빨간색
- Ch1 (잔여 공기):   초록색  (밝을수록 출고까지 여유 있음)
- Ch2 (경계 마스크): 파란색

실행:
  py visualize_grids.py --data-dir ./data --output-dir ./output/grid_images

옵션:
  --grid-size 64     그리드 해상도 (기본 64)
  --synthetic        합성 블록+레이아웃으로 렌더링
  --date 2026-04-15  환경 날짜 지정
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
from datetime import date, datetime
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
os.chdir(BASE)


def visualize_grids(args):
    import matplotlib
    matplotlib.use('Agg')  # GUI 없는 백엔드
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.font_manager as fm

    # 한글 폰트 설정 (Windows: Malgun Gothic)
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
    plt.rcParams['axes.unicode_minus'] = False

    from alloc_env.data_loader import load_workspaces, load_blocks
    from alloc_env.strategy import BaseGridStrategy
    from alloc_env.occupancy_grid import OccupancyGridRenderer
    from alloc_env.block_generator import SyntheticBlockGenerator

    data_dir = Path(args.data_dir)
    ws_csv  = str(data_dir / "선행건조 작업장 기준정보.csv")
    lot_csv = str(data_dir / "선행건조 지번 기준정보.csv")
    blk_csv = str(data_dir / "블록데이터.csv")

    # ── 1. 데이터 로드 ────────────────────────────────────────────
    strategy = BaseGridStrategy(step=5.0)
    workspaces = load_workspaces(ws_csv, lot_csv, strategy)
    blocks = load_blocks(blk_csv, workspaces)
    print(f"작업장 {len(workspaces)}개, 블록 {len(blocks)}개 로드")

    # ── 2. 환경 날짜 ──────────────────────────────────────────────
    if args.date:
        env_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        env_date = min(b.in_date for b in blocks) if blocks else date(2026, 4, 1)
    print(f"환경 날짜: {env_date}")

    # ── 3. 합성 모드 (선택) ───────────────────────────────────────
    if args.synthetic:
        gen = SyntheticBlockGenerator.from_csv(blk_csv, seed=42)
        workspaces = gen.generate_workspaces(workspaces)

        # 기배치 블록 합성 (각 작업장에 3~5개)
        n_pre = len(workspaces) * 4
        preplaced = gen.generate_preplaced(n_pre, workspaces, env_date)
        ws_map = {ws.code: ws for ws in workspaces}
        for ws_code, pp in preplaced:
            if ws_code in ws_map:
                ws_map[ws_code].add_pre_placement(pp)
        print(f"[Synthetic] 작업장 변형 + 기배치 {n_pre}개 생성")

    # ── 4. 렌더링 ─────────────────────────────────────────────────
    G = args.grid_size
    renderer = OccupancyGridRenderer(grid_size=G)
    grids = renderer.render_all(workspaces, env_date)  # (N, 3, G, G)
    print(f"그리드 shape: {grids.shape}")

    # ── 5. 출력 디렉토리 ──────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 6-A. 개별 작업장 이미지 저장 ──────────────────────────────
    individual_dir = out_dir / "individual"
    individual_dir.mkdir(exist_ok=True)

    for i, ws in enumerate(workspaces):
        grid = grids[i]  # (3, G, G)
        scale = renderer.compute_scale_value(ws)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(
            f"{ws.code} ({ws.name})  |  "
            f"{ws.length:.0f}×{ws.breadth:.0f}m  |  "
            f"scale={scale:.3f} m/px  |  date={env_date}",
            fontsize=13, fontweight='bold'
        )

        # Ch0: 점유 마스크
        ax = axes[0]
        ax.imshow(grid[0], cmap='Reds', vmin=0, vmax=1, origin='lower')
        ax.set_title("Ch0: 점유 마스크", fontsize=11)
        ax.set_xlabel(f"occupied={grid[0].sum():.0f}px")

        # Ch1: 잔여 공기
        ax = axes[1]
        im1 = ax.imshow(grid[1], cmap='YlGn', vmin=0, vmax=1, origin='lower')
        ax.set_title("Ch1: 잔여 출고 공기", fontsize=11)
        plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        # Ch2: 경계 마스크
        ax = axes[2]
        ax.imshow(grid[2], cmap='Blues', vmin=0, vmax=1, origin='lower')
        ax.set_title("Ch2: 작업장 경계", fontsize=11)
        boundary_ratio = grid[2].sum() / (G * G)
        ax.set_xlabel(f"경계 내부={boundary_ratio:.1%}")

        # RGB 합성 (3채널 → 컬러)
        ax = axes[3]
        rgb = np.stack([grid[0], grid[1], grid[2]], axis=-1)  # R=점유, G=잔여, B=경계
        ax.imshow(rgb, origin='lower')
        ax.set_title("RGB 합성 (R:점유, G:잔여, B:경계)", fontsize=11)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        img_path = individual_dir / f"{ws.code}_{G}x{G}.png"
        fig.savefig(str(img_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  저장: {img_path.name}")

    # ── 6-B. 전체 작업장 그리드 오버뷰 ────────────────────────────
    n_ws = len(workspaces)
    cols = min(6, n_ws)
    rows = (n_ws + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    fig.suptitle(
        f"전체 작업장 점유 그리드 오버뷰  |  {G}×{G}  |  date={env_date}",
        fontsize=14, fontweight='bold'
    )

    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i in range(len(axes)):
        ax = axes[i]
        if i < n_ws:
            ws = workspaces[i]
            grid = grids[i]
            rgb = np.stack([grid[0], grid[1], grid[2]], axis=-1)
            ax.imshow(rgb, origin='lower')
            occ = grid[0].sum() / max(grid[2].sum(), 1) * 100
            ax.set_title(f"{ws.code}\n{ws.length:.0f}×{ws.breadth:.0f}m\nocc={occ:.0f}%",
                         fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= n_ws:
            ax.set_visible(False)

    # 범례
    legend_patches = [
        mpatches.Patch(color='red', label='Ch0: 블록 점유'),
        mpatches.Patch(color='green', label='Ch1: 잔여 공기'),
        mpatches.Patch(color='blue', label='Ch2: 작업장 경계'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    overview_path = out_dir / f"grid_overview_{G}x{G}.png"
    fig.savefig(str(overview_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n오버뷰 저장: {overview_path}")

    # ── 7. 통계 요약 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  그리드 통계 요약 ({G}×{G})")
    print(f"{'='*60}")
    print(f"{'작업장':>8} {'크기(m)':>14} {'scale':>8} {'경계%':>6} {'점유%':>6} {'잔여공기':>8}")
    print(f"{'-'*60}")
    for i, ws in enumerate(workspaces):
        grid = grids[i]
        boundary = grid[2].sum() / (G * G) * 100
        occ_in_boundary = grid[0].sum() / max(grid[2].sum(), 1) * 100
        avg_ttl = grid[1][grid[1] > 0].mean() * 60 if grid[1].any() else 0
        scale = renderer.compute_scale_value(ws)
        print(f"{ws.code:>8} {ws.length:>6.0f}×{ws.breadth:<6.0f} "
              f"{scale:>7.3f} {boundary:>5.1f}% {occ_in_boundary:>5.1f}% "
              f"{avg_ttl:>6.1f}일")

    print(f"\n총 이미지 {n_ws + 1}개 저장 → {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="CNN 입력용 점유 그리드 시각화")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./output/grid_images")
    parser.add_argument("--grid-size", type=int, default=64,
                        help="그리드 해상도 (64 or 128)")
    parser.add_argument("--synthetic", action="store_true",
                        help="합성 블록+레이아웃으로 렌더링")
    parser.add_argument("--date", type=str, default=None,
                        help="환경 날짜 (YYYY-MM-DD)")
    args = parser.parse_args()
    visualize_grids(args)


if __name__ == "__main__":
    main()
