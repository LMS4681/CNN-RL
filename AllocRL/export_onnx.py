"""저장된 SB3 CNN+MaskablePPO 모델을 ONNX로 export하는 스크립트."""
import os
from pathlib import Path

# 스크립트 기준 디렉토리로 CWD 전환
BASE = Path(__file__).resolve().parent
os.chdir(BASE)

from train import export_to_onnx
from sb3_contrib import MaskablePPO
from alloc_env.alloc_env import BlockPlacementEnv
from alloc_env.data_loader import load_workspaces, load_blocks
from alloc_env.strategy import BaseGridStrategy

strategy = BaseGridStrategy(step=5.0)
data_dir = BASE / "data"
ws = load_workspaces(
    str(data_dir / "선행건조 작업장 기준정보.csv"),
    str(data_dir / "선행건조 지번 기준정보.csv"),
    strategy,
)
blocks = load_blocks(str(data_dir / "블록데이터.csv"), ws)

# CNN 관측 환경으로 생성 (Dict obs)
env = BlockPlacementEnv(blocks, ws, strategy)

out_dir = BASE / "output"
model = MaskablePPO.load(str(out_dir / "block_placement_ppo"))
onnx_path = str(out_dir / "block_placement_ppo.onnx")
export_to_onnx(model, env, onnx_path)
print(f"ONNX export OK: {onnx_path}")
