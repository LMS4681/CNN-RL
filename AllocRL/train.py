"""
블록 배치 강화학습 - CNN+MaskablePPO 학습 + ONNX export.

사용법:
    py train.py --data-dir ./data --timesteps 100000

의존성:
    pip install -r requirements.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Windows cp949 콘솔에서 Unicode 출력 에러 방지
if sys.platform == "win32" and os.environ.get("PYTHONIOENCODING") is None:
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np


def mask_fn(env):
    return env.action_masks()


def make_env(blocks, workspaces, strategy, use_synthetic=False, generator=None):
    """환경 팩토리 (SubprocVecEnv용)."""
    from alloc_env.alloc_env import BlockPlacementEnv

    def _init():
        return BlockPlacementEnv(
            blocks, workspaces, strategy,
            use_synthetic=use_synthetic,
            generator=generator,
        )
    return _init


def create_evaluation_env(blocks, workspaces, strategy, grid_size: int = 64):
    """CSV 원본 블록으로 평가하는 마스크 적용 환경을 생성합니다."""
    from sb3_contrib.common.wrappers import ActionMasker

    from alloc_env.alloc_env import BlockPlacementEnv

    env = BlockPlacementEnv(
        blocks,
        workspaces,
        strategy,
        use_synthetic=False,
        grid_size=grid_size,
    )
    return ActionMasker(env, mask_fn)


def train(args):
    """MaskablePPO 학습 실행."""
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    from alloc_env.alloc_env import BlockPlacementEnv
    from alloc_env.data_loader import (
        load_workspaces, load_blocks, apply_allowable_block_patterns,
    )
    from alloc_env.strategy import BaseGridStrategy
    from alloc_env.callbacks import AllocationCallback
    from alloc_env.block_generator import SyntheticBlockGenerator
    from alloc_env.cnn_extractor import OccupancyCnnExtractor

    data_dir = Path(args.data_dir)
    ws_csv   = str(data_dir / "선행건조 작업장 기준정보.csv")
    lot_csv  = str(data_dir / "선행건조 지번 기준정보.csv")
    blk_csv  = str(data_dir / "블록데이터.csv")

    print("=" * 60)
    print("  블록 배치 강화학습 - MaskablePPO")
    print("=" * 60)

    # ── 1. 데이터 로드 ────────────────────────────────────────────
    strategy = BaseGridStrategy(step=5.0)
    workspaces = load_workspaces(ws_csv, lot_csv, strategy)
    apply_allowable_block_patterns(workspaces)
    blocks = load_blocks(blk_csv, workspaces)

    print(f"블록 {len(blocks)}개, 작업장 {len(workspaces)}개")

    # ── 2. Synthetic 블록 생성기 ─────────────────────────────────
    generator = SyntheticBlockGenerator.from_csv(blk_csv)
    print("[Synthetic] CSV 분포 기반 블록 생성기 초기화 완료")

    # ── 3. 환경 생성 (학습: synthetic, 평가: CSV 원본) ────────────
    env = BlockPlacementEnv(
        blocks, workspaces, strategy,
        use_synthetic=True,
        generator=generator,
        synthetic_n_blocks=len(blocks),
        vary_layout=True,
        grid_size=args.grid_size,
    )
    env = ActionMasker(env, mask_fn)

    # 메모리 사용량 예측
    N = len(workspaces)
    G = args.grid_size
    obs_bytes = (10 + N * 3 * G * G + N * 2) * 4  # float32
    buffer_mb = obs_bytes * args.n_steps / 1024 / 1024
    print(f"Obs space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Rollout buffer 예상 메모리: {buffer_mb:.0f} MB (grid={G}×{G}, n_steps={args.n_steps})")

    # ── 3. 출력 디렉토리 사전 생성 ────────────────────────────────
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. 모델 생성 (CNN+MLP 하이브리드) ─────────────────────────
    policy_kwargs = {
        "features_extractor_class": OccupancyCnnExtractor,
        "features_extractor_kwargs": {
            "features_dim": 256,
            "cnn_out_dim": 64,
        },
    }
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(output_dir / "tb_logs"),
    )

    # ── 5. 콜백 설정 ──────────────────────────────────────────────
    callback = AllocationCallback(log_dir=args.output_dir, verbose=1)

    # ── 6. 학습 ──────────────────────────────────────────────────
    print(f"\n학습 시작: {args.timesteps} timesteps")
    print(f"TensorBoard: tensorboard --logdir {Path(args.output_dir) / 'tb_logs'}")
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True,
        callback=callback,
    )

    # ── 7. 모델 저장 ─────────────────────────────────────────────
    sb3_path = str(output_dir / "block_placement_ppo")
    model.save(sb3_path)
    print(f"\nSB3 모델 저장: {sb3_path}")

    # ── 8. ONNX export ───────────────────────────────────────────
    if args.export_onnx:
        onnx_path = str(output_dir / "block_placement_ppo.onnx")
        export_to_onnx(model, env, onnx_path)
        print(f"ONNX 모델 저장: {onnx_path}")

    # ── 9. 학습 결과 평가 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  학습 완료 - 최종 평가")
    print("=" * 60)
    eval_env = create_evaluation_env(
        blocks, workspaces, strategy, grid_size=args.grid_size
    )
    evaluate(model, eval_env, n_eval=args.n_eval)


def evaluate(model, env, n_eval: int = 5):
    """학습된 모델로 n_eval 에피소드 평가."""
    from sb3_contrib import MaskablePPO

    rewards = []
    terminal_rewards = []
    for ep in range(n_eval):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action_masks = env.action_masks() if hasattr(env, 'action_masks') else None
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        terminal_reward = info.get("terminal_reward", total_reward)
        terminal_rewards.append(terminal_reward)
        print(
            f"  Episode {ep+1}: "
            f"total = {total_reward:.2f}, terminal = {terminal_reward:.2f}"
        )

    mean_r = np.mean(rewards)
    mean_terminal = np.mean(terminal_rewards)
    print(
        f"\n  평균 reward: total={mean_r:.2f}, "
        f"terminal={mean_terminal:.2f} (n={n_eval})"
    )
    return mean_r


def export_to_onnx(model, env, onnx_path: str):
    """SB3 모델을 ONNX 형식으로 export (Dict obs 대응)."""
    import torch
    import onnx

    policy = model.policy
    obs_space = env.observation_space

    # Dict obs의 각 키별 더미 입력 생성
    dummy_obs = {}
    if hasattr(obs_space, 'spaces'):
        # Dict observation space
        for key, space in obs_space.spaces.items():
            dummy_obs[key] = torch.zeros(1, *space.shape, device=policy.device)
    else:
        # Flat observation space (fallback)
        dummy_obs = torch.zeros(1, obs_space.shape[0], device=policy.device)

    # Actor 네트워크만 export (추론에 필요한 부분)
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, block, grids, ws_meta):
            obs_dict = {"block": block, "grids": grids, "ws_meta": ws_meta}
            features = self.policy.extract_features(
                obs_dict, self.policy.pi_features_extractor
            )
            latent_pi = self.policy.mlp_extractor.forward_actor(features)
            return self.policy.action_net(latent_pi)

    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    # Dict obs를 개별 인자로 전달
    dummy_inputs = (
        dummy_obs["block"],
        dummy_obs["grids"],
        dummy_obs["ws_meta"],
    )

    torch.onnx.export(
        wrapper,
        dummy_inputs,
        onnx_path,
        input_names=["block", "grids", "ws_meta"],
        output_names=["action_logits"],
        dynamic_axes={
            "block": {0: "batch"},
            "grids": {0: "batch"},
            "ws_meta": {0: "batch"},
            "action_logits": {0: "batch"},
        },
        opset_version=18,
    )

    # 검증
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX inputs: {[inp.name for inp in onnx_model.graph.input]}")


def main():
    parser = argparse.ArgumentParser(description="블록 배치 RL 학습")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="CSV 데이터 디렉토리 경로")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="모델 출력 디렉토리")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="총 학습 타임스텝")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="학습률 (learning rate)")
    parser.add_argument("--n-steps", type=int, default=554,
                        help="PPO n_steps (에피소드 길이×2 권장)")
    parser.add_argument("--grid-size", type=int, default=64,
                        help="점유 그리드 해상도 (64 or 128, 메모리에 영향)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="미니배치 크기")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="PPO epochs per update")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="감가율 (discount factor)")
    parser.add_argument("--n-eval", type=int, default=5,
                        help="평가 에피소드 수")
    parser.add_argument("--export-onnx", action="store_true", default=True,
                        help="ONNX export 수행")
    parser.add_argument("--no-export-onnx", action="store_false", dest="export_onnx")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
