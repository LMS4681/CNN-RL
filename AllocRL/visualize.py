"""
학습 결과 시각화 스크립트.

사용법:
    py visualize.py --log-dir ./output
    py visualize.py --log-dir ./output --tensorboard   # TensorBoard 실행
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'   # 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False


def load_csv_log(log_path: str) -> dict:
    """training_log.csv를 읽어 딕셔너리로 반환."""
    data = {
        "episode": [], "timestep": [], "reward": [],
        "delayed_count": [], "dropout_count": [],
        "total_delay_days": [], "success_rate": [],
    }
    with open(log_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["episode"].append(int(row["episode"]))
            data["timestep"].append(int(row["timestep"]))
            data["reward"].append(float(row["reward"]))
            data["delayed_count"].append(int(row["delayed_count"]))
            data["dropout_count"].append(int(row["dropout_count"]))
            data["total_delay_days"].append(int(row["total_delay_days"]))
            data["success_rate"].append(float(row["success_rate"]))
    return data


def plot_training_results(data: dict, output_dir: str):
    """학습 결과를 4개 서브플롯으로 시각화."""
    episodes = data["episode"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("블록 배치 RL 학습 결과", fontsize=16, fontweight="bold")

    # 1. Reward 추이
    ax = axes[0, 0]
    ax.plot(episodes, data["reward"], alpha=0.3, color="steelblue", linewidth=0.8)
    if len(episodes) >= 10:
        window = max(1, len(episodes) // 20)
        smoothed = _moving_average(data["reward"], window)
        ax.plot(episodes[:len(smoothed)], smoothed,
                color="navy", linewidth=2, label=f"이동평균 (w={window})")
        ax.legend()
    ax.set_title("에피소드 보상 (Reward)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # 2. 지연/탈락 블록 수
    ax = axes[0, 1]
    ax.plot(episodes, data["delayed_count"],
            label="지연 블록", color="orange", alpha=0.6, linewidth=1)
    ax.plot(episodes, data["dropout_count"],
            label="탈락 블록", color="crimson", alpha=0.6, linewidth=1)
    if len(episodes) >= 10:
        window = max(1, len(episodes) // 20)
        sm_delay = _moving_average(data["delayed_count"], window)
        sm_drop = _moving_average(data["dropout_count"], window)
        ax.plot(episodes[:len(sm_delay)], sm_delay,
                color="darkorange", linewidth=2)
        ax.plot(episodes[:len(sm_drop)], sm_drop,
                color="darkred", linewidth=2)
    ax.set_title("지연 / 탈락 블록 수")
    ax.set_xlabel("Episode")
    ax.set_ylabel("블록 수")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 성공률
    ax = axes[1, 0]
    ax.plot(episodes, [s * 100 for s in data["success_rate"]],
            color="seagreen", alpha=0.4, linewidth=0.8)
    if len(episodes) >= 10:
        window = max(1, len(episodes) // 20)
        sm_success = _moving_average(
            [s * 100 for s in data["success_rate"]], window)
        ax.plot(episodes[:len(sm_success)], sm_success,
                color="darkgreen", linewidth=2, label=f"이동평균 (w={window})")
        ax.legend()
    ax.set_title("정상 배치 성공률 (%)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("성공률 (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)

    # 4. 총 지연 일수
    ax = axes[1, 1]
    ax.plot(episodes, data["total_delay_days"],
            color="mediumpurple", alpha=0.4, linewidth=0.8)
    if len(episodes) >= 10:
        window = max(1, len(episodes) // 20)
        sm_delay = _moving_average(data["total_delay_days"], window)
        ax.plot(episodes[:len(sm_delay)], sm_delay,
                color="indigo", linewidth=2, label=f"이동평균 (w={window})")
        ax.legend()
    ax.set_title("총 지연 일수 합")
    ax.set_xlabel("Episode")
    ax.set_ylabel("지연 일수")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(output_dir) / "training_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"그래프 저장: {save_path}")
    plt.show()


def _moving_average(values, window):
    """간단한 이동평균."""
    if window <= 1:
        return values
    result = []
    for i in range(len(values) - window + 1):
        result.append(sum(values[i:i+window]) / window)
    return result


def launch_tensorboard(log_dir: str):
    """TensorBoard를 실행합니다."""
    import subprocess
    tb_dir = str(Path(log_dir) / "tb_logs")
    print(f"TensorBoard 실행: {tb_dir}")
    print("브라우저에서 http://localhost:6006 을 열어주세요")
    subprocess.Popen(["tensorboard", "--logdir", tb_dir, "--port", "6006"])


def main():
    parser = argparse.ArgumentParser(description="RL 학습 결과 시각화")
    parser.add_argument("--log-dir", type=str, default="./output",
                        help="로그 디렉토리")
    parser.add_argument("--tensorboard", action="store_true",
                        help="TensorBoard 실행")
    args = parser.parse_args()

    if args.tensorboard:
        launch_tensorboard(args.log_dir)
        return

    csv_path = Path(args.log_dir) / "training_log.csv"
    if not csv_path.exists():
        print(f"로그 파일을 찾을 수 없습니다: {csv_path}")
        print("먼저 train.py로 학습을 실행해 주세요.")
        return

    data = load_csv_log(str(csv_path))
    if not data["episode"]:
        print("로그 데이터가 비어 있습니다.")
        return

    print(f"총 {len(data['episode'])} 에피소드 로드")
    plot_training_results(data, args.log_dir)


if __name__ == "__main__":
    main()
