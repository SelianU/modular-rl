"""
D-REX "Better-than-Demonstrator" 실험

Phase 4 학습 중 진짜 CartPole 보상으로 주기적으로 평가해서
D-REX 에이전트가 시연자 성능을 능가하는 시점을 추적합니다.

결과물: drex_long_results.png
"""
import random
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn

from modular_rl.networks import MLP, QHead
from modular_rl.algorithms import DQNConfig, ReplayBuffer, DQNAgent, QNetwork
from modular_rl.algorithms.reward import DemonstrationBuffer, RewardModel, RewardModelTrainer
from modular_rl.training import (
    GymEnvWrapper, Trainer, LearnedRewardWrapper,
    ConsoleLogger, MatplotlibLogger, CompositeLogger,
)
from modular_rl.training.logger import BaseLogger

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

ENV_NAME = "CartPole-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── helpers ──────────────────────────────────────────────────────────────────

def make_dqn_agent(environment, config, hidden_dims):
    backbone = MLP(environment.state_dim, hidden_dims)
    q_network = QNetwork(backbone, QHead(backbone.output_dim, environment.action_dim))
    target_backbone = MLP(environment.state_dim, hidden_dims)
    target_q_network = QNetwork(
        target_backbone,
        QHead(target_backbone.output_dim, environment.action_dim),
    )
    return DQNAgent(
        q_network=q_network, target_network=target_q_network,
        optimizer=torch.optim.Adam(q_network.parameters(), lr=config.learning_rate),
        criterion=nn.SmoothL1Loss(),
        replay_buffer=ReplayBuffer(config.buffer_size, config.device),
        config=config, action_dim=environment.action_dim,
    )


def true_eval(agent, n_episodes=30):
    """진짜 CartPole 보상(에피소드 길이)으로 평가."""
    environment = gym.make(ENV_NAME)
    rewards = []
    for _ in range(n_episodes):
        state, _ = environment.reset()
        episode_reward, done = 0, False
        while not done:
            action = agent.select_action(state, evaluation=True)
            state, reward, terminated, truncated, _ = environment.step(action)
            episode_reward += reward
            done = terminated or truncated
        rewards.append(episode_reward)
    environment.close()
    return np.array(rewards)


# ── custom logger: 학습 중 진짜 환경으로 주기적 평가 ─────────────────────────

class TrueEvalLogger(BaseLogger):
    """
    ConsoleLogger + 진짜 환경 주기적 평가.
    agent 속성은 Trainer 생성 후 외부에서 설정.
    """
    def __init__(
        self,
        agent_ref: list,          # [agent] — mutable container
        demonstrator_mean: float,
        eval_every_n_ep: int = 50,
        log_interval: int = 10,
        prefix: str = "",
    ):
        self._agent_ref       = agent_ref
        self.demonstrator_mean = demonstrator_mean
        self.eval_every        = eval_every_n_ep
        self.log_interval      = log_interval
        self.prefix            = prefix
        self.true_history: list = []   # (episode, step, mean_true_reward)
        self.surpassed_ep      = None
        self._recent: list     = []

    def log_episode(self, episode: int, step: int, metrics: dict):
        reward = metrics.get("reward", 0.0)
        self._recent.append(reward)
        if len(self._recent) > 100:
            self._recent.pop(0)

        if episode % self.log_interval == 0:
            parts = [
                f"{self.prefix}Ep {episode:4d}",
                f"Step {step:6d}",
                f"LearnedR {reward:7.2f}",
                f"Mean(100) {np.mean(self._recent):7.2f}",
            ]
            for metric_name, metric_value in metrics.items():
                if metric_name == "reward":
                    continue
                parts.append(f"{metric_name} {metric_value:.4f}")
            print(" | ".join(parts))

        # 진짜 환경 평가
        if episode % self.eval_every == 0:
            agent = self._agent_ref[0]
            rewards = true_eval(agent, n_episodes=30)
            mean_reward = rewards.mean()
            self.true_history.append((episode, step, mean_reward))

            surpass = ""
            if mean_reward > self.demonstrator_mean and self.surpassed_ep is None:
                self.surpassed_ep = episode
                surpass = "  ★ 시연자 초과!"

            print(
                f"  [True eval] Ep {episode:4d} | Step {step:6d} | "
                f"True mean: {mean_reward:6.1f} / Demo: {self.demonstrator_mean:.1f}{surpass}"
            )

    def close(self):
        pass


# ── Phase 1 ──────────────────────────────────────────────────────────────────

def phase1_train_reference():
    print("\n" + "═" * 60)
    print("Phase 1 │ 레퍼런스 DQN 학습 (진짜 보상, 30k steps)")
    print("═" * 60)
    config = DQNConfig(
        env_name=ENV_NAME, total_timesteps=30_000,
        buffer_size=15_000, batch_size=64, learning_rate=1e-3,
        learning_starts=500, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_steps=15_000, target_update_freq=200, log_interval=30,
    )
    environment = GymEnvWrapper(ENV_NAME)
    agent = make_dqn_agent(environment, config, hidden_dims=[128, 128])
    Trainer(agent, environment, config, ConsoleLogger(log_interval=30, prefix="Ref ")).train()
    return agent


# ── Phase 2 ──────────────────────────────────────────────────────────────────

def phase2_collect_demos(reference_agent, n_levels=6, n_per_level=15):
    print("\n" + "═" * 60)
    print("Phase 2 │ 랭킹된 시연 수집")
    print("═" * 60)
    demonstration_buffer = DemonstrationBuffer()
    # rank 0 = worst (eps=1.0, random), rank n_levels-1 = best (eps=0.0, greedy)
    epsilons = np.linspace(1.0, 0.0, n_levels)   # [1.0 … 0.0]

    environment = gym.make(ENV_NAME)
    lengths = {}
    for rank, eps in enumerate(epsilons):   # rank 0 → eps=1.0 (worst)
        episode_lengths = []
        for _ in range(n_per_level):
            states, actions = [], []
            state, _ = environment.reset()
            done = False
            while not done:
                if random.random() < eps:
                    action = environment.action_space.sample()
                else:
                    action = reference_agent.select_action(state, evaluation=True)
                states.append(state)
                actions.append(action)
                state, _, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
            demonstration_buffer.add_trajectory(
                np.array(states,  dtype=np.float32),
                np.array(actions, dtype=np.int64),
                rank=rank,
            )
            episode_lengths.append(len(states))
        lengths[rank] = episode_lengths
        print(f"  Rank {rank:2d} (ε={eps:.2f}) | {n_per_level} eps | "
              f"avg {np.mean(episode_lengths):.1f} / max {np.max(episode_lengths):.0f}")

    environment.close()
    # 최고 랭크(rank = n_levels-1)의 평균 = 시연자 기준 성능
    best_rank = n_levels - 1
    demonstrator_mean = float(np.mean(lengths[best_rank]))
    print(f"\n  시연자 기준 성능 (rank {best_rank}): {demonstrator_mean:.1f} steps")
    return demonstration_buffer, demonstrator_mean


# ── Phase 3 ──────────────────────────────────────────────────────────────────

def phase3_train_reward(demonstration_buffer, state_dim, n_epochs=1000):
    print("\n" + "═" * 60)
    print("Phase 3 │ 보상 모델 학습 (Bradley-Terry, 1000 epochs)")
    print("═" * 60)
    backbone = MLP(input_dim=state_dim, hidden_dims=[128, 128])
    reward_model = RewardModel(backbone, action_dim=0)
    trainer = RewardModelTrainer(
        reward_model=reward_model,
        optimizer=torch.optim.Adam(reward_model.parameters(), lr=1e-3),
        l2_coeff=1e-4,
        device=DEVICE,
    )
    losses = trainer.train(demonstration_buffer, n_epochs=n_epochs, batch_size=64, log_interval=200)
    print(f"  최종 loss: {losses[-1]:.4f}")
    return reward_model, losses


# ── Phase 4 ──────────────────────────────────────────────────────────────────

def phase4_train_drex(reward_model, demonstrator_mean):
    print("\n" + "═" * 60)
    print("Phase 4 │ D-REX 에이전트 학습 (학습된 보상, 300k steps)")
    print(f"         │ 시연자 기준: {demonstrator_mean:.1f} steps")
    print("═" * 60)

    config = DQNConfig(
        env_name=ENV_NAME, total_timesteps=300_000,
        buffer_size=50_000, batch_size=128, learning_rate=5e-4,
        learning_starts=1_000, epsilon_start=1.0, epsilon_end=0.01,
        epsilon_decay_steps=100_000, target_update_freq=500,
        log_interval=50,
    )

    base_environment = GymEnvWrapper(ENV_NAME)
    learned_environment = LearnedRewardWrapper(base_environment, reward_model, device=DEVICE)
    agent = make_dqn_agent(learned_environment, config, hidden_dims=[128, 128])

    agent_ref = [agent]   # mutable container for logger
    eval_logger = TrueEvalLogger(
        agent_ref=agent_ref,
        demonstrator_mean=demonstrator_mean,
        eval_every_n_ep=50,
        log_interval=config.log_interval,
        prefix="D-REX ",
    )

    Trainer(
        agent, learned_environment, config,
        eval_logger,
        save_path="checkpoints/drex_long_cartpole.pt",
    ).train()

    return agent, eval_logger.true_history, eval_logger.surpassed_ep


# ── 결과 플롯 ─────────────────────────────────────────────────────────────────

def plot_results(true_history, demonstrator_mean, reward_losses, surpassed_ep):
    episodes = [h[0] for h in true_history]
    means    = [h[2] for h in true_history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── 왼쪽: 진짜 성능 vs 시연자 ────────────────────────────────────────────
    ax1.plot(episodes, means, color="royalblue", linewidth=2, marker="o",
             markersize=4, label="D-REX (진짜 성능)")

    smooth_w = min(5, len(means))
    if len(means) >= smooth_w:
        sm = np.convolve(means, np.ones(smooth_w) / smooth_w, mode="valid")
        ax1.plot(episodes[smooth_w - 1:], sm, color="orange",
                 linewidth=2.5, label="Smoothed")

    ax1.axhline(demonstrator_mean, color="red", linestyle="--", linewidth=2,
                label=f"시연자 기준 ({demonstrator_mean:.0f})")
    ax1.axhline(195, color="gray", linestyle=":", linewidth=1.5,
                label="CartPole 해결 기준 (195)")
    ax1.axhline(500, color="green", linestyle=":", linewidth=1.5,
                label="CartPole 최대 (500)")

    if surpassed_ep:
        idx = next(i for i, h in enumerate(true_history) if h[0] >= surpassed_ep)
        ax1.annotate(
            f"★ 시연자 초과\n(ep {surpassed_ep})",
            xy=(true_history[idx][0], true_history[idx][2]),
            xytext=(true_history[idx][0] + max(episodes) * 0.05,
                    true_history[idx][2] + demonstrator_mean * 0.1),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10, color="red",
        )

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("True Episode Length (steps)")
    ax1.set_title("D-REX vs Demonstrator Performance")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 520)

    # ── 오른쪽: 보상 모델 손실 ───────────────────────────────────────────────
    ax2.plot(reward_losses, color="purple", alpha=0.5, linewidth=1)
    sm_loss = np.convolve(reward_losses, np.ones(30) / 30, mode="valid")
    ax2.plot(range(29, len(reward_losses)), sm_loss, color="purple",
             linewidth=2, label="Smoothed")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Bradley-Terry Loss")
    ax2.set_title("Reward Model Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("drex_long_results.png", dpi=150)
    plt.close()
    print("  → drex_long_results.png 저장")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    reference_agent = phase1_train_reference()
    demonstration_buffer, demonstrator_mean = phase2_collect_demos(
        reference_agent,
        n_levels=6,
        n_per_level=15,
    )
    reward_model, reward_losses = phase3_train_reward(
        demonstration_buffer,
        state_dim=4,
        n_epochs=1000,
    )
    drex_agent, true_history, surpassed_episode = phase4_train_drex(
        reward_model,
        demonstrator_mean,
    )

    print("\n" + "═" * 60)
    print("최종 결과")
    print("═" * 60)
    final_rewards = true_eval(drex_agent, n_episodes=50)
    print(f"  D-REX 최종 True Mean : {final_rewards.mean():.1f} ± {final_rewards.std():.1f}")
    print(f"  시연자 기준           : {demonstrator_mean:.1f}")
    print(
        f"  시연자 초과 시점      : ep {surpassed_episode}"
        if surpassed_episode
        else "  시연자 미초과 (학습 연장 필요)"
    )

    plot_results(true_history, demonstrator_mean, reward_losses, surpassed_episode)


if __name__ == "__main__":
    main()
