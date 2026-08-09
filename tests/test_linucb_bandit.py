import numpy as np

from eval.bandit_offline_eval import replay
from src.rl.linucb_bandit import (
    BanditAction,
    ContextEncoder,
    LinUCBBandit,
    reward_from_feedback,
)


def actions() -> tuple[BanditAction, BanditAction]:
    return (
        BanditAction("generic", "mid", "minimal"),
        BanditAction("personalized-art", "mid", "travel-poster"),
    )


def test_context_reward_and_linucb_matrix_update() -> None:
    context = ContextEncoder().encode("partner", 5, "casual", 0.5)
    assert context.shape == (ContextEncoder.dimension,)
    assert context[0] == 1.0
    assert reward_from_feedback(5, accept_count=4) == 1.0

    policy = LinUCBBandit(actions(), context.size, alpha=0.2)
    before_a, before_b = policy.parameters(actions()[0])
    policy.update(actions()[0], context, 0.75)
    after_a, after_b = policy.parameters(actions()[0])
    assert np.allclose(after_a, before_a + np.outer(context, context))
    assert np.allclose(after_b, before_b + 0.75 * context)


def test_linucb_converges_to_higher_reward_arm() -> None:
    rng = np.random.default_rng(2026)
    low_reward, high_reward = actions()
    policy = LinUCBBandit(actions(), context_dim=3, alpha=0.8)
    context = np.array([1.0, 0.4, 0.5])
    selected_history: list[BanditAction] = []

    for _ in range(1200):
        selected = policy.select(context)
        probability = 0.2 if selected == low_reward else 0.8
        reward = float(rng.random() < probability)
        policy.update(selected, context, reward)
        selected_history.append(selected)

    assert policy.select(context) == high_reward
    assert selected_history[-200:].count(high_reward) >= 190
    assert policy.counts[high_reward] > policy.counts[low_reward]


def test_offline_replay_reports_rewards_baseline_and_oracle_regret() -> None:
    low_reward, high_reward = actions()
    records = []
    for index in range(20):
        action = low_reward if index % 2 == 0 else high_reward
        reward = 0.2 if action == low_reward else 0.8
        records.append({
            "context": np.array([1.0, 0.5]),
            "action": action,
            "reward": reward,
            "baseline_reward": 0.5,
            "optimal_reward": 0.8,
        })
    report = replay(records, alpha=0.8)
    assert report["sessions"] == 20
    assert report["replay_accepted_sessions"] > 0
    assert np.isclose(report["observed_cumulative_reward"][-1], 10.0)
    assert np.isclose(report["fixed_mid_baseline_cumulative_reward"][-1], 10.0)
    assert report["cumulative_regret"] is not None



def test_reward_uses_clip_score_when_available() -> None:
    reward = reward_from_feedback(5, accept_count=2, edit_count=0, regenerate_count=0, clip_score=0.8)
    assert np.isclose(reward, 0.35 + 0.25 + 0.2)

