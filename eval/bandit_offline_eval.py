from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rl.linucb_bandit import BanditAction, LinUCBBandit


def load_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record["action"] = BanditAction.from_value(record["action"])
                record["context"] = np.asarray(record["context"], dtype=np.float64)
                record["reward"] = float(record["reward"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid bandit log line {line_number}: {error}") from error
            records.append(record)
    if not records:
        raise ValueError("bandit log contains no sessions")
    dimensions = {record["context"].shape for record in records}
    if len(dimensions) != 1 or next(iter(dimensions))[0] <= 0:
        raise ValueError("all logged contexts must have one matching positive dimension")
    return records


def replay(records: list[dict[str, Any]], *, alpha: float = 0.25) -> dict[str, Any]:
    actions = sorted({record["action"] for record in records})
    context_dim = int(records[0]["context"].size)
    policy = LinUCBBandit(actions, context_dim, alpha=alpha)
    observed_cumulative: list[float] = []
    replay_cumulative: list[float] = []
    baseline_cumulative: list[float] = []
    regret_cumulative: list[float] | None = [] if all("optimal_reward" in record for record in records) else None

    mid_rewards = [
        record["reward"] for record in records if record["action"].agency_bucket == "mid"
    ]
    baseline_mean = float(np.mean(mid_rewards)) if mid_rewards else 0.0
    observed_total = replay_total = baseline_total = regret_total = 0.0
    accepted = 0
    matches: list[bool] = []

    for record in records:
        observed_total += record["reward"]
        observed_cumulative.append(observed_total)
        baseline_reward = float(record.get("baseline_reward", baseline_mean))
        baseline_total += baseline_reward
        baseline_cumulative.append(baseline_total)

        selected = policy.select(record["context"])
        matched = selected == record["action"]
        matches.append(matched)
        if matched:
            reward = record["reward"]
            policy.update(selected, record["context"], reward)
            replay_total += reward
            accepted += 1
            if regret_cumulative is not None:
                regret_total += max(0.0, float(record["optimal_reward"]) - reward)
        replay_cumulative.append(replay_total)
        if regret_cumulative is not None:
            regret_cumulative.append(regret_total)

    return {
        "sessions": len(records),
        "actions": [json.loads(action.key) for action in actions],
        "replay_accepted_sessions": accepted,
        "replay_acceptance_rate": accepted / len(records),
        "observed_cumulative_reward": observed_cumulative,
        "replay_cumulative_reward": replay_cumulative,
        "fixed_mid_baseline_method": (
            "per-session logged baseline_reward"
            if all("baseline_reward" in record for record in records)
            else "mean observed reward among mid-agency logged actions"
        ),
        "fixed_mid_baseline_mean_reward": baseline_mean,
        "fixed_mid_baseline_cumulative_reward": baseline_cumulative,
        "cumulative_regret": regret_cumulative,
        "regret_note": (
            "Computed on replay-accepted sessions using logged optimal_reward."
            if regret_cumulative is not None
            else "Unavailable: counterfactual optimal_reward was not logged."
        ),
        "policy_match": matches,
        "final_policy": policy.to_dict(),
    }


def evaluate(log_path: str | Path, *, alpha: float = 0.25) -> dict[str, Any]:
    return replay(load_records(log_path), alpha=alpha)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline rejection-sampling replay for LinUCB logs")
    parser.add_argument("--log", type=Path, default=Path("experiments/bandit_log.jsonl"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.25)
    args = parser.parse_args()
    report = evaluate(args.log, alpha=args.alpha)
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()

