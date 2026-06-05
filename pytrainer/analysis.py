from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np

from environment import Environment
from learning_utils import BUCKET_COUNT, HIT_REWARD, JUMP_HIT_MULTIPLIER, JUMP_MISS_PENALTY, reward_function
from qlearning import QTable


WIDTH = 1000
HEIGHT = 1000
ACTIONS = ["PowerLeft", "PowerRight", "Jump", "No Power"]
NUM_STATES = BUCKET_COUNT * BUCKET_COUNT
BASE_DIR = Path(__file__).resolve().parent
TABLES_DIR = BASE_DIR / "tables"
DEFAULT_CHECKPOINT = TABLES_DIR / "hypertables" / "winner_1000000_hr15_a0p2_g0p9_e0p1.npy"
DEFAULT_HYPERTABLE_DIR = TABLES_DIR / "hypertables"
EVOLUTION_DIR = TABLES_DIR / "evolution"
DEFAULT_HIT_REWARDS = [10.0, 15.0, 25.0]
DEFAULT_ALPHAS = [0.05, 0.1, 0.2]
DEFAULT_GAMMAS = [0.90, 0.95, 0.99]
DEFAULT_EPSILONS = [0.01, 0.05, 0.1, 0.2]
WINNING_PARAMS = {
    "hit_reward": 15.0,
    "alpha": 0.2,
    "gamma": 0.9,
    "epsilon": 0.1,
}


def state_index(state_key: str) -> int:
    ball_bucket, agent_bucket = (int(value) for value in state_key.split("-"))
    return ball_bucket * BUCKET_COUNT + agent_bucket


def greedy_action_from_table(q_table: np.ndarray, state_key: str) -> str:
    row = q_table[state_index(state_key)]
    return ACTIONS[int(np.argmax(row))]


def slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class HyperParams:
    hit_reward: float
    alpha: float
    gamma: float
    epsilon: float
    jump_hit_multiplier: float = JUMP_HIT_MULTIPLIER
    jump_miss_penalty: float = JUMP_MISS_PENALTY

    def filename_stem(self) -> str:
        return (
            f"hr{slug_float(self.hit_reward)}"
            f"_a{slug_float(self.alpha)}"
            f"_g{slug_float(self.gamma)}"
            f"_e{slug_float(self.epsilon)}"
        )

    def reward_kwargs(self) -> dict[str, float]:
        return {
            "hit_reward": self.hit_reward,
            "jump_hit_multiplier": self.jump_hit_multiplier,
            "jump_miss_penalty": self.jump_miss_penalty,
        }

    def as_dict(self) -> dict[str, float]:
        return {
            "hit_reward": self.hit_reward,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "jump_hit_multiplier": self.jump_hit_multiplier,
            "jump_miss_penalty": self.jump_miss_penalty,
        }


FIXED_SCREENSHOT_PARAMS = HyperParams(**WINNING_PARAMS)
EVOLUTION_PARAMS = FIXED_SCREENSHOT_PARAMS
EVOLUTION_STEPS = (10, 100, 1000, 10_000, 100_000)


@dataclass
class EpisodeResult:
    steps: int
    total_reward: float
    hits: int
    first_hit_step: int | None
    terminated: bool


@dataclass
class PolicySummary:
    name: str
    episodes: int
    total_steps: int = 0
    total_reward: float = 0.0
    total_hits: int = 0
    episodes_with_hit: int = 0
    first_hit_steps: list[int] = field(default_factory=list)
    terminated_episodes: int = 0

    def add_episode(self, result: EpisodeResult) -> None:
        self.total_steps += result.steps
        self.total_reward += result.total_reward
        self.total_hits += result.hits
        if result.hits > 0:
            self.episodes_with_hit += 1
            if result.first_hit_step is not None:
                self.first_hit_steps.append(result.first_hit_step)
        if result.terminated:
            self.terminated_episodes += 1

    def as_metrics(self) -> dict[str, float | int | None]:
        avg_steps = self.total_steps / self.episodes if self.episodes else 0.0
        avg_reward_per_episode = self.total_reward / self.episodes if self.episodes else 0.0
        avg_reward_per_step = self.total_reward / self.total_steps if self.total_steps else 0.0
        hit_rate = self.episodes_with_hit / self.episodes if self.episodes else 0.0
        avg_hits_per_episode = self.total_hits / self.episodes if self.episodes else 0.0
        avg_first_hit = (
            sum(self.first_hit_steps) / len(self.first_hit_steps)
            if self.first_hit_steps
            else None
        )
        terminated_rate = self.terminated_episodes / self.episodes if self.episodes else 0.0
        return {
            "avg_steps": avg_steps,
            "avg_reward_per_episode": avg_reward_per_episode,
            "avg_reward_per_step": avg_reward_per_step,
            "hit_rate": hit_rate,
            "avg_hits_per_episode": avg_hits_per_episode,
            "avg_first_hit": avg_first_hit,
            "terminated_rate": terminated_rate,
            "total_reward": self.total_reward,
            "total_steps": self.total_steps,
            "total_hits": self.total_hits,
            "episodes_with_hit": self.episodes_with_hit,
            "terminated_episodes": self.terminated_episodes,
        }

    def as_row(self) -> dict[str, str]:
        metrics = self.as_metrics()
        return {
            "name": self.name,
            "avg_steps": f"{metrics['avg_steps']:.2f}",
            "avg_reward_per_episode": f"{metrics['avg_reward_per_episode']:.3f}",
            "avg_reward_per_step": f"{metrics['avg_reward_per_step']:.3f}",
            "hit_rate": f"{metrics['hit_rate']:.1%}",
            "avg_hits_per_episode": f"{metrics['avg_hits_per_episode']:.2f}",
            "avg_first_hit": "n/a" if metrics["avg_first_hit"] is None else f"{metrics['avg_first_hit']:.2f}",
            "terminated_rate": f"{metrics['terminated_rate']:.1%}",
        }


@dataclass
class SweepResult:
    index: int
    params: HyperParams
    table_path: Path
    summary: PolicySummary
    train_iterations: int
    eval_episodes: int
    eval_max_steps: int
    train_seed: int
    eval_seed: int

    def as_row(self) -> dict[str, object]:
        metrics = self.summary.as_metrics()
        return {
            "index": self.index,
            "table_file": self.table_path.name,
            "table_path": str(self.table_path),
            "train_iterations": self.train_iterations,
            "eval_episodes": self.eval_episodes,
            "eval_max_steps": self.eval_max_steps,
            "train_seed": self.train_seed,
            "eval_seed": self.eval_seed,
            **self.params.as_dict(),
            **metrics,
        }


class FrozenQPolicy:
    def __init__(self, name: str, q_table: np.ndarray):
        self.name = name
        self.q_table = q_table

    def reset(self, episode_seed: int) -> None:
        return None

    def select_action(self, state_key: str) -> str:
        return greedy_action_from_table(self.q_table, state_key)


class RandomPolicy:
    def __init__(self, name: str = "random"):
        self.name = name
        self.rng = random.Random()

    def reset(self, episode_seed: int) -> None:
        self.rng.seed(episode_seed)

    def select_action(self, state_key: str) -> str:
        return self.rng.choice(ACTIONS)


def load_trained_table(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    q_table = np.load(path)
    if q_table.shape != (NUM_STATES, len(ACTIONS)):
        raise ValueError(f"Checkpoint shape must be {(NUM_STATES, len(ACTIONS))}, got {q_table.shape}.")
    return q_table


def run_episode(policy, episode_seed: int, max_steps: int, reward_kwargs: dict[str, float] | None = None) -> EpisodeResult:
    policy.reset(episode_seed)
    env = Environment(WIDTH, HEIGHT, seed=episode_seed)
    reward_kwargs = reward_kwargs or {}

    total_reward = 0.0
    hits = 0
    first_hit_step = None

    for step in range(1, max_steps + 1):
        state_key = env.curr_state_key
        action = policy.select_action(state_key)
        prior_distance = abs(env.sim.ball_x - env.agent.center_x)
        result = env.take_action_and_observe(action)

        if not result.get("ball_remains_in_play", False):
            return EpisodeResult(
                steps=step,
                total_reward=total_reward,
                hits=hits,
                first_hit_step=first_hit_step,
                terminated=True,
            )

        reward = reward_function(
            result["state_key"],
            action,
            result.get("did_hit", False),
            previous_distance=prior_distance,
            current_distance=abs(env.sim.ball_x - env.agent.center_x),
            **reward_kwargs,
        )
        total_reward += reward
        if result.get("did_hit", False):
            hits += 1
            if first_hit_step is None:
                first_hit_step = step

    return EpisodeResult(
        steps=max_steps,
        total_reward=total_reward,
        hits=hits,
        first_hit_step=first_hit_step,
        terminated=False,
    )


def evaluate_policy(
    policy,
    episodes: int,
    max_steps: int,
    base_seed: int,
    reward_kwargs: dict[str, float] | None = None,
) -> PolicySummary:
    summary = PolicySummary(name=policy.name, episodes=episodes)
    for episode_index in range(episodes):
        episode_seed = base_seed + episode_index
        summary.add_episode(run_episode(policy, episode_seed, max_steps, reward_kwargs))
    return summary


def format_report(summaries: list[PolicySummary]) -> str:
    rows = [summary.as_row() for summary in summaries]
    headers = [
        "Policy",
        "Avg Steps",
        "Avg Reward/Ep",
        "Avg Reward/Step",
        "Hit Rate",
        "Avg Hits/Ep",
        "Avg First Hit",
        "Terminated",
    ]

    columns = {
        "Policy": [row["name"] for row in rows],
        "Avg Steps": [row["avg_steps"] for row in rows],
        "Avg Reward/Ep": [row["avg_reward_per_episode"] for row in rows],
        "Avg Reward/Step": [row["avg_reward_per_step"] for row in rows],
        "Hit Rate": [row["hit_rate"] for row in rows],
        "Avg Hits/Ep": [row["avg_hits_per_episode"] for row in rows],
        "Avg First Hit": [row["avg_first_hit"] for row in rows],
        "Terminated": [row["terminated_rate"] for row in rows],
    }
    widths = {header: max(len(header), max(len(value) for value in values)) for header, values in columns.items()}

    def fmt_row(values):
        return " | ".join(value.ljust(widths[header]) for header, value in zip(headers, values))

    lines = [fmt_row(headers)]
    lines.append("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        lines.append(
            fmt_row(
                [
                    row["name"],
                    row["avg_steps"],
                    row["avg_reward_per_episode"],
                    row["avg_reward_per_step"],
                    row["hit_rate"],
                    row["avg_hits_per_episode"],
                    row["avg_first_hit"],
                    row["terminated_rate"],
                ]
            )
        )
    return "\n".join(lines)


def build_hyperparameter_grid(
    hit_rewards: list[float] | None = None,
    alphas: list[float] | None = None,
    gammas: list[float] | None = None,
    epsilons: list[float] | None = None,
) -> list[HyperParams]:
    hit_rewards = hit_rewards or DEFAULT_HIT_REWARDS
    alphas = alphas or DEFAULT_ALPHAS
    gammas = gammas or DEFAULT_GAMMAS
    epsilons = epsilons or DEFAULT_EPSILONS
    return [
        HyperParams(hit_reward=hit_reward, alpha=alpha, gamma=gamma, epsilon=epsilon)
        for hit_reward, alpha, gamma, epsilon in product(hit_rewards, alphas, gammas, epsilons)
    ]


def train_hyperparameter_table(params: HyperParams, train_iterations: int, train_seed: int) -> QTable:
    qtable = QTable(
        alpha=params.alpha,
        gamma=params.gamma,
        epsilon=params.epsilon,
        hit_reward=params.hit_reward,
        jump_hit_multiplier=params.jump_hit_multiplier,
        jump_miss_penalty=params.jump_miss_penalty,
        seed=train_seed,
        checkpoint_prefix=params.filename_stem(),
    )
    for _ in range(train_iterations):
        qtable.explore()
    return qtable


def evaluate_table(
    q_table: np.ndarray,
    params: HyperParams,
    eval_episodes: int,
    eval_max_steps: int,
    eval_seed: int,
) -> PolicySummary:
    policy = FrozenQPolicy("trained", q_table)
    return evaluate_policy(policy, eval_episodes, eval_max_steps, eval_seed, params.reward_kwargs())


def run_hyperparameter_sweep(
    output_dir: Path = DEFAULT_HYPERTABLE_DIR,
    train_iterations: int = 1_000_000,
    eval_episodes: int = 100,
    eval_max_steps: int = 750,
    train_seed: int = 12345,
    eval_seed: int = 54321,
    hit_rewards: list[float] | None = None,
    alphas: list[float] | None = None,
    gammas: list[float] | None = None,
    epsilons: list[float] | None = None,
) -> list[SweepResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = build_hyperparameter_grid(hit_rewards, alphas, gammas, epsilons)
    sweep_results: list[SweepResult] = []

    for index, params in enumerate(grid, start=1):
        run_train_seed = train_seed + index
        qtable = train_hyperparameter_table(params, train_iterations, run_train_seed)
        table_path = output_dir / f"{index:03d}_{params.filename_stem()}.npy"
        np.save(table_path, qtable.q_table)
        summary = evaluate_table(qtable.q_table, params, eval_episodes, eval_max_steps, eval_seed)
        sweep_results.append(
            SweepResult(
                index=index,
                params=params,
                table_path=table_path,
                summary=summary,
                train_iterations=train_iterations,
                eval_episodes=eval_episodes,
                eval_max_steps=eval_max_steps,
                train_seed=run_train_seed,
                eval_seed=eval_seed,
            )
        )

    write_sweep_csv(output_dir / "legend.csv", sweep_results)
    top_five = sorted(
        sweep_results,
        key=sweep_sort_key,
    )[:5]
    write_top_five_reports(output_dir, top_five)
    return sweep_results


def train_winning_checkpoint(
    output_dir: Path = DEFAULT_HYPERTABLE_DIR,
    iterations: int = 1_000_000,
    seed: int = 12345,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    qtable = train_hyperparameter_table(FIXED_SCREENSHOT_PARAMS, iterations, seed)
    filepath = output_dir / f"winner_{iterations}_{FIXED_SCREENSHOT_PARAMS.filename_stem()}.npy"
    np.save(filepath, qtable.q_table)
    return filepath


def train_fixed_screenshot_table(
    iterations: int = 10_000,
    output_dir: Path = DEFAULT_HYPERTABLE_DIR,
    seed: int = 12345,
) -> tuple[Path, PolicySummary]:
    output_dir.mkdir(parents=True, exist_ok=True)
    qtable = train_hyperparameter_table(FIXED_SCREENSHOT_PARAMS, iterations, seed)
    filepath = output_dir / f"fixed_{iterations}_{FIXED_SCREENSHOT_PARAMS.filename_stem()}.npy"
    np.save(filepath, qtable.q_table)
    summary = evaluate_table(qtable.q_table, FIXED_SCREENSHOT_PARAMS, 100, 750, seed + 999)
    return filepath, summary


def train_evolution_checkpoints(
    output_dir: Path = EVOLUTION_DIR,
    steps: tuple[int, ...] = EVOLUTION_STEPS,
    seed: int = 12345,
    eval_episodes: int = 100,
    eval_max_steps: int = 750,
    eval_seed: int = 54321,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    qtable = train_hyperparameter_table(EVOLUTION_PARAMS, max(steps), seed)
    snapshots: list[dict[str, object]] = []
    step_set = set(steps)

    for iteration in range(1, max(steps) + 1):
        qtable.explore()
        if iteration not in step_set:
            continue

        checkpoint_path = output_dir / f"evolution_{iteration}_{EVOLUTION_PARAMS.filename_stem()}.npy"
        np.save(checkpoint_path, qtable.q_table)
        summary = evaluate_table(qtable.q_table, EVOLUTION_PARAMS, eval_episodes, eval_max_steps, eval_seed)
        metrics = summary.as_metrics()
        snapshots.append(
            {
                "iteration": iteration,
                "table_file": checkpoint_path.name,
                "table_path": str(checkpoint_path),
                "hit_rate": metrics["hit_rate"],
                "hit_rate_display": f"{metrics['hit_rate']:.1%}",
                "avg_reward_per_episode": metrics["avg_reward_per_episode"],
                "avg_reward_per_episode_display": f"{metrics['avg_reward_per_episode']:.3f}",
                "avg_steps": metrics["avg_steps"],
                "avg_reward_per_step": metrics["avg_reward_per_step"],
                "avg_hits_per_episode": metrics["avg_hits_per_episode"],
                "avg_first_hit": metrics["avg_first_hit"],
                "terminated_rate": metrics["terminated_rate"],
            }
        )

    return snapshots


def run_fixed_screenshot_10k(
    output_dir: Path = DEFAULT_HYPERTABLE_DIR,
    seed: int = 12345,
) -> tuple[Path, PolicySummary]:
    filepath, summary = train_fixed_screenshot_table(10_000, output_dir, seed)
    print(filepath)
    print()
    print(format_report([summary]))
    return filepath, summary


def sweep_sort_key(result: SweepResult):
    metrics = result.summary.as_metrics()
    avg_first_hit = metrics["avg_first_hit"]
    return (
        -metrics["hit_rate"],
        -metrics["avg_reward_per_episode"],
        math.inf if avg_first_hit is None else avg_first_hit,
        -metrics["avg_hits_per_episode"],
        result.index,
    )


def write_sweep_csv(path: Path, results: list[SweepResult]) -> None:
    rows = [result.as_row() for result in results]
    fieldnames = [
        "index",
        "table_file",
        "table_path",
        "hit_reward",
        "alpha",
        "gamma",
        "epsilon",
        "jump_hit_multiplier",
        "jump_miss_penalty",
        "train_iterations",
        "eval_episodes",
        "eval_max_steps",
        "train_seed",
        "eval_seed",
        "avg_steps",
        "avg_reward_per_episode",
        "avg_reward_per_step",
        "hit_rate",
        "avg_hits_per_episode",
        "avg_first_hit",
        "terminated_rate",
        "total_reward",
        "total_steps",
        "total_hits",
        "episodes_with_hit",
        "terminated_episodes",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_top_five_rows(results: list[SweepResult]) -> str:
    headers = ["Rank", "Table", "Hit Rate", "Avg Reward/Ep", "Hit Reward", "Alpha", "Gamma", "Epsilon"]
    rows = []
    for rank, result in enumerate(results, start=1):
        metrics = result.summary.as_metrics()
        rows.append(
            [
                str(rank),
                result.table_path.name,
                f"{metrics['hit_rate']:.1%}",
                f"{metrics['avg_reward_per_episode']:.3f}",
                f"{result.params.hit_reward:g}",
                f"{result.params.alpha:g}",
                f"{result.params.gamma:g}",
                f"{result.params.epsilon:g}",
            ]
        )

    widths = []
    for index, header in enumerate(headers):
        widest = len(header)
        for row in rows:
            widest = max(widest, len(row[index]))
        widths.append(widest)

    def fmt(values):
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    lines = [fmt(headers), "-+-".join("-" * width for width in widths)]
    for row in rows:
        lines.append(fmt(row))
    return "\n".join(lines)


def write_top_five_reports(output_dir: Path, top_five: list[SweepResult]) -> None:
    top_five_csv = output_dir / "top_5.csv"
    top_five_txt = output_dir / "top_5.txt"
    fieldnames = [
        "rank",
        "table_file",
        "hit_rate",
        "avg_reward_per_episode",
        "avg_reward_per_step",
        "avg_hits_per_episode",
        "avg_first_hit",
        "hit_reward",
        "alpha",
        "gamma",
        "epsilon",
    ]
    with top_five_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(top_five, start=1):
            metrics = result.summary.as_metrics()
            writer.writerow(
                {
                    "rank": rank,
                    "table_file": result.table_path.name,
                    "hit_rate": f"{metrics['hit_rate']:.4f}",
                    "avg_reward_per_episode": f"{metrics['avg_reward_per_episode']:.6f}",
                    "avg_reward_per_step": f"{metrics['avg_reward_per_step']:.6f}",
                    "avg_hits_per_episode": f"{metrics['avg_hits_per_episode']:.6f}",
                    "avg_first_hit": "" if metrics["avg_first_hit"] is None else f"{metrics['avg_first_hit']:.6f}",
                    "hit_reward": f"{result.params.hit_reward:g}",
                    "alpha": f"{result.params.alpha:g}",
                    "gamma": f"{result.params.gamma:g}",
                    "epsilon": f"{result.params.epsilon:g}",
                }
            )
    top_five_txt.write_text(format_top_five_rows(top_five))


def run_comparison(checkpoint: Path, episodes: int, max_steps: int, seed: int) -> str:
    trained_table = load_trained_table(checkpoint)
    uninitialized_table = np.zeros_like(trained_table)

    policies = [
        FrozenQPolicy(f"trained ({checkpoint.name})", trained_table),
        FrozenQPolicy("uninitialized", uninitialized_table),
        RandomPolicy(),
    ]

    summaries = [evaluate_policy(policy, episodes, max_steps, seed) for policy in policies]
    report = format_report(summaries)
    baseline = summaries[1]
    trained = summaries[0]
    delta_lines = [
        "",
        "Trained vs uninitialized",
        f"  Avg reward/episode delta: {trained.total_reward / trained.episodes - baseline.total_reward / baseline.episodes:+.3f}",
        f"  Hit rate delta: {(trained.episodes_with_hit / trained.episodes) - (baseline.episodes_with_hit / baseline.episodes):+.1%}",
        f"  Avg steps delta: {(trained.total_steps / trained.episodes) - (baseline.total_steps / baseline.episodes):+.2f}",
    ]
    return report + "\n" + "\n".join(delta_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, compare, and sweep Q-learning tables.")
    parser.add_argument("--mode", choices=("sweep", "compare", "winner", "fixed"), default="sweep", help="What this run should do.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Checkpoint for compare mode.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_HYPERTABLE_DIR, help="Directory to write hyper tables.")
    parser.add_argument("--train-iterations", type=int, default=1_000_000, help="Training iterations for each sweep table.")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Evaluation episodes for each table.")
    parser.add_argument("--eval-max-steps", type=int, default=750, help="Maximum steps per evaluation episode.")
    parser.add_argument("--train-seed", type=int, default=12345, help="Base seed for training runs.")
    parser.add_argument("--eval-seed", type=int, default=54321, help="Base seed for evaluation episodes.")
    parser.add_argument("--hit-rewards", type=parse_float_list, default=DEFAULT_HIT_REWARDS, help="Comma-separated hit reward values.")
    parser.add_argument("--alphas", type=parse_float_list, default=DEFAULT_ALPHAS, help="Comma-separated alpha values.")
    parser.add_argument("--gammas", type=parse_float_list, default=DEFAULT_GAMMAS, help="Comma-separated gamma values.")
    parser.add_argument("--epsilons", type=parse_float_list, default=DEFAULT_EPSILONS, help="Comma-separated epsilon values.")
    args = parser.parse_args()

    if args.mode == "compare":
        print(run_comparison(args.checkpoint, args.eval_episodes, args.eval_max_steps, args.eval_seed))
        return

    if args.mode == "winner":
        filepath = train_winning_checkpoint(args.output_dir, args.train_iterations, args.train_seed)
        print(filepath)
        return

    if args.mode == "fixed":
        run_fixed_screenshot_10k(args.output_dir, args.train_seed)
        return

    results = run_hyperparameter_sweep(
        output_dir=args.output_dir,
        train_iterations=args.train_iterations,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        hit_rewards=args.hit_rewards,
        alphas=args.alphas,
        gammas=args.gammas,
        epsilons=args.epsilons,
    )
    top_five = sorted(results, key=sweep_sort_key)[:5]
    print(format_top_five_rows(top_five))
    print()
    print(f"Wrote {len(results)} tables to {args.output_dir}")
    print(f"Legend: {args.output_dir / 'legend.csv'}")
    print(f"Top 5 report: {args.output_dir / 'top_5.txt'}")


if __name__ == "__main__":
    main()
