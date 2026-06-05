from environment import Environment
from learning_utils import (
    BUCKET_COUNT,
    HIT_REWARD,
    DISTANCE_REWARD_SCALE,
    JUMP_HIT_MULTIPLIER,
    JUMP_MISS_PENALTY,
    build_state_map,
    reward_function,
    random_unused_prefixed,
)
import os
import random
import time
from pathlib import Path
import numpy as np

NUM_STATES = BUCKET_COUNT * BUCKET_COUNT
BASE_DIR = Path(__file__).resolve().parent
TABLES_DIR = BASE_DIR / "tables"
SAVED_TABLES_DIR = TABLES_DIR / "saved_tables"

class QTable:
    def __init__(
        self,
        prior_table_path=None,
        alpha=0.1,
        gamma=0.95,
        epsilon=0.1,
        hit_reward=HIT_REWARD,
        jump_hit_multiplier=JUMP_HIT_MULTIPLIER,
        jump_miss_penalty=JUMP_MISS_PENALTY,
        distance_reward_scale=DISTANCE_REWARD_SCALE,
        seed=None,
        checkpoint_prefix=None,
    ):
        self.env = Environment(1000, 1000, seed=seed)
        self.key_map = build_state_map(BUCKET_COUNT)
        self.actions = ["PowerLeft", "PowerRight", "Jump", "No Power"]
        self.q_table = None
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.hit_reward = hit_reward
        self.jump_hit_multiplier = jump_hit_multiplier
        self.jump_miss_penalty = jump_miss_penalty
        self.distance_reward_scale = distance_reward_scale
        self.rng = random.Random(seed)

        if prior_table_path:
            self.q_table = np.load(prior_table_path)
            if self.q_table.shape != (NUM_STATES, len(self.actions)):
                raise ValueError(f"Checkpoint shape does not match the current {BUCKET_COUNT}x{BUCKET_COUNT} state space.")
            self.checkpoint_prefix, self.iterations = self._load_checkpoint_metadata(prior_table_path)
            self.first_checkpoint = False
        else:
            self.iterations = 0
            self.checkpoint_prefix = checkpoint_prefix or random_unused_prefixed()
            self.q_table = np.zeros((NUM_STATES, len(self.actions)))
            self.first_checkpoint = True

        self.current_state_key = self.env.get_sim_state_as_key()
        self.current_action, self.current_mode, self.current_greedy_action = self.select_action(self.current_state_key)

    def get_row_index(self, state_key):
        return self.key_map[state_key]

    def get_action_col(self, action):
        return self.actions.index(action)

    def _load_checkpoint_metadata(self, prior_table_path):
        stem = os.path.splitext(os.path.basename(prior_table_path))[0]
        parts = stem.split("_")
        if len(parts) >= 3 and parts[0] == "qtable":
            try:
                return parts[1], int(parts[2]) + 1
            except ValueError:
                pass
        return stem, 0

    def greedy_action(self, state_key):
        return self.actions[int(np.argmax(self.q_table[self.get_row_index(state_key)]))]

    def select_action(self, state_key):
        greedy_action = self.greedy_action(state_key)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.actions), "epsilon", greedy_action
        return greedy_action, "greedy", greedy_action

    def update_q_value(self, state, action, reward, next_state, alpha, gamma):
        state_row = self.get_row_index(state)
        next_state_row = self.get_row_index(next_state)
        action_col = self.get_action_col(action)
        old_value = self.q_table[state_row, action_col]
        future_reward = np.max(self.q_table[next_state_row])
        self.q_table[state_row, action_col] = (1 - alpha) * old_value + alpha * (reward + gamma * future_reward)

    def explore(self):
        self.current_action, self.current_mode, self.current_greedy_action = self.select_action(self.current_state_key)
        prior_distance = abs(self.env.sim.ball_x - self.env.agent.center_x)
        result = self.env.take_action_and_observe(self.current_action)
        if not result.get("ball_remains_in_play", False):
            self.current_state_key = self.env.curr_state_key
            return result

        next_state_key = result["state_key"]
        prior_state_key = self.current_state_key
        did_hit = result.get("did_hit", False)
        current_distance = abs(self.env.sim.ball_x - self.env.agent.center_x)
        reward = reward_function(
            next_state_key,
            self.current_action,
            did_hit,
            hit_reward=self.hit_reward,
            jump_hit_multiplier=self.jump_hit_multiplier,
            jump_miss_penalty=self.jump_miss_penalty,
            previous_distance=prior_distance,
            current_distance=current_distance,
            distance_reward_scale=self.distance_reward_scale,
        )
        self.update_q_value(prior_state_key, self.current_action, reward, next_state_key, self.alpha, self.gamma)
        result["reward"] = reward
        result["mode"] = self.current_mode
        result["greedy_action"] = self.current_greedy_action
        result["selected_action"] = self.current_action
        self.iterations += 1

        self.current_state_key = next_state_key
        self.current_action, self.current_mode, self.current_greedy_action = self.select_action(self.current_state_key)
        result["iterations"] = self.iterations
        return result

    def save_table(self):
        def name_my_checkpoint():
            return str(f"qtable_{self.checkpoint_prefix}_{self.iterations}")

        SAVED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
        filepath = SAVED_TABLES_DIR / f"{name_my_checkpoint()}.npy"
        np.save(filepath, self.q_table)
        self.iterations += 1
        self.first_checkpoint = False
        return str(filepath)


def run_save_new(n_iterations):
    qtable = QTable()
    started = time.perf_counter()
    for _ in range(n_iterations):
        qtable.explore()
    filepath = qtable.save_table()
    elapsed = time.perf_counter() - started
    print(f"Completed {n_iterations} iterations in {elapsed:.2f}s")
    print(filepath)
    return qtable
