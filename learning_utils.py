import os
import random
import string

BUCKET_COUNT = 40
HIT_REWARD = 10.0
JUMP_HIT_MULTIPLIER = 2.0
JUMP_MISS_PENALTY = -5.0
DISTANCE_REWARD_SCALE = 0.01

def random_unused_prefixed():
    invalid_options = load_known_prefixes()
    while True:
        prefix = "".join(random.choice(string.ascii_uppercase) for _ in range(2))
        if prefix not in invalid_options:
            return prefix

def load_known_prefixes():
    known_prefixes = set()
    if not os.path.isdir("saved_tables"):
        return known_prefixes
    files = os.listdir("saved_tables")
    for f in files:
        if "qtable" in f:
            fname_parts = f.split("_")
            known_prefixes.add(fname_parts[1])
    return known_prefixes

def build_state_map(bucket_count=BUCKET_COUNT):
    return {
        f"{ball_bucket}-{agent_bucket}": ball_bucket * bucket_count + agent_bucket
        for ball_bucket in range(bucket_count)
        for agent_bucket in range(bucket_count)
    }

def reward_function(
    state_key,
    action=None,
    did_hit=False,
    hit_reward=HIT_REWARD,
    jump_hit_multiplier=JUMP_HIT_MULTIPLIER,
    jump_miss_penalty=JUMP_MISS_PENALTY,
    previous_distance=None,
    current_distance=None,
    distance_reward_scale=DISTANCE_REWARD_SCALE,
):
    if did_hit:
        reward = hit_reward
        if action == "Jump":
            reward *= jump_hit_multiplier
        return reward

    if previous_distance is not None and current_distance is not None:
        distance_delta = previous_distance - current_distance
        reward = max(-0.5, min(0.5, distance_delta * distance_reward_scale))
    else:
        ball_bucket, agent_bucket = (int(value) for value in state_key.split("-"))
        distance = abs(ball_bucket - agent_bucket)
        if distance == 0:
            reward = 1.0
        elif distance <= 1:
            reward = 0.0
        else:
            reward = -1.0

    if action == "Jump":
        reward += jump_miss_penalty
    return reward
