import os
import random

from flask import Flask, jsonify, redirect, render_template, request
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

from environment import Environment
from learning_utils import BUCKET_COUNT

app = Flask(__name__)

WIDTH = 1000
HEIGHT = 1000
ACTIONS = ["PowerLeft", "PowerRight", "Jump", "No Power"]

ENV_STATE = {"env": None, "last_unique_key": None, "curr_action": "No Power", "trajectory": []}
QL_STATE = {"qtable": None, "trajectory": [], "updates": [], "load_token": None}
TABLE_DIRS = ("saved_tables", "hypertables")


def serialize_bbox(bbox):
    return [[[p1[0], p1[1]], [p2[0], p2[1]]] for p1, p2 in bbox]


def new_environment():
    env = Environment(WIDTH, HEIGHT)
    ENV_STATE["env"] = env
    ENV_STATE["last_unique_key"] = None
    ENV_STATE["curr_action"] = "No Power"
    ENV_STATE["trajectory"] = [{"x": env.sim.ball_x, "y": env.sim.ball_y}]
    return env


def get_environment():
    return ENV_STATE["env"] if ENV_STATE["env"] is not None else new_environment()


def environment_snapshot(env, ball_remains_in_play=True):
    return {
        "width": env.game_width,
        "height": env.game_height,
        "ball": {"x": env.sim.ball_x, "y": env.sim.ball_y, "vx": env.sim.ball_vx, "vy": env.sim.ball_vy},
        "agent": {"x": env.agent.center_x, "y": env.agent.center_y, "bbox": serialize_bbox(env.agent.bbox)},
        "walls": serialize_bbox(env.wall_bbox),
        "obstacles": [serialize_bbox(collider) for collider in env.sim.collision_segments[1:]],
        "trajectory": ENV_STATE["trajectory"],
        "state_key": env.curr_state_key,
        "episode_number": env.episode_number,
        "current_action": ENV_STATE["curr_action"],
        "ball_remains_in_play": ball_remains_in_play,
    }


def step_environment():
    env = get_environment()
    result = env.take_action_and_observe(ENV_STATE["curr_action"])

    if result.get("ball_remains_in_play", False):
        ENV_STATE["trajectory"].append({"x": env.sim.ball_x, "y": env.sim.ball_y})
        curr_key = result.get("state_key")
        if curr_key != ENV_STATE["last_unique_key"]:
            ENV_STATE["last_unique_key"] = curr_key
            ENV_STATE["curr_action"] = random.choice(ACTIONS)
    else:
        ENV_STATE["last_unique_key"] = None
        ENV_STATE["curr_action"] = random.choice(ACTIONS)
        ENV_STATE["trajectory"] = [{"x": env.sim.ball_x, "y": env.sim.ball_y}]

    return environment_snapshot(env, result.get("ball_remains_in_play", False))


def resolve_qlearning_path(load_token):
    if not load_token:
        return None
    if os.path.isfile(load_token):
        return load_token
    for table_dir in TABLE_DIRS:
        if os.path.isfile(os.path.join(table_dir, load_token)):
            return os.path.join(table_dir, load_token)
        if os.path.isfile(os.path.join(table_dir, f"{load_token}.npy")):
            return os.path.join(table_dir, f"{load_token}.npy")
        if os.path.isdir(table_dir):
            matches = [f for f in os.listdir(table_dir) if f.startswith(f"qtable_{load_token}_") and f.endswith(".npy")]
            if matches:
                return os.path.join(table_dir, sorted(matches)[-1])
    return None


def list_qlearning_tables():
    tables = []
    for table_dir in TABLE_DIRS:
        if not os.path.isdir(table_dir):
            continue
        for name in sorted(os.listdir(table_dir)):
            if not name.endswith(".npy"):
                continue
            if not (name.startswith("qtable_") or table_dir == "hypertables"):
                continue
            path = os.path.join(table_dir, name)
            try:
                shape = np.load(path, mmap_mode="r").shape
            except Exception:
                continue
            if shape != (BUCKET_COUNT * BUCKET_COUNT, len(ACTIONS)):
                continue
            tables.append(name)
    tables = sorted(dict.fromkeys(tables))
    return tables


def new_qlearning(load_token=None):
    from qlearning import QTable

    qtable = QTable(resolve_qlearning_path(load_token))
    QL_STATE["qtable"] = qtable
    QL_STATE["trajectory"] = [{"x": qtable.env.sim.ball_x, "y": qtable.env.sim.ball_y}]
    QL_STATE["updates"] = []
    QL_STATE["load_token"] = load_token
    return qtable


def get_qlearning():
    return QL_STATE["qtable"] if QL_STATE["qtable"] is not None else new_qlearning()


def qlearning_snapshot(qtable, ball_remains_in_play=True):
    return {
        "width": qtable.env.game_width,
        "height": qtable.env.game_height,
        "ball": {
            "x": qtable.env.sim.ball_x,
            "y": qtable.env.sim.ball_y,
            "vx": qtable.env.sim.ball_vx,
            "vy": qtable.env.sim.ball_vy,
        },
        "agent": {
            "x": qtable.env.agent.center_x,
            "y": qtable.env.agent.center_y,
            "bbox": serialize_bbox(qtable.env.agent.bbox),
        },
        "walls": serialize_bbox(qtable.env.wall_bbox),
        "obstacles": [serialize_bbox(collider) for collider in qtable.env.sim.collision_segments[1:]],
        "trajectory": QL_STATE["trajectory"],
        "state_key": qtable.current_state_key,
        "current_action": qtable.current_action,
        "current_mode": getattr(qtable, "current_mode", "greedy"),
        "current_greedy_action": getattr(qtable, "current_greedy_action", qtable.current_action),
        "episode_number": qtable.env.episode_number,
        "iterations": qtable.iterations,
        "ball_remains_in_play": ball_remains_in_play,
        "qtable_rows": QL_STATE["updates"][-10:],
        "qtable_columns": qtable.actions,
        "load_token": QL_STATE["load_token"],
        "available_tables": list_qlearning_tables(),
    }


def step_qlearning():
    qtable = get_qlearning()
    pre_state = qtable.current_state_key
    pre_action = qtable.current_action
    pre_row = qtable.get_row_index(pre_state)
    result = qtable.explore()

    if result.get("ball_remains_in_play", False):
        QL_STATE["trajectory"].append({"x": qtable.env.sim.ball_x, "y": qtable.env.sim.ball_y})
        if "reward" in result:
            QL_STATE["updates"].append(
                {
                    "row_index": pre_row,
                    "state_key": pre_state,
                    "mode": result.get("mode", getattr(qtable, "current_mode", "greedy")),
                    "greedy_action": result.get("greedy_action", pre_action),
                    "action": result.get("selected_action", pre_action),
                    "reward": result.get("reward"),
                    "values": [float(v) for v in qtable.q_table[pre_row].tolist()],
                    "iteration": qtable.iterations,
                }
            )
    else:
        QL_STATE["trajectory"] = [{"x": qtable.env.sim.ball_x, "y": qtable.env.sim.ball_y}]

    return qlearning_snapshot(qtable, result.get("ball_remains_in_play", False))


@app.route("/")
def index():
    return redirect("/debug-environment")


@app.route("/debug-environment")
def debug_environment():
    return render_template("environment_debug.html", width=WIDTH, height=HEIGHT)


@app.route("/debug-qlearning")
def debug_qlearning():
    return render_template("qlearning_debug.html", width=WIDTH, height=HEIGHT, available_tables=list_qlearning_tables())


@app.route("/api/environment/reset", methods=["POST"])
def api_environment_reset():
    return jsonify(environment_snapshot(new_environment()))


@app.route("/api/environment/step", methods=["POST"])
def api_environment_step():
    return jsonify(step_environment())


@app.route("/api/qlearning/reset", methods=["POST"])
def api_qlearning_reset():
    return jsonify(qlearning_snapshot(new_qlearning(request.args.get("load"))))


@app.route("/api/qlearning/load", methods=["POST"])
def api_qlearning_load():
    data = request.get_json(silent=True) or {}
    return jsonify(qlearning_snapshot(new_qlearning(data.get("load"))))


@app.route("/api/qlearning/step", methods=["POST"])
def api_qlearning_step():
    return jsonify(step_qlearning())


if __name__ == "__main__":
    app.run(port=8000, use_reloader=False)
