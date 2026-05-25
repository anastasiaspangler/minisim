# Minisim

Minisim is a small reinforcement-learning simulation for training a Q-learning agent to track and catch a moving ball. The project includes both the core simulation code and a Flask-based visualizer for inspecting the agent’s behavior and learned Q-table.

## Repository Structure

| File | Description |
| --- | --- |
| `analysis.py` | Batch training and evaluation entry points for comparing policies, running hyperparameter sweeps, and generating saved tables. |
| `environment.py` | Wraps the simulator in a learning environment and exposes the discrete state/action interface used by Q-learning. |
| `learning_utils.py` | Shared learning helpers, including state bucketing, reward shaping, and checkpoint naming utilities. |
| `qlearning.py` | Q-table implementation, action selection, Bellman updates, and table save/load behavior. |
| `simulator.py` | Core physics and collision logic for the ball, agent, and movement model. |
| `web_visualizer.py` | Flask app for stepping through the environment, viewing the agent, and inspecting recent Q-table updates. |

## Notes

- The current setup uses a discretized state space and reward shaping tuned for the agent’s horizontal movement model.
- Training outputs are written to `saved_tables/` and `hypertables/`.
- The web visualizer is intended for debugging and qualitative inspection, while `analysis.py` is used for quantitative comparison.

