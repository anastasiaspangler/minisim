import random
from learning_utils import BUCKET_COUNT
from simulator import Simulator, Agent, GutterBallError


def bucket_index(x, width):
    return min(BUCKET_COUNT - 1, max(0, int(x / (width / BUCKET_COUNT))))


class Environment:
    def __init__(self, game_width, game_height, seed=None, rng=None):
        self.game_width = game_width
        self.game_height = game_height
        self.rng = rng if rng is not None else random.Random(seed)
        self.agent_speed = 2
        self.agent_height = int(game_height * 0.05)
        self.agent_width = int(game_width * 0.08)
        self.wall_bbox = [((0, 0), (self.game_width, 0)), ((self.game_width, 0), (self.game_width, self.game_height - 5)), ((self.game_width, self.game_height - 5), (0, self.game_height - 5)), ((0, self.game_height - 5), (0, 0))]

        self.agent_bbox = self._agent_bbox()
        self.agent = Agent(self.game_width / 2, self.game_height - self.agent_height / 2, self.agent_bbox, self.agent_width)
        self.sim = Simulator(self.agent, self.agent_speed, self.game_width, self.game_height, rng=self.rng)
        self.sim.add_collider(self.wall_bbox)
        self.regen_obstacles()
        self.velocity_handler = self.sim.invertible_ball_velocity()
        self.curr_state_key = None

        self.episode_number = 0
        self.episode_hit_count = 0
        self.max_episode_hits = 4

        self.velocity_handler.__next__()
        self.curr_state_key = self.get_sim_state_as_key()

    def _agent_bbox(self):
        x = self.game_width / 2
        y = self.game_height - self.agent_height / 2
        w = self.agent_width / 2
        h = self.agent_height / 2
        return [((x - w, y - h), (x + w, y - h)), ((x + w, y - h), (x + w, y + h)), ((x + w, y + h), (x - w, y + h)), ((x - w, y + h), (x - w, y - h))]

    def regen_obstacles(self):
        side = max(1, int(self.agent_width * 0.8))
        bottom_clearance = self.agent_height * 4
        for _ in range(self.rng.randint(2, 5)):
            cx = self.rng.randint(side // 2, self.game_width - side // 2)
            cy = self.rng.randint(side // 2, self.game_height - 5 - bottom_clearance - side // 2)
            h = side / 2
            self.sim.add_collider([((cx - h, cy - h), (cx + h, cy - h)), ((cx + h, cy - h), (cx + h, cy + h)), ((cx + h, cy + h), (cx - h, cy + h)), ((cx - h, cy + h), (cx - h, cy - h))])

    def reset_simulation(self):
        self.sim.reset()
        self.sim.add_collider(self.wall_bbox)
        self.regen_obstacles()
        self.velocity_handler = self.sim.invertible_ball_velocity()
        self.velocity_handler.__next__()
        self.curr_state_key = self.get_sim_state_as_key()
        self.episode_hit_count = 0

    def end_episode(self, termination_reason):
        ended_hit_count = self.episode_hit_count
        self.episode_number += 1
        self.reset_simulation()
        return {
            "ball_remains_in_play": False,
            "termination_reason": termination_reason,
            "hits_this_episode": ended_hit_count,
        }

    def get_sim_state_as_key(self):
        ball_bucket = bucket_index(self.sim.ball_x, self.game_width)
        agent_bucket = bucket_index(self.agent.center_x, self.game_width)
        return f"{ball_bucket}-{agent_bucket}"

    def take_action_and_observe(self, action_col_key):
        try:
            if action_col_key == "PowerLeft":
                self.sim.move_agent_power(self.sim.agent_speed)
            elif action_col_key == "PowerRight":
                self.sim.move_agent_power(-self.sim.agent_speed)
            elif action_col_key == "Jump":
                self.sim.move_agent_jump()
            else:
                self.sim.move_agent_power(0.0)

            self.velocity_handler.__next__()
            self.curr_state_key = self.get_sim_state_as_key()
            if self.sim.agent_hit_ball_this_timestep:
                self.episode_hit_count += 1
                if self.episode_hit_count >= self.max_episode_hits:
                    return self.end_episode("hit_limit")

            return {
                "ball_remains_in_play": True,
                "state_key": self.curr_state_key,
                "episode_number": self.episode_number,
                "did_hit": self.sim.agent_hit_ball_this_timestep,
                "hits_this_episode": self.episode_hit_count,
            }
        except GutterBallError:
            return self.end_episode("gutter_ball")
