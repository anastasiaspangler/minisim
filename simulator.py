import math
import random

class GutterBallError(Exception):
    def __init__(self, message, x, y):
        super().__init__(message)  # Pass message to base Exception class
        self.x = x
        self.y = y

    def __str__(self):
        return f"Gutter ball at {self.x}, {self.y}"

class Agent:
    def __init__(self, home_x, home_y, bbox, width):
        self.home_x = home_x
        self.home_y = home_y
        self.center_x = home_x
        self.center_y = home_y

        self.bbox = bbox
        self.home_bbox = bbox
        self.velocity_function = None
        self.linear_velocity_function = None
        self.width = width

    def reset(self):
        self.center_x = self.home_x
        self.center_y = self.home_y
        self.bbox = self.home_bbox
        self.velocity_function = None
        self.linear_velocity_function = None

    def is_jumping(self):
        return self.velocity_function is not None

    def advance_position_per_jump_velocity(self):
        if self.velocity_function is None:
            raise Exception("Agent has no velocity function for jump defined.")
        old_x, old_y, proposed_x, proposed_y, vx, vy = next(self.velocity_function)

        # y values are greatest closest to bottom
        if proposed_y >= self.home_y:
            dy = self.home_y - self.center_y
            self.center_y = self.home_y
            self.bbox = [((x1, y1 + dy), (x2, y2 + dy)) for (x1, y1), (x2, y2) in self.bbox]
            self.velocity_function = None
        else:
            dy = proposed_y - old_y
            self.center_y = proposed_y
            self.bbox = [((x1, y1 + dy), (x2, y2 + dy)) for (x1, y1), (x2, y2) in self.bbox]


class Simulator:
    def __init__(self, agent: Agent, agent_speed, game_width, game_height, gravity=(9.8 * 20), ball_radius=8,
                 time_step=0.016, substeps=4, speed=400, rng=None):
        self.gravity = gravity
        self.air_drag = 0.999
        self.linear_drag = 0.90
        self.linear_brake = 0.50
        self.linear_turn_brake = 0.35
        self.linear_force_scale = 1.5
        self.linear_deadband = 0.15
        self.max_linear_velocity = 8.0
        self.rng = rng if rng is not None else random.Random()
        self.ball_radius = ball_radius
        self.time_step_interval = time_step
        self.substeps = substeps
        self.max_substep_displacement = self.ball_radius / 2
        self.ball_speed = speed
        self.agent_speed = agent_speed

        self.game_width = game_width
        self.game_height = game_height

        self.collision_segments = [] # includes walls
        self.agent = agent
        self.bottom_border = ((0, game_height - 5), (game_width, game_height - 5))

        self.agent_hit_ball_this_timestep = False

        self.ball_x = None
        self.ball_y = None
        self.ball_vx = None
        self.ball_vy = None

    def reset(self, clear_obstacles=True):
        if clear_obstacles:
            self.collision_segments = []
        self.agent.reset()
        self.ball_x = None
        self.ball_y = None
        self.ball_vx = None
        self.ball_vy = None

    def invertible_ball_velocity(self):
        def orient(a, b, c):
            return (
                    (b[0] - a[0]) * (c[1] - a[1])
                    - (b[1] - a[1]) * (c[0] - a[0])
            )

        def segments_intersect(_a, _b, c, d):
            o1 = orient(_a, _b, c)
            o2 = orient(_a, _b, d)
            o3 = orient(c, d, _a)
            o4 = orient(c, d, _b)

            return (o1 * o2 < 0) and (o3 * o4 < 0)

        def bbox_extents(bbox):
            xs = [point[0] for segment in bbox for point in segment]
            ys = [point[1] for segment in bbox for point in segment]
            return min(xs), max(xs), min(ys), max(ys)

        def point_in_bbox(point, bbox):
            x, y = point
            left, right, top, bottom = bbox_extents(bbox)
            return left < x < right and top < y < bottom

        def agent_side_segment(side):
            if side == "top":
                return self.agent.bbox[0]
            if side == "right":
                return self.agent.bbox[1]
            if side == "bottom":
                return self.agent.bbox[2]
            return self.agent.bbox[3]

        def side_name_for_segment(segment):
            for name, agent_segment in (
                ("top", self.agent.bbox[0]),
                ("right", self.agent.bbox[1]),
                ("bottom", self.agent.bbox[2]),
                ("left", self.agent.bbox[3]),
            ):
                if segment == agent_segment:
                    return name
            return None

        def preferred_agent_side(old_pos, new_pos, hits=None):
            if hits:
                dx = new_pos[0] - old_pos[0]
                dy = new_pos[1] - old_pos[1]
                if abs(dy) >= abs(dx):
                    preferred = "top" if dy > 0 else "bottom"
                else:
                    preferred = "right" if dx > 0 else "left"
                ordered_sides = [preferred, "top", "right", "bottom", "left"]
                for side in ordered_sides:
                    segment = agent_side_segment(side)
                    if segment in hits:
                        return segment
                return hits[0]

            x, y = new_pos
            left, right, top, bottom = bbox_extents(self.agent.bbox)
            distances = {
                "top": y - top,
                "bottom": bottom - y,
                "left": x - left,
                "right": right - x,
            }
            side = min(distances, key=distances.get)
            return agent_side_segment(side)

        def push_ball_outside_agent(point, segment, margin=0.5):
            x, y = point
            side = side_name_for_segment(segment)
            left, right, top, bottom = bbox_extents(self.agent.bbox)

            if side == "top":
                return x, top - margin
            if side == "right":
                return right + margin, y
            if side == "bottom":
                return x, bottom + margin
            if side == "left":
                return left - margin, y
            return x, y

        def find_collision(old_pos, new_pos):
            bottom_pt1, bottom_pt2 = self.bottom_border
            # check for ball out of bounds
            if segments_intersect(old_pos, new_pos, bottom_pt1, bottom_pt2):
                raise GutterBallError("Ball out of bounds!", old_pos[0], old_pos[1])
            # check for agent collision
            agent_hits = []
            for agent_p1, agent_p2 in self.agent.bbox:
                if segments_intersect(old_pos, new_pos, agent_p1, agent_p2):
                    agent_hits.append((agent_p1, agent_p2))
            if agent_hits:
                self.agent_hit_ball_this_timestep = True
                return preferred_agent_side(old_pos, new_pos, agent_hits)
            if point_in_bbox(new_pos, self.agent.bbox) or point_in_bbox(old_pos, self.agent.bbox):
                self.agent_hit_ball_this_timestep = True
                return preferred_agent_side(old_pos, new_pos)
            # check for obstacle collision (ex: sticky notes)
            for collider in self.collision_segments:
                for a, b in collider:
                    if segments_intersect(old_pos, new_pos, a, b):
                        return a, b
            return None

        def reflect_velocity(_vx, _vy, _a, _b, bounce=0.9):
            ax, ay = _a
            bx, by = _b

            ex = bx - ax
            ey = by - ay

            nx = -ey
            ny = ex

            length = math.hypot(nx, ny)
            if length == 0:
                return _vx, _vy

            nx /= length
            ny /= length

            if _vx * nx + _vy * ny > 0:
                nx = -nx
                ny = -ny

            dot = _vx * nx + _vy * ny

            return (
                (_vx - 2 * dot * nx) * bounce,
                (_vy - 2 * dot * ny) * bounce,
            )

        angle = self.rng.uniform(math.pi / 6, 5 * math.pi / 6)

        x = float(self.rng.randint(
            self.ball_radius,
            max(self.ball_radius + 1, self.game_width - self.ball_radius - 1),
        ))
        y = float(self.ball_radius)

        vx, vy = self.ball_speed * math.cos(angle), self.ball_speed * math.sin(angle)
        interp_angle = math.degrees(math.atan2(vy, vx))
        interp_angle = interp_angle % 360
        # print(f"vx: {vx}, vy: {vy}, heading: {interp_angle}")
        gen = self.velocity_gen(x, y, vx, vy)

        while True:
            # agent logic
            self.agent_hit_ball_this_timestep = False  # wipe saved_tables
            if self.agent.is_jumping():
                self.agent.advance_position_per_jump_velocity()

            # next position for ball given velocity
            old_x, old_y, proposed_x, proposed_y, vx, vy = next(gen)

            # check for and handle collisions
            collision = find_collision((old_x, old_y),(proposed_x, proposed_y))
            if collision:
                a, b = collision
                rx, ry = reflect_velocity(vx, vy, a, b)
                gen.close()

                # restart from a position guaranteed to be outside the agent body
                if collision in self.agent.bbox:
                    x, y = push_ball_outside_agent((old_x, old_y), collision)
                else:
                    x, y = old_x, old_y

                interp_angle = math.degrees(math.atan2(rx, ry))
                interp_angle = interp_angle % 360
                # print(f"REFLECTION: vx: {rx}, vy: {ry}, heading: {interp_angle}")

                gen = self.velocity_gen(x, y, vx=rx, vy=ry)

                self.ball_x = x
                self.ball_y = y
                self.ball_vx = rx
                self.ball_vy = ry
                yield x, y
            else:
                x, y = proposed_x, proposed_y
                self.ball_x = x
                self.ball_y = y
                self.ball_vx = vx
                self.ball_vy = vy
                yield x, y

    def velocity_gen(self, x, y, vx=None, vy=None, for_jump=False):
        jump_substeps = self.substeps
        def substeps_for_velocity(_vx, _vy):
            # substeps for finer grain computation of angles than a timestep might give
            max_speed = max(abs(_vx), abs(_vy))
            estimated_travel = max_speed * self.time_step_interval
            if estimated_travel <= 0:
                return jump_substeps
            return max(jump_substeps, int(math.ceil(estimated_travel / self.max_substep_displacement)))

        if for_jump:
            jump_substeps = max(self.substeps, 8)

        if (vx is None) and (vy is None) and for_jump:
            vx, vy = 0, -self.game_height * 0.14

        jump_gravity = self.gravity * 2 if for_jump else self.gravity

        while True:
            old_x, old_y = x, y

            substeps = substeps_for_velocity(vx, vy)
            sub_dt = self.time_step_interval / substeps

            for _ in range(substeps):
                vy += jump_gravity * sub_dt
                x += vx * sub_dt
                y += vy * sub_dt

                vx *= self.air_drag
                vy *= self.air_drag

            yield old_x, old_y, x, y, vx, vy

    def linear_velocity_gen(self, x):
        vx = 0.0
        power = yield x
        while True:
            if power == 0:
                vx *= self.linear_brake
                if abs(vx) < self.linear_deadband:
                    vx = 0.0
            else:
                if vx != 0 and (vx > 0) != (power > 0):
                    vx *= self.linear_turn_brake
                vx += power * self.linear_force_scale

            vx = max(-self.max_linear_velocity, min(self.max_linear_velocity, vx))
            x += vx

            min_x = self.agent.width / 2
            max_x = self.game_width - self.agent.width / 2
            if x < min_x:
                x = min_x
                vx = 0.0
            elif x > max_x:
                x = max_x
                vx = 0.0

            vx *= self.linear_drag
            power = yield x

    def add_collider(self, bbox_segments):
        # a bounding box is a list of segments: [ s1, s2, s3, s4 ]
        # each segment is a nested tuple ( (p1x, p1y), (p2x, p2y) )
        # assumes bbox is within bounds
        self.collision_segments.append(bbox_segments)

    def _ensure_linear_velocity_function(self):
        if self.agent.linear_velocity_function is None:
            self.agent.linear_velocity_function = self.linear_velocity_gen(self.agent.center_x)
            next(self.agent.linear_velocity_function)

    def move_agent_power(self, power):
        self._ensure_linear_velocity_function()
        old_x = self.agent.center_x
        new_x = self.agent.linear_velocity_function.send(power)
        dx = new_x - old_x
        self.agent.center_x = new_x
        self.agent.bbox = [((x1 + dx, y1), (x2 + dx, y2)) for (x1, y1), (x2, y2) in self.agent.bbox]

    def move_agent_jump(self):
        if not self.agent.is_jumping():
            self.agent.velocity_function = self.velocity_gen(self.agent.center_x, self.agent.center_y, None, None, True)
