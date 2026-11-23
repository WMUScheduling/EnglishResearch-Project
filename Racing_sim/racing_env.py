"""PPO racing environment on an irregular, multi-radius technical circuit."""

import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class RacingEnv(gym.Env):
    """A seeded single-lap racing task with a fixed technical circuit.

    The vehicle is controlled with discrete throttle and steering choices.
    Observations describe heading, lateral position, speed, and two points
    ahead on the route. All distances are measured in screen pixels.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    CONTROL_POINTS = np.array(
        [
            [864.4, 350.0],
            [875.3, 372.7],
            [877.2, 396.0],
            [869.9, 418.5],
            [854.9, 439.3],
            [834.7, 457.9],
            [811.8, 474.4],
            [788.5, 489.6],
            [765.5, 504.0],
            [742.9, 517.9],
            [719.8, 531.1],
            [695.3, 542.8],
            [669.0, 552.3],
            [641.0, 559.0],
            [612.0, 562.8],
            [583.1, 564.2],
            [554.7, 564.3],
            [527.1, 563.8],
            [500.0, 563.5],
            [473.0, 563.6],
            [445.5, 563.5],
            [417.5, 562.9],
            [388.9, 561.1],
            [359.9, 557.7],
            [330.6, 552.8],
            [301.1, 546.3],
            [271.3, 538.4],
            [241.5, 528.7],
            [212.5, 516.7],
            [186.0, 502.0],
            [164.2, 484.0],
            [149.3, 463.0],
            [142.9, 439.8],
            [145.1, 415.7],
            [154.8, 392.1],
            [169.0, 370.0],
            [184.4, 350.0],
            [197.9, 331.7],
            [207.6, 314.4],
            [213.1, 296.9],
            [215.9, 278.5],
            [218.4, 259.2],
            [222.9, 239.4],
            [231.4, 220.0],
            [244.6, 201.9],
            [262.1, 185.6],
            [282.7, 171.0],
            [305.3, 157.8],
            [329.0, 145.2],
            [353.6, 133.0],
            [379.5, 121.1],
            [407.1, 110.3],
            [436.6, 101.4],
            [467.8, 95.6],
            [500.0, 93.5],
            [532.2, 95.3],
            [563.6, 100.7],
            [593.5, 108.9],
            [621.4, 119.4],
            [647.2, 131.7],
            [670.6, 145.7],
            [691.1, 161.3],
            [708.4, 178.4],
            [722.3, 196.4],
            [733.4, 214.6],
            [743.0, 232.4],
            [753.1, 249.0],
            [765.6, 264.4],
            [781.9, 279.1],
            [802.0, 294.1],
            [824.4, 310.5],
            [846.4, 329.1],
        ],
        dtype=np.float64,
    )

    def __init__(self, render_mode=None, track="technical", progress_reward_scale=0.04, off_track_penalty_scale=1.0):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        if track not in ("oval", "technical", "complex"):
            raise ValueError("track must be 'oval', 'technical', or 'complex'")

        if progress_reward_scale <= 0:
            raise ValueError("progress_reward_scale must be positive")
        if off_track_penalty_scale < 0:
            raise ValueError("off_track_penalty_scale cannot be negative")

        self.track_name = track
        self.progress_reward_scale = float(progress_reward_scale)
        self.off_track_penalty_scale = float(off_track_penalty_scale)
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.window_closed = False

        self.track_half_width = {"oval": 50.0, "technical": 22.0, "complex": 18.0}[track]
        self.max_speed = 4.5
        self.max_steps = 4000
        self.max_departures = 6
        self.off_track_limit = 100
        self.max_steer_degrees = 4.0

        self.action_space = spaces.MultiDiscrete([3, 5])
        self.observation_space = spaces.Box(
            low=np.array(
                [-1, -1, -1, 0, -1, -1, -1, -1, -0.25, 0],
                dtype=np.float32,
            ),
            high=np.array(
                [1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.track_centerline = self._build_track(track)
        segment_vectors = np.roll(self.track_centerline, -1, axis=0) - self.track_centerline
        self.segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        self.segment_lengths = np.maximum(self.segment_lengths, 1e-9)
        self.segment_tangents = segment_vectors / self.segment_lengths[:, None]
        self.cumulative_lengths = np.concatenate(
            ([0.0], np.cumsum(self.segment_lengths[:-1]))
        )
        self.track_length = float(np.sum(self.segment_lengths))
        self.track_normals = np.column_stack(
            (-self.segment_tangents[:, 1], self.segment_tangents[:, 0])
        )

        self.steps = 0
        self.progress_distance = 0.0
        self.speed_sum = 0.0
        self.low_speed_steps = 0
        self.on_track_steps = 0
        self.off_track_count = 0
        self.off_track_steps = 0
        self.was_off_track = False
        self.on_track = True
        self.car_speed = 0.0
        self.car_angle = 0.0
        self.car_x = 0.0
        self.car_y = 0.0
        self.last_track_s = 0.0

    @staticmethod
    def _catmull_rom(points, samples_per_segment=8):
        samples = []
        count = len(points)
        for index in range(count):
            p0 = points[(index - 1) % count]
            p1 = points[index]
            p2 = points[(index + 1) % count]
            p3 = points[(index + 2) % count]
            for sample_index in range(samples_per_segment):
                t = sample_index / samples_per_segment
                t2 = t * t
                t3 = t2 * t
                point = 0.5 * (
                    2 * p1
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
                )
                samples.append(point)
        return np.asarray(samples, dtype=np.float64)

    def _build_track(self, track):
        if track == "oval":
            angles = np.linspace(0.0, 2 * math.pi, 36, endpoint=False)
            oval_points = np.column_stack(
                (500 + 300 * np.cos(angles), 350 + 205 * np.sin(angles))
            )
            return self._catmull_rom(oval_points)
        if track == "complex":
            angles = np.linspace(0.0, 2 * math.pi, 96, endpoint=False)
            radius = (
                1.0
                + 0.14 * np.sin(3 * angles + 0.35)
                + 0.08 * np.sin(5 * angles - 0.6)
                + 0.035 * np.cos(8 * angles + 0.2)
            )
            complex_points = np.column_stack(
                (
                    500 + 340 * radius * np.cos(angles),
                    350 + 230 * radius * np.sin(angles),
                )
            )
            return self._catmull_rom(complex_points, samples_per_segment=16)
        return self._catmull_rom(self.CONTROL_POINTS)

    @staticmethod
    def _wrap_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _project_to_track(self):
        position = np.array([self.car_x, self.car_y], dtype=np.float64)
        starts = self.track_centerline
        vectors = np.roll(starts, -1, axis=0) - starts
        relative = position - starts
        lengths_squared = np.einsum("ij,ij->i", vectors, vectors)
        fractions = np.clip(
            np.einsum("ij,ij->i", relative, vectors) / lengths_squared,
            0.0,
            1.0,
        )
        projections = starts + fractions[:, None] * vectors
        distances_squared = np.einsum(
            "ij,ij->i", position - projections, position - projections
        )
        index = int(np.argmin(distances_squared))
        projected = projections[index]
        offset_vector = position - projected
        lateral_offset = float(np.dot(offset_vector, self.track_normals[index]))
        route_s = float(
            self.cumulative_lengths[index]
            + fractions[index] * self.segment_lengths[index]
        )
        tangent = self.segment_tangents[index]
        return route_s, lateral_offset, tangent, projected

    def _point_at_s(self, route_s):
        route_s = route_s % self.track_length
        index = int(np.searchsorted(self.cumulative_lengths, route_s, side="right") - 1)
        index = max(0, min(index, len(self.track_centerline) - 1))
        fraction = (route_s - self.cumulative_lengths[index]) / self.segment_lengths[index]
        point = (
            self.track_centerline[index]
            + fraction
            * (self.track_centerline[(index + 1) % len(self.track_centerline)]
               - self.track_centerline[index])
        )
        return point

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        first_tangent = self.segment_tangents[0]
        start_normal = self.track_normals[0]
        start_offset = float(self.np_random.uniform(-8.0, 8.0))
        start_position = self.track_centerline[0] + start_normal * start_offset
        self.car_x, self.car_y = map(float, start_position)
        base_angle = math.degrees(math.atan2(-first_tangent[1], first_tangent[0]))
        self.car_angle = base_angle + float(self.np_random.uniform(-5.0, 5.0))
        self.car_speed = float(self.np_random.uniform(0.0, 0.4))
        self.steps = 0
        self.progress_distance = 0.0
        self.speed_sum = 0.0
        self.low_speed_steps = 0
        self.on_track_steps = 0
        self.off_track_count = 0
        self.off_track_steps = 0
        self.was_off_track = False
        self.on_track = True
        self.window_closed = False
        self.last_track_s, _, _, _ = self._project_to_track()
        return self._observation(), {}

    def step(self, action):
        throttle = int(action[0])
        steering = int(action[1])

        if throttle == 0:
            self.car_speed = max(0.0, self.car_speed - 0.16)
        elif throttle == 1:
            self.car_speed = max(0.0, self.car_speed - 0.025)
        else:
            self.car_speed = min(self.max_speed, self.car_speed + 0.085)

        steering_amount = steering - 2
        if self.car_speed > 0.05 and steering_amount:
            speed_factor = 0.25 + 0.75 * self.car_speed / self.max_speed
            self.car_angle += (
                steering_amount
                * self.max_steer_degrees
                * speed_factor
            )

        heading = math.radians(self.car_angle)
        self.car_x += math.cos(heading) * self.car_speed
        self.car_y -= math.sin(heading) * self.car_speed

        route_s, lateral_offset, _, _ = self._project_to_track()
        progress_delta = route_s - self.last_track_s
        if progress_delta < -self.track_length / 2:
            progress_delta += self.track_length
        elif progress_delta > self.track_length / 2:
            progress_delta -= self.track_length
        progress_delta = float(np.clip(progress_delta, -self.max_speed * 2, self.max_speed * 2))
        self.last_track_s = route_s
        self.progress_distance += progress_delta

        self.steps += 1
        self.on_track = abs(lateral_offset) <= self.track_half_width
        self.speed_sum += self.car_speed
        self.low_speed_steps += int(self.car_speed < 0.5)
        self.on_track_steps += int(self.on_track)
        if not self.on_track and not self.was_off_track:
            self.off_track_count += 1
        self.was_off_track = not self.on_track

        if self.on_track:
            self.off_track_steps = 0
        else:
            self.off_track_steps += 1

        reward = progress_delta * self.progress_reward_scale - 0.01
        if self.on_track:
            reward += 0.01 * self.car_speed
        else:
            reward -= 0.15 * self.off_track_penalty_scale
        if not self.on_track and self.off_track_steps == 1:
            reward -= 3.0 * self.off_track_penalty_scale

        finished = self.progress_distance >= self.track_length
        stranded = self.off_track_steps >= self.off_track_limit
        failed = self.off_track_count >= self.max_departures or stranded
        terminated = finished or failed
        truncated = self.steps >= self.max_steps and not terminated

        if finished:
            reward += 100.0
        if stranded:
            reward -= 20.0 * self.off_track_penalty_scale

        info = {
            "progress_laps": max(0.0, self.progress_distance / self.track_length),
            "off_track_count": self.off_track_count,
            "on_track": self.on_track,
            "stranded": stranded,
            "finished": finished,
            "is_success": finished,
            "episode_steps": self.steps,
            "average_speed": self.speed_sum / max(self.steps, 1),
            "low_speed_fraction": self.low_speed_steps / max(self.steps, 1),
            "on_track_fraction": self.on_track_steps / max(self.steps, 1),
            "lateral_offset": lateral_offset,
            "speed": self.car_speed,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def _observation(self):
        route_s, lateral_offset, tangent, projected = self._project_to_track()
        desired_angle = math.atan2(-tangent[1], tangent[0])
        heading = math.radians(self.car_angle)
        heading_error = self._wrap_angle(heading - desired_angle)

        forward = np.array([math.cos(heading), -math.sin(heading)])
        left = np.array([forward[1], -forward[0]])
        lookahead_features = []
        for distance in (55.0, 125.0):
            target = self._point_at_s(route_s + distance)
            relative = target - projected
            lookahead_features.extend(
                [
                    float(np.dot(relative, forward) / 140.0),
                    float(np.dot(relative, left) / 140.0),
                ]
            )

        values = np.array(
            [
                math.sin(heading_error),
                math.cos(heading_error),
                np.clip(lateral_offset / self.track_half_width, -1.0, 1.0),
                self.car_speed / self.max_speed,
                *lookahead_features,
                np.clip(self.progress_distance / self.track_length, -0.25, 2.0),
                self.off_track_count / self.max_departures,
            ],
            dtype=np.float32,
        )
        return np.clip(
            values,
            self.observation_space.low,
            self.observation_space.high,
        )

    def render(self):
        if self.render_mode != "human" or self.window_closed:
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((1200, 840))
            pygame.display.set_caption(f"{self.track_name.title()} Racing Track")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 28)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.window_closed = True
                return

        sample_indices = np.arange(0, len(self.track_centerline), 4)
        center_points = self.track_centerline[sample_indices]
        normals = self.track_normals[sample_indices]

        def to_screen(points):
            points = np.asarray(points, dtype=np.float64)
            return np.rint(
                (points - np.array([500.0, 350.0])) * 1.2
                + np.array([600.0, 420.0])
            ).astype(np.int32)

        centerline = to_screen(center_points)
        left_edge = to_screen(
            center_points + self.track_half_width * normals
        )
        right_edge = to_screen(
            center_points - self.track_half_width * normals
        )
        shoulder_left = to_screen(
            center_points + (self.track_half_width + 6.0) * normals
        )
        shoulder_right = to_screen(
            center_points - (self.track_half_width + 6.0) * normals
        )

        self.screen.fill((25, 43, 40))
        for x in range(0, 1200, 60):
            pygame.draw.line(self.screen, (32, 52, 47), (x, 0), (x, 840), 1)
        for y in range(0, 840, 60):
            pygame.draw.line(self.screen, (32, 52, 47), (0, y), (1200, y), 1)

        shoulder_polygon = np.concatenate(
            (shoulder_left, shoulder_right[::-1])
        )
        road_polygon = np.concatenate((left_edge, right_edge[::-1]))
        pygame.draw.polygon(self.screen, (13, 19, 22), shoulder_polygon)
        pygame.draw.polygon(self.screen, (58, 63, 68), road_polygon)
        pygame.draw.aalines(self.screen, (230, 234, 232), True, left_edge)
        pygame.draw.aalines(self.screen, (230, 234, 232), True, right_edge)

        dash_stride = max(
            1,
            int(
                34
                / max(float(np.mean(self.segment_lengths)) * 1.2 * 4, 1.0)
            ),
        )
        for index in range(0, len(centerline), dash_stride * 2):
            start_dash = centerline[index]
            end_dash = centerline[(index + dash_stride) % len(centerline)]
            pygame.draw.line(
                self.screen,
                (224, 194, 102),
                tuple(map(int, start_dash)),
                tuple(map(int, end_dash)),
                2,
            )

        start = self.track_centerline[0]
        tangent = self.segment_tangents[0]
        normal = np.array([-tangent[1], tangent[0]])
        start_a = to_screen(start - normal * self.track_half_width)
        start_b = to_screen(start + normal * self.track_half_width)
        pygame.draw.line(
            self.screen,
            (255, 213, 72),
            tuple(map(int, start_a)),
            tuple(map(int, start_b)),
            6,
        )

        car_image = pygame.Surface((41, 22), pygame.SRCALPHA)
        pygame.draw.rect(car_image, (232, 58, 55), (0, 0, 41, 22), border_radius=5)
        pygame.draw.rect(car_image, (170, 224, 243), (27, 4, 10, 14), border_radius=3)
        rotated_car = pygame.transform.rotate(car_image, self.car_angle)
        car_position = to_screen([self.car_x, self.car_y])
        car_rectangle = rotated_car.get_rect(center=tuple(map(int, car_position)))

        self.screen.blit(rotated_car, car_rectangle)

        progress = max(0.0, self.progress_distance / self.track_length)
        status = "ON TRACK" if self.on_track else "OFF TRACK"
        color = (125, 235, 170) if self.on_track else (255, 115, 115)
        panel = pygame.Surface((560, 94), pygame.SRCALPHA)
        panel.fill((12, 20, 23, 225))
        self.screen.blit(panel, (370, 350))
        title = self.font.render(
            f"{self.track_name.upper()} CIRCUIT",
            True,
            (238, 241, 238),
        )
        label = self.font.render(
            f"{status}   Speed {self.car_speed:.2f}   Lap {progress:.2f}",
            True,
            color,
        )
        detail = self.font.render(
            f"Departures {self.off_track_count}",
            True,
            (205, 214, 211),
        )
        self.screen.blit(title, (386, 360))
        self.screen.blit(label, (386, 391))
        self.screen.blit(detail, (782, 391))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
