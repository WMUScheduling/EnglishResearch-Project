"""PPO racing environment on a multi-turn closed circuit."""

import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class RacingEnv(gym.Env):
    """A deterministic, single-lap racing task with a fixed technical circuit.

    The vehicle is controlled with discrete throttle and steering choices.
    Observations describe heading, lateral position, speed, and two points
    ahead on the route. All distances are measured in screen pixels.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    CONTROL_POINTS = np.array(
        [
            [145, 530], [235, 565], [325, 540], [385, 480],
            [395, 410], [355, 350], [280, 330], [205, 365],
            [145, 335], [125, 270], [160, 210], [235, 180],
            [315, 195], [365, 250], [420, 305], [490, 285],
            [535, 220], [555, 155], [625, 125], [710, 145],
            [790, 185], [855, 240], [865, 305], [825, 355],
            [760, 355], [700, 325], [650, 350], [625, 405],
            [650, 470], [715, 505], [790, 515], [835, 560],
            [805, 605], [730, 620], [655, 600], [595, 555],
            [550, 505], [495, 485], [445, 520], [410, 575],
            [345, 610], [265, 605], [195, 580],
        ],
        dtype=np.float64,
    )

    def __init__(self, render_mode=None):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.window_closed = False

        self.track_half_width = 42.0
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

        self.track_centerline = self._build_track()
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

    def _build_track(self):
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
        self.car_x, self.car_y = map(float, self.track_centerline[0])
        first_tangent = self.segment_tangents[0]
        self.car_angle = math.degrees(math.atan2(-first_tangent[1], first_tangent[0]))
        self.car_speed = 0.0
        self.steps = 0
        self.progress_distance = 0.0
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
        if not self.on_track and not self.was_off_track:
            self.off_track_count += 1
        self.was_off_track = not self.on_track

        if self.on_track:
            self.off_track_steps = 0
        else:
            self.off_track_steps += 1

        reward = progress_delta * 0.04 - 0.01
        if self.on_track:
            reward += 0.01 * self.car_speed
        else:
            reward -= 0.15
        if not self.on_track and self.off_track_steps == 1:
            reward -= 3.0

        finished = self.progress_distance >= self.track_length
        stranded = self.off_track_steps >= self.off_track_limit
        failed = self.off_track_count >= self.max_departures or stranded
        terminated = finished or failed
        truncated = self.steps >= self.max_steps and not terminated

        if finished:
            reward += 100.0
        if stranded:
            reward -= 20.0

        info = {
            "progress_laps": max(0.0, self.progress_distance / self.track_length),
            "off_track_count": self.off_track_count,
            "on_track": self.on_track,
            "stranded": stranded,
            "finished": finished,
            "is_success": finished,
            "episode_steps": self.steps,
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
            self.screen = pygame.display.set_mode((1000, 700))
            pygame.display.set_caption("Multi-Turn PPO Racing Track")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.window_closed = True
                return

        left_edge = self.track_centerline + self.track_half_width * self.track_normals
        right_edge = self.track_centerline - self.track_half_width * self.track_normals
        road_polygon = np.concatenate((left_edge, right_edge[::-1])).astype(np.int32)

        self.screen.fill((48, 120, 62))
        pygame.draw.polygon(self.screen, (220, 220, 205), road_polygon)
        pygame.draw.lines(
            self.screen, (75, 75, 75), True, self.track_centerline.astype(np.int32), 78
        )
        pygame.draw.lines(
            self.screen, (245, 245, 245), True, left_edge.astype(np.int32), 3
        )
        pygame.draw.lines(
            self.screen, (245, 245, 245), True, right_edge.astype(np.int32), 3
        )

        start = self.track_centerline[0]
        tangent = self.segment_tangents[0]
        normal = np.array([-tangent[1], tangent[0]])
        start_a = start - normal * self.track_half_width
        start_b = start + normal * self.track_half_width
        pygame.draw.line(
            self.screen,
            (255, 210, 40),
            start_a.astype(int),
            start_b.astype(int),
            5,
        )

        car_image = pygame.Surface((34, 18), pygame.SRCALPHA)
        pygame.draw.rect(car_image, (220, 45, 45), (0, 0, 34, 18), border_radius=4)
        pygame.draw.rect(car_image, (160, 220, 240), (22, 3, 8, 12), border_radius=2)
        rotated_car = pygame.transform.rotate(car_image, self.car_angle)
        car_rectangle = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, car_rectangle)

        progress = max(0.0, self.progress_distance / self.track_length)
        status = "ON TRACK" if self.on_track else "OFF TRACK"
        color = (255, 255, 255) if self.on_track else (255, 80, 80)
        label = self.font.render(
            f"{status}  Speed: {self.car_speed:.2f}  Progress: {progress:.2f} laps"
            f"  Departures: {self.off_track_count}",
            True,
            color,
        )
        self.screen.blit(label, (18, 18))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
