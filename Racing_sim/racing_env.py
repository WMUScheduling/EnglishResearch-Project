import math

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class RacingEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.window_closed = False
        self.on_track = True

        # Each action has a throttle choice and a steering choice.
        self.action_space = spaces.MultiDiscrete([3, 3])

        # Observation: position, heading, speed, and departures.
        low = np.array([-2, -2, -1, -1, 0, 0], dtype=np.float32)
        high = np.array([2, 2, 1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        self.max_steps = 6000
        self.max_departures = 3
        self.off_track_limit = 120

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.car_x = 500.0
        self.car_y = 560.0
        self.car_angle = 0.0
        self.car_speed = 0.0
        self.on_track = True
        self.off_track_steps = 0

        self.off_track_count = 0
        self.was_off_track = False
        self.steps = 0
        self.progress = 0.0
        self.last_theta = self._track_angle()

        return self._observation(), {}

    def step(self, action):
        throttle = int(action[0])
        steering = int(action[1])

        if throttle == 1:
            self.car_speed = min(self.car_speed + 0.08, 5)
        elif throttle == 2:
            self.car_speed = max(self.car_speed - 0.12, 0)
        else:
            self.car_speed = max(self.car_speed - 0.03, 0)

        if self.car_speed > 0.2:
            if steering == 1:
                self.car_angle += 2.2 * (self.car_speed / 5)
            elif steering == 2:
                self.car_angle -= 2.2 * (self.car_speed / 5)

        radians = math.radians(self.car_angle)
        self.car_x += math.cos(radians) * self.car_speed
        self.car_y -= math.sin(radians) * self.car_speed

        outer_value = (
            ((self.car_x - 500) / 410) ** 2
            + ((self.car_y - 350) / 260) ** 2
        )
        inner_value = (
            ((self.car_x - 500) / 250) ** 2
            + ((self.car_y - 350) / 160) ** 2
        )
        on_track = outer_value <= 1 and inner_value >= 1

        if not on_track and not self.was_off_track:
            self.off_track_count += 1

        self.was_off_track = not on_track

        current_theta = self._track_angle()
        progress_change = (
            self.last_theta - current_theta + math.pi
        ) % (2 * math.pi) - math.pi
        self.last_theta = current_theta

        self.steps += 1
        reward = -0.002

        if on_track:
            self.off_track_steps = 0
        else: 
            self.off_track_steps += 1

        if on_track:
            self.on_track = on_track
            self.progress += progress_change
            reward += progress_change * 10
        else:
            reward -= 0.1

        if not on_track and self.off_track_count > 0:
            if not getattr(self, "penalized_this_departure", False):
                reward -= 1.0
                self.penalized_this_departure = True
        elif on_track:
            self.penalized_this_departure = False

        finished = self.progress >= 2 * math.pi
        stranded = self.off_track_steps >= self.off_track_limit
        failed = self.off_track_count >= self.max_departures or stranded
        terminated = finished or failed
        truncated = self.steps >= self.max_steps and not terminated

        if finished:
            reward += 100

        if stranded:
            reward -= 20

        info = {
            "progress_laps": self.progress / (2 * math.pi),
            "off_track_count": self.off_track_count,
            "on_track": on_track,
            "stranded": stranded,
        }

        return self._observation(), reward, terminated, truncated, info

    def _track_angle(self):
        return math.atan2(
            (self.car_y - 350) / 210,
            (self.car_x - 500) / 330,
        )

    def _observation(self):
        theta = self._track_angle()

        tangent_angle = math.atan2(
            210 * math.cos(theta),
            330 * math.sin(theta),
        )

        heading = math.radians(self.car_angle)
        heading_error = (
            heading - tangent_angle + math.pi
        ) % (2 * math.pi) - math.pi

        values = np.array(
            [
                (self.car_x - 500) / 410,
                (self.car_y - 350) / 260,
                math.cos(heading_error),
                math.sin(heading_error),
                self.car_speed / 5,
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
            pygame.display.set_caption("PPO Racing Agent")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 28)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.window_closed = True
                return

        self.screen.fill((48, 120, 62))

        pygame.draw.ellipse(self.screen, (85, 85, 85), (90, 90, 820, 520))
        pygame.draw.ellipse(self.screen, (48, 120, 62), (250, 190, 500, 320))

        car_image = pygame.Surface((40, 22), pygame.SRCALPHA)
        pygame.draw.rect(car_image, (220, 45, 45), (0, 0, 40, 22))
        pygame.draw.rect(car_image, (160, 220, 240), (25, 4, 10, 14))

        rotated_car = pygame.transform.rotate(car_image, self.car_angle)
        car_rectangle = rotated_car.get_rect(center=(self.car_x, self.car_y))
        self.screen.blit(rotated_car, car_rectangle)

        if self.on_track:
            status = "ON TRACK"
            status_color = (255, 255, 255)
        else:
            status = "OFF TRACK"
            status_color = (255, 80, 80)

        progress = max(0, self.progress / (2 * math.pi))
        label = self.font.render(
            f"{status}   Speed: {self.car_speed:.2f}    Progress: {progress:.2f} laps   Departures: {self.off_track_count}",
            True,
            status_color,
        )
        self.screen.blit(label, (20, 20))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
    