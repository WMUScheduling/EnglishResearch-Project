"""Keyboard demonstration of the technical racing circuit."""

import pygame

from racing_env import RacingEnv


def main():
    env = RacingEnv(render_mode="human")
    env.reset()
    running = True

    while running and not env.window_closed:
        env.render()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle = 2
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            throttle = 0
        else:
            throttle = 1

        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steering = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                steering = 0
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steering = 3
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                steering = 4
        else:
            steering = 2

        _, _, terminated, truncated, _ = env.step((throttle, steering))
        if terminated or truncated:
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
