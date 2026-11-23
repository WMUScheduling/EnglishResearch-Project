import math
import pygame

pygame.init()

screen = pygame.display.set_mode((1000, 700))
pygame.display.set_caption("Racing Simulation")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)

car_x = 500
car_y = 560
car_angle = 0
car_speed = 0

off_track_count = 0
was_off_track = False
on_track = True

running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP] or keys[pygame.K_w]:
        car_speed = min(car_speed + 0.08, 5)
    elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
        car_speed = max(car_speed - 0.12, 0)
    else:
        car_speed = max(car_speed - 0.03, 0)

    if car_speed > 0.2:
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            car_angle += 2.2 * (car_speed / 5)

        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            car_angle -= 2.2 * (car_speed / 5)

    radians = math.radians(car_angle)
    car_x += math.cos(radians) * car_speed
    car_y -= math.sin(radians) * car_speed

    outer_value = ((car_x - 500) / 410) ** 2 + ((car_y - 350) / 260) ** 2
    inner_value = ((car_x - 500) / 250) ** 2 + ((car_y - 350) / 160) ** 2

    on_track = outer_value <= 1 and inner_value >= 1

    if not on_track and not was_off_track:
        off_track_count += 1

    was_off_track = not on_track

    screen.fill((48, 120, 62))

    pygame.draw.ellipse(screen, (85, 85, 85), (90, 90, 820, 520))
    pygame.draw.ellipse(screen, (48, 120, 62), (250, 190, 500, 320))

    car_image = pygame.Surface((40, 22), pygame.SRCALPHA)
    pygame.draw.rect(car_image, (220, 45, 45), (0, 0, 40, 22))
    pygame.draw.rect(car_image, (160, 220, 240), (25, 4, 10, 14))

    rotated_car = pygame.transform.rotate(car_image, car_angle)
    car_rectangle = rotated_car.get_rect(center=(car_x, car_y))
    screen.blit(rotated_car, car_rectangle)

    label = font.render(
        "Up or W to accelerate   Left and Right to steer",
        True,
        (255, 255, 255)
    )
    screen.blit(label, (20, 20))

    if on_track:
        status = "ON TRACK"
        status_color = (255, 255, 255)
    else:
        status = "OFF TRACK"
        status_color = (255, 80, 80)

    status_label = font.render(
        f"Track status: {status}   Departures: {off_track_count}",
        True,
        status_color
    )
    screen.blit(status_label, (20, 52))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()