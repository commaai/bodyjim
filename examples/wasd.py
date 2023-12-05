#!/usr/bin/env python3
import argparse

from bodyjim import BodyEnv

import pygame


def run_wasd(body_ip, cameras):
  env = BodyEnv(body_ip, cameras, [], render_mode="human")
  env.reset()

  while True:
    env.render()

    action = [0, 0]
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        env.close()
        return

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
      action[0] += -1
    if keys[pygame.K_a]:
      action[1] += 1
    if keys[pygame.K_s]:
      action[0] += 1
    if keys[pygame.K_d]:
      action[1] += -1

    _, _, _, _, _ = env.step(action)


if __name__ == "__main__":
  parser = argparse.ArgumentParser("WASD controller for the body")
  parser.add_argument("body_ip", help="IP address of the body")
  parser.add_argument("cameras", nargs="*", default=["driver"], help="List of cameras to render")
  args = parser.parse_args()

  run_wasd(args.body_ip, args.cameras)
