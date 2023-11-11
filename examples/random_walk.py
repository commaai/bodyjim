#!/usr/bin/env python3
import argparse

from bodyjim import BodyEnv

import pygame


def run_random_walk(body_ip, cameras):
  env = BodyEnv(body_ip, cameras, [], render_mode="human")
  env.reset()

  while True:
    env.render()

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        env.close()
        return

    # sample from action space and perform a step
    action = env.action_space.sample()
    _, _, _, _, _ = env.step(action) # obs, reward, done, _, info


if __name__=="__main__":
  parser = argparse.ArgumentParser("Random walk")
  parser.add_argument("body_ip", help="IP address of the body")
  parser.add_argument("cameras", nargs="*", default=["driver"], help="List of cameras to render")
  args = parser.parse_args()

  run_random_walk(args.body_ip, args.cameras)
