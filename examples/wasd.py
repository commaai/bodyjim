#!/usr/bin/env python3
import argparse

from bodyjim import BodyEnv

import pygame


def run_wasd(body_ip, cameras):
  # initialize body environment
  # specify which cameras to stream (driver, road, wideRoad) and which services to subscribe to 
  # supported services: https://github.com/commaai/cereal/blob/master/log.capnp
  env = BodyEnv(body_ip, cameras, ["accelerometer", "gyroscope", "gpsLocation"], render_mode="human")
  # reset environment to intial state
  env.reset()

  while True:
    # render frame - in human rendering mode pygame window is created and updated in real time
    env.render()

    # generate action from keyboard
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

    # observation dict contains cameras at obs["cameras"][camera_name]
    # and other services at obs[service_name]
    # both action and observation spaces can be randomly sampled from
    obs, _, _, _, info = env.step(action)
    
    # observation dictionary structure is static, which means every field is always present, even if some messages havent been received yet.
    # in that case they will be filled with random values (can be examined if/when received using info["timestamps"][service_name])
    print("Acceleration:", obs["accelerometer"]["acceleration"]["v"], "updated at", info["timestamps"]["accelerometer"])
    print("Gyroscope:", obs["gyroscope"]["gyro"]["v"], "updated at", info["timestamps"]["gyroscope"])
    print("GPS: latitude", obs["gpsLocation"]["latitude"], "longitude", obs["gpsLocation"]["longitude"], "updated at", info["timestamps"]["gpsLocation"])


if __name__ == "__main__":
  parser = argparse.ArgumentParser("WASD controller for the body")
  parser.add_argument("body_ip", help="IP address of the body")
  parser.add_argument("cameras", nargs="*", default=["driver"], help="List of cameras to render")
  args = parser.parse_args()

  run_wasd(args.body_ip, args.cameras)
