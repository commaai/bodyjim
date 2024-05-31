#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnx
import pygame
import onnxruntime as ort
import gymnasium as gym
import time

from bodyjim import BodyEnv


MAX_SPEED = 0.8 # adjust between 0.0 and 1.0
USE_ACTION_T = 2

GPT_PATH = Path(__file__).parent / 'gpt2.onnx'
TOKENIZER_PATH = Path(__file__).parent / 'encoder.onnx'

INPUT_SHAPE = (256, 160)
FRAME_TOKENS = 163
FPS = 5 # model has been trained at 5 fps
CAMERA_FPS = 20

WHEEL_SPEEDS_RANGE = [-150, 150]
WHEEL_SPEEDS_VOCAB_SIZE = 512
WHEEL_SPEED_BINS = np.linspace(WHEEL_SPEEDS_RANGE[0], WHEEL_SPEEDS_RANGE[1], WHEEL_SPEEDS_VOCAB_SIZE)


def tokenize_wheel_speed(speed):
  speed = np.clip(speed, WHEEL_SPEEDS_RANGE[0], WHEEL_SPEEDS_RANGE[1])
  return np.digitize(speed, WHEEL_SPEED_BINS, right=True)


def detokenize_actions(actions):
  actions = np.clip(actions, 0, 9 - 1)
  ad, ws = np.divmod(actions, 3)
  return np.concatenate([ws, ad], axis=-1)


def wasd_to_xy(wasd):
  ws, ad = wasd[0], wasd[1]

  action = [0.0, 0.0]
  if ws == 0: # W
    action[0] += -1.0 * MAX_SPEED
  elif ws == 2: # S
    action[0] += 1.0 * MAX_SPEED

  if ad == 2: # A
    action[1] += 1 * MAX_SPEED
  elif ad == 0: # D
    action[1] += -1 * MAX_SPEED

  return action


class GPTRunner:
  def __init__(self):
    if not GPT_PATH.exists():
      print("Downloading GPT model...")
      os.system(f'wget -P {GPT_PATH.parent} {"https://huggingface.co/commaai/commabody-gpt2/resolve/main/gpt2.onnx"}')

    if not TOKENIZER_PATH.exists():
      print("Downloading tokenizer...")
      os.system(f'wget -P {TOKENIZER_PATH.parent} {"https://huggingface.co/commaai/commabody-gpt2/resolve/main/encoder.onnx"}')

    options = ort.SessionOptions()
    self.tokenizer_session = ort.InferenceSession(TOKENIZER_PATH, options, ['CUDAExecutionProvider'])
    self.gpt_session = ort.InferenceSession(GPT_PATH, options, ['CUDAExecutionProvider'])

    inputs = {input.name: input for input in self.gpt_session.get_inputs()}
    context_size = inputs['tokens'].shape[1]
    self.context = np.zeros((1, context_size), dtype=np.int64)
    self.last_action = np.zeros((1, 1), dtype=np.int64)

  def tokenize_frame(self, img):
    img = cv2.resize(img, INPUT_SHAPE)
    img = np.expand_dims(img, 0).astype(np.float32) # add batch dim
    img = img.transpose(0, 3, 1, 2) # NHWC -> NCHW
    img_tokens = self.tokenizer_session.run(None, {'img': img})[0].reshape(1, -1)
    return img_tokens

  def run(self, img: np.ndarray, wheel_speeds: dict):
    img_tokens = self.tokenize_frame(img)
    wheel_speeds = np.array([wheel_speeds["fl"], wheel_speeds["fr"]])
    wheel_speeds_tokens = np.expand_dims(tokenize_wheel_speed(wheel_speeds), 0)

    tokens = np.concatenate([img_tokens, wheel_speeds_tokens], axis=1)

    # drop first frame+wheel speed+action of the context, concatenate new tokens
    self.context = np.concatenate([self.context[:, FRAME_TOKENS:], self.last_action, tokens], axis=1)

    # run GPT model, get action plan
    plan_probs = self.gpt_session.run(None, {'tokens': self.context})[1].reshape(16, 9) # 16 actions
    plan = np.argmax(plan_probs, axis=1).reshape(-1, 1)
    action = plan[USE_ACTION_T]
    self.last_action = action.reshape(1, 1)

    plan = detokenize_actions(plan)

    return plan[USE_ACTION_T], plan


def roam(body_ip):
  env = BodyEnv(body_ip, ["driver"], ["carState"], render_mode="human")
  env = LagTracker(SkipFrame(env, control_freq=FPS), control_freq=FPS)
  obs, _ = env.reset()

  print("Loading model...")
  runner = GPTRunner()

  print("Running...")
  while True:
    env.render()

    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        env.close()
        return

    action, _ = runner.run(obs["cameras"]["driver"], obs["carState"]["wheelSpeeds"])

    obs, _, _, _, _ = env.step(wasd_to_xy(action))


class SkipFrame(gym.Wrapper):
  def __init__(self, env, control_freq: int):
    super().__init__(env)
    self.n = int(CAMERA_FPS / control_freq)
    print(f"Skipping {self.n} frames per action")

  def step(self, action):
    for _ in range(self.n):
      obs, reward, terminated, truncated, info = self.env.step(action)

    return obs, reward, terminated, truncated, info


class LagTracker(gym.Wrapper):
  """make sure the model is running fast enough"""

  def __init__(self, env, control_freq: int):
    super().__init__(env)
    self.dt = 1/control_freq

  def step(self, action):
    current = time.time()
    loop_time = 0
    if self.last is not None:
      loop_time = current - self.last
      if loop_time > self.dt:
        print(f"warning, lagging, max freq {1/loop_time} Hz, expected {1/self.dt} Hz")

    obs, reward, terminated, truncated, info = self.env.step(action)
    self.last = time.time()

    return obs, reward, terminated, truncated, info

  def reset(
      self,
      seed: Optional[int] = None,
      options: Optional[dict] = None,
    ):
    self.last = None
    return self.env.reset(seed=seed, options=options)


if __name__=="__main__":
  parser = argparse.ArgumentParser("Run model to roam around")
  parser.add_argument("body_ip", help="IP address of the body")
  args = parser.parse_args()

  roam(args.body_ip)
