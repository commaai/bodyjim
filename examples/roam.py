#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import pygame
import onnxruntime as ort

from bodyjim import BodyEnv
from bodyjim.wrappers import SkipFrame, LagTracker


@dataclass
class Config:
  max_speed: float = 1.0 # adjust between 0.0 and 1.0
  use_action_t: int = 2 # 0-15


GPT_PATH = Path(__file__).parent / 'gpt2.onnx'
TOKENIZER_PATH = Path(__file__).parent / 'encoder.onnx'
INPUT_SHAPE = (256, 160)
FRAME_TOKENS = 163
FPS = 5 # model has been trained at 5 fps
CAMERA_FPS = 20

WHEEL_SPEEDS_RANGE = [-150, 150]
WHEEL_SPEEDS_VOCAB_SIZE = 512
WHEEL_SPEED_BINS = np.linspace(WHEEL_SPEEDS_RANGE[0], WHEEL_SPEEDS_RANGE[1], WHEEL_SPEEDS_VOCAB_SIZE)


class GPTRunner:
  def __init__(self, config: Config = Config()):
    self.config = config
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

  def tokenize_wheel_speed(self, speed):
    speed = np.clip(speed, WHEEL_SPEEDS_RANGE[0], WHEEL_SPEEDS_RANGE[1])
    return np.digitize(speed, WHEEL_SPEED_BINS, right=True)

  def detokenize_actions(self, actions):
    actions = np.clip(actions, 0, 9 - 1)
    ad, ws = np.divmod(actions, 3)
    return np.concatenate([ws, ad], axis=-1)

  def wasd_to_xy(self, wasd):
    ws, ad = wasd[0], wasd[1]

    action = [0.0, 0.0]
    if ws == 0: # W
      action[0] += -1.0 * self.config.max_speed
    elif ws == 2: # S
      action[0] += 1.0 * self.config.max_speed

    if ad == 2: # A
      action[1] += 1 * self.config.max_speed
    elif ad == 0: # D
      action[1] += -1 * self.config.max_speed

    return action

  def run(self, img: np.ndarray, wheel_speeds: dict):
    img_tokens = self.tokenize_frame(img)
    wheel_speeds = np.array([wheel_speeds["fl"], wheel_speeds["fr"]])
    wheel_speeds_tokens = np.expand_dims(self.tokenize_wheel_speed(wheel_speeds), 0)

    tokens = np.concatenate([img_tokens, wheel_speeds_tokens], axis=1)

    # drop first frame+wheel speed+action of the context, concatenate new tokens
    self.context = np.concatenate([self.context[:, FRAME_TOKENS:], self.last_action, tokens], axis=1)

    # run GPT model, get action plan
    plan_probs = self.gpt_session.run(None, {'tokens': self.context})[1].reshape(16, 9) # 16 actions
    plan = np.argmax(plan_probs, axis=1).reshape(-1, 1)
    action = plan[self.config.use_action_t]
    self.last_action = action.reshape(1, 1)

    plan = self.detokenize_actions(plan)

    return plan[self.config.use_action_t], plan


def roam(body_ip):
  env = BodyEnv(body_ip, ["driver"], ["carState"], render_mode="human")
  n_skip_frames = int(CAMERA_FPS / FPS)
  env = LagTracker(SkipFrame(env, n=n_skip_frames), control_freq=FPS)
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
    obs, _, _, _, _ = env.step(runner.wasd_to_xy(action))
    overlay_wasd(env.unwrapped._last_observation["cameras"]["driver"], action)


def overlay_wasd(image, wasd_tokens, position=(100, 200), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=8, color=(255, 127, 14), thickness=10):
    sizes = {}
    for c in [' ', 'W', 'A', 'S', 'D']:
      w, h = cv2.getTextSize(c, font, font_scale, thickness)[0]
      sizes[c] = {"width": w, "height": h}
    
    ws_token_to_char = {0: 'W', 2: 'S', 1: None}
    ad_token_to_char = {2: 'A', 0: 'D', 1: None}
    
    interline_margin = 5
    positions = {} # compute where to draw each letter
    asd_position = (position[0], position[1] + sizes["W"]["height"] + interline_margin)
    positions["W"] = (position[0] + sizes[" "]["width"], position[1])
    positions["A"] = (asd_position[0], asd_position[1])
    positions["S"] = (positions["A"][0] + sizes["A"]["width"], asd_position[1])
    positions["D"] = (positions["S"][0] + sizes["S"]["width"], asd_position[1])

    for c in ['W', 'A', 'S', 'D']: # overlay wasd in white
      image = cv2.putText(image, c, positions[c], font, font_scale, (255, 255, 255), thickness)

    char_to_print = [ws_token_to_char[wasd_tokens[0]], ad_token_to_char[wasd_tokens[1]]] # overlay pressed letter in {color}
    for c in char_to_print:
      if c is not None:
        image = cv2.putText(image, c, positions[c], font, font_scale, color, thickness)

    return image


if __name__=="__main__":
  parser = argparse.ArgumentParser("Run model to roam around")
  parser.add_argument("body_ip", help="IP address of the body")
  args = parser.parse_args()

  roam(args.body_ip)
