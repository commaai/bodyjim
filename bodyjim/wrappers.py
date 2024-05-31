import time
from typing import Optional

import gymnasium as gym


class SkipFrame(gym.Wrapper):
  def __init__(self, env, n: int):
    super().__init__(env)
    self.n = n
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
