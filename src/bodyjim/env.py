from typing import Any, Dict, List, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from bodyjim.data_stream import DataStreamSession


class BaseBodyEnv(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

  def __init__(self, body_ip: str, cameras: List[str], services: List[str], render_mode=None):
    super().__init__()

    self.body_ip = body_ip
    self.cameras = cameras
    self.services = services

    self.data_stream: Optional[DataStreamSession] = None
    self.current_state = None

    self.observation_space = spaces.Dict({
      "cameras": spaces.Dict({
        cam: spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8) for cam in cameras
      }),
      **{service: spaces.Dict() for service in services}
    })
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    
    assert render_mode is None or render_mode in self.metadata["render_modes"]
    self.render_mode = render_mode

  def _initialize_data_stream(self):
    self.data_stream = DataStreamSession(cameras=self.cameras, services=self.services, webrtcd_host=self.body_ip)
    self.data_stream.start()

  @property
  def observation(self):


  def step(self, action):
    if self.data_stream is None:
      self._initialize_data_stream()
    
    frames, messages, valid, times = self.data_stream.receive()

  def reset(self, seed: int, options: Dict[str, Any]):
    if self.data_stream is None:
      self.data_stream.stop()
    self._initialize_data_stream()

  def render(self):
    pass

  def close(self):
    return
