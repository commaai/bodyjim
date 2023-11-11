import asyncio
from typing import Any, Dict, List, Optional, Tuple, Mapping

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
import numpy as np
import pygame

from bodyjim.data_stream import DataStreamSession, WebrtcdClient
from bodyjim.schema import space_from_schema

TICI_IMAGE_SIZE = (1208, 1928)


def update_obs_recursive(obs, new_obs):
  for key, value in new_obs.items():
    if isinstance(value, Mapping):
      obs[key] = update_obs_recursive(obs.get(key, {}), value)
    elif isinstance(value, List): # convert all lists to tuples
      obs[key] = tuple(value)
    else:
      obs[key] = value

  return obs


class BodyEnv(gym.Env):
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

  def __init__(self, body_ip: str, cameras: List[str], services: List[str], render_mode=None):
    super().__init__()

    # retrieve schema from device needed for observation space
    client = WebrtcdClient(body_ip)
    schema = asyncio.run(client.get_schema(services))

    self._cameras = cameras
    self._services = services

    self._client = client
    self._data_stream: Optional[DataStreamSession] = None
    self._last_observation: Optional[ObsType] = None
    self._last_info: Optional[Dict[str, Any]] = None
    self._window: Optional[pygame.Surface] = None

    self.observation_space = spaces.Dict({
      "cameras": spaces.Dict({
        cam: spaces.Box(low=0, high=255, shape=(*TICI_IMAGE_SIZE, 3), dtype=np.uint8) for cam in cameras
      }),
      **space_from_schema(schema)
    })
    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    assert render_mode is None or render_mode in self.metadata["render_modes"] # type: ignore
    self.render_mode = render_mode

  def _initialize_data_stream(self):
    if self._data_stream is not None:
      self._data_stream.stop()
    self._data_stream = DataStreamSession(self._client, self._cameras, self._services)
    self._data_stream.start()

  def _get_observation_and_info(self) -> ObsType:
    assert self._data_stream is not None
    frames, messages, valid, times = self._data_stream.receive()
    if self._last_observation is None:
      new_obs = self.observation_space.sample()
    else:
      new_obs = self._last_observation.copy()

    info = {"timestamps": times, "valid": valid}
    update_obs_recursive(new_obs, {
      "cameras": frames,
      **{service: messages[service] for service in self._services if service in messages and messages[service] is not None}
    })

    return new_obs, info

  @property
  def last_observation(self) -> Optional[ObsType]:
    return self._last_observation

  @property
  def last_info(self) -> Optional[Dict[str, Any]]:
    return self._last_info

  def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
    if self._data_stream is None:
      self._initialize_data_stream()

    assert self._data_stream is not None
    x, y = action
    self._data_stream.send(float(x), float(y))
    obs, info = self._get_observation_and_info()
    reward = self.reward(self._last_observation, action, obs)
    done = self.is_done(self._last_observation, action, obs)

    self._last_observation = obs
    self._last_info = info

    return obs, reward, done, False, info

  def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
    super().reset(seed=seed)

    if self._data_stream is None or (options is not None and options.get("reconnect", False)):
      self._initialize_data_stream()

    obs, info = self._get_observation_and_info()

    self._last_observation = obs
    self._last_info = info

    return obs, info

  def render(self):
    if self.render_mode is None:
      return

    assert self._last_observation is not None
    frames = [self._last_observation["cameras"][cam] for cam in self._cameras]
    image = np.hstack(frames)

    if self.render_mode == "rgb_array":
      return image
    else:
      black, white = (0, 0, 0), (255, 255, 255)
      top_margin, text_offset, single_cam_width = 50, 25, 800
      if self._window is None:
        pygame.init()
        pygame.display.init()
        scaling_factor = single_cam_width / (image.shape[1] / len(self._cameras))
        desired_camera_width, desired_camera_height = single_cam_width * len(self._cameras), scaling_factor * image.shape[0]
        self._window = pygame.display.set_mode((desired_camera_width, desired_camera_height + top_margin))
        self._image_surface = pygame.Surface((image.shape[1], image.shape[0]))
        self._surface = pygame.Surface((desired_camera_width, desired_camera_height))

      width, _ = self._window.get_size()

      self._window.fill(black)

      font = pygame.font.SysFont("Arial", 36)
      for i, cam in enumerate(self._cameras):
        text = font.render(f"{cam}", True, white, black)
        text_rect = text.get_rect()
        text_rect.center = (int((i + 0.5)* (width / len(self._cameras))), text_offset)
        self._window.blit(text, text_rect)

      pygame.surfarray.blit_array(self._image_surface, image.swapaxes(0, 1))
      pygame.transform.scale(self._image_surface, self._surface.get_size(), dest_surface=self._surface)
      self._window.blit(self._surface, (0, top_margin))
      pygame.display.flip()

  def close(self):
    if self._window is not None:
      pygame.display.quit()
      pygame.quit()
      self._window = None
      self._suface = None
    if self._data_stream is not None:
      self._data_stream.stop()
      self._data_stream = None

  def reward(self, obs: Optional[ObsType], action: ActType, next_obs: ObsType) -> float:
    return 0.0 # to be implemented in subclass

  def is_done(self, obs: Optional[ObsType], action: ActType, next_obs: ObsType) -> bool:
    return False # to be implemented in subclass
