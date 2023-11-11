import asyncio
import json
import time
from typing import Any, Awaitable, Dict, List, Tuple

import aiohttp
import aiortc
import numpy as np

from openpilot.tools.bodyteleop.webrtc import WebRTCOfferBuilder

WEBRTCD_PORT=5001


class HttpConnectionProvider:
  def __init__(self, address: str, port: int, incoming_services: List[str], outgoing_services: List[str]):
    self.url = f"http://{address}:{port}/stream"
    self.incoming_services = incoming_services
    self.outgoing_services = outgoing_services

  async def __call__(self, offer):
    async with aiohttp.ClientSession() as session:
      body = {
        'sdp': offer.sdp, 'cameras': offer.video,
        'bridge_services_in': self.incoming_services, 'bridge_services_out': self.outgoing_services
      }
      async with session.post(self.url, json=body) as resp:
        payload = await resp.json()
        answer = aiortc.RTCSessionDescription(**payload)
        return answer


class DataStreamSession:
  frame_rate = 20.0

  def __init__(self, cameras: List[str], services: List[str], webrtcd_host: str = "localhost"):
    provider = HttpConnectionProvider(address=webrtcd_host, port=WEBRTCD_PORT, incoming_services=["testJoystick"], outgoing_services=services)
    builder = WebRTCOfferBuilder(provider)
    for cam in cameras:
      builder.offer_to_receive_video_stream(cam)
    builder.add_messaging()
    self.stream = builder.stream()
    self.stream.set_messaging_handler(self.new_message_handler)

    self.camera_tracks = None
    self.channel = None

    self.cameras = cameras
    self.requested_services = services
    self.message_storage = {service: None for service in services}
    self.message_log_mono_times = {service: None for service in services}
    self.message_validity = {service: False for service in services}
    self.message_recv_times = {service: 0 for service in services}
    self.last_recv_time = 0

  def start(self):
    asyncio.run(self._connect_async())

  def stop(self):
    asyncio.run(self._disconnect_async())

  def receive(self):
    return asyncio.run(self._receive_async())
  
  def send(self, x: float, y: float):
    msg = {"type": "testJoystick", "data": {"axes": [x, y], "buttons": [False]}}
    data = json.dumps(msg).encode()
    self.channel.send(data)

  async def _connect_async(self):
    await self.stream.start()
    await self.stream.wait_for_connection()
    self.camera_tracks = {cam: self.stream.get_incoming_video_track(cam, buffered=False) for cam in self.cameras}
    self.channel = self.stream.get_messaging_channel()

  async def _disconnect_async(self):
    await self.stream.stop()
    await self.wait_for_disconnection()

  async def _receive_async(self) -> Tuple[Dict[str, np.array], Dict[str, Any], Dict[str, bool], Dict[str, int]]:
    camera_coroutines: List[Awaitable[np.array]] = []
    for cam in self.cameras:
      cor = self.camera_tracks[cam].recv()
      camera_coroutines.append(cor)

    frames = await asyncio.gather(*camera_coroutines)
    self.last_recv_time = time.time()

    return (
      {cam: frame for cam, frame in zip(self.cameras, frames)},
      {service: self.message_storage[service] for service in self.requested_services},
      {service: self.message_validity[service] for service in self.requested_services},
      {service: self.message_log_mono_times[service] for service in self.requested_services},
    )

  async def new_message_handler(self, data: bytes):
    msg = json.loads(data)
    msg_type, msg_time, msg_valid, msg_data = msg['type'], msg['logMonoTime'], msg['valid'], msg['data']

    self.message_storage[msg_type] = msg_data
    self.message_validity[msg_type] = msg_valid
    self.message_log_mono_times[msg_type] = msg_time
    self.message_recv_times[msg_type] = time.time()
