import asyncio
import json
import time
import threading
from typing import Any, Awaitable, Dict, List, Tuple, Optional

import aiohttp
import aiortc
import numpy as np

from teleoprtc import WebRTCOfferBuilder

CONNECT_TIMEOUT_SECONDS = 3.0
RECEIVE_TIMEOUT_SECONDS = 2.0


class WebrtcdClient:
  def __init__(self, address: str, port: int = 5001):
    self._url = f"http://{address}:{port}"

  async def request_stream(self, sdp, cameras, incoming_services, outgoing_service):
    endpoint = self._url + "/stream"
    body = {
      'sdp': sdp, 'cameras': cameras,
      'bridge_services_in': incoming_services, 'bridge_services_out': outgoing_service
    }
    async with aiohttp.ClientSession(raise_for_status=True) as session:
      async with session.post(endpoint, json=body) as resp:
        payload = await resp.json()
        return payload

  async def get_schema(self, services):
    endpoint = self._url + "/schema"
    params = {"services": ",".join(services)}
    async with aiohttp.ClientSession(raise_for_status=True) as session:
      async with session.get(endpoint, params=params) as resp:
        payload = await resp.json()
        return payload


class DataStreamSession:
  def __init__(self, client: WebrtcdClient, cameras: List[str], services: List[str]):
    incoming_services, outgoing_services = ["testJoystick"], services
    async def connection_provider(offer):
      response = await client.request_stream(offer.sdp, offer.video, incoming_services, outgoing_services)
      return aiortc.RTCSessionDescription(**response)

    builder = WebRTCOfferBuilder(connection_provider)
    for cam in cameras:
      builder.offer_to_receive_video_stream(cam)
    builder.add_messaging()
    self._stream = builder.stream()
    self._stream.set_message_handler(self._new_message_handler)
    self._client = client
    self._event_loop: Optional[asyncio.AbstractEventLoop] = None
    self._runner_thread: Optional[threading.Thread] = None

    self._camera_tracks: Dict[str, aiortc.MediaStreamTrack] = {}
    self._channel: Optional[aiortc.RTCDataChannel] = None
    self._message_schema: Optional[Dict[str, Any]] = None

    self._cameras = cameras
    self._requested_services = services
    self._message_storage: Dict[str, Optional[Any]] = {service: None for service in services}
    self._message_log_mono_times: Dict[str, Optional[int]] = {service: None for service in services}
    self._message_validity: Dict[str, bool] = {service: False for service in services}
    self._message_recv_times: Dict[str, float] = {service: 0.0 for service in services}
    self._last_recv_time = 0.0

  @property
  def last_recv_time(self) -> float:
    return self._last_recv_time

  @property
  def message_schema(self) -> Optional[Dict[str, Any]]:
    return self._message_schema

  def start(self):
    def _background_loop_run(loop):
      asyncio.set_event_loop(loop)
      loop.run_forever()
    
    self._event_loop = asyncio.new_event_loop()
    self._runner_thread = threading.Thread(target=_background_loop_run, args=(self._event_loop,), daemon=True)
    self._runner_thread.start()
    future = asyncio.run_coroutine_threadsafe(self._connect_async(), self._event_loop)
    future.result(CONNECT_TIMEOUT_SECONDS)

  def stop(self):
    assert self._event_loop is not None, "Session not started"
    future = asyncio.run_coroutine_threadsafe(self._disconnect_async(), self._event_loop)
    future.result(CONNECT_TIMEOUT_SECONDS)
    # stop() neeed to be called threadsafe
    self._event_loop.call_soon_threadsafe(self._event_loop.stop)
    self._runner_thread.join()
    self._runner_thread = None
    self._event_loop.close()

  def receive(self) -> Tuple[Dict[str, np.array], Dict[str, Any], Dict[str, Optional[bool]], Dict[str, Optional[int]]]:
    assert self._event_loop is not None, "Session not started"
    future = asyncio.run_coroutine_threadsafe(self._receive_async(), self._event_loop)
    return future.result(RECEIVE_TIMEOUT_SECONDS)

  def send(self, x: float, y: float):
    assert self._channel is not None and self._event_loop is not None, "Session not started"
    msg = {"type": "testJoystick", "data": {"axes": [x, y], "buttons": [False]}}
    data = json.dumps(msg).encode()
    self._event_loop.call_soon_threadsafe(self._channel.send, data)

  async def _connect_async(self):
    await self._stream.start()
    await self._stream.wait_for_connection()
    self._camera_tracks = {cam: self._stream.get_incoming_video_track(cam, buffered=False) for cam in self._cameras}
    self._channel = self._stream.get_messaging_channel()

  async def _disconnect_async(self):
    await self._stream.stop()

  async def _receive_async(self) -> Tuple[Dict[str, np.array], Dict[str, Any], Dict[str, Optional[bool]], Dict[str, Optional[int]]]:
    camera_coroutines: List[Awaitable[np.array]] = []
    for cam in self._cameras:
      assert cam in self._camera_tracks
      cor = self._camera_tracks[cam].recv()
      camera_coroutines.append(cor)

    frames = await asyncio.gather(*camera_coroutines)
    self._last_recv_time = time.time()

    return (
      {cam: frame.to_ndarray(format="rgb24") for cam, frame in zip(self._cameras, frames)},
      {service: self._message_storage[service] for service in self._requested_services},
      {service: self._message_validity[service] for service in self._requested_services},
      {service: self._message_log_mono_times[service] for service in self._requested_services},
    )

  async def _new_message_handler(self, data: bytes):
    try:
      msg = json.loads(data)
      msg_type, msg_time, msg_valid, msg_data = msg['type'], msg['logMonoTime'], msg['valid'], msg['data']

      self._message_storage[msg_type] = msg_data
      self._message_validity[msg_type] = msg_valid
      self._message_log_mono_times[msg_type] = msg_time
      self._message_recv_times[msg_type] = time.time()
    except Exception as e:
      print("Error parsing message", e)
