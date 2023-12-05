#!/usr/bin/env python3
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from bodyjim.env import BodyEnv, TICI_IMAGE_SIZE
from bodyjim.data_stream import DataStreamSession, WebrtcdClient


class MockedDataStreamSession:
  def __init__(self, mocked_data_stream, mocked_client):
    self.mocked_data_stream = mocked_data_stream
    self.mocked_client = mocked_client

  def __enter__(self):
    self._data_stream_patcher = patch('bodyjim.env.DataStreamSession', return_value=self.mocked_data_stream)
    self._client_patcher = patch('bodyjim.env.WebrtcdClient', return_value=self.mocked_client)
    self._data_stream_patcher.start()
    self._client_patcher.start()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self._data_stream_patcher.stop()
    self._client_patcher.stop()


class TestEnv(unittest.TestCase):
  def mocked_data_stream(self, cameras, messages):
    mock = MagicMock(spec=DataStreamSession)
    frames = {cam: np.zeros((*TICI_IMAGE_SIZE, 3), dtype=np.uint8) for cam in cameras}
    times = {msg_type: 0 for msg_type in messages}
    valid = {msg_type: True for msg_type in messages}
    mock.receive.return_value = (frames, messages, valid, times)

    return mock

  def mocked_schema_client(self, schema):
    mock = MagicMock(spec=WebrtcdClient)
    mock.get_schema.return_value = schema
    return mock

  def setUp(self):
    # example data
    self.message_a, message_a_schema = {'a': 1}, {'a': 'int32'}
    self.message_b, message_b_schema = {'b': 1.0, 'c': 2.0}, {'b': 'float32', 'c': 'float32'}
    self.cameras, self.services = ['driver'], ['message_a', 'message_b']
    self.schema = {
      'message_a': message_a_schema,
      'message_b': message_b_schema,
    }

  def test_step_and_reset_all_messages(self):
    msgs = {'message_a': self.message_a, 'message_b': self.message_b}
    with MockedDataStreamSession(self.mocked_data_stream(self.cameras, msgs), self.mocked_schema_client(self.schema)) as mock_helper:
      env = BodyEnv('localhost', self.cameras, self.services)
      obs, _, _, _, _ = env.step((0.5, 1.0))
      self.assertTrue(env.observation_space.contains(obs))
      self.assertEqual(obs.get('message_a', None), self.message_a)
      self.assertEqual(obs.get('message_b', None), self.message_b)

      obs, _ = env.reset()
      self.assertTrue(env.observation_space.contains(obs))

      mock_helper.mocked_data_stream.send.assert_called_once_with(0.5, 1.0)

  def test_step_some_messages(self):
    msgs = {'message_a': self.message_a}
    with MockedDataStreamSession(self.mocked_data_stream(self.cameras, msgs), self.mocked_schema_client(self.schema)):
      env = BodyEnv('localhost', self.cameras, self.services)
      obs, _, _, _, _ = env.step((0.0, 0.0))
      self.assertTrue(env.observation_space.contains(obs))
      self.assertEqual(obs.get('message_a', None), self.message_a)
      self.assertTrue('message_b' in obs)

  def test_step_no_messages(self):
    msgs = {}
    with MockedDataStreamSession(self.mocked_data_stream(self.cameras, msgs), self.mocked_schema_client(self.schema)):
      env = BodyEnv('localhost', self.cameras, self.services)
      obs, _, _, _, _ = env.step((0.0, 0.0))
      self.assertTrue(env.observation_space.contains(obs))
      self.assertTrue('message_a' in obs)
      self.assertTrue('message_b' in obs)

  def test_step_union_fields(self):
    # some fields are present, some are not
    modified_msg_b = {'b': 1.0} # missing c field
    msgs = {'message_b': modified_msg_b}
    with MockedDataStreamSession(self.mocked_data_stream(self.cameras, msgs), self.mocked_schema_client(self.schema)):
      env = BodyEnv('localhost', self.cameras, self.services)
      obs, _, _, _, _ = env.step((0.0, 0.0))
      self.assertTrue(env.observation_space.contains(obs))
      self.assertTrue('message_a' in obs)
      self.assertTrue('message_b' in obs)
      self.assertEqual(obs['message_b']['b'], 1.0)

  def test_rgb_render(self):
    msgs = {'message_a': self.message_a, 'message_b': self.message_b}
    rd_cameras = ['driver', 'road']
    with MockedDataStreamSession(self.mocked_data_stream(rd_cameras, msgs), self.mocked_schema_client(self.schema)):
      env = BodyEnv('localhost', rd_cameras, self.services, render_mode='rgb_array')
      env.step((0.0, 0.0))
      image = env.render()
      self.assertEqual(image.shape, (TICI_IMAGE_SIZE[0], 2*TICI_IMAGE_SIZE[1], 3))
      self.assertTrue(np.all(image == 0))

  def test_stop(self):
    msgs = {'message_a': self.message_a, 'message_b': self.message_b}
    with MockedDataStreamSession(self.mocked_data_stream(self.cameras, msgs), self.mocked_schema_client(self.schema)) as mock_helper:
      env = BodyEnv('localhost', self.cameras, self.services)
      env.reset()
      env.close()
      mock_helper.mocked_data_stream.stop.assert_called_once()

  def test_msg_list_of_structs_field(self):
    # for some reason gymnasium spaces.Sequence doesn't like lists of dicts
    schema = {'message_a': [self.schema['message_a']]}
    msg = {'message_a': [self.message_a]}
    with MockedDataStreamSession(self.mocked_data_stream(self.cameras, msg), self.mocked_schema_client(schema)):
      env = BodyEnv('localhost', self.cameras, self.services)
      obs, _, _, _, _ = env.step((0.0, 0.0))
      self.assertTrue(env.observation_space.contains(obs))

  def test_none_msg(self):
    msg = {'message_a': None}
    with MockedDataStreamSession(self.mocked_data_stream(self.cameras, msg), self.mocked_schema_client(self.schema)):
      env = BodyEnv('localhost', self.cameras, self.services)
      obs, _, _, _, _ = env.step((0.0, 0.0))
      self.assertTrue(env.observation_space.contains(obs))
      self.assertFalse(obs['message_a'] is None)


if __name__ == '__main__':
  unittest.main()