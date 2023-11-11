#!/usr/bin/env python
import unittest

from gymnasium import spaces

from bodyjim.schema import space_from_schema


class TestSpace(unittest.TestCase):
  def test_bool(self):
    schema = {
      'bool_field': 'bool',
    }
    space = space_from_schema(schema)
    self.assertTrue('bool_field' in space.spaces)
    self.assertTrue(isinstance(space['bool_field'], spaces.Discrete))

    for b in (True, False):
      example_dict = {'bool_field': b}
      self.assertTrue(space.contains(example_dict))

  def test_integers(self):
    schema = {
      'int8_field': 'int8',
      'int16_field': 'int16',
      'int32_field': 'int32',
      'int64_field': 'int64',
      'uint8_field': 'uint8',
      'uint16_field': 'uint16',
      'uint32_field': 'uint32',
      'uint64_field': 'uint64',
    }
    space = space_from_schema(schema)
    for key in schema:
      self.assertTrue(key in space.spaces)

    example_dict = {key: 0 for key in schema.keys()}
    self.assertTrue(space.contains(example_dict))

  def test_floats(self):
    schema = {
      'float32_field': 'float32',
      'float64_field': 'float64',
    }
    space = space_from_schema(schema)
    for key in schema:
      self.assertTrue(key in space.spaces)

    example_dict = {key: 0.0 for key in schema.keys()}
    self.assertTrue(space.contains(example_dict))

  def test_text(self):
    schema = {
      'text_field': 'text',
      'data_field': 'data',
    }
    space = space_from_schema(schema)
    for key in schema:
      self.assertTrue(key in space.spaces)

    example_dict = {'text_field': 'test', 'data_field': [0xff, 0x7f, 0x00, 0x80]}
    self.assertTrue(space.contains(example_dict))

  def test_nested_structs(self):
    schema = {
      'nested_struct': {
        'int_field': 'int32',
        'nested_nested_struct': {
          'float_field': 'float32',
        },
      }
    }
    space = space_from_schema(schema)
    self.assertTrue('nested_struct' in space.spaces)
    self.assertTrue(isinstance(space['nested_struct'], spaces.Dict))
    self.assertTrue('nested_nested_struct' in space['nested_struct'].spaces)
    self.assertTrue(isinstance(space['nested_struct']['nested_nested_struct'], spaces.Dict))

    example_dict = {'nested_struct': {'int_field': 0, 'nested_nested_struct': {'float_field': 0.0}}}
    self.assertTrue(space.contains(example_dict))

  def test_lists(self):
    primitive_schema = {
      'list_field': ['int32'],
    }
    list_in_list_schema = {
      'nested_list_field': [['float32']],
    }
    struct_in_list_schema = {
      'nested_struct_list_field': [{'int_field': 'int32'}],
    }

    primitive_space = space_from_schema(primitive_schema)
    self.assertTrue('list_field' in primitive_space.spaces)
    self.assertTrue(isinstance(primitive_space['list_field'], spaces.Sequence))

    list_in_list_space = space_from_schema(list_in_list_schema)
    self.assertTrue('nested_list_field' in list_in_list_space.spaces)
    self.assertTrue(isinstance(list_in_list_space['nested_list_field'], spaces.Sequence))

    struct_in_list_space = space_from_schema(struct_in_list_schema)
    self.assertTrue('nested_struct_list_field' in struct_in_list_space.spaces)
    self.assertTrue(isinstance(struct_in_list_space['nested_struct_list_field'], spaces.Sequence))

    primitive_example_dict = {'list_field': (0, 1, 2, 3)}
    self.assertTrue(primitive_space.contains(primitive_example_dict))

    list_in_list_example_dict = {'nested_list_field': ((0.0, 1.0), (2.0, 3.0))}
    self.assertTrue(list_in_list_space.contains(list_in_list_example_dict))

    struct_in_list_example_dict = {'nested_struct_list_field': ({'int_field': 0}, {'int_field': 1})}
    self.assertTrue(struct_in_list_space.contains(struct_in_list_example_dict))


if __name__ == '__main__':
  unittest.main()