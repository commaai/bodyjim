import string
from typing import Any, Dict

from gymnasium import spaces
import numpy as np


def space_from_schema(schema: Any) -> spaces.Space:
  if isinstance(schema, dict):
    return dict_space_from_schema(schema)
  elif isinstance(schema, list):
    assert len(schema) == 1
    subspace = space_from_schema(schema[0])
    stack = not isinstance(subspace, spaces.Dict)
    return spaces.Sequence(space_from_schema(schema[0]), stack=stack)
  elif schema == "text":
    return spaces.Text(max_length=1000, charset=string.printable)
  elif schema == "data":
    return spaces.Sequence(spaces.Discrete(256), stack=True)
  elif schema == "anyPointer":
    return spaces.Text(max_length=10)
  elif schema == "bool":
    return spaces.Discrete(2)
  elif "int" in schema:
    int_type = getattr(np, schema)
    #return spaces.Discrete(np.iinfo(int_type).max - np.iinfo(int_type).min + 1, start=np.iinfo(int_type).min)
    return spaces.Box(low=np.iinfo(int_type).min, high=np.iinfo(int_type).max, shape=(), dtype=int_type)
  elif "float" in schema:
    float_type = getattr(np, schema)
    return spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=float_type)
  else:
    raise ValueError(f"Unknown type {schema}")


def dict_space_from_schema(schema_dict: Dict[str, Any]) -> spaces.Space:
  output_dict = {}
  for key, value in schema_dict.items():
    output_dict[key] = space_from_schema(value)
      
  space = spaces.Dict(output_dict)
  return space
