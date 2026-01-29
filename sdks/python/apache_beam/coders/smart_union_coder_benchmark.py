#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Benchmark comparing SmartUnionCoder vs FastPrimitivesCoder for Union types.

This benchmarks the encode-decode performance of SmartUnionCoder (which uses
O(1) type discrimination via _OrderedUnionCoder) against FastPrimitivesCoder
(which falls back to pickle for union values).

Run as:
  python -m apache_beam.coders.smart_union_coder_benchmark

"""

# pytype: skip-file

import argparse
import logging
import random
import re
import string
import struct

from apache_beam.coders import coders
from apache_beam.coders import typecoders
from apache_beam.coders.coders import Coder
from apache_beam.coders.coders import FastPrimitivesCoder
from apache_beam.coders.smart_union_coder import build_smart_union_coder
from apache_beam.tools import utils
from apache_beam.typehints import typehints

# ---------------------------------------------------------------------------
# Custom classes and coders for benchmarking custom-class unions
# ---------------------------------------------------------------------------


class BenchPoint:
  __slots__ = ('x', 'y')

  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __eq__(self, other):
    return type(other) is BenchPoint and self.x == other.x and self.y == other.y


class BenchPointCoder(Coder):
  def encode(self, value):
    return struct.pack('>dd', value.x, value.y)

  def decode(self, encoded):
    x, y = struct.unpack('>dd', encoded)
    return BenchPoint(x, y)

  def is_deterministic(self):
    return True

  @classmethod
  def from_type_hint(cls, typehint, registry):
    return cls()


class BenchColor:
  __slots__ = ('r', 'g', 'b')

  def __init__(self, r, g, b):
    self.r = r
    self.g = g
    self.b = b

  def __eq__(self, other):
    return (
        type(other) is BenchColor and self.r == other.r and
        self.g == other.g and self.b == other.b)


class BenchColorCoder(Coder):
  def encode(self, value):
    return bytes([value.r, value.g, value.b])

  def decode(self, encoded):
    return BenchColor(encoded[0], encoded[1], encoded[2])

  def is_deterministic(self):
    return True

  @classmethod
  def from_type_hint(cls, typehint, registry):
    return cls()


class BenchRecord:
  """A larger object with many attributes to amplify pickle overhead."""
  __slots__ = tuple(f'f{i}' for i in range(20))

  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)

  def __eq__(self, other):
    return type(other) is BenchRecord and all(
        getattr(self, f'f{i}') == getattr(other, f'f{i}') for i in range(20))


class BenchRecordCoder(Coder):
  """Encodes BenchRecord as 20 packed doubles."""
  def encode(self, value):
    return struct.pack('>20d', *(getattr(value, f'f{i}') for i in range(20)))

  def decode(self, encoded):
    vals = struct.unpack('>20d', encoded)
    return BenchRecord(**{f'f{i}': vals[i] for i in range(20)})

  def is_deterministic(self):
    return True

  @classmethod
  def from_type_hint(cls, typehint, registry):
    return cls()


typecoders.registry._register_coder_internal(BenchPoint, BenchPointCoder)
typecoders.registry._register_coder_internal(BenchColor, BenchColorCoder)
typecoders.registry._register_coder_internal(BenchRecord, BenchRecordCoder)

# ---------------------------------------------------------------------------
# Value generators — each produces a random value from a union type
# ---------------------------------------------------------------------------


def _random_string(length):
  return ''.join(
      random.choice(string.ascii_letters + string.digits)
      for _ in range(length))


def int_or_str():
  """Union[int, str]"""
  if random.random() < 0.5:
    return random.randint(0, 127)
  return _random_string(8)


def int_str_float_bytes():
  """Union[int, str, float, bytes]"""
  r = random.random()
  if r < 0.25:
    return random.randint(0, 127)
  elif r < 0.5:
    return _random_string(8)
  elif r < 0.75:
    return random.random() * 100
  else:
    return _random_string(8).encode('ascii')


def int_str_list_int():
  """Union[int, str, List[int]]"""
  r = random.random()
  if r < 0.33:
    return random.randint(0, 127)
  elif r < 0.66:
    return _random_string(8)
  else:
    return [random.randint(0, 127) for _ in range(10)]


def int_str_list_int_list_str():
  """Union[int, str, List[int], List[str]]"""
  r = random.random()
  if r < 0.25:
    return random.randint(0, 127)
  elif r < 0.5:
    return _random_string(8)
  elif r < 0.75:
    return [random.randint(0, 127) for _ in range(10)]
  else:
    return [_random_string(4) for _ in range(10)]


def int_tuple2_tuple3():
  """Union[int, Tuple[int, int], Tuple[str, str, str]]"""
  r = random.random()
  if r < 0.33:
    return random.randint(0, 127)
  elif r < 0.66:
    return (random.randint(0, 127), random.randint(0, 127))
  else:
    return (_random_string(4), _random_string(4), _random_string(4))


def point_or_int():
  """Union[BenchPoint, int]"""
  if random.random() < 0.5:
    return BenchPoint(random.random() * 100, random.random() * 100)
  return random.randint(0, 127)


def point_only():
  """BenchPoint only — custom class, no primitives."""
  return BenchPoint(random.random() * 100, random.random() * 100)


def color_only():
  """BenchColor only — custom class, no primitives."""
  return BenchColor(
      random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def point_or_color():
  """Union[BenchPoint, BenchColor]"""
  if random.random() < 0.5:
    return BenchPoint(random.random() * 100, random.random() * 100)
  return BenchColor(
      random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def point_color_or_int():
  """Union[BenchPoint, BenchColor, int]"""
  r = random.random()
  if r < 0.33:
    return BenchPoint(random.random() * 100, random.random() * 100)
  elif r < 0.66:
    return BenchColor(
        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
  else:
    return random.randint(0, 127)


def record_only():
  """BenchRecord only — large custom object (20 float fields)."""
  return BenchRecord(**{f'f{i}': random.random() * 1000 for i in range(20)})


def record_or_int():
  """Union[BenchRecord, int]"""
  if random.random() < 0.5:
    return BenchRecord(**{f'f{i}': random.random() * 1000 for i in range(20)})
  return random.randint(0, 127)


# ---------------------------------------------------------------------------
# Benchmark factory
# ---------------------------------------------------------------------------


def union_benchmark_factory(coder, generate_fn, label):
  """Creates a benchmark that encodes and decodes a list of union values.

  Args:
    coder: coder to use to encode elements.
    generate_fn: callable that generates a single union element.
    label: descriptive label for the benchmark.
  """
  class UnionBenchmark(object):
    def __init__(self, num_elements_per_benchmark):
      self._coder = coders.IterableCoder(coder)
      self._list = [generate_fn() for _ in range(num_elements_per_benchmark)]

    def __call__(self):
      _ = self._coder.decode(self._coder.encode(self._list))

  UnionBenchmark.__name__ = label
  return UnionBenchmark


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------


def _build_benchmarks():
  """Build all benchmark pairs (SmartUnion vs FastPrimitives)."""
  scenarios = [
      # --- Primitives only (FastPrimitivesCoder has native fast paths) ---
      (
          'int|str',
          [int, str],
          int_or_str,
      ),
      (
          'int|str|float|bytes',
          [int, str, float, bytes],
          int_str_float_bytes,
      ),
      # --- Containers ---
      (
          'int|str|List[int]|List[str]',
          [int, str, typehints.List[int], typehints.List[str]],
          int_str_list_int_list_str,
      ),
      # --- Custom classes (FastPrimitivesCoder falls back to pickle) ---
      (
          'BenchPoint (single)',
          [BenchPoint],
          point_only,
      ),
      (
          'BenchColor (single)',
          [BenchColor],
          color_only,
      ),
      (
          'BenchPoint|int',
          [BenchPoint, int],
          point_or_int,
      ),
      (
          'BenchPoint|BenchColor',
          [BenchPoint, BenchColor],
          point_or_color,
      ),
      (
          'BenchPoint|BenchColor|int',
          [BenchPoint, BenchColor, int],
          point_color_or_int,
      ),
      # --- Large custom object (20 fields) ---
      (
          'BenchRecord (single, 20 fields)',
          [BenchRecord],
          record_only,
      ),
      (
          'BenchRecord|int (20 fields)',
          [BenchRecord, int],
          record_or_int,
      ),
  ]

  benchmarks = []
  for name, type_hints, gen_fn in scenarios:
    smart_coder = build_smart_union_coder(type_hints)
    fp_coder = FastPrimitivesCoder()
    benchmarks.append(
        union_benchmark_factory(
            smart_coder, gen_fn, f'{name}, SmartUnionCoder'))
    benchmarks.append(
        union_benchmark_factory(
            fp_coder, gen_fn, f'{name}, FastPrimitivesCoder'))

  return benchmarks


def run_benchmarks(num_runs, input_size, seed, verbose, filter_regex='.*'):
  random.seed(seed)

  benchmarks = _build_benchmarks()
  suite = [
      utils.BenchmarkConfig(b, input_size, num_runs) for b in benchmarks
      if re.search(filter_regex, b.__name__, flags=re.I)
  ]
  utils.run_benchmarks(suite, verbose=verbose)


if __name__ == '__main__':
  logging.basicConfig()

  parser = argparse.ArgumentParser(
      description='Benchmark SmartUnionCoder vs FastPrimitivesCoder')
  parser.add_argument('--filter', default='.*')
  parser.add_argument('--num_runs', default=20, type=int)
  parser.add_argument('--num_elements_per_benchmark', default=1000, type=int)
  parser.add_argument('--seed', default=42, type=int)
  options = parser.parse_args()

  run_benchmarks(
      options.num_runs,
      options.num_elements_per_benchmark,
      options.seed,
      verbose=True,
      filter_regex=options.filter)
