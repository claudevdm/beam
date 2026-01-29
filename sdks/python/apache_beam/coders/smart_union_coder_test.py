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

"""Tests for SmartUnionCoder."""

import json
import struct
import unittest
from typing import NamedTuple

from apache_beam.coders.coders import BytesCoder
from apache_beam.coders.coders import Coder
from apache_beam.coders.coders import FastPrimitivesCoder
from apache_beam.coders.coders import FloatCoder
from apache_beam.coders.coders import ListCoder
from apache_beam.coders.coders import MapCoder
from apache_beam.coders.coders import _OrderedUnionCoder
from apache_beam.coders.coders import StrUtf8Coder
from apache_beam.coders.coders import TupleCoder
from apache_beam.coders.coders import VarIntCoder
from apache_beam.coders.row_coder import RowCoder
from apache_beam.coders.smart_union_coder import ArityDiscriminatingTupleCoder
from apache_beam.coders.smart_union_coder import build_smart_union_coder
from apache_beam.coders import typecoders
from apache_beam.pvalue import Row
from apache_beam.typehints import typehints
from apache_beam.typehints.row_type import RowTypeConstraint

# Custom classes and coders for testing user-defined type handling.


class Point:
  """A simple 2D point."""
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __eq__(self, other):
    return isinstance(other, Point) and self.x == other.x and self.y == other.y


class Color:
  """An RGB color."""
  def __init__(self, r, g, b):
    self.r = r
    self.g = g
    self.b = b

  def __eq__(self, other):
    return (
        isinstance(other, Color) and self.r == other.r and self.g == other.g and
        self.b == other.b)


class Label:
  """A text label with a name."""
  def __init__(self, name):
    self.name = name

  def __eq__(self, other):
    return isinstance(other, Label) and self.name == other.name


class PointCoder(Coder):
  """Encodes Point as two big-endian doubles."""
  def encode(self, value):
    return struct.pack('>dd', value.x, value.y)

  def decode(self, encoded):
    x, y = struct.unpack('>dd', encoded)
    return Point(x, y)

  def is_deterministic(self):
    return True


class ColorCoder(Coder):
  """Encodes Color as three bytes (r, g, b)."""
  def encode(self, value):
    return bytes([value.r, value.g, value.b])

  def decode(self, encoded):
    return Color(encoded[0], encoded[1], encoded[2])

  def is_deterministic(self):
    return True


class LabelCoder(Coder):
  """Encodes Label as JSON."""
  def encode(self, value):
    return json.dumps({"name": value.name}).encode('utf-8')

  def decode(self, encoded):
    d = json.loads(encoded.decode('utf-8'))
    return Label(d["name"])

  def is_deterministic(self):
    return True


# Register custom coders with the global registry.
typecoders.registry._register_coder_internal(Point, PointCoder)
typecoders.registry._register_coder_internal(Color, ColorCoder)
typecoders.registry._register_coder_internal(Label, LabelCoder)


class SmartUnionCoderTest(unittest.TestCase):
  """Tests for SmartUnionCoder."""
  def _roundtrip(self, coder, value):
    """Encode and decode a value, returning the decoded result."""
    encoded = coder.encode(value)
    decoded = coder.decode(encoded)
    return decoded

  def test_basic_different_types(self):
    """Test discrimination between different primitive types."""
    coder = build_smart_union_coder([int, str])

    # Test int
    self.assertEqual(42, self._roundtrip(coder, 42))
    self.assertEqual(-100, self._roundtrip(coder, -100))
    self.assertEqual(0, self._roundtrip(coder, 0))

    # Test str
    self.assertEqual("hello", self._roundtrip(coder, "hello"))
    self.assertEqual("", self._roundtrip(coder, ""))
    self.assertEqual(
        "unicode: \u00e9\u00e8",
        self._roundtrip(coder, "unicode: \u00e9\u00e8"))

  def test_three_types(self):
    """Test discrimination between three types."""
    coder = build_smart_union_coder([int, str, bytes])

    self.assertEqual(42, self._roundtrip(coder, 42))
    self.assertEqual("hello", self._roundtrip(coder, "hello"))
    self.assertEqual(b"bytes", self._roundtrip(coder, b"bytes"))

  def test_list_int_vs_str(self):
    """Test Union[int, str] with primitive discrimination."""
    coder = build_smart_union_coder([int, str])

    # These should use direct type discrimination
    self.assertEqual(42, self._roundtrip(coder, 42))
    self.assertEqual("hello", self._roundtrip(coder, "hello"))

  def test_same_outer_type_lists(self):
    """Test merging List[int] and List[str] into List[int|str]."""
    coder = build_smart_union_coder([typehints.List[int], typehints.List[str]])

    # Pure int list
    self.assertEqual([1, 2, 3], self._roundtrip(coder, [1, 2, 3]))

    # Pure str list
    self.assertEqual(["a", "b"], self._roundtrip(coder, ["a", "b"]))

    # Mixed list (allowed after merge!)
    self.assertEqual([1, "a", 2, "b"], self._roundtrip(coder, [1, "a", 2, "b"]))

  def test_mixed_outer_types(self):
    """Test Union[int, str, List[int], List[str]]."""
    coder = build_smart_union_coder(
        [int, str, typehints.List[int], typehints.List[str]])

    # Primitives
    self.assertEqual(42, self._roundtrip(coder, 42))
    self.assertEqual("hello", self._roundtrip(coder, "hello"))

    # Lists
    self.assertEqual([1, 2, 3], self._roundtrip(coder, [1, 2, 3]))
    self.assertEqual(["a", "b"], self._roundtrip(coder, ["a", "b"]))

    # Mixed list (merged)
    self.assertEqual([1, "a"], self._roundtrip(coder, [1, "a"]))

  def test_tuple_same_arity(self):
    """Test same-arity tuples: Tuple[int, int] | Tuple[str, str]."""
    coder = build_smart_union_coder(
        [typehints.Tuple[int, int], typehints.Tuple[str, str]])

    self.assertEqual((1, 2), self._roundtrip(coder, (1, 2)))
    self.assertEqual(("a", "b"), self._roundtrip(coder, ("a", "b")))

    # Mixed tuple (allowed after position merge!)
    self.assertEqual((1, "b"), self._roundtrip(coder, (1, "b")))
    self.assertEqual(("a", 2), self._roundtrip(coder, ("a", 2)))

  def test_tuple_different_arities(self):
    """Test tuples with different arities."""
    coder = build_smart_union_coder([
        typehints.Tuple[int],
        typehints.Tuple[int, int],
        typehints.Tuple[int, int, int]
    ])

    # Should be ArityDiscriminatingTupleCoder
    self.assertIsInstance(coder, ArityDiscriminatingTupleCoder)

    # Test each arity
    self.assertEqual((1, ), self._roundtrip(coder, (1, )))
    self.assertEqual((1, 2), self._roundtrip(coder, (1, 2)))
    self.assertEqual((1, 2, 3), self._roundtrip(coder, (1, 2, 3)))

  def test_tuple_mixed_arities_and_types(self):
    """Test Union[Tuple[int], Tuple[int, int], Tuple[str, str]]."""
    coder = build_smart_union_coder([
        typehints.Tuple[int],
        typehints.Tuple[int, int],
        typehints.Tuple[str, str]
    ])

    # arity=1: single coder
    self.assertEqual((42, ), self._roundtrip(coder, (42, )))

    # arity=2: merged positions (int|str, int|str)
    self.assertEqual((1, 2), self._roundtrip(coder, (1, 2)))
    self.assertEqual(("a", "b"), self._roundtrip(coder, ("a", "b")))
    self.assertEqual((1, "b"), self._roundtrip(coder, (1, "b")))

  def test_dict_merging(self):
    """Test merging Dict[str, int] and Dict[str, str]."""
    coder = build_smart_union_coder(
        [typehints.Dict[str, int], typehints.Dict[str, str]])

    self.assertEqual({"a": 1, "b": 2}, self._roundtrip(coder, {"a": 1, "b": 2}))
    self.assertEqual({"x": "y"}, self._roundtrip(coder, {"x": "y"}))

    # Mixed values (allowed after merge!)
    self.assertEqual({
        "a": 1, "b": "two"
    },
                     self._roundtrip(coder, {
                         "a": 1, "b": "two"
                     }))

  def test_nested_lists(self):
    """Test List[List[int]] and List[List[str]]."""
    coder = build_smart_union_coder([
        typehints.List[typehints.List[int]],
        typehints.List[typehints.List[str]]
    ])

    self.assertEqual([[1, 2], [3, 4]], self._roundtrip(coder, [[1, 2], [3, 4]]))
    self.assertEqual([["a", "b"]], self._roundtrip(coder, [["a", "b"]]))

    # Mixed inner lists (allowed!)
    self.assertEqual([[1, "a"]], self._roundtrip(coder, [[1, "a"]]))

  def test_determinism(self):
    """Test that deterministic coders produce deterministic output."""
    # int and str coders are deterministic
    coder = build_smart_union_coder([int, str])
    self.assertTrue(coder.is_deterministic())

    # Same value should always produce same bytes
    self.assertEqual(coder.encode(42), coder.encode(42))
    self.assertEqual(coder.encode("hello"), coder.encode("hello"))

  def test_as_deterministic_coder(self):
    """Test conversion to deterministic coder."""
    coder = build_smart_union_coder([int, str])
    det_coder = coder.as_deterministic_coder("test_step")

    # Should work the same
    self.assertEqual(42, self._roundtrip(det_coder, 42))
    self.assertEqual("hello", self._roundtrip(det_coder, "hello"))

  def test_empty_list(self):
    """Test encoding/decoding empty lists."""
    coder = build_smart_union_coder([typehints.List[int]])
    self.assertEqual([], self._roundtrip(coder, []))

  def test_empty_tuple(self):
    """Test encoding/decoding empty tuples."""
    coder = build_smart_union_coder([typehints.Tuple[()]])
    self.assertEqual((), self._roundtrip(coder, ()))

  def test_large_values(self):
    """Test with large values."""
    coder = build_smart_union_coder([int, str, typehints.List[int]])

    # Large int (within 64-bit range - VarIntCoder limitation)
    large_int = 2**62
    self.assertEqual(large_int, self._roundtrip(coder, large_int))

    # Large string
    large_str = "x" * 10000
    self.assertEqual(large_str, self._roundtrip(coder, large_str))

    # Large list
    large_list = list(range(1000))
    self.assertEqual(large_list, self._roundtrip(coder, large_list))

  def test_single_type_returns_simple_coder(self):
    """Test that single type returns the simple coder, not a union."""
    coder = build_smart_union_coder([int])

    # Should be a simple VarIntCoder, not SmartUnionCoder
    self.assertIsInstance(coder, VarIntCoder)

  def test_repr(self):
    """Test string representation."""
    # All unions now use _OrderedUnionCoder at the outer level
    coder1 = build_smart_union_coder([int, str])
    repr_str1 = repr(coder1)
    self.assertIn("_OrderedUnionCoder", repr_str1)

    # Mixed unions also use _OrderedUnionCoder
    coder2 = build_smart_union_coder([int, typehints.List[str]])
    repr_str2 = repr(coder2)
    self.assertIn("_OrderedUnionCoder", repr_str2)

  def test_equality(self):
    """Test coder equality."""
    coder1 = build_smart_union_coder([int, str])
    coder2 = build_smart_union_coder([int, str])
    coder3 = build_smart_union_coder([int, bytes])

    self.assertEqual(coder1, coder2)
    self.assertNotEqual(coder1, coder3)

  def test_hash(self):
    """Test coder hashing."""
    coder1 = build_smart_union_coder([int, str])
    coder2 = build_smart_union_coder([int, str])

    # Same coders should have same hash
    self.assertEqual(hash(coder1), hash(coder2))

    # Can be used in sets/dicts
    s = {coder1}
    self.assertIn(coder2, s)


class NamedTupleUnionTest(unittest.TestCase):
  """Tests for SmartUnionCoder with NamedTuples."""
  def _roundtrip(self, coder, value):
    encoded = coder.encode(value)
    decoded = coder.decode(encoded)
    return decoded

  def test_different_namedtuples(self):
    """Test Union of different NamedTuple types."""
    class Person(NamedTuple):
      name: str
      age: int

    class Order(NamedTuple):
      id: int
      total: float

    # Build coder using the concrete types
    coder = _OrderedUnionCoder(
        (Person, TupleCoder([StrUtf8Coder(), VarIntCoder()])),
        (Order, TupleCoder([VarIntCoder(), VarIntCoder()])),  # Simplified
        fallback_coder=None,
    )

    # Test Person
    person = Person("Alice", 30)
    decoded = self._roundtrip(coder, person)
    # Note: decoded will be a tuple, not a NamedTuple
    # This is a limitation - we'd need to reconstruct the NamedTuple
    self.assertEqual(("Alice", 30), decoded)

    # Test Order
    order = Order(123, 99)
    decoded = self._roundtrip(coder, order)
    self.assertEqual((123, 99), decoded)

  def test_namedtuple_type_discrimination(self):
    """Test that different NamedTuple types are correctly discriminated."""
    class TypeA(NamedTuple):
      x: int

    class TypeB(NamedTuple):
      x: int

    # Even though both have same structure, they're different types
    self.assertIsNot(TypeA, TypeB)
    self.assertIsNot(type(TypeA(1)), type(TypeB(1)))

    # _OrderedUnionCoder uses type() for discrimination, so this works
    coder = _OrderedUnionCoder(
        (TypeA, TupleCoder([VarIntCoder()])),
        (TypeB, TupleCoder([VarIntCoder()])),
        fallback_coder=None,
    )

    # Can distinguish between them
    a = TypeA(42)
    b = TypeB(42)

    # Encode produces different bytes (different tag)
    self.assertNotEqual(coder.encode(a), coder.encode(b))


class ArityDiscriminatingTupleCoderTest(unittest.TestCase):
  """Tests for ArityDiscriminatingTupleCoder."""
  def _roundtrip(self, coder, value):
    encoded = coder.encode(value)
    decoded = coder.decode(encoded)
    return decoded

  def test_basic_arity_discrimination(self):
    """Test basic arity discrimination."""
    coder = ArityDiscriminatingTupleCoder({
        1: TupleCoder([VarIntCoder()]),
        2: TupleCoder([VarIntCoder(), VarIntCoder()]),
        3: TupleCoder([VarIntCoder(), VarIntCoder(), VarIntCoder()]),
    })

    self.assertEqual((1, ), self._roundtrip(coder, (1, )))
    self.assertEqual((1, 2), self._roundtrip(coder, (1, 2)))
    self.assertEqual((1, 2, 3), self._roundtrip(coder, (1, 2, 3)))

  def test_arity_zero(self):
    """Test arity 0 (empty tuple)."""
    coder = ArityDiscriminatingTupleCoder({
        0: TupleCoder([]),
        1: TupleCoder([VarIntCoder()]),
    })

    self.assertEqual((), self._roundtrip(coder, ()))
    self.assertEqual((42, ), self._roundtrip(coder, (42, )))

  def test_repr(self):
    """Test string representation."""
    coder = ArityDiscriminatingTupleCoder({
        1: TupleCoder([VarIntCoder()]),
        2: TupleCoder([VarIntCoder(), VarIntCoder()]),
    })
    repr_str = repr(coder)
    self.assertIn("ArityDiscriminatingTupleCoder", repr_str)

  def test_determinism(self):
    """Test determinism."""
    coder = ArityDiscriminatingTupleCoder({
        1: TupleCoder([VarIntCoder()]),
        2: TupleCoder([VarIntCoder(), VarIntCoder()]),
    })
    self.assertTrue(coder.is_deterministic())

  def test_to_type_hint(self):
    """Test type hint generation."""
    coder = ArityDiscriminatingTupleCoder({
        1: TupleCoder([VarIntCoder()]),
        2: TupleCoder([VarIntCoder(), StrUtf8Coder()]),
    })
    hint = coder.to_type_hint()
    # Should be a Union of tuple types
    self.assertIsInstance(hint, typehints.UnionConstraint)


class EdgeCaseTest(unittest.TestCase):
  """Edge case tests for SmartUnionCoder."""
  def _roundtrip(self, coder, value):
    encoded = coder.encode(value)
    decoded = coder.decode(encoded)
    return decoded

  def test_negative_integers(self):
    """Test with negative integers."""
    coder = build_smart_union_coder([int, str])
    self.assertEqual(-42, self._roundtrip(coder, -42))
    # VarIntCoder is limited to 64-bit signed integers
    self.assertEqual(-(2**62), self._roundtrip(coder, -(2**62)))

  def test_unicode_strings(self):
    """Test with unicode strings."""
    coder = build_smart_union_coder([int, str])
    self.assertEqual("\u4e2d\u6587", self._roundtrip(coder, "\u4e2d\u6587"))
    self.assertEqual(
        "\U0001f600", self._roundtrip(coder, "\U0001f600"))  # Emoji

  def test_nested_empty_structures(self):
    """Test with nested empty structures."""
    coder = build_smart_union_coder([typehints.List[typehints.List[int]]])
    self.assertEqual([[]], self._roundtrip(coder, [[]]))
    self.assertEqual([[], []], self._roundtrip(coder, [[], []]))

  def test_deeply_nested_union(self):
    """Test with deeply nested structure."""
    # List[List[List[int|str]]]
    inner_union = build_smart_union_coder([int, str])
    coder = ListCoder(ListCoder(ListCoder(inner_union)))

    value = [[[1, "a", 2], ["b", 3]], [[4]]]
    self.assertEqual(value, self._roundtrip(coder, value))


class CoderStructureTest(unittest.TestCase):
  """Tests that verify the expected coder tree structure."""
  def test_two_primitives(self):
    """int | str -> _OrderedUnionCoder[(int, VarInt), (str, StrUtf8)]"""
    coder = build_smart_union_coder([int, str])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    self.assertEqual(len(coder._coder_types), 2)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[int], VarIntCoder)
    self.assertIsInstance(types[str], StrUtf8Coder)
    self.assertIsNone(coder._fallback_coder)

  def test_three_primitives(self):
    """int | str | bytes -> _OrderedUnionCoder with three entries."""
    coder = build_smart_union_coder([int, str, bytes])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[int], VarIntCoder)
    self.assertIsInstance(types[str], StrUtf8Coder)
    self.assertIsInstance(types[bytes], BytesCoder)

  def test_primitives_and_merged_lists(self):
    """int | str | List[int] | List[str]
    -> _OrderedUnionCoder[
         (int, VarInt),
         (str, StrUtf8),
         (list, ListCoder(_OrderedUnionCoder[(int, VarInt), (str, StrUtf8)]))]
    """
    coder = build_smart_union_coder(
        [int, str, typehints.List[int], typehints.List[str]])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}

    self.assertIsInstance(types[int], VarIntCoder)
    self.assertIsInstance(types[str], StrUtf8Coder)

    # list entry should be ListCoder wrapping a union of element types
    list_coder = types[list]
    self.assertIsInstance(list_coder, ListCoder)
    elem_coder = list_coder._elem_coder
    self.assertIsInstance(elem_coder, _OrderedUnionCoder)
    elem_types = {t: c for t, c in elem_coder._coder_types}
    self.assertIsInstance(elem_types[int], VarIntCoder)
    self.assertIsInstance(elem_types[str], StrUtf8Coder)

  def test_merged_lists_only(self):
    """List[int] | List[str] -> ListCoder(_OrderedUnionCoder[int, str])
    No outer union needed since there's only one outer type (list).
    """
    coder = build_smart_union_coder([typehints.List[int], typehints.List[str]])
    # Single outer type -> no wrapping _OrderedUnionCoder
    self.assertIsInstance(coder, ListCoder)
    elem_coder = coder._elem_coder
    self.assertIsInstance(elem_coder, _OrderedUnionCoder)
    elem_types = {t: c for t, c in elem_coder._coder_types}
    self.assertIsInstance(elem_types[int], VarIntCoder)
    self.assertIsInstance(elem_types[str], StrUtf8Coder)

  def test_merged_dicts(self):
    """Dict[str, int] | Dict[str, float]
    -> MapCoder(StrUtf8, _OrderedUnionCoder[int, float])
    Keys are the same type so no union needed there.
    """
    coder = build_smart_union_coder(
        [typehints.Dict[str, int], typehints.Dict[str, float]])
    self.assertIsInstance(coder, MapCoder)
    # Key coder: both are str, so single StrUtf8Coder (no union)
    self.assertIsInstance(coder._key_coder, StrUtf8Coder)
    # Value coder: int | float -> union
    self.assertIsInstance(coder._value_coder, _OrderedUnionCoder)
    val_types = {t: c for t, c in coder._value_coder._coder_types}
    self.assertIsInstance(val_types[int], VarIntCoder)
    self.assertIsInstance(val_types[float], FloatCoder)

  def test_dict_with_different_keys_and_values(self):
    """Dict[str, int] | Dict[int, str]
    -> MapCoder(_OrderedUnionCoder[str, int], _OrderedUnionCoder[int, str])
    """
    coder = build_smart_union_coder(
        [typehints.Dict[str, int], typehints.Dict[int, str]])
    self.assertIsInstance(coder, MapCoder)
    self.assertIsInstance(coder._key_coder, _OrderedUnionCoder)
    self.assertIsInstance(coder._value_coder, _OrderedUnionCoder)

  def test_tuple_same_arity_structure(self):
    """Tuple[int, str] | Tuple[float, bytes] -> TupleCoder[Union, Union]."""
    coder = build_smart_union_coder(
        [typehints.Tuple[int, str], typehints.Tuple[float, bytes]])
    self.assertIsInstance(coder, TupleCoder)
    self.assertEqual(len(coder._coders), 2)
    # Position 0: int | float
    self.assertIsInstance(coder._coders[0], _OrderedUnionCoder)
    pos0_types = {t: c for t, c in coder._coders[0]._coder_types}
    self.assertIn(int, pos0_types)
    self.assertIn(float, pos0_types)
    # Position 1: str | bytes
    self.assertIsInstance(coder._coders[1], _OrderedUnionCoder)
    pos1_types = {t: c for t, c in coder._coders[1]._coder_types}
    self.assertIn(str, pos1_types)
    self.assertIn(bytes, pos1_types)

  def test_tuple_same_arity_shared_position_type(self):
    """Tuple[int, str] | Tuple[int, bytes]
    -> TupleCoder[VarInt, _OrderedUnionCoder[str, bytes]]
    Position 0 has the same type so no union needed there.
    """
    coder = build_smart_union_coder(
        [typehints.Tuple[int, str], typehints.Tuple[int, bytes]])
    self.assertIsInstance(coder, TupleCoder)
    # Position 0: both int -> plain VarIntCoder
    self.assertIsInstance(coder._coders[0], VarIntCoder)
    # Position 1: str | bytes -> union
    self.assertIsInstance(coder._coders[1], _OrderedUnionCoder)

  def test_tuple_different_arities_structure(self):
    """Tuple[int] | Tuple[int, str] -> ArityDiscriminatingTupleCoder."""
    coder = build_smart_union_coder(
        [typehints.Tuple[int], typehints.Tuple[int, str]])
    self.assertIsInstance(coder, ArityDiscriminatingTupleCoder)
    self.assertIn(1, coder._arity_to_coder)
    self.assertIn(2, coder._arity_to_coder)
    # Arity 1
    self.assertIsInstance(coder._arity_to_coder[1], TupleCoder)
    self.assertEqual(len(coder._arity_to_coder[1]._coders), 1)
    # Arity 2
    self.assertIsInstance(coder._arity_to_coder[2], TupleCoder)
    self.assertEqual(len(coder._arity_to_coder[2]._coders), 2)

  def test_tuple_mixed_arities_merged_positions(self):
    """Tuple[int] | Tuple[int, int] | Tuple[str, str].

    Arity 2 merges positions into unions.
    """
    coder = build_smart_union_coder([
        typehints.Tuple[int],
        typehints.Tuple[int, int],
        typehints.Tuple[str, str]
    ])
    self.assertIsInstance(coder, ArityDiscriminatingTupleCoder)
    # Arity 1: single coder, no merge needed
    self.assertIsInstance(coder._arity_to_coder[1], TupleCoder)
    # Arity 2: merged positions
    arity2 = coder._arity_to_coder[2]
    self.assertIsInstance(arity2, TupleCoder)
    self.assertIsInstance(arity2._coders[0], _OrderedUnionCoder)
    self.assertIsInstance(arity2._coders[1], _OrderedUnionCoder)

  def test_primitives_and_tuples_outer_union(self):
    """int | Tuple[str, str]
    -> _OrderedUnionCoder[(int, VarInt), (tuple, TupleCoder[StrUtf8, StrUtf8])]
    """
    coder = build_smart_union_coder([int, typehints.Tuple[str, str]])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[int], VarIntCoder)
    self.assertIsInstance(types[tuple], TupleCoder)

  def test_nested_lists_structure(self):
    """List[List[int]] | List[List[str]]
    -> ListCoder(ListCoder(_OrderedUnionCoder[int, str]))
    Merging happens recursively at each level.
    """
    coder = build_smart_union_coder([
        typehints.List[typehints.List[int]],
        typehints.List[typehints.List[str]]
    ])
    self.assertIsInstance(coder, ListCoder)
    inner = coder._elem_coder
    self.assertIsInstance(inner, ListCoder)
    self.assertIsInstance(inner._elem_coder, _OrderedUnionCoder)

  def test_single_type_no_wrapper(self):
    """int -> VarIntCoder (no union wrapper)."""
    coder = build_smart_union_coder([int])
    self.assertIsInstance(coder, VarIntCoder)

  def test_single_list_type_no_wrapper(self):
    """List[int] -> ListCoder(VarInt) (no union wrapper)."""
    coder = build_smart_union_coder([typehints.List[int]])
    self.assertIsInstance(coder, ListCoder)
    self.assertIsInstance(coder._elem_coder, VarIntCoder)

  def test_generated_row_with_primitive_structure(self):
    """GeneratedClassRowTypeConstraint(x=int) | str
    -> _OrderedUnionCoder[(BeamSchema_xxx, RowCoder), (str, StrUtf8)]
    from_fields creates GeneratedClassRowTypeConstraint with a unique
    NamedTuple _user_type, which is used directly for discrimination.
    """
    row_hint = RowTypeConstraint.from_fields([("x", int)])
    coder = build_smart_union_coder([row_hint, str])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    # The key is the generated NamedTuple class, not Row
    self.assertIsInstance(types[row_hint._user_type], RowCoder)
    self.assertIsInstance(types[str], StrUtf8Coder)

  def test_two_generated_rows_discriminable(self):
    """Two GeneratedClassRowTypeConstraints have distinct _user_type classes,
    so both can be discriminated — no fallback needed.
    """
    row_a = RowTypeConstraint.from_fields([("x", int)])
    row_b = RowTypeConstraint.from_fields([("y", str)])
    coder = build_smart_union_coder([row_a, row_b, int])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[row_a._user_type], RowCoder)
    self.assertIsInstance(types[row_b._user_type], RowCoder)
    self.assertIsInstance(types[int], VarIntCoder)
    self.assertIsNone(coder._fallback_coder)

  def test_raw_row_with_primitive_structure(self):
    """Plain RowTypeConstraint(user_type=Row) | str
    -> _OrderedUnionCoder[(Row, RowCoder), (str, StrUtf8)]
    """
    row_hint = RowTypeConstraint(fields=[("x", int)], user_type=Row)
    coder = build_smart_union_coder([row_hint, str])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[Row], RowCoder)
    self.assertIsInstance(types[str], StrUtf8Coder)

  def test_multiple_raw_rows_fallback_structure(self):
    """Multiple plain RowTypeConstraints (user_type=Row) can't be
    discriminated — fall back to FastPrimitivesCoder.
    """
    row_a = RowTypeConstraint(fields=[("x", int)], user_type=Row)
    row_b = RowTypeConstraint(fields=[("y", str)], user_type=Row)
    coder = build_smart_union_coder([row_a, row_b, int])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[int], VarIntCoder)
    self.assertNotIn(Row, types)
    self.assertIsInstance(coder._fallback_coder, FastPrimitivesCoder)


class BeamRowUnionTest(unittest.TestCase):
  """Tests for SmartUnionCoder with beam.Row types."""
  def _roundtrip(self, coder, value):
    encoded = coder.encode(value)
    decoded = coder.decode(encoded)
    return decoded

  def test_single_beam_row_with_primitive(self):
    """Single raw Row type in union uses RowCoder, not pickle fallback."""
    row_hint = RowTypeConstraint(fields=[("x", int), ("y", str)], user_type=Row)
    coder = build_smart_union_coder([row_hint, int])

    # Should be _OrderedUnionCoder with Row mapped to RowCoder
    self.assertIsInstance(coder, _OrderedUnionCoder)

    # int roundtrip
    self.assertEqual(42, self._roundtrip(coder, 42))

    # Row roundtrip — RowCoder decodes to NamedTuple, not beam.Row
    row = Row(x=10, y="hello")
    decoded = self._roundtrip(coder, row)
    self.assertEqual(decoded.x, 10)
    self.assertEqual(decoded.y, "hello")

  def test_single_beam_row_with_list(self):
    """Single raw Row type discriminated from List by type(value)."""
    row_hint = RowTypeConstraint(fields=[("a", int)], user_type=Row)
    coder = build_smart_union_coder([row_hint, typehints.List[int]])

    self.assertEqual([1, 2, 3], self._roundtrip(coder, [1, 2, 3]))

    row = Row(a=99)
    decoded = self._roundtrip(coder, row)
    self.assertEqual(decoded.a, 99)

  def test_single_beam_row_only(self):
    """Single raw Row type hint returns RowCoder directly."""
    row_hint = RowTypeConstraint(fields=[("x", int)], user_type=Row)
    coder = build_smart_union_coder([row_hint])

    self.assertIsInstance(coder, RowCoder)

    row = Row(x=42)
    decoded = self._roundtrip(coder, row)
    self.assertEqual(decoded.x, 42)

  def test_multiple_generated_rows_discriminable(self):
    """Multiple GeneratedClassRowTypeConstraints have unique _user_type
    classes and can be discriminated without fallback."""
    row_hint_a = RowTypeConstraint.from_fields([("x", int)])
    row_hint_b = RowTypeConstraint.from_fields([("y", str)])
    coder = build_smart_union_coder([row_hint_a, row_hint_b])

    # Both generated types are discriminable — no fallback needed
    self.assertIsInstance(coder, _OrderedUnionCoder)

  def test_multiple_raw_rows_uses_fallback(self):
    """Multiple plain RowTypeConstraint(user_type=Row) fall back."""
    row_hint_a = RowTypeConstraint(fields=[("x", int)], user_type=Row)
    row_hint_b = RowTypeConstraint(fields=[("y", str)], user_type=Row)
    coder = build_smart_union_coder([row_hint_a, row_hint_b])

    # Both share user_type=Row — can't discriminate
    self.assertIsInstance(coder, FastPrimitivesCoder)

  def test_multiple_beam_rows_with_primitive(self):
    """Multiple generated Row types with a primitive — all discriminable."""
    row_hint_a = RowTypeConstraint.from_fields([("x", int)])
    row_hint_b = RowTypeConstraint.from_fields([("y", str)])
    coder = build_smart_union_coder([row_hint_a, row_hint_b, int])

    self.assertIsInstance(coder, _OrderedUnionCoder)

    # int should roundtrip correctly
    self.assertEqual(42, self._roundtrip(coder, 42))


class UserNamedTupleUnionTest(unittest.TestCase):
  """Tests for SmartUnionCoder with user-defined NamedTuple Row types."""
  def _roundtrip(self, coder, value):
    encoded = coder.encode(value)
    decoded = coder.decode(encoded)
    return decoded

  def test_two_user_namedtuples_discriminable(self):
    """Two user-defined NamedTuples have distinct _user_type classes,
    so both can be discriminated — no fallback needed."""
    class PersonRow(NamedTuple):
      name: str
      age: int

    class OrderRow(NamedTuple):
      id: int
      total: float

    person_hint = RowTypeConstraint.from_user_type(PersonRow)
    order_hint = RowTypeConstraint.from_user_type(OrderRow)

    # Verify these are plain RowTypeConstraint with user's own class
    self.assertIs(person_hint._user_type, PersonRow)
    self.assertIs(order_hint._user_type, OrderRow)

    coder = build_smart_union_coder([person_hint, order_hint])
    self.assertIsInstance(coder, _OrderedUnionCoder)

    # Both user types should be in the coder — no fallback
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[PersonRow], RowCoder)
    self.assertIsInstance(types[OrderRow], RowCoder)
    self.assertIsNone(coder._fallback_coder)

  def test_two_user_namedtuples_roundtrip(self):
    """Two user-defined NamedTuples roundtrip correctly through the union."""
    class PersonRow(NamedTuple):
      name: str
      age: int

    class OrderRow(NamedTuple):
      id: int
      total: float

    person_hint = RowTypeConstraint.from_user_type(PersonRow)
    order_hint = RowTypeConstraint.from_user_type(OrderRow)

    coder = build_smart_union_coder([person_hint, order_hint])

    person = PersonRow(name="Alice", age=30)
    decoded_person = self._roundtrip(coder, person)
    self.assertEqual(decoded_person.name, "Alice")
    self.assertEqual(decoded_person.age, 30)

    order = OrderRow(id=123, total=99.5)
    decoded_order = self._roundtrip(coder, order)
    self.assertEqual(decoded_order.id, 123)
    self.assertAlmostEqual(decoded_order.total, 99.5)

  def test_user_namedtuple_with_primitive(self):
    """User NamedTuple | int — both discriminable by type()."""
    class PersonRow(NamedTuple):
      name: str
      age: int

    person_hint = RowTypeConstraint.from_user_type(PersonRow)
    coder = build_smart_union_coder([person_hint, int])

    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[PersonRow], RowCoder)
    self.assertIsInstance(types[int], VarIntCoder)
    self.assertIsNone(coder._fallback_coder)

    # Roundtrip both
    self.assertEqual(42, self._roundtrip(coder, 42))
    person = PersonRow(name="Bob", age=25)
    decoded = self._roundtrip(coder, person)
    self.assertEqual(decoded.name, "Bob")
    self.assertEqual(decoded.age, 25)

  def test_user_namedtuple_with_generated_row(self):
    """User NamedTuple | GeneratedClassRowTypeConstraint — both discriminable
    since each has a distinct _user_type class."""
    class PersonRow(NamedTuple):
      name: str
      age: int

    person_hint = RowTypeConstraint.from_user_type(PersonRow)
    generated_hint = RowTypeConstraint.from_fields([("x", int), ("y", str)])

    # Verify distinct _user_type classes
    self.assertIsNot(person_hint._user_type, generated_hint._user_type)

    coder = build_smart_union_coder([person_hint, generated_hint])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[PersonRow], RowCoder)
    self.assertIsInstance(types[generated_hint._user_type], RowCoder)
    self.assertIsNone(coder._fallback_coder)

  def test_user_namedtuple_with_raw_row(self):
    """User NamedTuple | beam.Row — both discriminable since user NamedTuple
    has its own class and there's only one raw Row."""
    class PersonRow(NamedTuple):
      name: str
      age: int

    person_hint = RowTypeConstraint.from_user_type(PersonRow)
    raw_hint = RowTypeConstraint(fields=[("x", int)], user_type=Row)

    coder = build_smart_union_coder([person_hint, raw_hint])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[PersonRow], RowCoder)
    self.assertIsInstance(types[Row], RowCoder)
    self.assertIsNone(coder._fallback_coder)


class DeterministicOrderingTest(unittest.TestCase):
  """Tests that coder structure is deterministic regardless of input order."""
  def test_different_input_order_same_coder(self):
    """int | str and str | int produce the same coder structure."""
    coder1 = build_smart_union_coder([int, str])
    coder2 = build_smart_union_coder([str, int])

    # Both should have the same _coder_types order
    self.assertEqual(coder1._coder_types, coder2._coder_types)
    self.assertEqual(coder1, coder2)

  def test_different_input_order_same_encoding(self):
    """Values encode to same bytes regardless of type hint order."""
    coder1 = build_smart_union_coder([int, str])
    coder2 = build_smart_union_coder([str, int])

    # Same value should produce identical bytes
    self.assertEqual(coder1.encode(42), coder2.encode(42))
    self.assertEqual(coder1.encode("hello"), coder2.encode("hello"))

    # Cross-decode: encode with one, decode with other
    self.assertEqual(42, coder2.decode(coder1.encode(42)))
    self.assertEqual("hello", coder1.decode(coder2.encode("hello")))

  def test_three_types_different_orders(self):
    """Three types in any order produce identical coders."""
    orders = [
        [int, str, bytes],
        [str, int, bytes],
        [bytes, str, int],
        [int, bytes, str],
        [str, bytes, int],
        [bytes, int, str],
    ]
    coders = [build_smart_union_coder(order) for order in orders]

    # All coders should be equal
    for i, c1 in enumerate(coders):
      for j, c2 in enumerate(coders):
        self.assertEqual(
            c1, c2, f"Coders for order {i} and {j} differ: {c1} vs {c2}")

    # All should produce identical encoding
    for value in [42, "hello", b"bytes"]:
      encoded = [c.encode(value) for c in coders]
      self.assertTrue(
          all(e == encoded[0] for e in encoded),
          f"Different encodings for {value}: {encoded}")

  def test_list_element_types_deterministic(self):
    """List[int] | List[str] order doesn't affect element coder order."""
    coder1 = build_smart_union_coder([typehints.List[int], typehints.List[str]])
    coder2 = build_smart_union_coder([typehints.List[str], typehints.List[int]])

    self.assertEqual(coder1, coder2)

    # Same encoding for lists
    self.assertEqual(coder1.encode([1, 2]), coder2.encode([1, 2]))
    self.assertEqual(coder1.encode(["a"]), coder2.encode(["a"]))

  def test_dict_types_deterministic(self):
    """Dict type order doesn't affect key/value coder order."""
    coder1 = build_smart_union_coder(
        [typehints.Dict[str, int], typehints.Dict[int, str]])
    coder2 = build_smart_union_coder(
        [typehints.Dict[int, str], typehints.Dict[str, int]])

    self.assertEqual(coder1, coder2)

  def test_tuple_position_types_deterministic(self):
    """Tuple position types are ordered deterministically."""
    coder1 = build_smart_union_coder(
        [typehints.Tuple[int, str], typehints.Tuple[str, int]])
    coder2 = build_smart_union_coder(
        [typehints.Tuple[str, int], typehints.Tuple[int, str]])

    self.assertEqual(coder1, coder2)

  def test_mixed_types_deterministic(self):
    """Complex mix of types produces deterministic coder."""
    coder1 = build_smart_union_coder(
        [int, str, typehints.List[int], typehints.List[str]])
    coder2 = build_smart_union_coder(
        [typehints.List[str], str, typehints.List[int], int])

    self.assertEqual(coder1, coder2)

    # Cross-encode/decode
    for value in [42, "hi", [1, 2], ["a", "b"]]:
      self.assertEqual(value, coder2.decode(coder1.encode(value)))
      self.assertEqual(value, coder1.decode(coder2.encode(value)))

  def test_custom_classes_deterministic(self):
    """Custom classes are ordered deterministically by module.qualname."""
    coder1 = build_smart_union_coder([Point, Color, int])
    coder2 = build_smart_union_coder([int, Color, Point])
    coder3 = build_smart_union_coder([Color, int, Point])

    self.assertEqual(coder1, coder2)
    self.assertEqual(coder2, coder3)


class CustomClassUnionTest(unittest.TestCase):
  """Tests for SmartUnionCoder with user-defined classes and coders."""
  def _roundtrip(self, coder, value):
    encoded = coder.encode(value)
    decoded = coder.decode(encoded)
    return decoded

  def test_custom_class_with_primitive_structure(self):
    """Point | int -> _OrderedUnionCoder[(Point, PointCoder), (int, VarInt)]"""
    coder = build_smart_union_coder([Point, int])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[Point], PointCoder)
    self.assertIsInstance(types[int], VarIntCoder)

  def test_custom_class_with_primitive_roundtrip(self):
    """Point | int roundtrips both types correctly."""
    coder = build_smart_union_coder([Point, int])

    self.assertEqual(42, self._roundtrip(coder, 42))
    self.assertEqual(Point(1.0, 2.0), self._roundtrip(coder, Point(1.0, 2.0)))

  def test_two_custom_classes_structure(self):
    """Point | Color -> _OrderedUnionCoder with both custom coders."""
    coder = build_smart_union_coder([Point, Color])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[Point], PointCoder)
    self.assertIsInstance(types[Color], ColorCoder)
    self.assertIsNone(coder._fallback_coder)

  def test_two_custom_classes_roundtrip(self):
    """Point | Color roundtrips both types correctly."""
    coder = build_smart_union_coder([Point, Color])

    p = Point(3.14, 2.72)
    c = Color(255, 128, 0)
    self.assertEqual(p, self._roundtrip(coder, p))
    self.assertEqual(c, self._roundtrip(coder, c))

  def test_three_custom_classes(self):
    """Point | Color | Label -> _OrderedUnionCoder with three entries."""
    coder = build_smart_union_coder([Point, Color, Label])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[Point], PointCoder)
    self.assertIsInstance(types[Color], ColorCoder)
    self.assertIsInstance(types[Label], LabelCoder)

    self.assertEqual(Point(0, 0), self._roundtrip(coder, Point(0, 0)))
    self.assertEqual(Color(0, 0, 0), self._roundtrip(coder, Color(0, 0, 0)))
    self.assertEqual(Label("test"), self._roundtrip(coder, Label("test")))

  def test_custom_class_with_container_structure(self):
    """Point | List[int] | str
    -> _OrderedUnionCoder[
         (Point, PointCoder),
         (list, ListCoder(VarInt)),
         (str, StrUtf8)]
    """
    coder = build_smart_union_coder([Point, typehints.List[int], str])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[Point], PointCoder)
    self.assertIsInstance(types[list], ListCoder)
    self.assertIsInstance(types[str], StrUtf8Coder)

  def test_custom_class_with_container_roundtrip(self):
    """Point | List[int] | str roundtrips all types."""
    coder = build_smart_union_coder([Point, typehints.List[int], str])

    self.assertEqual(Point(1, 2), self._roundtrip(coder, Point(1, 2)))
    self.assertEqual([1, 2, 3], self._roundtrip(coder, [1, 2, 3]))
    self.assertEqual("hello", self._roundtrip(coder, "hello"))

  def test_custom_class_with_merged_lists(self):
    """Point | List[int] | List[str]
    -> _OrderedUnionCoder[
         (Point, PointCoder),
         (list, ListCoder(_OrderedUnionCoder[int, str]))]
    """
    coder = build_smart_union_coder(
        [Point, typehints.List[int], typehints.List[str]])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[Point], PointCoder)
    list_coder = types[list]
    self.assertIsInstance(list_coder, ListCoder)
    self.assertIsInstance(list_coder._elem_coder, _OrderedUnionCoder)

  def test_custom_class_single_type(self):
    """Point alone -> PointCoder (no union wrapper)."""
    coder = build_smart_union_coder([Point])
    self.assertIsInstance(coder, PointCoder)

  def test_custom_classes_with_tuples(self):
    """Point | Color | Tuple[int, str]
    -> _OrderedUnionCoder[
         (Point, PointCoder),
         (Color, ColorCoder),
         (tuple, TupleCoder[VarInt, StrUtf8])]
    """
    coder = build_smart_union_coder([Point, Color, typehints.Tuple[int, str]])
    self.assertIsInstance(coder, _OrderedUnionCoder)
    types = {t: c for t, c in coder._coder_types}
    self.assertIsInstance(types[Point], PointCoder)
    self.assertIsInstance(types[Color], ColorCoder)
    self.assertIsInstance(types[tuple], TupleCoder)

    self.assertEqual(Point(1, 2), self._roundtrip(coder, Point(1, 2)))
    self.assertEqual(
        Color(10, 20, 30), self._roundtrip(coder, Color(10, 20, 30)))
    self.assertEqual((42, "x"), self._roundtrip(coder, (42, "x")))


if __name__ == '__main__':
  unittest.main()
