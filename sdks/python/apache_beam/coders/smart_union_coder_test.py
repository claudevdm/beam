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

import unittest
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

from apache_beam.coders.smart_union_coder import ArityDiscriminatingTupleCoder
from apache_beam.coders.smart_union_coder import build_smart_union_coder
from apache_beam.coders.coders import _OrderedUnionCoder
from apache_beam.coders.coders import StrUtf8Coder
from apache_beam.coders.coders import TupleCoder
from apache_beam.coders.coders import VarIntCoder
from apache_beam.typehints import typehints


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
    self.assertEqual("unicode: \u00e9\u00e8", self._roundtrip(coder, "unicode: \u00e9\u00e8"))

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
    coder = build_smart_union_coder([
        typehints.List[int],
        typehints.List[str]
    ])

    # Pure int list
    self.assertEqual([1, 2, 3], self._roundtrip(coder, [1, 2, 3]))

    # Pure str list
    self.assertEqual(["a", "b"], self._roundtrip(coder, ["a", "b"]))

    # Mixed list (allowed after merge!)
    self.assertEqual([1, "a", 2, "b"], self._roundtrip(coder, [1, "a", 2, "b"]))

  def test_mixed_outer_types(self):
    """Test Union[int, str, List[int], List[str]]."""
    coder = build_smart_union_coder([
        int,
        str,
        typehints.List[int],
        typehints.List[str]
    ])

    # Primitives
    self.assertEqual(42, self._roundtrip(coder, 42))
    self.assertEqual("hello", self._roundtrip(coder, "hello"))

    # Lists
    self.assertEqual([1, 2, 3], self._roundtrip(coder, [1, 2, 3]))
    self.assertEqual(["a", "b"], self._roundtrip(coder, ["a", "b"]))

    # Mixed list (merged)
    self.assertEqual([1, "a"], self._roundtrip(coder, [1, "a"]))

  def test_tuple_same_arity(self):
    """Test Tuple[int, int] and Tuple[str, str] - same arity, merge positions."""
    coder = build_smart_union_coder([
        typehints.Tuple[int, int],
        typehints.Tuple[str, str]
    ])

    self.assertEqual((1, 2), self._roundtrip(coder, (1, 2)))
    self.assertEqual(("a", "b"), self._roundtrip(coder, ("a", "b")))

    # Mixed tuple (allowed after position merge!)
    self.assertEqual((1, "b"), self._roundtrip(coder, (1, "b")))
    self.assertEqual(("a", 2), self._roundtrip(coder, ("a", 2)))

  def test_tuple_different_arities(self):
    """Test tuples with different arities using ArityDiscriminatingTupleCoder."""
    coder = build_smart_union_coder([
        typehints.Tuple[int],
        typehints.Tuple[int, int],
        typehints.Tuple[int, int, int]
    ])

    # Should be ArityDiscriminatingTupleCoder
    self.assertIsInstance(coder, ArityDiscriminatingTupleCoder)

    # Test each arity
    self.assertEqual((1,), self._roundtrip(coder, (1,)))
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
    self.assertEqual((42,), self._roundtrip(coder, (42,)))

    # arity=2: merged positions (int|str, int|str)
    self.assertEqual((1, 2), self._roundtrip(coder, (1, 2)))
    self.assertEqual(("a", "b"), self._roundtrip(coder, ("a", "b")))
    self.assertEqual((1, "b"), self._roundtrip(coder, (1, "b")))

  def test_dict_merging(self):
    """Test merging Dict[str, int] and Dict[str, str]."""
    coder = build_smart_union_coder([
        typehints.Dict[str, int],
        typehints.Dict[str, str]
    ])

    self.assertEqual({"a": 1, "b": 2}, self._roundtrip(coder, {"a": 1, "b": 2}))
    self.assertEqual({"x": "y"}, self._roundtrip(coder, {"x": "y"}))

    # Mixed values (allowed after merge!)
    self.assertEqual({"a": 1, "b": "two"}, self._roundtrip(coder, {"a": 1, "b": "two"}))

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
    from apache_beam.coders.coders import VarIntCoder
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

    self.assertEqual((1,), self._roundtrip(coder, (1,)))
    self.assertEqual((1, 2), self._roundtrip(coder, (1, 2)))
    self.assertEqual((1, 2, 3), self._roundtrip(coder, (1, 2, 3)))

  def test_arity_zero(self):
    """Test arity 0 (empty tuple)."""
    coder = ArityDiscriminatingTupleCoder({
        0: TupleCoder([]),
        1: TupleCoder([VarIntCoder()]),
    })

    self.assertEqual((), self._roundtrip(coder, ()))
    self.assertEqual((42,), self._roundtrip(coder, (42,)))

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
    self.assertEqual("\U0001f600", self._roundtrip(coder, "\U0001f600"))  # Emoji

  def test_nested_empty_structures(self):
    """Test with nested empty structures."""
    coder = build_smart_union_coder([
        typehints.List[typehints.List[int]]
    ])
    self.assertEqual([[]], self._roundtrip(coder, [[]]))
    self.assertEqual([[], []], self._roundtrip(coder, [[], []]))

  def test_deeply_nested_union(self):
    """Test with deeply nested structure."""
    # List[List[List[int|str]]]
    inner_union = build_smart_union_coder([int, str])
    from apache_beam.coders.coders import ListCoder
    coder = ListCoder(ListCoder(ListCoder(inner_union)))

    value = [[[1, "a", 2], ["b", 3]], [[4]]]
    self.assertEqual(value, self._roundtrip(coder, value))


if __name__ == '__main__':
  unittest.main()
