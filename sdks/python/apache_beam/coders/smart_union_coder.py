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

"""SmartUnionCoder: Efficient encoding for Union types.

This module provides a coder that handles Union types by:
1. Grouping types by their outer Python type (list, tuple, dict, class)
2. Using O(1) type discrimination for different outer types
3. Merging same-outer types recursively (e.g., List[int] | List[str] -> 
   List[int|str])
4. Handling different tuple arities via len() discrimination
"""

from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from apache_beam.coders import coder_impl
from apache_beam.coders.coders import Coder
from apache_beam.coders.coders import FastCoder
from apache_beam.coders.coders import FastPrimitivesCoder
from apache_beam.coders.coders import ListCoder
from apache_beam.coders.coders import MapCoder
from apache_beam.coders.coders import TupleCoder
from apache_beam.coders.coders import _OrderedUnionCoder
from apache_beam.pvalue import Row
from apache_beam.typehints import typehints
from apache_beam.typehints.row_type import RowTypeConstraint

__all__ = ['build_smart_union_coder', 'ArityDiscriminatingTupleCoder']


def _type_sort_key(t) -> str:
  """Return a deterministic sort key for a type or type hint.

  This ensures that Union coders have a consistent wire format regardless of
  the order type hints are passed to build_smart_union_coder. Without this,
  [int, str] and [str, int] would produce different tag bytes for the same
  values.

  This matters for streaming pipeline updates: when a running streaming
  pipeline is updated, the new pipeline version must be able to decode state
  that was encoded by the previous version. If the coder wire format changes
  (e.g., because type hints are ordered differently in the new code), existing
  state becomes unreadable.

  The key is designed to be:
  1. Deterministic across runs (no id() or hash())
  2. Unique enough to distinguish types we care about
  3. Stable across Python versions

  For primitive types: uses __name__ (e.g., "int", "str")
  For TypeConstraints: uses repr() which includes inner types
  For classes: uses __module__.__qualname__ for uniqueness
  """
  if isinstance(t, type):
    # Use module + qualname for classes to handle same-named classes
    # in different modules (e.g., mymodule.Point vs othermodule.Point)
    module = getattr(t, '__module__', '')
    qualname = getattr(t, '__qualname__', t.__name__)
    return f"{module}.{qualname}"
  # TypeConstraints (ListConstraint, TupleConstraint, etc.) and other hints
  # repr() is deterministic and includes inner type info
  return repr(t)


class ArityDiscriminatingTupleCoderImpl(coder_impl.StreamCoderImpl):
  """CoderImpl for tuples with different arities.

  Uses len(value) to discriminate between tuple coders of different arities.
  """
  def __init__(
      self,
      arity_to_coder_impl: Dict[int, coder_impl.CoderImpl],
      fallback_coder_impl: Optional[coder_impl.CoderImpl] = None):
    """Initialize ArityDiscriminatingTupleCoderImpl.

    Args:
      arity_to_coder_impl: Maps tuple length -> coder_impl
      fallback_coder_impl: Optional fallback for unrecognized arities
    """
    self._arity_to_coder_impl = arity_to_coder_impl
    self._fallback_coder_impl = fallback_coder_impl

  def encode_to_stream(self, value, out, nested):
    arity = len(value)
    coder_impl = self._arity_to_coder_impl.get(arity)
    if coder_impl is not None:
      # Write arity as varint, then encode tuple
      out.write_var_int64(arity)
      coder_impl.encode_to_stream(value, out, nested)
    elif self._fallback_coder_impl is not None:
      out.write_var_int64(-1)  # Sentinel for fallback
      self._fallback_coder_impl.encode_to_stream(value, out, nested)
    else:
      raise ValueError(
          f"Cannot encode tuple of arity {arity}. "
          f"Known arities: {list(self._arity_to_coder_impl.keys())}")

  def decode_from_stream(self, in_stream, nested):
    arity = in_stream.read_var_int64()
    if arity == -1:
      if self._fallback_coder_impl is None:
        raise ValueError("No fallback coder configured")
      return self._fallback_coder_impl.decode_from_stream(in_stream, nested)
    coder_impl = self._arity_to_coder_impl.get(arity)
    if coder_impl is None:
      raise ValueError(f"Unknown arity: {arity}")
    return coder_impl.decode_from_stream(in_stream, nested)


class ArityDiscriminatingTupleCoder(FastCoder):
  """Coder for tuples with different arities.

  Discriminates between tuple types based on len(value).
  For example: Union[Tuple[int], Tuple[int, int], Tuple[str, str, str]]
  """
  def __init__(
      self,
      arity_to_coder: Dict[int, Coder],
      fallback_coder: Optional[Coder] = None):
    """Initialize ArityDiscriminatingTupleCoder.

    Args:
      arity_to_coder: Maps tuple arity -> coder for that arity
      fallback_coder: Optional fallback for unrecognized arities
    """
    self._arity_to_coder = arity_to_coder
    self._fallback_coder = fallback_coder

  def _create_impl(self):
    return ArityDiscriminatingTupleCoderImpl(
        {
            arity: coder.get_impl()
            for arity, coder in self._arity_to_coder.items()
        },
        fallback_coder_impl=self._fallback_coder.get_impl()
        if self._fallback_coder else None)

  def is_deterministic(self) -> bool:
    return (
        all(c.is_deterministic() for c in self._arity_to_coder.values()) and (
            self._fallback_coder is None or
            self._fallback_coder.is_deterministic()))

  def as_deterministic_coder(self, step_label, error_message=None):
    if self.is_deterministic():
      return self
    return ArityDiscriminatingTupleCoder(
        {
            arity: coder.as_deterministic_coder(step_label, error_message)
            for arity, coder in self._arity_to_coder.items()
        },
        fallback_coder=self._fallback_coder.as_deterministic_coder(
            step_label, error_message) if self._fallback_coder else None)

  def to_type_hint(self):
    # Return Union of all tuple types
    hints = []
    for _, coder in sorted(self._arity_to_coder.items()):
      hint = coder.to_type_hint()
      hints.append(hint)
    if len(hints) == 1:
      return hints[0]
    return typehints.Union[tuple(hints)]

  def __repr__(self):
    return 'ArityDiscriminatingTupleCoder[%s]' % ', '.join(
        f'{arity}: {coder}'
        for arity, coder in sorted(self._arity_to_coder.items()))

  def __eq__(self, other):
    return (
        type(self) == type(other) and
        self._arity_to_coder == other._arity_to_coder and
        self._fallback_coder == other._fallback_coder)

  def __hash__(self):
    return hash((
        type(self),
        tuple(sorted(self._arity_to_coder.items())),
        self._fallback_coder))


class _SmartUnionCoderFactory:
  """Factory class for registry integration.

  This class only exists to provide the from_type_hint() method for the
  coder registry. The actual coders returned are _OrderedUnionCoder,
  ArityDiscriminatingTupleCoder, or container coders with nested unions.
  """
  @classmethod
  def from_type_hint(cls, typehint, registry):
    """Build a coder from a UnionConstraint type hint.

    This is called by the coder registry when it encounters a Union type.

    Args:
      typehint: A UnionConstraint (e.g., Union[int, str, List[int]])
      registry: The coder registry

    Returns:
      A coder that can handle any of the union's types.
    """
    union_types = list(typehint.union_types)
    return build_smart_union_coder(union_types, registry)


# Mapping from TypeConstraint class to Python runtime type for discrimination.
# Used by _OrderedUnionCoder which does `type(value) is t` at runtime.
#
# Why IterableTypeConstraint maps to `list`:
# -----------------------------------------
# At runtime, Beam's IterableCoder decodes values as Python lists, not abstract
# iterables. This is controlled by the flag
# `_iterable_coder_uses_abstract_iterable_by_default` which is set to False
# (see coder_impl.py:1449).
#
# The decoding flow is:
# 1. SequenceCoderImpl.decode_from_stream (coder_impl.py:1350-1376) reads
#    elements into a Python list: `elements = [...]` (line 1355) or
#    `elements = []` with `elements.append(...)` (lines 1360-1364).
# 2. IterableCoderImpl._construct_from_sequence (coder_impl.py:1462-1466)
#    returns the list directly when `_use_abstract_iterable` is False
#    (the default).
#
# This means that even when a PCollection has type hint Iterable[T], the actual
# runtime values are Python lists. Therefore, for type discrimination in union
# coders using `type(value) is t`, we must map IterableTypeConstraint to `list`.
#
# Note: convert_to_python_type() from native_type_compatibility.py returns
# `collections.abc.Iterable` for IterableTypeConstraint, which would NOT work
# for runtime type discrimination since `type([1,2,3]) is
# collections.abc.Iterable` is False.
_CONSTRAINT_TO_PYTHON_TYPE: Dict[type, type] = {
    typehints.ListConstraint: list,
    typehints.TupleConstraint: tuple,
    typehints.DictConstraint: dict,
    typehints.IterableTypeConstraint: list,
}


def _get_outer_type(typehint) -> Optional[type]:
  """Get the outer Python type for a type hint.

  Returns:
    The Python type (list, tuple, dict, class) for discrimination.
    Returns None only if truly unable to determine the type.
  """
  # For concrete types (int, str, custom classes, NamedTuples), return the type
  # itself
  if isinstance(typehint, type):
    return typehint
  # Look up TypeConstraint class in mapping
  # Note: beam.Row types (RowTypeConstraint) are not in this mapping.
  # Single Row types in a union are handled directly in
  # build_smart_union_coder() by mapping to the Row class.
  # Multiple Row types fall back to
  # FastPrimitivesCoder since all beam.Row instances share the same class.
  return _CONSTRAINT_TO_PYTHON_TYPE.get(type(typehint))


def _get_tuple_arity(typehint) -> Optional[int]:
  """Get the arity of a tuple type hint."""
  if isinstance(typehint, typehints.TupleConstraint):
    return len(typehint.tuple_types)
  return None


def build_smart_union_coder(
    type_hints: List[Any],
    registry=None,
    fallback_coder: Optional[Coder] = None) -> Coder:
  """Build a SmartUnionCoder from a list of type hints.

  This function analyzes the type hints and builds an efficient coder that:
  1. Groups types by outer Python type
  2. Merges same-outer types recursively
  3. Handles different tuple arities

  Args:
    type_hints: List of type hints to union
    registry: Coder registry (uses default if None)
    fallback_coder: Optional fallback for unrecognized types

  Returns:
    A coder that can handle any of the given types.

  Example:
    # Different outer types - direct discrimination
    coder = build_smart_union_coder([int, str, List[int]])

    # Same outer types - merged
    coder = build_smart_union_coder([List[int], List[str]])
    # Returns ListCoder(_OrderedUnionCoder[int, str])

    # Different tuple arities
    coder = build_smart_union_coder([Tuple[int], Tuple[int, int]])
    # Returns ArityDiscriminatingTupleCoder
  """
  if registry is None:
    from apache_beam.coders import typecoders  # pylint: disable=reimported
    registry = typecoders.registry

  if not type_hints:
    raise ValueError("Cannot build union coder from empty type hints")

  if len(type_hints) == 1:
    return registry.get_coder(type_hints[0])

  # Handle RowTypeConstraint hints using _user_type for discrimination.
  # - GeneratedClassRowTypeConstraint has _user_type = BeamSchema_xxx (unique
  #   per schema), so each can be discriminated by type(value) is
  #   BeamSchema_xxx.
  # - Plain RowTypeConstraint has _user_type = Row. A single one can be
  #   discriminated by type(value) is Row, but multiple cannot since all
  #   beam.Row instances share the same class.
  raw_row_hints = [
      h for h in type_hints
      if isinstance(h, RowTypeConstraint) and h._user_type is Row
  ]
  single_raw_row = len(raw_row_hints) == 1

  # Group by outer Python type for discrimination
  by_outer_type: Dict[Optional[type], List[Any]] = defaultdict(list)
  for hint in type_hints:
    if isinstance(hint, RowTypeConstraint):
      if hint._user_type is Row:
        # Raw beam.Row — can only discriminate if there's exactly one
        outer = Row if single_raw_row else None
      else:
        # Generated NamedTuple class — unique per schema, always discriminable
        outer = hint._user_type
    else:
      outer = _get_outer_type(hint)
    by_outer_type[outer].append(hint)

  # Build (type, coder) pairs for each group
  coder_types: List[Tuple[type, Coder]] = []

  for outer_type, hints in by_outer_type.items():
    if outer_type is None:
      # Can't determine outer type - use fallback
      if fallback_coder is None:
        fallback_coder = FastPrimitivesCoder()
      continue

    if len(hints) == 1:
      # Single type in group - use its coder directly
      coder = registry.get_coder(hints[0])
    else:
      # Multiple types with same outer - need to merge
      coder = _merge_same_outer_types(outer_type, hints, registry)

    coder_types.append((outer_type, coder))

  if len(coder_types) == 0:
    # All types had unknown outer type - use fallback only
    if fallback_coder is None:
      fallback_coder = FastPrimitivesCoder()
    return fallback_coder

  if len(coder_types) == 1 and fallback_coder is None:
    # Only one outer type - return its coder directly (no union needed)
    return coder_types[0][1]

  # Sort coder_types for deterministic wire format.
  # This ensures [int, str] and [str, int] produce the same coder with
  # the same tag bytes. Critical for streaming pipeline updates: state
  # encoded by the previous pipeline version must remain decodable.
  coder_types.sort(key=lambda x: _type_sort_key(x[0]))

  # Use _OrderedUnionCoder for O(1) type discrimination
  # This is Cython-optimized and handles all cases where outer types differ
  return _OrderedUnionCoder(*coder_types, fallback_coder=fallback_coder)


def _merge_same_outer_types(
    outer_type: type, hints: List[Any], registry) -> Coder:
  """Merge type hints that have the same outer Python type.

  Args:
    outer_type: The common outer type (list, tuple, dict, or class)
    hints: List of type hints with this outer type
    registry: Coder registry

  Returns:
    A merged coder for the types.
  """
  if outer_type == list:
    # List[A], List[B] -> ListCoder(SmartUnionCoder[A, B])
    element_types = []
    for hint in hints:
      if isinstance(hint, typehints.ListConstraint):
        element_types.append(hint.inner_type)
      elif isinstance(hint, typehints.IterableTypeConstraint):
        element_types.append(hint.inner_type)
      else:
        raise ValueError(f"Expected list type hint, got {hint}")

    # Deduplicate and sort for deterministic ordering
    element_types = list(dict.fromkeys(element_types))
    element_types.sort(key=_type_sort_key)
    if len(element_types) == 1:
      element_coder = registry.get_coder(element_types[0])
    else:
      element_coder = build_smart_union_coder(element_types, registry)
    return ListCoder(element_coder)

  if outer_type == tuple:
    # Group by arity
    by_arity: Dict[int, List[Any]] = defaultdict(list)
    for hint in hints:
      arity = _get_tuple_arity(hint)
      if arity is not None:
        by_arity[arity].append(hint)
      else:
        raise ValueError(f"Expected tuple type hint, got {hint}")

    if len(by_arity) == 1:
      # All same arity - merge positions
      arity = list(by_arity.keys())[0]
      return _merge_tuple_positions(by_arity[arity], registry)
    else:
      # Different arities - use ArityDiscriminatingTupleCoder
      arity_to_coder: Dict[int, Coder] = {}
      for arity, arity_hints in by_arity.items():
        if len(arity_hints) == 1:
          arity_to_coder[arity] = registry.get_coder(arity_hints[0])
        else:
          arity_to_coder[arity] = _merge_tuple_positions(arity_hints, registry)
      return ArityDiscriminatingTupleCoder(arity_to_coder)

  if outer_type == dict:
    # Dict[K1, V1] | Dict[K2, V2] -> MapCoder(Union[K1, K2], Union[V1, V2])
    key_types = []
    value_types = []
    for hint in hints:
      if isinstance(hint, typehints.DictConstraint):
        key_types.append(hint.key_type)
        value_types.append(hint.value_type)
      else:
        raise ValueError(f"Expected dict type hint, got {hint}")

    # Deduplicate and sort for deterministic ordering
    key_types = list(dict.fromkeys(key_types))
    key_types.sort(key=_type_sort_key)
    value_types = list(dict.fromkeys(value_types))
    value_types.sort(key=_type_sort_key)
    key_coder = (
        registry.get_coder(key_types[0]) if len(key_types) == 1 else
        build_smart_union_coder(key_types, registry))
    value_coder = (
        registry.get_coder(value_types[0]) if len(value_types) == 1 else
        build_smart_union_coder(value_types, registry))
    return MapCoder(key_coder, value_coder)

  # For classes - can't merge, need direct discrimination
  # This shouldn't happen if _get_outer_type returns the class type
  raise ValueError(f"Cannot merge types with outer type {outer_type}: {hints}")


def _merge_tuple_positions(tuple_hints: List[Any], registry) -> Coder:
  """Merge tuple hints with the same arity by merging each position.

  Example:
    Tuple[int, str], Tuple[float, bytes]
    -> TupleCoder[SmartUnionCoder[int, float], SmartUnionCoder[str, bytes]]
  """
  if not tuple_hints:
    raise ValueError("Cannot merge empty tuple hints")

  arity = _get_tuple_arity(tuple_hints[0])
  if arity is None:
    raise ValueError(f"Expected tuple type hint, got {tuple_hints[0]}")

  # Collect types at each position
  position_types: List[List[Any]] = [[] for _ in range(arity)]
  for hint in tuple_hints:
    if not isinstance(hint, typehints.TupleConstraint):
      raise ValueError(f"Expected TupleConstraint, got {hint}")
    if len(hint.tuple_types) != arity:
      raise ValueError(
          f"Arity mismatch: expected {arity}, got {len(hint.tuple_types)}")
    for i, t in enumerate(hint.tuple_types):
      position_types[i].append(t)

  # Build coder for each position
  position_coders = []
  for types in position_types:
    # Deduplicate and sort for deterministic ordering
    unique_types = list(dict.fromkeys(types))
    unique_types.sort(key=_type_sort_key)
    if len(unique_types) == 1:
      position_coders.append(registry.get_coder(unique_types[0]))
    else:
      position_coders.append(build_smart_union_coder(unique_types, registry))

  return TupleCoder(position_coders)


# Register the factory for UnionConstraint type hints
# This makes Union[A, B, ...] automatically use _OrderedUnionCoder
# Import here to avoid circular imports during module initialization
from apache_beam.coders import typecoders  # pylint: disable=wrong-import-position

typecoders.registry.register_coder(
    typehints.UnionConstraint, _SmartUnionCoderFactory)
