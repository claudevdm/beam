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

"""Structured expression AST for safe metric computation.

Provides expression tree nodes that can be constructed programmatically,
serialized to/from JSON dicts, and evaluated against a field context.

Example usage::

  from apache_beam.ml.anomaly.safe_eval import (
      FieldRef, Literal, Div, Add, Eq, IfExpr)

  # clicks / impressions
  expr = Div(FieldRef('clicks'), FieldRef('impressions'))
  result = expr.evaluate({'clicks': 50, 'impressions': 1000})
  # result = 0.05

  # IF(status == 'success', 1, 0)
  expr = IfExpr(
      condition=Eq(FieldRef('status'), Literal('success')),
      true_value=Literal(1),
      false_value=Literal(0))
  result = expr.evaluate({'status': 'success'})
  # result = 1
"""

import operator

# --- Operator registries ---

_BINARY_OPS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
    '//': operator.floordiv,
    '%': operator.mod,
}

_COMPARE_OPS = {
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge,
}

# --- Base class ---


class Expr:
  """Base class for all expression nodes."""
  def evaluate(self, context):
    """Evaluate this expression against a dict of field values.

    Args:
      context: Dict mapping field names to values.

    Returns:
      The result of evaluating the expression.
    """
    raise NotImplementedError

  def to_dict(self):
    """Serialize to a plain dict suitable for JSON."""
    raise NotImplementedError

  @staticmethod
  def from_dict(d):
    """Deserialize an expression from a plain dict.

    Args:
      d: Dict with a ``'type'`` key indicating the node type.

    Returns:
      An ``Expr`` instance.
    """
    if not isinstance(d, dict):
      raise ValueError(f"Expected dict, got {type(d).__name__}: {d}")

    node_type = d.get('type')
    if node_type is None:
      raise ValueError(f"Expression dict missing 'type' key: {d}")

    if node_type == 'field':
      return FieldRef(d['name'])
    elif node_type == 'literal':
      return Literal(d['value'])
    elif node_type == 'bin_op':
      return BinOp(
          op=d['op'],
          left=Expr.from_dict(d['left']),
          right=Expr.from_dict(d['right']))
    elif node_type == 'compare':
      return Compare(
          op=d['op'],
          left=Expr.from_dict(d['left']),
          right=Expr.from_dict(d['right']))
    elif node_type == 'if':
      return IfExpr(
          condition=Expr.from_dict(d['condition']),
          true_value=Expr.from_dict(d['true_value']),
          false_value=Expr.from_dict(d['false_value']))
    elif node_type == 'negate':
      return Negate(Expr.from_dict(d['operand']))
    else:
      raise ValueError(f"Unknown expression type: {node_type}")


# --- Leaf nodes ---


class FieldRef(Expr):
  """Reference to a named field in the evaluation context.

  Args:
    name: The field name.
  """
  def __init__(self, name):
    self.name = name

  def evaluate(self, context):
    if self.name not in context:
      raise KeyError(
          f"Field '{self.name}' not found. "
          f"Available: {list(context.keys())}")
    return context[self.name]

  def to_dict(self):
    return {'type': 'field', 'name': self.name}

  def __repr__(self):
    return f"FieldRef({self.name!r})"

  def __eq__(self, other):
    return isinstance(other, FieldRef) and self.name == other.name


class Literal(Expr):
  """A constant value (number or string).

  Args:
    value: The constant value.
  """
  def __init__(self, value):
    self.value = value

  def evaluate(self, context):
    return self.value

  def to_dict(self):
    return {'type': 'literal', 'value': self.value}

  def __repr__(self):
    return f"Literal({self.value!r})"

  def __eq__(self, other):
    return isinstance(other, Literal) and self.value == other.value


# --- Unary operations ---


class Negate(Expr):
  """Unary negation.

  Args:
    operand: The expression to negate.
  """
  def __init__(self, operand):
    self.operand = operand

  def evaluate(self, context):
    return -self.operand.evaluate(context)

  def to_dict(self):
    return {'type': 'negate', 'operand': self.operand.to_dict()}

  def __repr__(self):
    return f"Negate({self.operand!r})"

  def __eq__(self, other):
    return isinstance(other, Negate) and self.operand == other.operand


# --- Binary operations ---


class BinOp(Expr):
  """Binary arithmetic operation.

  Args:
    op: Operator string: ``'+', '-', '*', '/', '//', '%'``.
    left: Left operand expression.
    right: Right operand expression.
  """
  def __init__(self, op, left, right):
    if op not in _BINARY_OPS:
      raise ValueError(
          f"Unknown binary operator '{op}'. "
          f"Supported: {list(_BINARY_OPS.keys())}")
    self.op = op
    self.left = left
    self.right = right

  def evaluate(self, context):
    left_val = self.left.evaluate(context)
    right_val = self.right.evaluate(context)
    return _BINARY_OPS[self.op](left_val, right_val)

  def to_dict(self):
    return {
        'type': 'bin_op',
        'op': self.op,
        'left': self.left.to_dict(),
        'right': self.right.to_dict(),
    }

  def __repr__(self):
    return f"BinOp({self.op!r}, {self.left!r}, {self.right!r})"

  def __eq__(self, other):
    return (
        isinstance(other, BinOp) and self.op == other.op and
        self.left == other.left and self.right == other.right)


class Compare(Expr):
  """Comparison operation.

  Args:
    op: Operator string: ``'==', '!=', '<', '<=', '>', '>='``.
    left: Left operand expression.
    right: Right operand expression.
  """
  def __init__(self, op, left, right):
    if op not in _COMPARE_OPS:
      raise ValueError(
          f"Unknown comparison operator '{op}'. "
          f"Supported: {list(_COMPARE_OPS.keys())}")
    self.op = op
    self.left = left
    self.right = right

  def evaluate(self, context):
    left_val = self.left.evaluate(context)
    right_val = self.right.evaluate(context)
    return _COMPARE_OPS[self.op](left_val, right_val)

  def to_dict(self):
    return {
        'type': 'compare',
        'op': self.op,
        'left': self.left.to_dict(),
        'right': self.right.to_dict(),
    }

  def __repr__(self):
    return f"Compare({self.op!r}, {self.left!r}, {self.right!r})"

  def __eq__(self, other):
    return (
        isinstance(other, Compare) and self.op == other.op and
        self.left == other.left and self.right == other.right)


# --- Conditional ---


class IfExpr(Expr):
  """Conditional expression: if condition then true_value else false_value.

  Args:
    condition: An expression that evaluates to a boolean.
    true_value: Expression returned when condition is truthy.
    false_value: Expression returned when condition is falsy.
  """
  def __init__(self, condition, true_value, false_value):
    self.condition = condition
    self.true_value = true_value
    self.false_value = false_value

  def evaluate(self, context):
    if self.condition.evaluate(context):
      return self.true_value.evaluate(context)
    else:
      return self.false_value.evaluate(context)

  def to_dict(self):
    return {
        'type': 'if',
        'condition': self.condition.to_dict(),
        'true_value': self.true_value.to_dict(),
        'false_value': self.false_value.to_dict(),
    }

  def __repr__(self):
    return (
        f"IfExpr({self.condition!r}, "
        f"{self.true_value!r}, {self.false_value!r})")

  def __eq__(self, other):
    return (
        isinstance(other, IfExpr) and self.condition == other.condition and
        self.true_value == other.true_value and
        self.false_value == other.false_value)


# --- Convenience constructors ---


def Add(left, right):
  """Create an addition expression."""
  return BinOp('+', left, right)


def Sub(left, right):
  """Create a subtraction expression."""
  return BinOp('-', left, right)


def Mul(left, right):
  """Create a multiplication expression."""
  return BinOp('*', left, right)


def Div(left, right):
  """Create a division expression."""
  return BinOp('/', left, right)


def Mod(left, right):
  """Create a modulo expression."""
  return BinOp('%', left, right)


def Eq(left, right):
  """Create an equality comparison."""
  return Compare('==', left, right)


def Neq(left, right):
  """Create a not-equal comparison."""
  return Compare('!=', left, right)


def Gt(left, right):
  """Create a greater-than comparison."""
  return Compare('>', left, right)


def Lt(left, right):
  """Create a less-than comparison."""
  return Compare('<', left, right)


def Gte(left, right):
  """Create a greater-than-or-equal comparison."""
  return Compare('>=', left, right)


def Lte(left, right):
  """Create a less-than-or-equal comparison."""
  return Compare('<=', left, right)
