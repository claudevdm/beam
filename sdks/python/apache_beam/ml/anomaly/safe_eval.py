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

Provides expression tree nodes that can be constructed programmatically
or parsed from Python expression strings, and evaluated against a field
context.

Example usage::

  from apache_beam.ml.anomaly.safe_eval import Expr, FieldRef, Literal, Div

  # Parse from string (preferred for JSON configs)
  expr = Expr.from_string("clicks / impressions")
  result = expr.evaluate({'clicks': 50, 'impressions': 1000})
  # result = 0.05

  # Parse conditional expression
  expr = Expr.from_string("1 if status == 'success' else 0")
  result = expr.evaluate({'status': 'success'})
  # result = 1

  # Programmatic construction (for Python code)
  expr = Div(FieldRef('clicks'), FieldRef('impressions'))
"""

import ast
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

# --- AST mapping for from_string ---

_AST_BINOP_MAP = {
    ast.Add: '+',
    ast.Sub: '-',
    ast.Mult: '*',
    ast.Div: '/',
    ast.FloorDiv: '//',
    ast.Mod: '%',
}

_AST_CMPOP_MAP = {
    ast.Eq: '==',
    ast.NotEq: '!=',
    ast.Lt: '<',
    ast.LtE: '<=',
    ast.Gt: '>',
    ast.GtE: '>=',
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

  def field_refs(self):
    """Return the set of field names referenced by this expression."""
    raise NotImplementedError

  @staticmethod
  def from_string(text):
    """Parse a Python expression string into an Expr tree.

    Supports field references (bare names), literals (numbers, strings),
    arithmetic (``+, -, *, /, //, %``), comparisons
    (``==, !=, <, <=, >, >=``), unary negation, and ``if/else``.

    Examples::

      Expr.from_string("clicks / impressions")
      Expr.from_string("1 if status == 'success' else 0")
      Expr.from_string("(a + b) / total")

    Args:
      text: A Python expression string.

    Returns:
      An ``Expr`` instance.

    Raises:
      ValueError: If the expression uses unsupported Python constructs.
      SyntaxError: If the string is not valid Python syntax.
    """
    tree = ast.parse(text, mode='eval')
    return _ast_to_expr(tree.body)


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

  def field_refs(self):
    return {self.name}

  def __str__(self):
    return self.name

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

  def field_refs(self):
    return set()

  def __str__(self):
    return repr(self.value)

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

  def field_refs(self):
    return self.operand.field_refs()

  def __str__(self):
    return f"(-{self.operand})"

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

  def field_refs(self):
    return self.left.field_refs() | self.right.field_refs()

  def __str__(self):
    return f"({self.left} {self.op} {self.right})"

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

  def field_refs(self):
    return self.left.field_refs() | self.right.field_refs()

  def __str__(self):
    return f"({self.left} {self.op} {self.right})"

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

  def field_refs(self):
    return (
        self.condition.field_refs() | self.true_value.field_refs()
        | self.false_value.field_refs())

  def __str__(self):
    return f"({self.true_value} if {self.condition} else {self.false_value})"

  def __repr__(self):
    return (
        f"IfExpr({self.condition!r}, "
        f"{self.true_value!r}, {self.false_value!r})")

  def __eq__(self, other):
    return (
        isinstance(other, IfExpr) and self.condition == other.condition and
        self.true_value == other.true_value and
        self.false_value == other.false_value)


# --- AST-to-Expr conversion ---


def _ast_to_expr(node):
  """Convert a Python AST node to an Expr tree."""
  if isinstance(node, ast.Name):
    return FieldRef(node.id)

  if isinstance(node, ast.Constant):
    if not isinstance(node.value, (int, float, str)):
      raise ValueError(
          f"Unsupported literal type: {type(node.value).__name__}. "
          f"Only int, float, and str literals are supported.")
    return Literal(node.value)

  if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
    return Negate(_ast_to_expr(node.operand))

  if isinstance(node, ast.BinOp):
    op_str = _AST_BINOP_MAP.get(type(node.op))
    if op_str is None:
      raise ValueError(
          f"Unsupported binary operator: {type(node.op).__name__}. "
          f"Supported: +, -, *, /, //, %")
    return BinOp(op_str, _ast_to_expr(node.left), _ast_to_expr(node.right))

  if isinstance(node, ast.Compare):
    if len(node.ops) != 1 or len(node.comparators) != 1:
      raise ValueError(
          "Chained comparisons not supported (e.g., a < b < c). "
          "Use (a < b) and separate expressions instead.")
    op_str = _AST_CMPOP_MAP.get(type(node.ops[0]))
    if op_str is None:
      raise ValueError(
          f"Unsupported comparison: {type(node.ops[0]).__name__}. "
          f"Supported: ==, !=, <, <=, >, >=")
    return Compare(
        op_str, _ast_to_expr(node.left), _ast_to_expr(node.comparators[0]))

  if isinstance(node, ast.IfExp):
    return IfExpr(
        _ast_to_expr(node.test),
        _ast_to_expr(node.body),
        _ast_to_expr(node.orelse))

  raise ValueError(
      f"Unsupported expression: {ast.dump(node)}. "
      f"Only field names, literals, arithmetic (+,-,*,/,//,%), "
      f"comparisons (==,!=,<,<=,>,>=), negation, and "
      f"if/else expressions are supported.")


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


def Eq(left, right):
  """Create an equality comparison."""
  return Compare('==', left, right)


def Gt(left, right):
  """Create a greater-than comparison."""
  return Compare('>', left, right)
