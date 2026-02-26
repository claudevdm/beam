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

import unittest

from apache_beam.ml.anomaly.safe_eval import Add
from apache_beam.ml.anomaly.safe_eval import BinOp
from apache_beam.ml.anomaly.safe_eval import Compare
from apache_beam.ml.anomaly.safe_eval import Div
from apache_beam.ml.anomaly.safe_eval import Eq
from apache_beam.ml.anomaly.safe_eval import Expr
from apache_beam.ml.anomaly.safe_eval import FieldRef
from apache_beam.ml.anomaly.safe_eval import Gt
from apache_beam.ml.anomaly.safe_eval import IfExpr
from apache_beam.ml.anomaly.safe_eval import Literal
from apache_beam.ml.anomaly.safe_eval import Mul
from apache_beam.ml.anomaly.safe_eval import Negate
from apache_beam.ml.anomaly.safe_eval import Sub


class FieldRefTest(unittest.TestCase):
  def test_evaluate(self):
    expr = FieldRef('amount')
    self.assertEqual(expr.evaluate({'amount': 42.0}), 42.0)

  def test_missing_field(self):
    expr = FieldRef('missing')
    with self.assertRaises(KeyError):
      expr.evaluate({'amount': 42.0})

  def test_field_refs(self):
    self.assertEqual(FieldRef('x').field_refs(), {'x'})

  def test_str(self):
    self.assertEqual(str(FieldRef('clicks')), 'clicks')


class LiteralTest(unittest.TestCase):
  def test_int(self):
    self.assertEqual(Literal(5).evaluate({}), 5)

  def test_float(self):
    self.assertAlmostEqual(Literal(3.14).evaluate({}), 3.14)

  def test_string(self):
    self.assertEqual(Literal('hello').evaluate({}), 'hello')

  def test_field_refs(self):
    self.assertEqual(Literal(5).field_refs(), set())

  def test_str_int(self):
    self.assertEqual(str(Literal(42)), '42')

  def test_str_string(self):
    self.assertEqual(str(Literal('hello')), "'hello'")


class BinOpTest(unittest.TestCase):
  def test_add(self):
    expr = Add(FieldRef('a'), FieldRef('b'))
    self.assertEqual(expr.evaluate({'a': 3, 'b': 4}), 7)

  def test_sub(self):
    expr = Sub(FieldRef('a'), Literal(3))
    self.assertEqual(expr.evaluate({'a': 10}), 7)

  def test_mul(self):
    expr = Mul(FieldRef('a'), FieldRef('b'))
    self.assertEqual(expr.evaluate({'a': 5, 'b': 6}), 30)

  def test_div(self):
    expr = Div(FieldRef('clicks'), FieldRef('impressions'))
    self.assertAlmostEqual(
        expr.evaluate({
            'clicks': 50, 'impressions': 1000
        }), 0.05)

  def test_floor_div(self):
    expr = BinOp('//', FieldRef('a'), FieldRef('b'))
    self.assertEqual(expr.evaluate({'a': 7, 'b': 2}), 3)

  def test_mod(self):
    expr = BinOp('%', FieldRef('a'), Literal(3))
    self.assertEqual(expr.evaluate({'a': 7}), 1)

  def test_precedence_via_nesting(self):
    # a + (b * c) = 1 + 6 = 7
    expr = Add(FieldRef('a'), Mul(FieldRef('b'), FieldRef('c')))
    self.assertEqual(expr.evaluate({'a': 1, 'b': 2, 'c': 3}), 7)

  def test_nested_different(self):
    # (a + b) * c = 3 * 3 = 9
    expr = Mul(Add(FieldRef('a'), FieldRef('b')), FieldRef('c'))
    self.assertEqual(expr.evaluate({'a': 1, 'b': 2, 'c': 3}), 9)

  def test_division_by_zero(self):
    expr = Div(FieldRef('a'), Literal(0))
    with self.assertRaises(ZeroDivisionError):
      expr.evaluate({'a': 1})

  def test_invalid_operator(self):
    with self.assertRaises(ValueError):
      BinOp('**', FieldRef('a'), FieldRef('b'))

  def test_field_refs(self):
    expr = Div(FieldRef('clicks'), FieldRef('impressions'))
    self.assertEqual(expr.field_refs(), {'clicks', 'impressions'})

  def test_str(self):
    expr = Div(FieldRef('clicks'), FieldRef('impressions'))
    self.assertEqual(str(expr), '(clicks / impressions)')


class CompareTest(unittest.TestCase):
  def test_eq_true(self):
    expr = Eq(FieldRef('a'), Literal(5))
    self.assertTrue(expr.evaluate({'a': 5}))

  def test_eq_false(self):
    expr = Eq(FieldRef('a'), Literal(5))
    self.assertFalse(expr.evaluate({'a': 3}))

  def test_string_eq(self):
    expr = Eq(FieldRef('status'), Literal('success'))
    self.assertTrue(expr.evaluate({'status': 'success'}))
    self.assertFalse(expr.evaluate({'status': 'failure'}))

  def test_gt(self):
    expr = Gt(FieldRef('a'), Literal(5))
    self.assertTrue(expr.evaluate({'a': 7}))
    self.assertFalse(expr.evaluate({'a': 3}))

  def test_invalid_operator(self):
    with self.assertRaises(ValueError):
      Compare('===', FieldRef('a'), Literal(1))

  def test_field_refs(self):
    expr = Eq(FieldRef('status'), Literal('success'))
    self.assertEqual(expr.field_refs(), {'status'})


class IfExprTest(unittest.TestCase):
  def test_true_branch(self):
    expr = IfExpr(
        Eq(FieldRef('status'), Literal('success')), Literal(1), Literal(0))
    self.assertEqual(expr.evaluate({'status': 'success'}), 1)

  def test_false_branch(self):
    expr = IfExpr(
        Eq(FieldRef('status'), Literal('success')), Literal(1), Literal(0))
    self.assertEqual(expr.evaluate({'status': 'failure'}), 0)

  def test_with_arithmetic(self):
    expr = IfExpr(
        Gt(FieldRef('a'), Literal(10)),
        Mul(FieldRef('a'), Literal(2)),
        Add(FieldRef('a'), Literal(1)))
    self.assertEqual(expr.evaluate({'a': 15}), 30)
    self.assertEqual(expr.evaluate({'a': 5}), 6)

  def test_nested(self):
    expr = IfExpr(
        Gt(FieldRef('a'), Literal(10)),
        IfExpr(Gt(FieldRef('b'), Literal(5)), Literal(3), Literal(2)),
        Literal(1))
    self.assertEqual(expr.evaluate({'a': 15, 'b': 7}), 3)
    self.assertEqual(expr.evaluate({'a': 15, 'b': 3}), 2)
    self.assertEqual(expr.evaluate({'a': 5, 'b': 7}), 1)

  def test_field_refs(self):
    expr = IfExpr(
        Eq(FieldRef('status'), Literal('success')),
        FieldRef('amount'),
        Literal(0))
    self.assertEqual(expr.field_refs(), {'status', 'amount'})


class NegateTest(unittest.TestCase):
  def test_negate(self):
    expr = Negate(FieldRef('a'))
    self.assertEqual(expr.evaluate({'a': 5}), -5)

  def test_field_refs(self):
    self.assertEqual(Negate(FieldRef('a')).field_refs(), {'a'})


class FromStringTest(unittest.TestCase):
  """Tests for Expr.from_string() — Python expression parsing."""
  def test_field_ref(self):
    expr = Expr.from_string("clicks")
    self.assertEqual(expr, FieldRef('clicks'))

  def test_int_literal(self):
    expr = Expr.from_string("42")
    self.assertEqual(expr, Literal(42))

  def test_float_literal(self):
    expr = Expr.from_string("3.14")
    self.assertEqual(expr, Literal(3.14))

  def test_string_literal(self):
    expr = Expr.from_string("'success'")
    self.assertEqual(expr, Literal('success'))

  def test_division(self):
    expr = Expr.from_string("clicks / impressions")
    self.assertEqual(expr, Div(FieldRef('clicks'), FieldRef('impressions')))

  def test_addition(self):
    expr = Expr.from_string("a + b")
    self.assertEqual(expr, Add(FieldRef('a'), FieldRef('b')))

  def test_subtraction(self):
    expr = Expr.from_string("a - 3")
    self.assertEqual(expr, Sub(FieldRef('a'), Literal(3)))

  def test_multiplication(self):
    expr = Expr.from_string("a * b")
    self.assertEqual(expr, Mul(FieldRef('a'), FieldRef('b')))

  def test_floor_div(self):
    expr = Expr.from_string("a // b")
    self.assertEqual(expr, BinOp('//', FieldRef('a'), FieldRef('b')))

  def test_modulo(self):
    expr = Expr.from_string("a % 3")
    self.assertEqual(expr, BinOp('%', FieldRef('a'), Literal(3)))

  def test_comparison_eq(self):
    expr = Expr.from_string("status == 'success'")
    self.assertEqual(expr, Eq(FieldRef('status'), Literal('success')))

  def test_comparison_gt(self):
    expr = Expr.from_string("a > 5")
    self.assertEqual(expr, Gt(FieldRef('a'), Literal(5)))

  def test_negation(self):
    expr = Expr.from_string("-a")
    self.assertEqual(expr, Negate(FieldRef('a')))

  def test_if_else(self):
    expr = Expr.from_string("1 if status == 'success' else 0")
    expected = IfExpr(
        Eq(FieldRef('status'), Literal('success')), Literal(1), Literal(0))
    self.assertEqual(expr, expected)

  def test_parenthesized(self):
    expr = Expr.from_string("(a + b) / c")
    expected = Div(Add(FieldRef('a'), FieldRef('b')), FieldRef('c'))
    self.assertEqual(expr, expected)

  def test_precedence(self):
    # Python precedence: a + b * c = a + (b * c)
    expr = Expr.from_string("a + b * c")
    expected = Add(FieldRef('a'), Mul(FieldRef('b'), FieldRef('c')))
    self.assertEqual(expr, expected)

  def test_nested_if(self):
    expr = Expr.from_string("3 if a > 10 else (2 if b > 5 else 1)")
    self.assertEqual(expr.evaluate({'a': 15, 'b': 7}), 3)
    self.assertEqual(expr.evaluate({'a': 15, 'b': 3}), 3)
    self.assertEqual(expr.evaluate({'a': 5, 'b': 7}), 2)
    self.assertEqual(expr.evaluate({'a': 5, 'b': 3}), 1)

  def test_syntax_error(self):
    with self.assertRaises(SyntaxError):
      Expr.from_string("a +")

  def test_unsupported_function_call(self):
    with self.assertRaises(ValueError):
      Expr.from_string("abs(a)")

  def test_unsupported_attribute(self):
    with self.assertRaises(ValueError):
      Expr.from_string("a.b")

  def test_unsupported_power(self):
    with self.assertRaises(ValueError):
      Expr.from_string("a ** 2")

  def test_chained_comparison(self):
    with self.assertRaises(ValueError):
      Expr.from_string("a < b < c")


class RoundTripTest(unittest.TestCase):
  """Test that complex expressions survive str/from_string round-trips."""
  def test_cuj2_ratio(self):
    expr = Expr.from_string("clicks / impressions")
    roundtripped = Expr.from_string(str(expr))
    self.assertEqual(roundtripped, expr)

  def test_cuj3_derived_field(self):
    expr = Expr.from_string("1 if status == 'success' else 0")
    roundtripped = Expr.from_string(str(expr))
    self.assertEqual(roundtripped, expr)

  def test_complex_nested(self):
    expr = Expr.from_string("(a + b) / (1 if c > 0 else 1)")
    roundtripped = Expr.from_string(str(expr))
    self.assertEqual(roundtripped, expr)
    ctx = {'a': 10, 'b': 20, 'c': 5}
    self.assertEqual(expr.evaluate(ctx), roundtripped.evaluate(ctx))

  def test_negation_roundtrip(self):
    expr = Expr.from_string("-a")
    roundtripped = Expr.from_string(str(expr))
    self.assertEqual(roundtripped, expr)


class MetricPatternTest(unittest.TestCase):
  """Tests for the specific expression patterns used in the three CUJs."""
  def test_cuj1_no_expression_needed(self):
    # CUJ 1 uses a single measure, no metric_expr
    pass

  def test_cuj2_ratio(self):
    expr = Expr.from_string("clicks / impressions")
    self.assertAlmostEqual(
        expr.evaluate({
            'clicks': 250.0, 'impressions': 10000.0
        }), 0.025)

  def test_cuj3_flag_derivation(self):
    expr = Expr.from_string("1 if status == 'success' else 0")
    self.assertEqual(expr.evaluate({'status': 'success'}), 1)
    self.assertEqual(expr.evaluate({'status': 'error'}), 0)

  def test_cuj3_ratio(self):
    expr = Expr.from_string("successes / total")
    self.assertAlmostEqual(
        expr.evaluate({
            'successes': 920.0, 'total': 1000.0
        }), 0.92)


if __name__ == '__main__':
  unittest.main()
