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

  def test_to_dict(self):
    self.assertEqual(FieldRef('x').to_dict(), {'type': 'field', 'name': 'x'})

  def test_from_dict(self):
    expr = Expr.from_dict({'type': 'field', 'name': 'x'})
    self.assertEqual(expr, FieldRef('x'))


class LiteralTest(unittest.TestCase):
  def test_int(self):
    self.assertEqual(Literal(5).evaluate({}), 5)

  def test_float(self):
    self.assertAlmostEqual(Literal(3.14).evaluate({}), 3.14)

  def test_string(self):
    self.assertEqual(Literal('hello').evaluate({}), 'hello')

  def test_to_dict(self):
    self.assertEqual(Literal(5).to_dict(), {'type': 'literal', 'value': 5})

  def test_from_dict(self):
    expr = Expr.from_dict({'type': 'literal', 'value': 42})
    self.assertEqual(expr, Literal(42))


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

  def test_to_dict(self):
    expr = Add(FieldRef('a'), Literal(1))
    self.assertEqual(
        expr.to_dict(),
        {
            'type': 'bin_op',
            'op': '+',
            'left': {
                'type': 'field', 'name': 'a'
            },
            'right': {
                'type': 'literal', 'value': 1
            },
        })

  def test_from_dict(self):
    d = {
        'type': 'bin_op',
        'op': '/',
        'left': {
            'type': 'field', 'name': 'a'
        },
        'right': {
            'type': 'field', 'name': 'b'
        },
    }
    expr = Expr.from_dict(d)
    self.assertEqual(expr, Div(FieldRef('a'), FieldRef('b')))


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

  def test_to_dict(self):
    expr = Eq(FieldRef('x'), Literal(1))
    self.assertEqual(
        expr.to_dict(),
        {
            'type': 'compare',
            'op': '==',
            'left': {
                'type': 'field', 'name': 'x'
            },
            'right': {
                'type': 'literal', 'value': 1
            },
        })

  def test_from_dict(self):
    d = {
        'type': 'compare',
        'op': '==',
        'left': {
            'type': 'field', 'name': 'status'
        },
        'right': {
            'type': 'literal', 'value': 'success'
        },
    }
    expr = Expr.from_dict(d)
    self.assertEqual(expr, Eq(FieldRef('status'), Literal('success')))


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

  def test_to_dict(self):
    expr = IfExpr(Eq(FieldRef('x'), Literal(1)), Literal(10), Literal(0))
    self.assertEqual(
        expr.to_dict(),
        {
            'type': 'if',
            'condition': {
                'type': 'compare',
                'op': '==',
                'left': {
                    'type': 'field', 'name': 'x'
                },
                'right': {
                    'type': 'literal', 'value': 1
                }
            },
            'true_value': {
                'type': 'literal', 'value': 10
            },
            'false_value': {
                'type': 'literal', 'value': 0
            },
        })

  def test_from_dict(self):
    d = {
        'type': 'if',
        'condition': {
            'type': 'compare',
            'op': '==',
            'left': {
                'type': 'field', 'name': 'status'
            },
            'right': {
                'type': 'literal', 'value': 'success'
            }
        },
        'true_value': {
            'type': 'literal', 'value': 1
        },
        'false_value': {
            'type': 'literal', 'value': 0
        },
    }
    expr = Expr.from_dict(d)
    expected = IfExpr(
        Eq(FieldRef('status'), Literal('success')), Literal(1), Literal(0))
    self.assertEqual(expr, expected)


class NegateTest(unittest.TestCase):
  def test_negate(self):
    expr = Negate(FieldRef('a'))
    self.assertEqual(expr.evaluate({'a': 5}), -5)

  def test_to_from_dict(self):
    expr = Negate(Literal(3))
    d = expr.to_dict()
    self.assertEqual(
        d, {
            'type': 'negate', 'operand': {
                'type': 'literal', 'value': 3
            }
        })
    self.assertEqual(Expr.from_dict(d), expr)


class RoundTripTest(unittest.TestCase):
  """Test that complex expressions survive to_dict/from_dict round-trips."""
  def test_cuj2_ratio(self):
    expr = Div(FieldRef('clicks'), FieldRef('impressions'))
    self.assertEqual(Expr.from_dict(expr.to_dict()), expr)

  def test_cuj3_derived_field(self):
    expr = IfExpr(
        Eq(FieldRef('status'), Literal('success')), Literal(1), Literal(0))
    self.assertEqual(Expr.from_dict(expr.to_dict()), expr)

  def test_complex_nested(self):
    # (a + b) / IF(c > 0, c, 1)
    expr = Div(
        Add(FieldRef('a'), FieldRef('b')),
        IfExpr(Gt(FieldRef('c'), Literal(0)), FieldRef('c'), Literal(1)))
    roundtripped = Expr.from_dict(expr.to_dict())
    self.assertEqual(roundtripped, expr)
    # Also verify evaluation matches
    ctx = {'a': 10, 'b': 20, 'c': 5}
    self.assertEqual(expr.evaluate(ctx), roundtripped.evaluate(ctx))


class FromDictErrorTest(unittest.TestCase):
  def test_not_a_dict(self):
    with self.assertRaises(ValueError):
      Expr.from_dict("not a dict")

  def test_missing_type(self):
    with self.assertRaises(ValueError):
      Expr.from_dict({'name': 'x'})

  def test_unknown_type(self):
    with self.assertRaises(ValueError):
      Expr.from_dict({'type': 'unknown_node'})


class MetricPatternTest(unittest.TestCase):
  """Tests for the specific expression patterns used in the three CUJs."""
  def test_cuj1_no_expression_needed(self):
    # CUJ 1 uses a single measure, no metric_expr
    pass

  def test_cuj2_ratio(self):
    expr = Div(FieldRef('clicks'), FieldRef('impressions'))
    self.assertAlmostEqual(
        expr.evaluate({
            'clicks': 250.0, 'impressions': 10000.0
        }), 0.025)

  def test_cuj3_flag_derivation(self):
    expr = IfExpr(
        Eq(FieldRef('status'), Literal('success')), Literal(1), Literal(0))
    self.assertEqual(expr.evaluate({'status': 'success'}), 1)
    self.assertEqual(expr.evaluate({'status': 'error'}), 0)

  def test_cuj3_ratio(self):
    expr = Div(FieldRef('successes'), FieldRef('total'))
    self.assertAlmostEqual(
        expr.evaluate({
            'successes': 920.0, 'total': 1000.0
        }), 0.92)


if __name__ == '__main__':
  unittest.main()
