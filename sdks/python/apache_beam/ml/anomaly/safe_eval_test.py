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

import pickle
import unittest

from apache_beam.ml.anomaly.safe_eval import Expr


class ExprTest(unittest.TestCase):
  """Core tests for the Expr compiled expression."""

  # --- field references ---

  def test_field_ref(self):
    expr = Expr("amount")
    self.assertEqual(expr({'amount': 42.0}), 42.0)

  def test_missing_field(self):
    expr = Expr("missing")
    with self.assertRaises(NameError):
      expr({'amount': 42.0})

  def test_field_refs(self):
    self.assertEqual(Expr("x").field_refs(), {'x'})

  def test_field_refs_multiple(self):
    self.assertEqual(
        Expr("clicks / impressions").field_refs(), {'clicks', 'impressions'})

  # --- literals ---

  def test_int_literal(self):
    self.assertEqual(Expr("42")({}), 42)

  def test_float_literal(self):
    self.assertAlmostEqual(Expr("3.14")({}), 3.14)

  def test_string_literal(self):
    self.assertEqual(Expr("'hello'")({}), 'hello')

  # --- arithmetic ---

  def test_add(self):
    self.assertEqual(Expr("a + b")({'a': 3, 'b': 4}), 7)

  def test_sub(self):
    self.assertEqual(Expr("a - 3")({'a': 10}), 7)

  def test_mul(self):
    self.assertEqual(Expr("a * b")({'a': 5, 'b': 6}), 30)

  def test_div(self):
    self.assertAlmostEqual(
        Expr("clicks / impressions")({
            'clicks': 50, 'impressions': 1000
        }), 0.05)

  def test_floor_div(self):
    self.assertEqual(Expr("a // b")({'a': 7, 'b': 2}), 3)

  def test_mod(self):
    self.assertEqual(Expr("a % 3")({'a': 7}), 1)

  def test_division_by_zero(self):
    with self.assertRaises(ZeroDivisionError):
      Expr("a / 0")({'a': 1})

  def test_precedence(self):
    # Python precedence: a + b * c = a + (b * c)
    self.assertEqual(Expr("a + b * c")({'a': 1, 'b': 2, 'c': 3}), 7)

  def test_parenthesized(self):
    self.assertEqual(Expr("(a + b) / c")({'a': 6, 'b': 4, 'c': 2}), 5.0)

  # --- comparisons ---

  def test_eq_true(self):
    self.assertTrue(Expr("a == 5")({'a': 5}))

  def test_eq_false(self):
    self.assertFalse(Expr("a == 5")({'a': 3}))

  def test_string_eq(self):
    self.assertTrue(Expr("status == 'success'")({'status': 'success'}))
    self.assertFalse(Expr("status == 'success'")({'status': 'failure'}))

  def test_gt(self):
    self.assertTrue(Expr("a > 5")({'a': 7}))
    self.assertFalse(Expr("a > 5")({'a': 3}))

  # --- negation ---

  def test_negate(self):
    self.assertEqual(Expr("-a")({'a': 5}), -5)

  # --- if/else ---

  def test_if_else_true(self):
    expr = Expr("1 if status == 'success' else 0")
    self.assertEqual(expr({'status': 'success'}), 1)

  def test_if_else_false(self):
    expr = Expr("1 if status == 'success' else 0")
    self.assertEqual(expr({'status': 'failure'}), 0)

  def test_nested_if(self):
    expr = Expr("3 if a > 10 else (2 if b > 5 else 1)")
    self.assertEqual(expr({'a': 15, 'b': 7}), 3)
    self.assertEqual(expr({'a': 5, 'b': 7}), 2)
    self.assertEqual(expr({'a': 5, 'b': 3}), 1)


class ValidationTest(unittest.TestCase):
  """Tests for AST validation — reject unsafe constructs."""
  def test_syntax_error(self):
    with self.assertRaises(SyntaxError):
      Expr("a +")

  def test_unsupported_function_call(self):
    with self.assertRaises(ValueError):
      Expr("abs(a)")

  def test_unsupported_attribute(self):
    with self.assertRaises(ValueError):
      Expr("a.b")

  def test_unsupported_power(self):
    with self.assertRaises(ValueError):
      Expr("a ** 2")

  def test_chained_comparison(self):
    with self.assertRaises(ValueError):
      Expr("a < b < c")

  def test_unsupported_literal_type(self):
    with self.assertRaises(ValueError):
      Expr("None")

  def test_no_builtins_access(self):
    expr = Expr("a")
    with self.assertRaises(NameError):
      expr({})  # 'a' not in context


class SerializationTest(unittest.TestCase):
  """Tests for str(), from_string(), pickle round-trips."""
  def test_str(self):
    self.assertEqual(str(Expr("clicks / impressions")), "clicks / impressions")

  def test_from_string(self):
    expr = Expr.from_string("clicks / impressions")
    self.assertAlmostEqual(expr({'clicks': 50, 'impressions': 1000}), 0.05)

  def test_equality(self):
    self.assertEqual(Expr("clicks / impressions"), Expr("clicks / impressions"))

  def test_inequality(self):
    self.assertNotEqual(Expr("a + b"), Expr("a - b"))

  def test_str_roundtrip(self):
    text = "clicks / impressions"
    expr = Expr(text)
    roundtripped = Expr(str(expr))
    self.assertEqual(roundtripped, expr)

  def test_pickle_roundtrip(self):
    expr = Expr("1 if status == 'success' else 0")
    restored = pickle.loads(pickle.dumps(expr))
    self.assertEqual(restored({'status': 'success'}), 1)
    self.assertEqual(restored({'status': 'error'}), 0)
    self.assertEqual(restored.field_refs(), {'status'})

  def test_hash(self):
    # Expr is hashable (can be used in sets/dicts)
    s = {Expr("a + b"), Expr("a + b"), Expr("a - b")}
    self.assertEqual(len(s), 2)


class MetricPatternTest(unittest.TestCase):
  """Tests for the specific expression patterns used in the three CUJs."""
  def test_cuj1_no_expression_needed(self):
    # CUJ 1 uses a single measure, no measure_combiner
    pass

  def test_cuj2_ratio(self):
    expr = Expr.from_string("clicks / impressions")
    self.assertAlmostEqual(
        expr({
            'clicks': 250.0, 'impressions': 10000.0
        }), 0.025)

  def test_cuj3_flag_derivation(self):
    expr = Expr.from_string("1 if status == 'success' else 0")
    self.assertEqual(expr({'status': 'success'}), 1)
    self.assertEqual(expr({'status': 'error'}), 0)

  def test_cuj3_ratio(self):
    expr = Expr.from_string("successes / total")
    self.assertAlmostEqual(expr({'successes': 920.0, 'total': 1000.0}), 0.92)


if __name__ == '__main__':
  unittest.main()
