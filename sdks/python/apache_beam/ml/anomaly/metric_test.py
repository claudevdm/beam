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

import json
import unittest

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.transforms.window import TimestampedValue

from apache_beam.ml.anomaly.metric import AggOp
from apache_beam.ml.anomaly.metric import AggregationSpec
from apache_beam.ml.anomaly.metric import ComputeMetric
from apache_beam.ml.anomaly.metric import DerivedField
from apache_beam.ml.anomaly.metric import MeasureSpec
from apache_beam.ml.anomaly.metric import MetricSpec
from apache_beam.ml.anomaly.metric import WindowSpec
from apache_beam.ml.anomaly.metric import WindowType
from apache_beam.ml.anomaly.safe_eval import Div
from apache_beam.ml.anomaly.safe_eval import Eq
from apache_beam.ml.anomaly.safe_eval import Expr
from apache_beam.ml.anomaly.safe_eval import FieldRef
from apache_beam.ml.anomaly.safe_eval import IfExpr
from apache_beam.ml.anomaly.safe_eval import Literal

# ---- Spec construction and validation tests ----


class MetricSpecValidationTest(unittest.TestCase):
  def test_valid_single_measure(self):
    spec = MetricSpec(
        name='revenue',
        aggregation=AggregationSpec(
            measures=[MeasureSpec(field='amount', op=AggOp.SUM, alias='total')
                      ]),
        _run_init=True)
    self.assertEqual(spec.name, 'revenue')

  def test_multiple_measures_without_expr_raises(self):
    with self.assertRaises(ValueError):
      MetricSpec(
          name='test',
          aggregation=AggregationSpec(
              measures=[
                  MeasureSpec(field='a', op=AggOp.SUM, alias='sum_a'),
                  MeasureSpec(field='*', op=AggOp.COUNT, alias='count'),
              ]),
          _run_init=True)

  def test_no_measures_raises(self):
    with self.assertRaises(ValueError):
      MetricSpec(
          name='test', aggregation=AggregationSpec(measures=[]), _run_init=True)

  def test_sliding_without_period_raises(self):
    with self.assertRaises(ValueError):
      MetricSpec(
          name='test',
          aggregation=AggregationSpec(
              window=WindowSpec(type=WindowType.SLIDING, size_sec=3600),
              measures=[MeasureSpec(field='a', op=AggOp.SUM, alias='total')]),
          _run_init=True)

  def test_sliding_with_period_ok(self):
    spec = MetricSpec(
        name='test',
        aggregation=AggregationSpec(
            window=WindowSpec(
                type=WindowType.SLIDING, size_sec=3600, period_sec=60),
            measures=[MeasureSpec(field='a', op=AggOp.SUM, alias='total')]),
        _run_init=True)
    self.assertEqual(spec.aggregation.window.period_sec, 60)

  def test_metric_expr_must_be_expr(self):
    with self.assertRaises(TypeError):
      MetricSpec(
          name='test',
          aggregation=AggregationSpec(
              measures=[
                  MeasureSpec(field='a', op=AggOp.SUM, alias='x'),
                  MeasureSpec(field='*', op=AggOp.COUNT, alias='y'),
              ]),
          metric_expr="x / y",  # string, not Expr
          _run_init=True)


# ---- JSON round-trip tests ----


class MetricSpecSerializationTest(unittest.TestCase):
  def _round_trip(self, spec):
    d = spec.to_dict()
    json_str = json.dumps(d)
    d2 = json.loads(json_str)
    spec2 = MetricSpec.from_dict(d2)
    return spec2

  def test_cuj1_round_trip(self):
    spec = MetricSpec(
        name='revenue',
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_sec=3600),
            measures=[
                MeasureSpec(
                    field='transaction_amount', op=AggOp.SUM, alias='revenue')
            ]),
        _run_init=True)
    spec2 = self._round_trip(spec)
    self.assertEqual(spec2.name, 'revenue')
    self.assertEqual(spec2.aggregation.window.size_sec, 3600)
    self.assertEqual(len(spec2.aggregation.measures), 1)
    self.assertEqual(spec2.aggregation.measures[0].op, AggOp.SUM)

  def test_cuj2_round_trip(self):
    spec = MetricSpec(
        name='ctr',
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_sec=86400),
            group_by=['campaign_type', 'user_segment', 'browser_version'],
            measures=[
                MeasureSpec(field='is_click', op=AggOp.SUM, alias='clicks'),
                MeasureSpec(field='*', op=AggOp.COUNT, alias='impressions'),
            ]),
        metric_expr=Div(FieldRef('clicks'), FieldRef('impressions')),
        _run_init=True)
    spec2 = self._round_trip(spec)
    self.assertEqual(spec2.name, 'ctr')
    self.assertEqual(
        spec2.aggregation.group_by,
        ['campaign_type', 'user_segment', 'browser_version'])
    self.assertEqual(len(spec2.aggregation.measures), 2)
    # Verify expression round-tripped
    self.assertAlmostEqual(
        spec2.metric_expr.evaluate({
            'clicks': 50, 'impressions': 1000
        }), 0.05)

  def test_cuj3_round_trip(self):
    spec = MetricSpec(
        name='success_rate',
        derived_fields=[
            DerivedField(
                name='is_success',
                expression=IfExpr(
                    Eq(FieldRef('status'), Literal('success')),
                    Literal(1),
                    Literal(0)))
        ],
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_sec=86400),
            group_by=['brand_name', 'category'],
            measures=[
                MeasureSpec(
                    field='is_success', op=AggOp.SUM, alias='successes'),
                MeasureSpec(field='*', op=AggOp.COUNT, alias='total'),
            ]),
        metric_expr=Div(FieldRef('successes'), FieldRef('total')),
        _run_init=True)
    spec2 = self._round_trip(spec)
    self.assertEqual(spec2.name, 'success_rate')
    self.assertEqual(len(spec2.derived_fields), 1)
    # Verify derived field expression round-tripped
    df_expr = spec2.derived_fields[0].expression
    self.assertEqual(df_expr.evaluate({'status': 'success'}), 1)
    self.assertEqual(df_expr.evaluate({'status': 'error'}), 0)


# ---- Pipeline tests ----


class ComputeMetricCUJ1Test(unittest.TestCase):
  """CUJ 1: SUM(transaction_amount) over 1h, no grouping."""
  def test_global_sum(self):
    input_data = [
        # Window [0, 3600): timestamps 100, 200, 300
        TimestampedValue({'transaction_amount': 100.0}, 100),
        TimestampedValue({'transaction_amount': 200.0}, 200),
        TimestampedValue({'transaction_amount': 50.0}, 300),
        # Window [3600, 7200): timestamp 4000
        TimestampedValue({'transaction_amount': 75.0}, 4000),
    ]
    spec = MetricSpec(
        name='revenue',
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_sec=3600),
            measures=[
                MeasureSpec(
                    field='transaction_amount', op=AggOp.SUM, alias='revenue')
            ]),
        _run_init=True)

    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input_data)
          | ComputeMetric(spec)
          | beam.Map(lambda row: row.value))
      assert_that(result, equal_to([350.0, 75.0]))


class ComputeMetricCUJ2Test(unittest.TestCase):
  """CUJ 2: SUM(is_click)/COUNT(*) over 1h, grouped by campaign_type."""
  def test_grouped_ratio(self):
    input_data = [
        TimestampedValue({
            'campaign_type': 'search', 'is_click': 1
        }, 100),
        TimestampedValue({
            'campaign_type': 'search', 'is_click': 0
        }, 200),
        TimestampedValue({
            'campaign_type': 'search', 'is_click': 1
        }, 300),
        TimestampedValue({
            'campaign_type': 'display', 'is_click': 0
        }, 100),
        TimestampedValue({
            'campaign_type': 'display', 'is_click': 0
        }, 200),
    ]
    spec = MetricSpec(
        name='ctr',
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_sec=3600),
            group_by=['campaign_type'],
            measures=[
                MeasureSpec(field='is_click', op=AggOp.SUM, alias='clicks'),
                MeasureSpec(field='*', op=AggOp.COUNT, alias='impressions'),
            ]),
        metric_expr=Div(FieldRef('clicks'), FieldRef('impressions')),
        _run_init=True)

    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input_data)
          | ComputeMetric(spec)
          | beam.Map(lambda kv: (kv[0], round(kv[1].value, 4))))

      expected = [
          # search: 2 clicks / 3 impressions
          (('search', ), round(2 / 3, 4)),
          # display: 0 clicks / 2 impressions
          (('display', ), 0.0),
      ]
      assert_that(result, equal_to(expected))


class ComputeMetricCUJ3Test(unittest.TestCase):
  """CUJ 3: IF(status=='success',1,0) -> SUM/COUNT, grouped."""
  def test_derived_field_and_ratio(self):
    input_data = [
        TimestampedValue({
            'brand_name': 'Nike', 'category': 'shoes', 'status': 'success'
        },
                         100),
        TimestampedValue({
            'brand_name': 'Nike', 'category': 'shoes', 'status': 'success'
        },
                         200),
        TimestampedValue({
            'brand_name': 'Nike', 'category': 'shoes', 'status': 'error'
        },
                         300),
        TimestampedValue({
            'brand_name': 'Adidas', 'category': 'shirts', 'status': 'success'
        },
                         100),
        TimestampedValue({
            'brand_name': 'Adidas', 'category': 'shirts', 'status': 'error'
        },
                         200),
    ]
    spec = MetricSpec(
        name='success_rate',
        derived_fields=[
            DerivedField(
                name='is_success',
                expression=IfExpr(
                    Eq(FieldRef('status'), Literal('success')),
                    Literal(1),
                    Literal(0)))
        ],
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_sec=3600),
            group_by=['brand_name', 'category'],
            measures=[
                MeasureSpec(
                    field='is_success', op=AggOp.SUM, alias='successes'),
                MeasureSpec(field='*', op=AggOp.COUNT, alias='total'),
            ]),
        metric_expr=Div(FieldRef('successes'), FieldRef('total')),
        _run_init=True)

    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input_data)
          | ComputeMetric(spec)
          | beam.Map(lambda kv: (kv[0], round(kv[1].value, 4))))

      expected = [
          # Nike shoes: 2/3
          (('Nike', 'shoes'), round(2 / 3, 4)),
          # Adidas shirts: 1/2
          (('Adidas', 'shirts'), 0.5),
      ]
      assert_that(result, equal_to(expected))


class ComputeMetricMiscTest(unittest.TestCase):
  def test_count_only(self):
    input_data = [
        TimestampedValue({'x': 1}, 100),
        TimestampedValue({'x': 2}, 200),
        TimestampedValue({'x': 3}, 300),
    ]
    spec = MetricSpec(
        name='count',
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_sec=3600),
            measures=[MeasureSpec(field='*', op=AggOp.COUNT, alias='total')]),
        _run_init=True)

    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input_data)
          | ComputeMetric(spec)
          | beam.Map(lambda row: row.value))
      assert_that(result, equal_to([3.0]))

  def test_mean_aggregation(self):
    input_data = [
        TimestampedValue({'score': 10.0}, 100),
        TimestampedValue({'score': 20.0}, 200),
        TimestampedValue({'score': 30.0}, 300),
    ]
    spec = MetricSpec(
        name='avg_score',
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_sec=3600),
            measures=[MeasureSpec(field='score', op=AggOp.MEAN, alias='avg')]),
        _run_init=True)

    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input_data)
          | ComputeMetric(spec)
          | beam.Map(lambda row: row.value))
      assert_that(result, equal_to([20.0]))

  def test_min_max(self):
    input_data = [
        TimestampedValue({'latency': 50.0}, 100),
        TimestampedValue({'latency': 10.0}, 200),
        TimestampedValue({'latency': 90.0}, 300),
    ]
    spec = MetricSpec(
        name='latency_range',
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_sec=3600),
            measures=[
                MeasureSpec(field='latency', op=AggOp.MIN, alias='min_l'),
                MeasureSpec(field='latency', op=AggOp.MAX, alias='max_l'),
            ]),
        metric_expr=Div(FieldRef('max_l'), FieldRef('min_l')),
        _run_init=True)

    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input_data)
          | ComputeMetric(spec)
          | beam.Map(lambda row: row.value))
      assert_that(result, equal_to([9.0]))


class ComputeMetricWindowMetadataTest(unittest.TestCase):
  """Verify window_start and window_end are embedded in output Rows."""
  def test_global_metric_has_window_bounds(self):
    input_data = [
        TimestampedValue({'amount': 100.0}, 100),
        TimestampedValue({'amount': 200.0}, 200),
    ]
    spec = MetricSpec(
        name='revenue',
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_sec=3600),
            measures=[MeasureSpec(field='amount', op=AggOp.SUM,
                                  alias='total')]),
        _run_init=True)

    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input_data)
          | ComputeMetric(spec)
          | beam.Map(lambda row: (row.value, row.window_start, row.window_end)))
      # Window [0, 3600) — all timestamps fall in this window
      assert_that(result, equal_to([(300.0, 0.0, 3600.0)]))

  def test_keyed_metric_has_window_bounds(self):
    input_data = [
        TimestampedValue({
            'group': 'A', 'val': 10.0
        }, 100),
        TimestampedValue({
            'group': 'B', 'val': 20.0
        }, 100),
    ]
    spec = MetricSpec(
        name='grouped',
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_sec=3600),
            group_by=['group'],
            measures=[MeasureSpec(field='val', op=AggOp.SUM, alias='total')]),
        _run_init=True)

    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input_data)
          | ComputeMetric(spec)
          | beam.Map(
              lambda kv:
              (kv[0], kv[1].value, kv[1].window_start, kv[1].window_end)))
      assert_that(
          result,
          equal_to([
              (('A', ), 10.0, 0.0, 3600.0),
              (('B', ), 20.0, 0.0, 3600.0),
          ]))


class ComputeMetricEndToEndTest(unittest.TestCase):
  """Test ComputeMetric -> AnomalyDetection integration."""
  def test_global_metric_to_anomaly_detection(self):
    from apache_beam.ml.anomaly.detectors.zscore import ZScore
    from apache_beam.ml.anomaly.transforms import AnomalyDetection

    # Create data in multiple windows: several normal, one anomalous
    input_data = []
    # Windows 0-9: normal revenue ~100 each
    for window_idx in range(10):
      base_ts = window_idx * 10
      input_data.append(TimestampedValue({'amount': 100.0}, base_ts + 1))

    # Window 10: anomalous revenue
    input_data.append(TimestampedValue({'amount': 10000.0}, 101))

    spec = MetricSpec(
        name='revenue',
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_sec=10),
            measures=[MeasureSpec(field='amount', op=AggOp.SUM,
                                  alias='total')]),
        _run_init=True)

    detector = ZScore(features=['value'], _run_init=True)

    from apache_beam.transforms.window import GlobalWindows

    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input_data)
          | ComputeMetric(spec)
          | beam.WindowInto(GlobalWindows())
          | AnomalyDetection(detector)
          | beam.Map(lambda r: r.predictions[0].score is not None))
      # All results should have a score (not None)
      assert_that(result, equal_to([True] * 11))

  def test_keyed_metric_to_anomaly_detection(self):
    from apache_beam.ml.anomaly.detectors.zscore import ZScore
    from apache_beam.ml.anomaly.transforms import AnomalyDetection

    input_data = []
    for i in range(5):
      input_data.append(TimestampedValue({'group': 'A', 'val': 10.0}, i))
      input_data.append(TimestampedValue({'group': 'B', 'val': 20.0}, i))

    spec = MetricSpec(
        name='grouped_sum',
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_sec=1),
            group_by=['group'],
            measures=[MeasureSpec(field='val', op=AggOp.SUM, alias='total')]),
        _run_init=True)

    detector = ZScore(features=['value'], _run_init=True)

    from apache_beam.transforms.window import GlobalWindows

    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input_data)
          | ComputeMetric(spec)
          | beam.WindowInto(GlobalWindows())
          | AnomalyDetection(detector)
          | beam.Map(lambda kv: (kv[0], kv[1].example.value)))
      # Each (key, AnomalyResult) should have the right metric value
      expected = []
      for _ in range(5):
        expected.append((('A', ), 10.0))
        expected.append((('B', ), 20.0))
      assert_that(result, equal_to(expected))


if __name__ == '__main__':
  unittest.main()
