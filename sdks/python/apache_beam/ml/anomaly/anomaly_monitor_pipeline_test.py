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
from unittest import mock

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
from apache_beam.transforms.window import TimestampedValue

from apache_beam.ml.anomaly.anomaly_monitor_pipeline import _parse_detector_spec
from apache_beam.ml.anomaly.anomaly_monitor_pipeline import _parse_metric_spec
from apache_beam.ml.anomaly.anomaly_monitor_pipeline import AnomalyMonitorOptions
from apache_beam.ml.anomaly.anomaly_monitor_pipeline import _LogAnomalyResult
from apache_beam.ml.anomaly.metric import AggOp
from apache_beam.ml.anomaly.metric import AggregationSpec
from apache_beam.ml.anomaly.metric import ComputeMetric
from apache_beam.ml.anomaly.metric import MeasureSpec
from apache_beam.ml.anomaly.metric import MetricSpec
from apache_beam.ml.anomaly.metric import WindowSpec
from apache_beam.ml.anomaly.metric import WindowType
from apache_beam.ml.anomaly.transforms import AnomalyDetection
from apache_beam.options.pipeline_options import PipelineOptions


class ParseMetricSpecTest(unittest.TestCase):
  """Test _parse_metric_spec from JSON strings."""
  def test_cuj1_revenue(self):
    spec_json = json.dumps({
        'name': 'revenue',
        'aggregation': {
            'window': {
                'type': 'fixed', 'size_seconds': 3600
            },
            'measures': [{
                'field': 'transaction_amount', 'agg': 'SUM', 'alias': 'revenue'
            }],
        },
    })
    spec = _parse_metric_spec(spec_json)
    self.assertEqual(spec.name, 'revenue')
    self.assertEqual(spec.aggregation.window.size_seconds, 3600)
    self.assertEqual(len(spec.aggregation.measures), 1)
    self.assertEqual(spec.aggregation.measures[0].agg, AggOp.SUM)

  def test_cuj2_ctr(self):
    spec_json = json.dumps({
        'name': 'ctr',
        'aggregation': {
            'window': {
                'type': 'fixed', 'size_seconds': 86400
            },
            'group_by': ['campaign_type', 'user_segment'],
            'measures': [
                {
                    'field': 'is_click', 'agg': 'SUM', 'alias': 'clicks'
                },
                {
                    'field': '*', 'agg': 'COUNT', 'alias': 'impressions'
                },
            ],
        },
        'measure_combiner': {
            'expression': 'clicks / impressions'
        },
    })
    spec = _parse_metric_spec(spec_json)
    self.assertEqual(spec.name, 'ctr')
    self.assertEqual(
        spec.aggregation.group_by, ['campaign_type', 'user_segment'])
    self.assertEqual(len(spec.aggregation.measures), 2)
    self.assertAlmostEqual(
        spec.measure_combiner({
            'clicks': 50, 'impressions': 1000
        }), 0.05)

  def test_cuj3_success_rate(self):
    spec_json = json.dumps({
        'name': 'success_rate',
        'derived_fields': [{
            'name': 'is_success',
            'expression': "1 if status == 'success' else 0",
        }],
        'aggregation': {
            'window': {
                'type': 'fixed', 'size_seconds': 86400
            },
            'group_by': ['brand_name', 'category'],
            'measures': [
                {
                    'field': 'is_success', 'agg': 'SUM', 'alias': 'successes'
                },
                {
                    'field': '*', 'agg': 'COUNT', 'alias': 'total'
                },
            ],
        },
        'measure_combiner': {
            'expression': 'successes / total'
        },
    })
    spec = _parse_metric_spec(spec_json)
    self.assertEqual(spec.name, 'success_rate')
    self.assertEqual(len(spec.derived_fields), 1)
    self.assertEqual(len(spec.aggregation.group_by), 2)

  def test_invalid_json_raises(self):
    with self.assertRaises(json.JSONDecodeError):
      _parse_metric_spec('not valid json')


class ParseDetectorSpecTest(unittest.TestCase):
  """Test _parse_detector_spec from JSON strings."""
  def test_zscore_default(self):
    spec_json = json.dumps({'type': 'ZScore'})
    detector = _parse_detector_spec(spec_json)
    from apache_beam.ml.anomaly.detectors.zscore import ZScore
    self.assertIsInstance(detector, ZScore)
    self.assertEqual(detector._features, ['value'])

  def test_iqr_detector(self):
    spec_json = json.dumps({'type': 'IQR'})
    detector = _parse_detector_spec(spec_json)
    from apache_beam.ml.anomaly.detectors.iqr import IQR
    self.assertIsInstance(detector, IQR)
    self.assertEqual(detector._features, ['value'])

  def test_robust_zscore_detector(self):
    spec_json = json.dumps({'type': 'RobustZScore'})
    detector = _parse_detector_spec(spec_json)
    from apache_beam.ml.anomaly.detectors.robust_zscore import RobustZScore
    self.assertIsInstance(detector, RobustZScore)
    self.assertEqual(detector._features, ['value'])

  def test_unknown_detector_raises(self):
    spec_json = json.dumps({
        'type': 'NonExistentDetector',
        'config': {},
    })
    with self.assertRaises(ValueError):
      _parse_detector_spec(spec_json)

  def test_empty_config(self):
    spec_json = json.dumps({
        'type': 'ZScore',
        'config': {},
    })
    detector = _parse_detector_spec(spec_json)
    from apache_beam.ml.anomaly.detectors.zscore import ZScore
    self.assertIsInstance(detector, ZScore)

  def test_zscore_window_size(self):
    spec_json = json.dumps({
        'type': 'ZScore',
        'config': {
            'window_size': 500
        },
    })
    detector = _parse_detector_spec(spec_json)
    from apache_beam.ml.anomaly.detectors.zscore import ZScore
    self.assertIsInstance(detector, ZScore)
    self.assertEqual(detector._sub_stat_tracker._window_size, 500)
    self.assertEqual(detector._stdev_tracker._window_size, 500)

  def test_iqr_window_size(self):
    spec_json = json.dumps({
        'type': 'IQR',
        'config': {
            'window_size': 200
        },
    })
    detector = _parse_detector_spec(spec_json)
    from apache_beam.ml.anomaly.detectors.iqr import IQR
    self.assertIsInstance(detector, IQR)
    self.assertEqual(detector._q1_tracker._window_size, 200)

  def test_robust_zscore_window_size(self):
    spec_json = json.dumps({
        'type': 'RobustZScore',
        'config': {
            'window_size': 300
        },
    })
    detector = _parse_detector_spec(spec_json)
    from apache_beam.ml.anomaly.detectors.robust_zscore import RobustZScore
    self.assertIsInstance(detector, RobustZScore)
    self.assertEqual(
        detector._mad_tracker._median_tracker._quantile_tracker._window_size,
        300)
    self.assertEqual(
        detector._mad_tracker._diff_median_tracker._quantile_tracker.
        _window_size,
        300)

  def test_window_size_does_not_override_explicit_tracker(self):
    """If a user sets both window_size and an explicit tracker, the explicit
    tracker takes precedence."""
    spec_json = json.dumps({
        'type': 'ZScore',
        'config': {
            'window_size': 500,
            'sub_stat_tracker': {
                'type': 'IncSlidingMeanTracker', 'config': {
                    'window_size': 100
                }
            },
        },
    })
    detector = _parse_detector_spec(spec_json)
    # Explicit tracker wins over window_size shorthand
    self.assertEqual(detector._sub_stat_tracker._window_size, 100)
    # stdev_tracker gets the shorthand value since it wasn't set explicitly
    self.assertEqual(detector._stdev_tracker._window_size, 500)

  def test_zscore_default_window_size(self):
    """Without window_size, detectors use the default (1000)."""
    spec_json = json.dumps({'type': 'ZScore'})
    detector = _parse_detector_spec(spec_json)
    self.assertEqual(detector._sub_stat_tracker._window_size, 1000)
    self.assertEqual(detector._stdev_tracker._window_size, 1000)


class AnomalyMonitorOptionsTest(unittest.TestCase):
  """Test pipeline options parsing."""
  def test_required_options(self):
    options = PipelineOptions([
        '--table=project:dataset.table',
        '--metric_spec={"name":"test","aggregation":{"measures":[{"field":"x","agg":"SUM","alias":"y"}]}}',
        '--detector_spec={"type":"ZScore","config":{"features":["value"]}}',
    ])
    monitor = options.view_as(AnomalyMonitorOptions)
    self.assertEqual(monitor.table, 'project:dataset.table')
    self.assertIn('test', monitor.metric_spec)

  def test_default_values(self):
    options = PipelineOptions([
        '--table=project:dataset.table',
        '--metric_spec={}',
        '--detector_spec={}',
    ])
    monitor = options.view_as(AnomalyMonitorOptions)
    self.assertEqual(monitor.poll_interval_sec, 60)
    self.assertEqual(monitor.change_function, 'APPENDS')
    self.assertAlmostEqual(monitor.buffer_sec, 15.0)
    self.assertAlmostEqual(monitor.start_offset_sec, 60.0)
    self.assertAlmostEqual(monitor.duration_sec, 0.0)
    self.assertIsNone(monitor.temp_dataset)


class LogAnomalyResultTest(unittest.TestCase):
  """Test the logging DoFn."""
  def test_unkeyed_result(self):
    from apache_beam.ml.anomaly.base import AnomalyPrediction
    from apache_beam.ml.anomaly.base import AnomalyResult

    result = AnomalyResult(
        example=beam.Row(
            value=42.0, window_start=1700000000.0, window_end=1700003600.0),
        predictions=[AnomalyPrediction(score=2.5, label=0)])

    dofn = _LogAnomalyResult()
    output = list(dofn.process(result))
    self.assertEqual(len(output), 1)
    self.assertEqual(output[0], result)

  def test_keyed_result(self):
    from apache_beam.ml.anomaly.base import AnomalyPrediction
    from apache_beam.ml.anomaly.base import AnomalyResult

    result = AnomalyResult(
        example=beam.Row(
            value=42.0, window_start=1700000000.0, window_end=1700003600.0),
        predictions=[AnomalyPrediction(score=5.0, label=1)])
    element = (('search', ), result)

    dofn = _LogAnomalyResult()
    output = list(dofn.process(element))
    self.assertEqual(len(output), 1)
    self.assertEqual(output[0], element)


class EndToEndPipelineTest(unittest.TestCase):
  """Integration test: metric JSON -> ComputeMetric -> AnomalyDetection.

  These tests skip ReadBigQueryChangeHistory (which requires GCP credentials)
  and directly test the metric + anomaly detection pipeline stages.
  """
  def test_cuj1_revenue_e2e(self):
    metric_json = json.dumps({
        'name': 'revenue',
        'aggregation': {
            'window': {
                'type': 'fixed', 'size_seconds': 10
            },
            'measures': [
                {
                    'field': 'amount', 'agg': 'SUM', 'alias': 'total'
                },
            ],
        },
    })
    detector_json = json.dumps({'type': 'ZScore'})
    metric_spec = _parse_metric_spec(metric_json)
    detector = _parse_detector_spec(detector_json)

    input_data = []
    for i in range(10):
      input_data.append(TimestampedValue({'amount': 100.0}, i * 10 + 1))
    # Anomalous window
    input_data.append(TimestampedValue({'amount': 10000.0}, 101))

    from apache_beam.transforms.window import GlobalWindows

    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input_data)
          | ComputeMetric(metric_spec)
          | beam.WindowInto(GlobalWindows())
          | AnomalyDetection(detector)
          | beam.Map(lambda r: r.predictions[0].score is not None))
      assert_that(result, equal_to([True] * 11))

  def test_cuj2_grouped_ctr_e2e(self):
    metric_json = json.dumps({
        'name': 'ctr',
        'aggregation': {
            'window': {
                'type': 'fixed', 'size_seconds': 10
            },
            'group_by': ['campaign_type'],
            'measures': [
                {
                    'field': 'is_click', 'agg': 'SUM', 'alias': 'clicks'
                },
                {
                    'field': '*', 'agg': 'COUNT', 'alias': 'impressions'
                },
            ],
        },
        'measure_combiner': {
            'expression': 'clicks / impressions'
        },
    })
    detector_json = json.dumps({'type': 'ZScore'})
    metric_spec = _parse_metric_spec(metric_json)
    detector = _parse_detector_spec(detector_json)

    input_data = []
    for i in range(5):
      base_ts = i * 10 + 1
      input_data.append(
          TimestampedValue({
              'campaign_type': 'search', 'is_click': 1
          }, base_ts))
      input_data.append(
          TimestampedValue({
              'campaign_type': 'search', 'is_click': 0
          },
                           base_ts + 1))
      input_data.append(
          TimestampedValue({
              'campaign_type': 'display', 'is_click': 0
          }, base_ts))

    from apache_beam.transforms.window import GlobalWindows

    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input_data)
          | ComputeMetric(metric_spec)
          | beam.WindowInto(GlobalWindows())
          | AnomalyDetection(detector)
          |
          beam.Map(lambda kv: (kv[0], kv[1].predictions[0].score is not None)))
      # All results should have scores
      expected = []
      for _ in range(5):
        expected.append((('search', ), True))
        expected.append((('display', ), True))
      assert_that(result, equal_to(expected))

  def test_cuj3_success_rate_e2e(self):
    metric_json = json.dumps({
        'name': 'success_rate',
        'derived_fields': [{
            'name': 'is_success',
            'expression': "1 if status == 'success' else 0",
        }],
        'aggregation': {
            'window': {
                'type': 'fixed', 'size_seconds': 10
            },
            'group_by': ['region'],
            'measures': [
                {
                    'field': 'is_success', 'agg': 'SUM', 'alias': 'successes'
                },
                {
                    'field': '*', 'agg': 'COUNT', 'alias': 'total'
                },
            ],
        },
        'measure_combiner': {
            'expression': 'successes / total'
        },
    })
    detector_json = json.dumps({'type': 'ZScore'})
    metric_spec = _parse_metric_spec(metric_json)
    detector = _parse_detector_spec(detector_json)

    input_data = []
    for i in range(5):
      base_ts = i * 10 + 1
      input_data.append(
          TimestampedValue({
              'region': 'US', 'status': 'success'
          }, base_ts))
      input_data.append(
          TimestampedValue({
              'region': 'US', 'status': 'error'
          }, base_ts + 1))

    from apache_beam.transforms.window import GlobalWindows

    with TestPipeline() as p:
      result = (
          p
          | beam.Create(input_data)
          | ComputeMetric(metric_spec)
          | beam.WindowInto(GlobalWindows())
          | AnomalyDetection(detector)
          | beam.Map(lambda kv: (kv[0], round(kv[1].example.value, 2))))
      # Each window: 1 success / 2 total = 0.5
      expected = [(('US', ), 0.5)] * 5
      assert_that(result, equal_to(expected))


class SpecRoundTripFromGcloudTest(unittest.TestCase):
  """Test that specs survive the gcloud --parameters JSON encoding."""
  def test_metric_spec_as_gcloud_parameter(self):
    """Simulate what happens when gcloud passes --parameters metric_spec=..."""
    original_spec = MetricSpec(
        name='revenue',
        aggregation=AggregationSpec(
            window=WindowSpec(type=WindowType.FIXED, size_seconds=3600),
            measures=[
                MeasureSpec(
                    field='transaction_amount', agg=AggOp.SUM, alias='revenue')
            ]),
        _run_init=True)
    # Serialize to JSON (what the user types on the command line)
    json_str = json.dumps(original_spec.to_dict())
    # Deserialize (what the pipeline does)
    restored = _parse_metric_spec(json_str)
    self.assertEqual(restored.name, original_spec.name)
    self.assertEqual(
        restored.aggregation.window.size_seconds,
        original_spec.aggregation.window.size_seconds)
    self.assertEqual(
        restored.aggregation.measures[0].field,
        original_spec.aggregation.measures[0].field)

  def test_detector_spec_as_gcloud_parameter(self):
    """Simulate detector spec from gcloud --parameters."""
    json_str = '{"type":"ZScore"}'
    detector = _parse_detector_spec(json_str)
    from apache_beam.ml.anomaly.detectors.zscore import ZScore
    self.assertIsInstance(detector, ZScore)
    self.assertEqual(detector._features, ['value'])


if __name__ == '__main__':
  unittest.main()
