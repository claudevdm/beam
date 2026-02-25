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

"""Anomaly monitoring pipeline for BigQuery tables.

Reads streaming CDC data from BigQuery, computes a configurable windowed
metric, runs anomaly detection, and logs results.

Designed to be run as a Dataflow Flex Template or locally with DirectRunner.

Usage (Flex Template)::

    gcloud dataflow flex-template run "sales-monitor-$(date +%Y%m%d)" \\
        --template-file-gcs-location "gs://bucket/anomaly_monitor.json" \\
        --parameters table="project:dataset.table" \\
        --parameters metric_spec='{"name":"revenue","aggregation":{"window":{"type":"fixed","size_sec":3600},"measures":[{"field":"transaction_amount","op":"SUM","alias":"revenue"}]}}' \\
        --parameters detector_spec='{"type":"ZScore","config":{"features":["value"]}}' \\
        --region us-central1

Usage (PrismRunner)::

    python -m apache_beam.ml.anomaly.anomaly_monitor_pipeline \\
        --table=project:dataset.table \\
        --metric_spec='{"name":"revenue","aggregation":{"window":{"type":"fixed","size_sec":3600},"measures":[{"field":"transaction_amount","op":"SUM","alias":"revenue"}]}}' \\
        --detector_spec='{"type":"ZScore","config":{"features":["value"]}}' \\
        --runner=PrismRunner

Usage (DataflowRunner)::

    python -m apache_beam.ml.anomaly.anomaly_monitor_pipeline \\
        --table=project:dataset.table \\
        --metric_spec='<json>' \\
        --detector_spec='<json>' \\
        --runner=DataflowRunner \\
        --project=my-project \\
        --region=us-central1 \\
        --temp_location=gs://bucket/temp \\
        --staging_location=gs://bucket/staging \\
        --setup_file=./setup.py


metric_spec JSON Reference
==========================

Top-level ``metric_spec`` object::

    {
      "name": "<metric_name>",
      "aggregation": { ... },           # required
      "derived_fields": [ ... ],         # optional, pre-aggregation
      "metric_expr": { ... },            # optional (required if >1 measure)
      "output_field": "value"            # optional, default "value"
    }

aggregation
-----------
::

    "aggregation": {
      "window": {
        "type": "fixed" | "sliding",
        "size_sec": <int>,               # window size in seconds
        "period_sec": <int>              # slide period (required for sliding)
      },
      "group_by": ["field1", "field2"],  # optional, omit for global agg
      "measures": [
        {"field": "<col>", "op": "<OP>", "alias": "<name>"},
        ...
      ]
    }

Aggregation operators (``op``): ``SUM``, ``COUNT``, ``MIN``, ``MAX``, ``MEAN``.

For ``COUNT``, the ``field`` value is ignored (use ``"*"`` by convention)
— it counts all rows in the group.

derived_fields
--------------
Computed before aggregation. Each entry creates a new column available to
measures::

    "derived_fields": [
      {
        "name": "is_success",
        "expression": {
          "type": "if",
          "condition": {"type": "compare", "op": "==",
                        "left": {"type": "field", "name": "status"},
                        "right": {"type": "literal", "value": "success"}},
          "true_value": {"type": "literal", "value": 1},
          "false_value": {"type": "literal", "value": 0}
        }
      }
    ]

metric_expr
-----------
Post-aggregation expression that combines measure aliases into a single
value. Required when there are multiple measures (e.g., ratio metrics).

Expression types:

- ``{"type": "field", "name": "<alias>"}`` — reference a measure alias
- ``{"type": "literal", "value": <number|string>}`` — constant value
- ``{"type": "bin_op", "op": <op>, "left": <expr>, "right": <expr>}``
  where ``op`` is one of: ``+``, ``-``, ``*``, ``/``, ``//``, ``%``
- ``{"type": "compare", "op": <op>, "left": <expr>, "right": <expr>}``
  where ``op`` is one of: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``
- ``{"type": "negate", "operand": <expr>}`` — unary negation
- ``{"type": "if", "condition": <expr>, "true_value": <expr>,
  "false_value": <expr>}`` — conditional expression


detector_spec JSON Reference
=============================

Top-level ``detector_spec`` object::

    {"type": "<DetectorName>", "config": { ... }}

The ``type`` must be a registered ``@specifiable`` detector class name.
``config`` keys map to that class's ``__init__`` parameters plus inherited
``AnomalyDetector`` parameters.

Common AnomalyDetector parameters (all detectors)::

    "config": {
      "features": ["value"],                # input feature names (required)
      "threshold_criterion": { ... },       # optional, see below
      "model_id": "<string>"                # optional detector ID
    }

Available detectors
-------------------

**ZScore** — ``|value - mean| / stdev`` (default threshold: 3)::

    {"type": "ZScore", "config": {"features": ["value"]}}

**IQR** — Interquartile Range (default threshold: 1.5)::

    {"type": "IQR", "config": {"features": ["value"]}}

**RobustZScore** — Modified Z-Score using median/MAD (default threshold: 3.5)::

    {"type": "RobustZScore", "config": {"features": ["value"]}}

threshold_criterion
-------------------
Override the default threshold by nesting a specifiable threshold object.

**FixedThreshold** — static cutoff (scores >= cutoff are outliers)::

    "threshold_criterion": {
      "type": "FixedThreshold",
      "config": {"cutoff": 10}
    }

**QuantileThreshold** — dynamic cutoff at a quantile of observed scores::

    "threshold_criterion": {
      "type": "QuantileThreshold",
      "config": {"quantile": 0.95}
    }

Both accept optional ``normal_label`` (default 0), ``outlier_label``
(default 1), and ``missing_label`` (default -2).


Examples
--------

Simple SUM metric with ZScore::

    --metric_spec='{"name":"revenue","aggregation":{"window":{"type":"fixed","size_sec":3600},"measures":[{"field":"transaction_amount","op":"SUM","alias":"revenue"}]}}'
    --detector_spec='{"type":"ZScore","config":{"features":["value"]}}'

Grouped ratio metric (CTR) with ZScore::

    --metric_spec='{"name":"ctr","aggregation":{"window":{"type":"fixed","size_sec":10},"group_by":["campaign_type","browser_version"],"measures":[{"field":"is_click","op":"SUM","alias":"clicks"},{"field":"*","op":"COUNT","alias":"impressions"}]},"metric_expr":{"type":"bin_op","op":"/","left":{"type":"field","name":"clicks"},"right":{"type":"field","name":"impressions"}}}'
    --detector_spec='{"type":"ZScore","config":{"features":["value"]}}'

Derived field + ratio + custom threshold::

    --metric_spec='{"name":"success_rate","derived_fields":[{"name":"is_success","expression":{"type":"if","condition":{"type":"compare","op":"==","left":{"type":"field","name":"status"},"right":{"type":"literal","value":"success"}},"true_value":{"type":"literal","value":1},"false_value":{"type":"literal","value":0}}}],"aggregation":{"window":{"type":"fixed","size_sec":10},"group_by":["brand_name","category"],"measures":[{"field":"is_success","op":"SUM","alias":"successes"},{"field":"*","op":"COUNT","alias":"total"}]},"metric_expr":{"type":"bin_op","op":"/","left":{"type":"field","name":"successes"},"right":{"type":"field","name":"total"}}}'
    --detector_spec='{"type":"ZScore","config":{"features":["value"],"threshold_criterion":{"type":"FixedThreshold","config":{"cutoff":10}}}}'
"""

import datetime
import json
import logging
import time

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from apache_beam.ml.anomaly.metric import ComputeMetric
from apache_beam.ml.anomaly.metric import MetricSpec
from apache_beam.ml.anomaly.specifiable import Spec
from apache_beam.ml.anomaly.specifiable import Specifiable
from apache_beam.ml.anomaly.transforms import AnomalyDetection

# Import detectors so they register with @specifiable before from_spec.
from apache_beam.ml.anomaly.detectors import zscore  # noqa: F401
from apache_beam.ml.anomaly.detectors import iqr  # noqa: F401
from apache_beam.ml.anomaly.detectors import robust_zscore  # noqa: F401

_LOGGER = logging.getLogger(__name__)


class _LogAnomalyResult(beam.DoFn):
  """Logs each AnomalyResult at WARNING level for visibility in Dataflow."""
  def process(self, element):
    # Handle both keyed (key, AnomalyResult) and unkeyed (AnomalyResult)
    if isinstance(element, tuple) and len(element) == 2:
      key, result = element
      prediction = result.predictions[0]
    else:
      key = None
      result = element
      prediction = result.predictions[0]

    if prediction.label == 1:
      tag = '!! OUTLIER !!'
    elif prediction.label == 0:
      tag = 'NORMAL'
    else:
      tag = 'WARMUP'

    # Format window bounds from the Row metadata (set by ComputeMetric).
    example = result.example
    ws = datetime.datetime.fromtimestamp(
        example.window_start, tz=datetime.timezone.utc).strftime('%H:%M:%S')
    we = datetime.datetime.fromtimestamp(
        example.window_end, tz=datetime.timezone.utc).strftime('%H:%M:%S')
    window_str = f'{ws}-{we}'

    if key is not None:
      _LOGGER.warning(
          '[%s] window=%s key=%s value=%.2f score=%s label=%s',
          tag,
          window_str,
          key,
          example.value,
          prediction.score,
          prediction.label)
    else:
      _LOGGER.warning(
          '[%s] window=%s value=%.2f score=%s label=%s',
          tag,
          window_str,
          example.value,
          prediction.score,
          prediction.label)
    yield element


class AnomalyMonitorOptions(PipelineOptions):
  """Pipeline options for the anomaly monitor."""
  @classmethod
  def _add_argparse_args(cls, parser):
    parser.add_argument(
        '--table',
        default=None,
        help='BigQuery table to monitor. '
        'Format: project:dataset.table')
    parser.add_argument(
        '--metric_spec',
        default=None,
        help='JSON string defining the metric computation. '
        'See MetricSpec.from_dict() for schema.')
    parser.add_argument(
        '--detector_spec',
        default=None,
        help='JSON string defining the anomaly detector. '
        'Format: {"type":"ZScore","config":{"features":["value"]}}')
    parser.add_argument(
        '--poll_interval_sec',
        type=int,
        default=60,
        help='Seconds between BigQuery CDC polls. Default 60.')
    parser.add_argument(
        '--change_function',
        default='APPENDS',
        choices=['APPENDS', 'CHANGES'],
        help='BigQuery change function to use. Default APPENDS.')
    parser.add_argument(
        '--buffer_sec',
        type=float,
        default=15.0,
        help='Safety buffer behind now() in seconds. Default 15.')
    parser.add_argument(
        '--start_offset_sec',
        type=float,
        default=60.0,
        help='Start reading from this many seconds ago. Default 60.')
    parser.add_argument(
        '--duration_sec',
        type=float,
        default=0.0,
        help='How long to run in seconds. 0 means run forever. Default 0.')
    parser.add_argument(
        '--temp_dataset',
        default=None,
        help='BigQuery dataset for temp tables. If unset, auto-created.')


def _parse_metric_spec(json_str):
  """Parse a MetricSpec from a JSON string."""
  d = json.loads(json_str)
  return MetricSpec.from_dict(d)


def _dict_to_spec(d):
  """Recursively convert nested dicts with ``type`` keys into Spec objects.

  ``json.loads`` produces plain dicts, but ``Specifiable.from_spec`` expects
  ``Spec`` objects for nested specifiables (e.g. ``threshold_criterion``
  inside a detector config).  Without this conversion the nested dict passes
  through ``_specifiable_from_spec_helper`` unchanged and the detector
  receives a raw dict instead of the expected ``ThresholdFn`` instance.
  """
  if isinstance(d, dict) and 'type' in d:
    config = d.get('config', {})
    if config:
      config = {k: _dict_to_spec(v) for k, v in config.items()}
    return Spec(type=d['type'], config=config)
  if isinstance(d, list):
    return [_dict_to_spec(item) for item in d]
  return d


def _parse_detector_spec(json_str):
  """Parse an anomaly detector from a JSON Spec string.

  The JSON should have the form::

      {"type": "ZScore", "config": {"features": ["value"]}}

  Nested specifiable objects (e.g. ``threshold_criterion``) are supported::

      {"type": "ZScore", "config": {
          "features": ["value"],
          "threshold_criterion": {"type": "FixedThreshold", "config": {"cutoff": 10}}
      }}

  The ``type`` field must match a registered @specifiable detector class
  (e.g. ZScore, IQR, RobustZScore).
  """
  d = json.loads(json_str)
  spec = _dict_to_spec(d)
  return Specifiable.from_spec(spec, _run_init=True)


def build_pipeline(pipeline, options):
  """Construct the anomaly monitoring pipeline.

  Args:
    pipeline: A beam.Pipeline instance.
    options: AnomalyMonitorOptions with table, metric_spec, etc.

  Returns:
    The final PCollection (for testing).
  """
  from apache_beam.io.gcp.bigquery_change_history import (
      ReadBigQueryChangeHistory)

  metric_spec = _parse_metric_spec(options.metric_spec)
  detector = _parse_detector_spec(options.detector_spec)

  start_time = time.time() - options.start_offset_sec
  stop_time = (
      time.time() + options.duration_sec if options.duration_sec > 0 else None)

  _LOGGER.info('Anomaly Monitor Pipeline')
  _LOGGER.info('  Table: %s', options.table)
  _LOGGER.info('  Metric: %s', metric_spec.name)
  _LOGGER.info('  Detector: %s', type(detector).__name__)
  _LOGGER.info('  Poll interval: %d sec', options.poll_interval_sec)
  _LOGGER.info('  Change function: %s', options.change_function)

  cdc_kwargs = dict(
      table=options.table,
      poll_interval_sec=options.poll_interval_sec,
      start_time=start_time,
      change_function=options.change_function,
      buffer_sec=options.buffer_sec,
      trace=False)
  if stop_time is not None:
    cdc_kwargs['stop_time'] = stop_time
  if options.temp_dataset:
    cdc_kwargs['temp_dataset'] = options.temp_dataset

  rows = pipeline | 'ReadCDC' >> ReadBigQueryChangeHistory(**cdc_kwargs)
  metrics = rows | 'ComputeMetric' >> ComputeMetric(metric_spec)

  # Rewindow into GlobalWindows so the anomaly detector sees the full
  # stream of window results as a time series, not isolated per-window.
  from apache_beam.transforms.window import GlobalWindows
  global_metrics = metrics | 'Rewindow' >> beam.WindowInto(GlobalWindows())

  anomalies = global_metrics | 'DetectAnomalies' >> AnomalyDetection(detector)
  _ = anomalies | 'LogResults' >> beam.ParDo(_LogAnomalyResult())

  return anomalies


def run(argv=None):
  """Main entry point."""
  options = PipelineOptions(argv)
  monitor_options = options.view_as(AnomalyMonitorOptions)

  for required_opt in ('table', 'metric_spec', 'detector_spec'):
    if getattr(monitor_options, required_opt) is None:
      raise ValueError(f'--{required_opt} is required')

  options.view_as(SetupOptions).save_main_session = True

  with beam.Pipeline(options=options) as p:
    build_pipeline(p, monitor_options)


if __name__ == '__main__':
  logging.getLogger().setLevel(logging.WARNING)
  run()
