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

"""Configurable metric computation for anomaly detection pipelines.

This module provides a ``MetricSpec`` configuration system and a
``ComputeMetric`` PTransform that computes windowed, grouped metrics from
raw row dicts (e.g., from ``ReadBigQueryChangeHistory``). The output is
suitable for feeding directly into ``AnomalyDetection``.

Example usage::

  from apache_beam.ml.anomaly.metric import (
      MetricSpec, AggregationSpec, WindowSpec, MeasureSpec,
      DerivedField, WindowType, AggOp, ComputeMetric)
  from apache_beam.ml.anomaly.safe_eval import (
      FieldRef, Literal, Div, Eq, IfExpr)
  from apache_beam.ml.anomaly.transforms import AnomalyDetection
  from apache_beam.ml.anomaly.detectors.zscore import ZScore

  # CUJ 1: Total revenue per hour
  spec = MetricSpec(
      name='revenue',
      aggregation=AggregationSpec(
          window=WindowSpec(type=WindowType.FIXED, size_sec=3600),
          measures=[MeasureSpec(
              field='transaction_amount', op=AggOp.SUM, alias='revenue')],
      ),
  )
  result = cdc_rows | ComputeMetric(spec) | AnomalyDetection(
      ZScore(features=['value']))

  # CUJ 2: CTR grouped by dimensions
  spec = MetricSpec(
      name='ctr',
      aggregation=AggregationSpec(
          window=WindowSpec(type=WindowType.FIXED, size_sec=86400),
          group_by=['campaign_type', 'user_segment'],
          measures=[
              MeasureSpec(field='is_click', op=AggOp.SUM, alias='clicks'),
              MeasureSpec(field='*', op=AggOp.COUNT, alias='impressions'),
          ],
      ),
      metric_expr=Div(FieldRef('clicks'), FieldRef('impressions')),
  )

  # CUJ 3: Success rate with derived field
  spec = MetricSpec(
      name='success_rate',
      derived_fields=[
          DerivedField(
              name='is_success',
              expression=IfExpr(
                  Eq(FieldRef('status'), Literal('success')),
                  Literal(1), Literal(0))),
      ],
      aggregation=AggregationSpec(
          window=WindowSpec(type=WindowType.FIXED, size_sec=86400),
          group_by=['brand_name', 'category'],
          measures=[
              MeasureSpec(field='is_success', op=AggOp.SUM, alias='successes'),
              MeasureSpec(field='*', op=AggOp.COUNT, alias='total'),
          ],
      ),
      metric_expr=Div(FieldRef('successes'), FieldRef('total')),
  )
"""

import dataclasses
from enum import Enum
from typing import Any
from typing import Optional
from typing import Tuple

import apache_beam as beam
from apache_beam.transforms import combiners
from apache_beam.transforms import window as beam_window

from apache_beam.ml.anomaly.safe_eval import Expr
from apache_beam.ml.anomaly.specifiable import specifiable


class WindowType(Enum):
  """Window type for metric aggregation."""
  FIXED = 'fixed'
  SLIDING = 'sliding'


class AggOp(Enum):
  """Aggregation operator."""
  SUM = 'SUM'
  COUNT = 'COUNT'
  MIN = 'MIN'
  MAX = 'MAX'
  MEAN = 'MEAN'


@dataclasses.dataclass(frozen=True)
class WindowSpec:
  """Window configuration for metric aggregation.

  Args:
    type: FIXED or SLIDING window.
    size_sec: Window size in seconds.
    period_sec: Slide period in seconds (required for SLIDING, ignored for
      FIXED).
  """
  type: WindowType = WindowType.FIXED
  size_sec: int = 3600
  period_sec: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class DerivedField:
  """Pre-aggregation column derivation via structured expression.

  Args:
    name: Name of the new field to create.
    expression: An ``Expr`` tree, e.g.
      ``IfExpr(Eq(FieldRef('status'), Literal('success')),
      Literal(1), Literal(0))``.
  """
  name: str
  expression: Expr


@dataclasses.dataclass(frozen=True)
class MeasureSpec:
  """A single aggregation measure.

  Args:
    field: Input field name to aggregate, or ``'*'`` for COUNT.
    op: The aggregation operator.
    alias: Output name for this measure's result.
  """
  field: str
  op: AggOp
  alias: str


@dataclasses.dataclass(frozen=True)
class AggregationSpec:
  """Windowed grouped aggregation configuration.

  Args:
    window: Window configuration.
    group_by: Field names for grouping. Empty list means global aggregation.
    measures: List of aggregation measures.
  """
  window: WindowSpec = dataclasses.field(default_factory=WindowSpec)
  group_by: list = dataclasses.field(default_factory=list)
  measures: list = dataclasses.field(default_factory=list)


@specifiable
class MetricSpec:
  """Complete metric computation specification.

  Defines how to transform raw row dicts into a single numeric metric value
  suitable for anomaly detection.

  Args:
    name: Human-readable metric name.
    aggregation: Windowed grouped aggregation spec.
    derived_fields: Optional pre-aggregation derived fields.
    metric_expr: Optional post-aggregation ``Expr`` operating on measure
      aliases. Required when there are multiple measures.
    output_field: Name of the output field in the resulting beam.Row.
  """
  def __init__(
      self,
      name,
      aggregation,
      derived_fields=None,
      metric_expr=None,
      output_field='value',
  ):
    self.name = name
    self.aggregation = aggregation
    self.derived_fields = derived_fields or []
    self.metric_expr = metric_expr
    self.output_field = output_field
    self._validate()

  def _validate(self):
    agg = self.aggregation
    if not agg.measures:
      raise ValueError("MetricSpec requires at least one measure")
    if self.metric_expr is None and len(agg.measures) > 1:
      raise ValueError(
          "metric_expr is required when there are multiple measures. "
          f"Got {len(agg.measures)} measures: "
          f"{[m.alias for m in agg.measures]}")
    if (agg.window.type == WindowType.SLIDING and
        agg.window.period_sec is None):
      raise ValueError("period_sec is required for SLIDING windows")
    for df in self.derived_fields:
      if not isinstance(df.expression, Expr):
        raise TypeError(
            f"DerivedField.expression must be an Expr, "
            f"got {type(df.expression).__name__}")
    if self.metric_expr is not None and not isinstance(self.metric_expr, Expr):
      raise TypeError(
          f"metric_expr must be an Expr, "
          f"got {type(self.metric_expr).__name__}")

  def to_dict(self):
    """Serialize to a plain dict suitable for JSON."""
    result = {
        'name': self.name,
        'aggregation': {
            'window': {
                'type': self.aggregation.window.type.value,
                'size_sec': self.aggregation.window.size_sec,
                'period_sec': self.aggregation.window.period_sec,
            },
            'group_by': list(self.aggregation.group_by),
            'measures': [{
                'field': m.field, 'op': m.op.value, 'alias': m.alias
            } for m in self.aggregation.measures],
        },
        'output_field': self.output_field,
    }
    if self.derived_fields:
      result['derived_fields'] = [{
          'name': df.name, 'expression': df.expression.to_dict()
      } for df in self.derived_fields]
    if self.metric_expr is not None:
      result['metric_expr'] = self.metric_expr.to_dict()
    return result

  @classmethod
  def from_dict(cls, d):
    """Construct a MetricSpec from a plain dict (e.g., loaded from JSON).

    Args:
      d: Dictionary with keys matching the MetricSpec constructor.

    Returns:
      MetricSpec instance.
    """
    agg_dict = d['aggregation']
    window_dict = agg_dict.get('window', {})
    window = WindowSpec(
        type=WindowType(window_dict.get('type', 'fixed')),
        size_sec=window_dict.get('size_sec', 3600),
        period_sec=window_dict.get('period_sec'),
    )
    measures = [
        MeasureSpec(field=m['field'], op=AggOp(m['op']), alias=m['alias'])
        for m in agg_dict.get('measures', [])
    ]
    derived_fields = None
    if 'derived_fields' in d and d['derived_fields']:
      derived_fields = [
          DerivedField(
              name=df['name'], expression=Expr.from_dict(df['expression']))
          for df in d['derived_fields']
      ]
    metric_expr = None
    if 'metric_expr' in d and d['metric_expr'] is not None:
      metric_expr = Expr.from_dict(d['metric_expr'])
    return cls(
        name=d['name'],
        aggregation=AggregationSpec(
            window=window,
            group_by=agg_dict.get('group_by', []),
            measures=measures,
        ),
        derived_fields=derived_fields,
        metric_expr=metric_expr,
        output_field=d.get('output_field', 'value'),
        _run_init=True,
    )


# ---------------------------------------------------------------------------
# Internal CombineFn and DoFns
# ---------------------------------------------------------------------------


class _SumCombineFn(beam.CombineFn):
  """Simple sum combiner."""
  def create_accumulator(self):
    return 0

  def add_input(self, accumulator, element):
    return accumulator + element

  def merge_accumulators(self, accumulators):
    return sum(accumulators)

  def extract_output(self, accumulator):
    return accumulator


class _MinCombineFn(beam.CombineFn):
  """Min combiner."""
  def create_accumulator(self):
    return float('inf')

  def add_input(self, accumulator, element):
    return min(accumulator, element)

  def merge_accumulators(self, accumulators):
    return min(accumulators)

  def extract_output(self, accumulator):
    return accumulator


class _MaxCombineFn(beam.CombineFn):
  """Max combiner."""
  def create_accumulator(self):
    return float('-inf')

  def add_input(self, accumulator, element):
    return max(accumulator, element)

  def merge_accumulators(self, accumulators):
    return max(accumulators)

  def extract_output(self, accumulator):
    return accumulator


def _get_combiner_for_op(op):
  """Map AggOp enum to a Beam CombineFn instance."""
  if op == AggOp.SUM:
    return _SumCombineFn()
  elif op == AggOp.COUNT:
    return combiners.CountCombineFn()
  elif op == AggOp.MIN:
    return _MinCombineFn()
  elif op == AggOp.MAX:
    return _MaxCombineFn()
  elif op == AggOp.MEAN:
    return combiners.MeanCombineFn()
  else:
    raise ValueError(f"Unknown aggregation operator: {op}")


class _MetricCombineFn(beam.CombineFn):
  """CombineFn that applies multiple aggregations to different fields of input
  dicts in a single pass.

  Each measure extracts a specific field from the input dict and feeds it
  to its own sub-combiner. The output is a dict mapping measure aliases to
  aggregated values.
  """
  def __init__(self, measures):
    self._measures = measures
    self._combiners = [_get_combiner_for_op(m.op) for m in measures]

  def create_accumulator(self):
    return [c.create_accumulator() for c in self._combiners]

  def add_input(self, accumulator, element):
    result = []
    for i, (measure, comb) in enumerate(zip(self._measures, self._combiners)):
      if measure.op == AggOp.COUNT:
        result.append(comb.add_input(accumulator[i], element))
      else:
        field_val = element.get(measure.field)
        if field_val is not None:
          result.append(comb.add_input(accumulator[i], field_val))
        else:
          result.append(accumulator[i])
    return result

  def merge_accumulators(self, accumulators):
    accumulators = list(accumulators)
    if not accumulators:
      return self.create_accumulator()
    if len(accumulators) == 1:
      return accumulators[0]
    merged = []
    for i, comb in enumerate(self._combiners):
      per_combiner_accs = [acc[i] for acc in accumulators]
      merged.append(comb.merge_accumulators(per_combiner_accs))
    return merged

  def extract_output(self, accumulator):
    return {
        measure.alias: comb.extract_output(acc)
        for measure, comb, acc in zip(
            self._measures, self._combiners, accumulator)
    }


class _ApplyDerivedFields(beam.DoFn):
  """DoFn that evaluates derived field expressions on each input dict."""
  def __init__(self, derived_fields):
    self._derived_fields = derived_fields

  def process(self, element):
    row = dict(element)
    for df in self._derived_fields:
      row[df.name] = df.expression.evaluate(row)
    yield row


class _ApplyMetricExpr(beam.DoFn):
  """DoFn that evaluates a post-aggregation expression on combined results."""
  def __init__(self, metric_expr, output_field, is_keyed):
    self._metric_expr = metric_expr
    self._output_field = output_field
    self._is_keyed = is_keyed

  def process(self, element, window=beam.DoFn.WindowParam):
    if self._is_keyed:
      key, agg_dict = element
    else:
      agg_dict = element

    if self._metric_expr is not None:
      value = float(self._metric_expr.evaluate(agg_dict))
    else:
      value = float(next(iter(agg_dict.values())))

    row = beam.Row(
        **{self._output_field: value},
        window_start=float(window.start),
        window_end=float(window.end))

    if self._is_keyed:
      yield (key, row)
    else:
      yield row


class ComputeMetric(beam.PTransform):
  """Transforms raw row dicts into metric beam.Rows for anomaly detection.

  Takes a ``PCollection[dict]`` with event-time timestamps and produces
  either ``PCollection[beam.Row]`` (for global aggregation) or
  ``PCollection[tuple[key, beam.Row]]`` (for grouped aggregation).

  The output is directly compatible with ``AnomalyDetection``.

  Args:
    metric_spec: A ``MetricSpec`` defining the metric computation.
  """
  def __init__(self, metric_spec):
    super().__init__()
    self._spec = metric_spec

  def expand(self, pcoll):
    spec = self._spec
    agg = spec.aggregation

    # Step 1: Apply derived fields
    if spec.derived_fields:
      pcoll = pcoll | 'DerivedFields' >> beam.ParDo(
          _ApplyDerivedFields(spec.derived_fields))

    # Step 2: Apply windowing
    if agg.window.type == WindowType.FIXED:
      window_fn = beam_window.FixedWindows(agg.window.size_sec)
    elif agg.window.type == WindowType.SLIDING:
      window_fn = beam_window.SlidingWindows(
          agg.window.size_sec, agg.window.period_sec)
    else:
      raise ValueError(f"Unknown window type: {agg.window.type}")

    windowed = pcoll | 'Window' >> beam.WindowInto(window_fn)

    # Step 3: Aggregate
    combine_fn = _MetricCombineFn(agg.measures)
    is_keyed = bool(agg.group_by)

    if is_keyed:
      group_by_fields = agg.group_by

      def extract_key(row_dict):
        return tuple(row_dict.get(f) for f in group_by_fields)

      keyed = windowed | 'ExtractKey' >> beam.WithKeys(extract_key)
      aggregated = keyed | 'Combine' >> beam.CombinePerKey(combine_fn)
    else:
      aggregated = (
          windowed
          | 'Combine' >> beam.CombineGlobally(combine_fn).without_defaults())

    # Step 4: Apply metric expression and set output type hints
    metric_dofn = _ApplyMetricExpr(
        spec.metric_expr, spec.output_field, is_keyed)

    if is_keyed:
      # AnomalyDetection checks isinstance(element_type, TupleConstraint)
      # to detect keyed input. We must annotate the output type.
      result = aggregated | 'MetricExpr' >> beam.ParDo(
          metric_dofn).with_output_types(Tuple[Any, beam.Row])
    else:
      result = aggregated | 'MetricExpr' >> beam.ParDo(metric_dofn)

    return result
