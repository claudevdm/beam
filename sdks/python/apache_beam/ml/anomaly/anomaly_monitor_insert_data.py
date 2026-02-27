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

"""Insert test data for anomaly monitoring CUJs.

Creates a BigQuery table (with change history enabled) and continuously
inserts rows matching one of the three CUJ schemas. Use alongside
anomaly_monitor_pipeline.py to test streaming anomaly detection.

Data is generated in a repeating cycle: a long stretch of **normal** data
followed by a brief **anomaly** burst, simulating a real production system
that runs fine for minutes and then experiences a short-lived incident.
The cycle repeats until Ctrl+C so the pipeline always has fresh data.

Defaults produce ~95% normal / ~5% anomaly:
  200 normal batches (~3.3 min) → 10 anomaly batches (~10 sec) → repeat.
  With 10s metric windows, ZScore sees ~20 stable windows before each
  anomaly, giving it strong statistics to detect the drop.

Quick-start tips:
  - Always use --truncate for clean runs (drops + recreates the table).
  - Start the insert script FIRST, wait ~30s, then start the pipeline.
  - Expected output: [WARMUP] → [NORMAL] → [!! OUTLIER !!] → [NORMAL].

Usage:
  # ── CUJ 1: Revenue monitoring (MEAN transaction amount by region) ──
  # Anomaly: US region revenue crashes to $0.01 while EU/APAC stay normal.
  # Terminal 1: insert data
  python -m apache_beam.ml.anomaly.anomaly_monitor_insert_data \
      --project=dataflow-twest --dataset=cdc --table=cuj1_revenue \
      --cuj=revenue --rows_per_batch=10 --interval_sec=1 --truncate

  # Terminal 2: run pipeline
  python -m apache_beam.ml.anomaly.anomaly_monitor_pipeline \
      --project=dataflow-twest \
      --table='dataflow-twest:cdc.cuj1_revenue' \
      --metric_spec='{"name":"revenue","aggregation":{"window":{"type":"fixed","size_seconds":10},"group_by":["region"],"measures":[{"field":"transaction_amount","agg":"MEAN","alias":"revenue"}]}}' \
      --detector_spec='{"type":"ZScore"}' \
      --poll_interval_sec=30 --start_offset_sec=30 --duration_sec=980 \
      --runner=PrismRunner


  # ── CUJ 2: CTR monitoring (grouped by campaign_type, browser_version) ──
  # Terminal 1:
  python -m apache_beam.ml.anomaly.anomaly_monitor_insert_data \
      --project=dataflow-twest --dataset=cdc --table=cuj2_ctr \
      --cuj=ctr --rows_per_batch=500 --interval_sec=1 --truncate

  # Terminal 2:
  python -m apache_beam.ml.anomaly.anomaly_monitor_pipeline \
      --project=dataflow-twest \
      --table='dataflow-twest:cdc.cuj2_ctr' \
      --metric_spec='{"name":"ctr","aggregation":{"window":{"type":"fixed","size_seconds":10},"group_by":["campaign_type","browser_version"],"measures":[{"field":"is_click","agg":"SUM","alias":"clicks"},{"field":"is_click","agg":"COUNT","alias":"impressions"}]},"measure_combiner":{"expression":"clicks / impressions"}}' \
      --detector_spec='{"type":"ZScore"}' \
      --poll_interval_sec=30 --start_offset_sec=30 --duration_sec=980 \
      --runner=PrismRunner


  # ── CUJ 3: Success rate monitoring (derived field + grouped) ──
  # Terminal 1:
  python -m apache_beam.ml.anomaly.anomaly_monitor_insert_data \
      --project=dataflow-twest --dataset=cdc --table=cuj3_success \
      --cuj=success_rate --rows_per_batch=500 --interval_sec=1 --truncate

  # Terminal 2:
  python -m apache_beam.ml.anomaly.anomaly_monitor_pipeline \
      --project=dataflow-twest \
      --table='dataflow-twest:cdc.cuj3_success' \
      --metric_spec='{"name":"success_rate","derived_fields":[{"name":"is_success","expression":"1 if status == '"'"'success'"'"' else 0"}],"aggregation":{"window":{"type":"fixed","size_seconds":10},"group_by":["brand_name","category"],"measures":[{"field":"is_success","agg":"SUM","alias":"successes"},{"field":"is_success","agg":"COUNT","alias":"total"}]},"measure_combiner":{"expression":"successes / total"}}' \
      --detector_spec='{"type":"RobustZScore","config":{"threshold_criterion":{"type":"FixedThreshold","config":{"cutoff":10}}}}' \
      --poll_interval_sec=30 --start_offset_sec=30 --duration_sec=980 \
      --runner=PrismRunner
"""

import argparse
import logging
import random
import time
import uuid

from apache_beam.io.gcp.bigquery_tools import BigQueryWrapper
from apache_beam.io.gcp.internal.clients import bigquery

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Table schemas (DDL)
# ---------------------------------------------------------------------------

_DDL_REVENUE = (
    'CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` '
    '(transaction_id INT64, transaction_amount FLOAT64, '
    'customer_id STRING, region STRING) '
    'OPTIONS (enable_change_history = true)')

_DDL_CTR = (
    'CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` '
    '(impression_id INT64, campaign_type STRING, user_segment STRING, '
    'browser_version STRING, is_click INT64) '
    'OPTIONS (enable_change_history = true)')

_DDL_SUCCESS_RATE = (
    'CREATE TABLE IF NOT EXISTS `{project}.{dataset}.{table}` '
    '(request_id INT64, brand_name STRING, category STRING, '
    'status STRING, latency_ms FLOAT64) '
    'OPTIONS (enable_change_history = true)')

# ---------------------------------------------------------------------------
# Row generators — normal
# ---------------------------------------------------------------------------

_REGIONS = ['US', 'EU', 'APAC']
_CAMPAIGNS = ['search', 'display', 'video']
_SEGMENTS = ['new', 'returning', 'premium']
_BROWSERS = ['chrome', 'firefox', 'safari']
_BRANDS = ['Nike', 'Adidas', 'Puma']
_CATEGORIES = ['shoes', 'shirts', 'accessories']


def _gen_revenue_normal(base_id, count):
  """Normal revenue: $40-$60 per transaction."""
  rows = []
  for i in range(count):
    rows.append({
        'transaction_id': base_id + i,
        'transaction_amount': round(random.uniform(40, 60), 2),
        'customer_id': f'cust_{random.randint(1000, 9999)}',
        'region': random.choice(_REGIONS),
    })
  return rows


def _gen_revenue_anomaly(base_id, count):
  """Anomalous revenue: $0.01 for US region, normal for others.

  Simulates a regional payment gateway outage — only US revenue crashes
  while EU and APAC continue normally.
  """
  rows = []
  for i in range(count):
    region = random.choice(_REGIONS)
    rows.append({
        'transaction_id': base_id + i,
        'transaction_amount': 0.01 if region == 'US' else round(
            random.uniform(40, 60), 2),
        'customer_id': f'cust_{random.randint(1000, 9999)}',
        'region': region,
    })
  return rows


def _gen_ctr_normal(base_id, count):
  """Normal CTR: ~20% click rate."""
  rows = []
  for i in range(count):
    rows.append({
        'impression_id': base_id + i,
        'campaign_type': random.choice(_CAMPAIGNS),
        'user_segment': random.choice(_SEGMENTS),
        'browser_version': random.choice(_BROWSERS),
        'is_click': 1 if random.random() < 0.20 else 0,
    })
  return rows


def _gen_ctr_anomaly(base_id, count):
  """Anomalous CTR: 0% clicks for search+chrome, normal for others.

  Rows keep random groups so non-target groups maintain stable row counts.
  Only the (search, chrome) group has its clicks zeroed out.
  """
  rows = []
  for i in range(count):
    campaign = random.choice(_CAMPAIGNS)
    browser = random.choice(_BROWSERS)
    is_anomaly_group = campaign == 'search' and browser == 'chrome'
    rows.append({
        'impression_id': base_id + i,
        'campaign_type': campaign,
        'user_segment': random.choice(_SEGMENTS),
        'browser_version': browser,
        'is_click': 0 if is_anomaly_group else
        (1 if random.random() < 0.20 else 0),
    })
  return rows


def _gen_success_normal(base_id, count):
  """Normal success rate: ~95% success."""
  rows = []
  for i in range(count):
    rows.append({
        'request_id': base_id + i,
        'brand_name': random.choice(_BRANDS),
        'category': random.choice(_CATEGORIES),
        'status': 'success' if random.random() < 0.95 else 'error',
        'latency_ms': round(random.uniform(50, 200), 1),
    })
  return rows


def _gen_success_anomaly(base_id, count):
  """Anomalous success rate: 0% success for Nike+shoes, normal for others.

  Rows keep random groups so non-target groups maintain stable row counts.
  Only the (Nike, shoes) group has its status forced to 'error'.
  """
  rows = []
  for i in range(count):
    brand = random.choice(_BRANDS)
    category = random.choice(_CATEGORIES)
    is_anomaly_group = brand == 'Nike' and category == 'shoes'
    rows.append({
        'request_id': base_id + i,
        'brand_name': brand,
        'category': category,
        'status': 'error' if is_anomaly_group else
        ('success' if random.random() < 0.95 else 'error'),
        'latency_ms': round(random.uniform(500, 3000), 1)
        if is_anomaly_group else round(random.uniform(50, 200), 1),
    })
  return rows


_CUJ_CONFIG = {
    'revenue': {
        'ddl': _DDL_REVENUE,
        'normal': _gen_revenue_normal,
        'anomaly': _gen_revenue_anomaly,
    },
    'ctr': {
        'ddl': _DDL_CTR,
        'normal': _gen_ctr_normal,
        'anomaly': _gen_ctr_anomaly,
    },
    'success_rate': {
        'ddl': _DDL_SUCCESS_RATE,
        'normal': _gen_success_normal,
        'anomaly': _gen_success_anomaly,
    },
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _create_table(bq_wrapper, project, dataset, table, ddl_template):
  ddl = ddl_template.format(project=project, dataset=dataset, table=table)
  job_id = f'beam_anomaly_ddl_{uuid.uuid4().hex[:8]}'
  reference = bigquery.JobReference(jobId=job_id, projectId=project)
  request = bigquery.BigqueryJobsInsertRequest(
      projectId=project,
      job=bigquery.Job(
          configuration=bigquery.JobConfiguration(
              query=bigquery.JobConfigurationQuery(
                  query=ddl, useLegacySql=False)),
          jobReference=reference))
  response = bq_wrapper._start_job(request)
  bq_wrapper.wait_for_bq_job(response.jobReference, sleep_duration_sec=2)
  _LOGGER.info('Table %s.%s.%s ready', project, dataset, table)


def _drop_table(bq_wrapper, project, dataset, table):
  """Drop the table so it can be recreated with a clean change history."""
  dml = f'DROP TABLE IF EXISTS `{project}.{dataset}.{table}`'
  job_id = f'beam_anomaly_drop_{uuid.uuid4().hex[:8]}'
  reference = bigquery.JobReference(jobId=job_id, projectId=project)
  request = bigquery.BigqueryJobsInsertRequest(
      projectId=project,
      job=bigquery.Job(
          configuration=bigquery.JobConfiguration(
              query=bigquery.JobConfigurationQuery(
                  query=dml, useLegacySql=False)),
          jobReference=reference))
  response = bq_wrapper._start_job(request)
  bq_wrapper.wait_for_bq_job(response.jobReference, sleep_duration_sec=2)
  _LOGGER.info('Dropped table %s.%s.%s', project, dataset, table)


def main():
  parser = argparse.ArgumentParser(
      description='Insert test data for anomaly monitoring CUJs.')
  parser.add_argument(
      '--project', default='dataflow-twest', help='GCP project ID')
  parser.add_argument('--dataset', default='cdc', help='BigQuery dataset')
  parser.add_argument('--table', required=True, help='BigQuery table name')
  parser.add_argument(
      '--cuj',
      required=True,
      choices=['revenue', 'ctr', 'success_rate'],
      help='Which CUJ data to generate')
  parser.add_argument(
      '--rows_per_batch',
      type=int,
      default=10,
      help='Rows per insert batch (default: 10)')
  parser.add_argument(
      '--interval_sec',
      type=float,
      default=1.0,
      help='Seconds between batches (default: 1.0)')
  parser.add_argument(
      '--normal_batches',
      type=int,
      default=200,
      help='Normal batches per cycle before an anomaly burst. '
      'With interval_sec=1 and 10s windows this gives ~20 '
      'normal windows per cycle. (default: 200)')
  parser.add_argument(
      '--anomaly_batches',
      type=int,
      default=10,
      help='Anomaly batches per cycle. With interval_sec=1 and '
      '10s windows this gives ~1 anomaly window. (default: 10)')
  parser.add_argument(
      '--truncate',
      action='store_true',
      default=False,
      help='Drop and recreate the table before inserting. '
      'Gives a clean change history. '
      'Recommended for clean validation runs.')
  args = parser.parse_args()

  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s %(levelname)s %(name)s: %(message)s')

  config = _CUJ_CONFIG[args.cuj]
  bq_wrapper = BigQueryWrapper()

  if args.truncate:
    _drop_table(bq_wrapper, args.project, args.dataset, args.table)

  _create_table(
      bq_wrapper, args.project, args.dataset, args.table, config['ddl'])

  cycle_len = args.normal_batches + args.anomaly_batches
  anomaly_pct = 100.0 * args.anomaly_batches / cycle_len

  _LOGGER.info('CUJ: %s', args.cuj)
  _LOGGER.info('Table: %s.%s.%s', args.project, args.dataset, args.table)
  _LOGGER.info(
      'Cycle: %d normal + %d anomaly = %d batches (%.1f%% anomaly)',
      args.normal_batches,
      args.anomaly_batches,
      cycle_len,
      anomaly_pct)
  _LOGGER.info(
      'Inserting %d rows every %.1f sec — Ctrl+C to stop',
      args.rows_per_batch,
      args.interval_sec)

  batch_num = 0
  total_rows = 0
  next_time = time.monotonic()
  try:
    while True:
      base_id = batch_num * args.rows_per_batch
      pos = batch_num % cycle_len
      is_anomaly = pos >= args.normal_batches

      if is_anomaly:
        rows = config['anomaly'](base_id, args.rows_per_batch)
        label = 'ANOMALY'
      else:
        rows = config['normal'](base_id, args.rows_per_batch)
        label = 'normal'

      # Log cycle transitions.
      if pos == 0 and batch_num > 0:
        cycle_num = batch_num // cycle_len
        _LOGGER.info('── Cycle %d ──', cycle_num + 1)

      bq_wrapper.insert_rows(args.project, args.dataset, args.table, rows)
      total_rows += len(rows)
      _LOGGER.info(
          'Batch %d [%s]: inserted %d rows (total: %d)',
          batch_num,
          label,
          len(rows),
          total_rows)

      batch_num += 1
      next_time += args.interval_sec
      sleep_for = next_time - time.monotonic()
      if sleep_for > 0:
        time.sleep(sleep_for)
  except KeyboardInterrupt:
    _LOGGER.info('Interrupted by user')

  _LOGGER.info('Done. Inserted %d rows in %d batches.', total_rows, batch_num)


if __name__ == '__main__':
  main()
