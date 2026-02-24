#!/usr/bin/env python

"""Insert rows periodically into a BigQuery change-history-enabled table.

Creates the table if it doesn't exist, then inserts rows at a steady rate.
Use this alongside bigquery_change_history_pipeline.py to test the streaming
ReadBigQueryChangeHistory source.

Usage:
  python -m apache_beam.io.gcp.bigquery_change_history_insert_data \
      --project=apache-beam-testing \
      --dataset=cdc \
      --table=ch_demo \
      --rows_per_batch=3 \
      --interval_sec=5 \
      --num_batches=0     # 0 = infinite

The table schema is: (id INT64, name STRING, value FLOAT64)
with enable_change_history = true.
"""

import argparse
import logging
import random
import time
import uuid

from apache_beam.io.gcp.bigquery_tools import BigQueryWrapper
from apache_beam.io.gcp.internal.clients import bigquery

_LOGGER = logging.getLogger(__name__)


def create_table_if_needed(bq_wrapper, project, dataset, table):
  """Create a change-history-enabled table via DDL if it doesn't exist."""
  ddl = (
      f'CREATE TABLE IF NOT EXISTS '
      f'`{project}.{dataset}.{table}` '
      f'(id INT64, name STRING, value FLOAT64) '
      f'OPTIONS (enable_change_history = true)')

  job_id = f'beam_ch_ddl_{uuid.uuid4().hex[:8]}'
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
  _LOGGER.info(
      'Table %s.%s.%s ready (with change history enabled)',
      project,
      dataset,
      table)


def insert_batch(
    bq_wrapper, project, dataset, table, batch_num, rows_per_batch):
  """Insert a batch of rows with incrementing IDs."""
  base_id = batch_num * rows_per_batch
  rows = []
  for i in range(rows_per_batch):
    row_id = base_id + i
    rows.append({
        'id': row_id,
        'name': f'row_{row_id}_{uuid.uuid4().hex[:4]}',
        'value': round(random.uniform(0, 100), 2),
    })

  bq_wrapper.insert_rows(project, dataset, table, rows)
  ids = [r['id'] for r in rows]
  _LOGGER.info(
      'Batch %d: inserted %d rows (ids: %s)', batch_num, len(rows), ids)
  return rows


def main():
  parser = argparse.ArgumentParser(
      description='Insert rows periodically into a BQ change-history table.')
  parser.add_argument(
      '--project', default='apache-beam-testing', help='GCP project ID')
  parser.add_argument('--dataset', default='cdc', help='BigQuery dataset')
  parser.add_argument('--table', default='ch_demo', help='BigQuery table name')
  parser.add_argument(
      '--rows_per_batch',
      type=int,
      default=3,
      help='Number of rows per insert batch')
  parser.add_argument(
      '--interval_sec', type=float, default=5.0, help='Seconds between batches')
  parser.add_argument(
      '--num_batches',
      type=int,
      default=0,
      help='Number of batches (0 = infinite)')
  args = parser.parse_args()

  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s %(levelname)s %(name)s: %(message)s')

  bq_wrapper = BigQueryWrapper()

  # Create table if needed
  create_table_if_needed(bq_wrapper, args.project, args.dataset, args.table)

  _LOGGER.info(
      'Starting periodic insert: %d rows every %.1f sec into %s.%s.%s',
      args.rows_per_batch,
      args.interval_sec,
      args.project,
      args.dataset,
      args.table)

  batch_num = 0
  total_rows = 0
  try:
    while args.num_batches == 0 or batch_num < args.num_batches:
      insert_batch(
          bq_wrapper,
          args.project,
          args.dataset,
          args.table,
          batch_num,
          args.rows_per_batch)
      total_rows += args.rows_per_batch
      batch_num += 1

      if args.num_batches == 0 or batch_num < args.num_batches:
        time.sleep(args.interval_sec)
  except KeyboardInterrupt:
    _LOGGER.info('Interrupted by user')

  _LOGGER.info('Done. Inserted %d rows in %d batches.', total_rows, batch_num)


if __name__ == '__main__':
  main()
