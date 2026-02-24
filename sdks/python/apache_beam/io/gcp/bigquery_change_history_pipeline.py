#!/usr/bin/env python

"""Test pipeline for ReadBigQueryChangeHistory.

Runs ReadBigQueryChangeHistory against a table and prints every row it reads.
Use alongside bigquery_change_history_insert_data.py to test streaming.

Usage (terminal 1 — insert data):
  python -m apache_beam.io.gcp.bigquery_change_history_insert_data \
      --project=apache-beam-testing --dataset=cdc --table=ch_demo \
      --rows_per_batch=3 --interval_sec=5

Usage (terminal 2 — DirectRunner, for local testing):
  python -m apache_beam.io.gcp.bigquery_change_history_pipeline \
      --project=apache-beam-testing --dataset=cdc --table=ch_demo \
      --poll_interval_sec=30 --change_function=APPENDS \
      --runner=DirectRunner

Usage (terminal 2 — DataflowRunner):
  python -m apache_beam.io.gcp.bigquery_change_history_pipeline \
      --project=apache-beam-testing --dataset=cdc --table=ch_demo \
      --poll_interval_sec=30 --change_function=APPENDS \
      --runner=DataflowRunner \
      --region=us-central1 \
      --temp_location=gs://apache-beam-testing-temp/staging \
      --staging_location=gs://apache-beam-testing-temp/staging \
      --setup_file=./setup.py

The pipeline will run until stop_time is reached (default: 5 minutes from now)
or until you Ctrl-C / drain the Dataflow job.
"""

import argparse
import logging
import time

import apache_beam as beam
from apache_beam.io.gcp.bigquery_change_history import ReadBigQueryChangeHistory
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import WorkerOptions

_LOGGER = logging.getLogger(__name__)


class LogRow(beam.DoFn):
  """DoFn that logs each row to WARNING so it appears in Dataflow logs."""
  def __init__(self):
    self._count = 0

  def process(self, row):
    self._count += 1
    row_id = row.get('id', '?')
    name = row.get('name', '?')
    value = row.get('value', '?')
    change_type = row.get('change_type', '?')
    change_ts = row.get('change_timestamp', '?')
    _LOGGER.warning(
        'ROW [%s] id=%s name=%s value=%s change_ts=%s',
        change_type,
        row_id,
        name,
        value,
        change_ts)
    yield row


def main():
  parser = argparse.ArgumentParser(
      description='Run ReadBigQueryChangeHistory and print rows.')
  parser.add_argument(
      '--project', default='apache-beam-testing', help='GCP project ID')
  parser.add_argument('--dataset', default='cdc', help='BigQuery dataset')
  parser.add_argument('--table', default='ch_demo', help='BigQuery table name')
  parser.add_argument(
      '--poll_interval_sec', type=int, default=30, help='Seconds between polls')
  parser.add_argument(
      '--change_function',
      default='APPENDS',
      choices=['APPENDS', 'CHANGES'],
      help='BQ change function to use')
  parser.add_argument(
      '--buffer_sec',
      type=float,
      default=10.0,
      help='Safety buffer behind now() in seconds')
  parser.add_argument(
      '--duration_sec',
      type=float,
      default=3000.0,
      help='How long to run (seconds). Default 300 (5 min)')
  parser.add_argument(
      '--start_offset_sec',
      type=float,
      default=60.0,
      help='Start reading from this many seconds ago')
  parser.add_argument(
      '--temp_dataset', default='beam_ch_temp', help='Dataset for temp tables')
  # Runner and Dataflow-specific args
  parser.add_argument(
      '--runner',
      default='PrismRunner',
      help='Pipeline runner (DirectRunner or DataflowRunner)')
  parser.add_argument('--region', default='us-central1', help='Dataflow region')
  parser.add_argument(
      '--temp_location',
      default=None,
      help='GCS temp location (e.g. gs://bucket/temp)')
  parser.add_argument(
      '--staging_location',
      default=None,
      help='GCS staging location (e.g. gs://bucket/staging)')
  parser.add_argument(
      '--num_workers', type=int, default=1, help='Number of Dataflow workers')
  parser.add_argument(
      '--max_num_workers',
      type=int,
      default=2,
      help='Max number of Dataflow workers')
  parser.add_argument(
      '--machine_type', default='n1-standard-2', help='Worker machine type')
  parser.add_argument('--job_name', default=None, help='Dataflow job name')
  args, pipeline_args = parser.parse_known_args()

  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s %(levelname)s %(name)s: %(message)s')

  table = f'{args.project}:{args.dataset}.{args.table}'
  start_time = time.time() - args.start_offset_sec
  stop_time = time.time() + args.duration_sec

  _LOGGER.info('Starting ReadBigQueryChangeHistory pipeline')
  _LOGGER.info('  Runner: %s', args.runner)
  _LOGGER.info('  Table: %s', table)
  _LOGGER.info('  Change function: %s', args.change_function)
  _LOGGER.info('  Poll interval: %d sec', args.poll_interval_sec)
  _LOGGER.info('  Buffer: %.1f sec', args.buffer_sec)
  _LOGGER.info('  Duration: %.0f sec', args.duration_sec)

  # Build pipeline options
  options = PipelineOptions(pipeline_args)
  options.view_as(StandardOptions).runner = args.runner
  options.view_as(StandardOptions).streaming = True
  options.view_as(GoogleCloudOptions).project = args.project

  if args.runner == 'DataflowRunner':
    options.view_as(GoogleCloudOptions).region = args.region
    if args.temp_location:
      options.view_as(GoogleCloudOptions).temp_location = args.temp_location
    if args.staging_location:
      options.view_as(
          GoogleCloudOptions).staging_location = args.staging_location
    if args.job_name:
      options.view_as(GoogleCloudOptions).job_name = args.job_name
    options.view_as(WorkerOptions).num_workers = args.num_workers
    options.view_as(WorkerOptions).max_num_workers = args.max_num_workers
    options.view_as(WorkerOptions).machine_type = args.machine_type

  with beam.Pipeline(options=options) as p:
    rows = (
        p
        | ReadBigQueryChangeHistory(
            table=table,
            poll_interval_sec=args.poll_interval_sec,
            start_time=start_time,
            stop_time=stop_time,
            change_function=args.change_function,
            buffer_sec=args.buffer_sec,
            project=args.project,
            temp_dataset=args.temp_dataset,
            batch_arrow_read=True,
            trace=True))

    # Log every row
    _ = rows | 'LogRows' >> beam.ParDo(LogRow())

  _LOGGER.info('Pipeline complete.')


if __name__ == '__main__':
  _LOGGER.setLevel(logging.INFO)
  main()
