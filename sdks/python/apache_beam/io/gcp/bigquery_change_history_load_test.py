#!/usr/bin/env python

"""Load test framework for ReadBigQueryChangeHistory.

Runs a single Beam pipeline with two branches:
  - Writer: PeriodicImpulse -> generate rows -> WriteToBigQuery (source table)
  - Reader: ReadBigQueryChangeHistory (source table) -> WriteToBigQuery (sink)

After the pipeline completes, validates correctness via SQL queries comparing
source vs sink tables (row counts, missing IDs, duplicates, extras).

Supports preset profiles for convenience and full CLI configurability.

Usage (small profile, PrismRunner):
  python -m apache_beam.io.gcp.bigquery_change_history_load_test \
      --project=apache-beam-testing --dataset=cdc --profile=small

Usage (medium profile, DataflowRunner):
  python -m apache_beam.io.gcp.bigquery_change_history_load_test \
      --project=apache-beam-testing --dataset=cdc --profile=medium \
      --temp_location=gs://apache-beam-testing-temp/staging \
      --staging_location=gs://apache-beam-testing-temp/staging \
      --sdk_location=dist/apache_beam-2.72.0.dev0.tar.gz

Usage (custom):
  python -m apache_beam.io.gcp.bigquery_change_history_load_test \
      --project=apache-beam-testing --dataset=cdc \
      --rows_per_impulse=20 --insert_interval_sec=2.0 \
      --poll_interval_sec=30 --duration_sec=600 \
      --runner=PrismRunner
"""

import argparse
import logging
import random
import sys
import time
import uuid

import apache_beam as beam
from apache_beam.io.gcp.bigquery_change_history import ReadBigQueryChangeHistory
from apache_beam.io.gcp.bigquery_tools import BigQueryWrapper
from apache_beam.io.gcp.internal.clients import bigquery
from apache_beam.options.pipeline_options import GoogleCloudOptions
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.options.pipeline_options import WorkerOptions
from apache_beam.transforms.periodicsequence import PeriodicImpulse

_LOGGER = logging.getLogger(__name__)

# Preset profiles: name -> config overrides
PROFILES = {
    'small': {
        'rows_per_impulse': 5,
        'insert_interval_sec': 2.0,
        'poll_interval_sec': 60,
        'buffer_sec': 10,
        'payload_bytes': 0,
        'duration_sec': 180,
        'drain_polls': 2,
        'runner': 'PrismRunner',
        'write_method': 'STREAMING_INSERTS',
    },
    'medium': {
        'rows_per_impulse': 500,
        'insert_interval_sec': 0.5,
        'poll_interval_sec': 90,
        'buffer_sec': 10,
        'payload_bytes': 10240,
        'duration_sec': 900,
        'drain_polls': 8,
        'runner': 'DataflowRunner',
        'write_method': 'STORAGE_WRITE_API',
    },
    'large': {
        'rows_per_impulse': 100,
        'insert_interval_sec': 0.5,
        'poll_interval_sec': 60,
        'buffer_sec': 10,
        'payload_bytes': 4096,
        'duration_sec': 3600,
        'drain_polls': 8,
        'runner': 'DataflowRunner',
        'write_method': 'STORAGE_WRITE_API',
    },
}

SOURCE_SCHEMA = 'id:INTEGER,name:STRING,value:FLOAT,payload:STRING'
SINK_SCHEMA = ('id:INTEGER,name:STRING,value:FLOAT,payload:STRING,'
               'change_type:STRING,change_timestamp:TIMESTAMP')


def _coerce_datetimes(row):
  """Convert datetime.datetime values to Beam Timestamps for BQ write.

  Arrow's .as_py() returns datetime.datetime for TIMESTAMP columns, but
  StorageWriteToBigQuery's RowCoder expects objects with a .micros attribute
  (Beam Timestamp). Convert datetime fields so the RowCoder can encode them.
  """
  import datetime
  from apache_beam.utils.timestamp import Timestamp
  out = {}
  for k, v in row.items():
    if isinstance(v, datetime.datetime):
      out[k] = Timestamp.from_utc_datetime(v.replace(tzinfo=datetime.timezone.utc)
                                           if v.tzinfo is None else v)
    else:
      out[k] = v
  return out


# =============================================================================
# Table setup
# =============================================================================

def create_source_table(bq_wrapper, project, dataset, table):
  """Create a change-history-enabled source table via DDL."""
  ddl = (
      f'CREATE TABLE IF NOT EXISTS '
      f'`{project}.{dataset}.{table}` '
      f'(id INT64, name STRING, value FLOAT64, payload STRING) '
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
      'Source table %s.%s.%s created (with change history enabled)',
      project, dataset, table)


# =============================================================================
# Row generation (stateless — IDs derived from impulse timestamp)
# =============================================================================

def generate_rows(
    impulse_ts, start_time, interval, rows_per_impulse, payload_bytes):
  """Generate rows with deterministic IDs based on impulse position.

  Each impulse timestamp maps to a unique batch number, and each batch
  produces a contiguous range of IDs. This avoids the need for state
  and allows PeriodicImpulse to split freely across workers.

  payload_bytes controls the size of a random string padding field to
  increase row size for exercising multi-stream reads.
  """
  batch_num = int(round((impulse_ts - start_time) / interval))
  base_id = batch_num * rows_per_impulse
  for i in range(rows_per_impulse):
    row_id = base_id + i
    row = {
        'id': row_id,
        'name': f'row_{row_id}_{uuid.uuid4().hex[:4]}',
        'value': round(random.uniform(0, 100), 2),
        'payload': uuid.uuid4().hex * (payload_bytes // 32 + 1)
                   if payload_bytes > 0 else '',
    }
    yield row


# =============================================================================
# Validation queries
# =============================================================================

def run_validation_query(bq_wrapper, project, sql):
  """Execute a validation SQL query and return result rows."""
  job_id = f'beam_ch_validate_{uuid.uuid4().hex[:8]}'
  reference = bigquery.JobReference(jobId=job_id, projectId=project)
  request = bigquery.BigqueryJobsInsertRequest(
      projectId=project,
      job=bigquery.Job(
          configuration=bigquery.JobConfiguration(
              query=bigquery.JobConfigurationQuery(
                  query=sql, useLegacySql=False)),
          jobReference=reference))
  response = bq_wrapper._start_job(request)
  bq_wrapper.wait_for_bq_job(response.jobReference, sleep_duration_sec=2)

  # Read results — pass location from job response so BQ can find the job
  job_ref = response.jobReference
  response = bq_wrapper._get_query_results(
      project, job_ref.jobId, location=job_ref.location)
  return response.rows if response.rows else []


def _cell_val(cell):
  """Extract the string value from a BQ response cell."""
  v = cell.v
  return v.string_value if hasattr(v, 'string_value') else str(v)


def validate_results(
    bq_wrapper, project, source_table, sink_table,
    start_time=None, reader_stop_time=None):
  """Run all validation queries and return a results dict.

  Args:
    bq_wrapper: BigQueryWrapper instance.
    project: GCP project ID.
    source_table: Fully qualified source table (project.dataset.table).
    sink_table: Fully qualified sink table (project.dataset.table).
    start_time: Pipeline start time (epoch seconds), for APPENDS() diagnostic.
    reader_stop_time: Reader stop time (epoch seconds), for APPENDS() diag.

  Returns:
    Dict with keys: source_count, sink_count, missing_count, duplicate_count,
    extra_count, appends_count, all_pass.
  """
  results = {}

  # a) Row counts
  count_sql = (
      f'SELECT '
      f'(SELECT COUNT(*) FROM `{source_table}`) AS source_count, '
      f'(SELECT COUNT(*) FROM `{sink_table}`) AS sink_count')
  _LOGGER.info('Running row count query...')
  count_rows = run_validation_query(bq_wrapper, project, count_sql)
  if count_rows:
    row = count_rows[0]
    results['source_count'] = int(_cell_val(row.f[0]))
    results['sink_count'] = int(_cell_val(row.f[1]))
  else:
    results['source_count'] = 0
    results['sink_count'] = 0

  # b) Missing IDs (in source but not sink) — count + sample
  # Use COUNT for accuracy (avoids pagination limits of EXCEPT DISTINCT).
  missing_count_sql = (
      f'SELECT COUNT(*) FROM ('
      f'SELECT id FROM `{source_table}` '
      f'EXCEPT DISTINCT '
      f'SELECT id FROM `{sink_table}`)')
  _LOGGER.info('Running missing IDs query...')
  missing_count_rows = run_validation_query(
      bq_wrapper, project, missing_count_sql)
  results['missing_count'] = (
      int(_cell_val(missing_count_rows[0].f[0]))
      if missing_count_rows else 0)
  # Get a small sample for debugging
  missing_sample_sql = (
      f'SELECT id FROM ('
      f'SELECT id FROM `{source_table}` '
      f'EXCEPT DISTINCT '
      f'SELECT id FROM `{sink_table}`) ORDER BY id LIMIT 10')
  missing_sample_rows = run_validation_query(
      bq_wrapper, project, missing_sample_sql)
  results['missing_sample'] = [
      int(_cell_val(r.f[0]))
      for r in missing_sample_rows] if missing_sample_rows else []

  # c) Duplicate IDs in sink
  dup_count_sql = (
      f'SELECT COUNT(*) FROM ('
      f'SELECT id FROM `{sink_table}` GROUP BY id HAVING COUNT(*) > 1)')
  _LOGGER.info('Running duplicate IDs query...')
  dup_count_rows = run_validation_query(bq_wrapper, project, dup_count_sql)
  results['duplicate_count'] = (
      int(_cell_val(dup_count_rows[0].f[0]))
      if dup_count_rows else 0)
  dup_sample_sql = (
      f'SELECT id, COUNT(*) AS cnt '
      f'FROM `{sink_table}` GROUP BY id HAVING cnt > 1 '
      f'ORDER BY id LIMIT 10')
  dup_sample_rows = run_validation_query(bq_wrapper, project, dup_sample_sql)
  results['duplicate_sample'] = [
      (int(_cell_val(r.f[0])), int(_cell_val(r.f[1])))
      for r in dup_sample_rows] if dup_sample_rows else []

  # d) Extra IDs (in sink but not source) — count + sample
  extra_count_sql = (
      f'SELECT COUNT(*) FROM ('
      f'SELECT id FROM `{sink_table}` '
      f'EXCEPT DISTINCT '
      f'SELECT id FROM `{source_table}`)')
  _LOGGER.info('Running extra IDs query...')
  extra_count_rows = run_validation_query(
      bq_wrapper, project, extra_count_sql)
  results['extra_count'] = (
      int(_cell_val(extra_count_rows[0].f[0]))
      if extra_count_rows else 0)
  extra_sample_sql = (
      f'SELECT id FROM ('
      f'SELECT id FROM `{sink_table}` '
      f'EXCEPT DISTINCT '
      f'SELECT id FROM `{source_table}`) ORDER BY id LIMIT 10')
  extra_sample_rows = run_validation_query(
      bq_wrapper, project, extra_sample_sql)
  results['extra_sample'] = [
      int(_cell_val(r.f[0]))
      for r in extra_sample_rows] if extra_sample_rows else []

  # e) APPENDS() diagnostic — run APPENDS() now for the full reader window.
  #    Compares with source_count to detect APPENDS() visibility lag.
  results['appends_count'] = None
  if start_time is not None and reader_stop_time is not None:
    import datetime as _dt
    start_iso = _dt.datetime.fromtimestamp(
        start_time, tz=_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    end_iso = _dt.datetime.fromtimestamp(
        reader_stop_time, tz=_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    appends_sql = (
        f"SELECT COUNT(*) FROM APPENDS("
        f"TABLE `{source_table}`, "
        f"TIMESTAMP '{start_iso}', "
        f"TIMESTAMP '{end_iso}')")
    _LOGGER.info('Running APPENDS() diagnostic query...')
    _LOGGER.info('  Time range: %s to %s', start_iso, end_iso)
    try:
      appends_rows = run_validation_query(bq_wrapper, project, appends_sql)
      results['appends_count'] = (
          int(_cell_val(appends_rows[0].f[0])) if appends_rows else 0)
      _LOGGER.info('APPENDS() count: %d', results['appends_count'])
    except Exception as e:
      _LOGGER.warning('APPENDS() diagnostic query failed: %s', e)

  # Overall pass/fail
  results['all_pass'] = (
      results['source_count'] == results['sink_count'] and
      results['missing_count'] == 0 and
      results['duplicate_count'] == 0 and
      results['extra_count'] == 0)

  return results


def delete_table(bq_wrapper, project, dataset, table):
  """Delete a table, ignoring errors if it doesn't exist."""
  try:
    bq_wrapper._delete_table(project, dataset, table)
    _LOGGER.info('Deleted table %s.%s.%s', project, dataset, table)
  except Exception as e:
    _LOGGER.warning('Failed to delete %s.%s.%s: %s', project, dataset, table, e)


# =============================================================================
# Report
# =============================================================================

def print_report(args, source_fq, sink_fq, validation, duration_actual):
  """Print the final load test report."""
  print('\n' + '=' * 60)
  print('Load Test Report')
  print('=' * 60)
  print(f'  Runner:           {args.runner}')
  print(f'  Duration:         {args.duration_sec}s (actual: {duration_actual:.1f}s)')
  row_size = f'~{args.payload_bytes + 50} bytes/row' if args.payload_bytes > 0 else '~50 bytes/row'
  print(f'  Writer:           {args.rows_per_impulse} rows every '
        f'{args.insert_interval_sec}s ({args.write_method}, {row_size})')
  print(f'  Reader:           poll every {args.poll_interval_sec}s, '
        f'buffer {args.buffer_sec}s, drain {args.drain_polls} polls')
  print(f'  Sink:             {args.write_method}')
  print(f'  Source table:     {source_fq}')
  print(f'  Sink table:       {sink_fq}')
  print()

  if validation:
    src = validation['source_count']
    snk = validation['sink_count']
    count_match = src == snk
    missing_count = validation['missing_count']
    missing_sample = validation['missing_sample']
    dup_count = validation['duplicate_count']
    dup_sample = validation['duplicate_sample']
    extra_count = validation['extra_count']
    extra_sample = validation['extra_sample']

    print('  Validation:')
    print(f'    Source rows:      {src}')
    print(f'    Sink rows:        {snk}')
    print(f'    Row count match:  {"PASS" if count_match else "FAIL"}')
    print(f'    Missing IDs:      {"PASS" if missing_count == 0 else "FAIL"} '
          f'({missing_count})')
    if missing_sample:
      print(f'      First 10: {missing_sample}')
    print(f'    Duplicate IDs:    {"PASS" if dup_count == 0 else "FAIL"} '
          f'({dup_count})')
    if dup_sample:
      print(f'      First 10: {dup_sample}')
    print(f'    Extra IDs:        {"PASS" if extra_count == 0 else "FAIL"} '
          f'({extra_count})')
    if extra_sample:
      print(f'      First 10: {extra_sample}')

    # Flush diagnostics
    flush_diag = validation.get('flush_diag', {})
    appends_count = validation.get('appends_count')
    if flush_diag or appends_count is not None:
      print()
      print('  Flush Diagnostics:')
      imm = flush_diag.get('source_count_immediate')
      if imm is not None:
        unflushed = src - imm
        print(f'    Source rows (immediate):  {imm}'
              f'  ({unflushed} unflushed at pipeline end)'
              if unflushed > 0 else
              f'    Source rows (immediate):  {imm}'
              f'  (all flushed at pipeline end)')
      if appends_count is not None:
        gap = src - appends_count
        print(f'    APPENDS() count (post-settle): {appends_count}'
              f'  ({gap} not visible in APPENDS)'
              if gap > 0 else
              f'    APPENDS() count (post-settle): {appends_count}'
              f'  (all visible in APPENDS)')
      # Interpretation
      if imm is not None and appends_count is not None:
        print()
        if imm < src and appends_count >= src:
          print('    -> Writer-side delay: Dataflow had not flushed all rows')
          print('       to BQ when pipeline ended. APPENDS() sees them now.')
        elif imm >= src and appends_count < src:
          print('    -> APPENDS() visibility lag: all rows committed to BQ')
          print('       but APPENDS() TVF does not yet show them.')
        elif imm < src and appends_count < src:
          print('    -> Both: writer-side flush delay AND APPENDS() lag.')
        else:
          print('    -> No flush delay detected at validation time.')

    print()
    print(f'  RESULT: {"PASS" if validation["all_pass"] else "FAIL"}')
  else:
    print('  Validation: SKIPPED')

  print('=' * 60)


# =============================================================================
# Main
# =============================================================================

def parse_args():
  parser = argparse.ArgumentParser(
      description='Load test for ReadBigQueryChangeHistory.')

  # Core
  parser.add_argument(
      '--project', default='apache-beam-testing', help='GCP project ID')
  parser.add_argument('--dataset', default='cdc', help='BigQuery dataset')
  parser.add_argument(
      '--profile',
      choices=list(PROFILES.keys()),
      default=None,
      help='Preset profile (overrides individual args)')

  # Writer config
  parser.add_argument(
      '--rows_per_impulse', type=int, default=3,
      help='Rows generated per PeriodicImpulse tick')
  parser.add_argument(
      '--insert_interval_sec', type=float, default=5.0,
      help='Seconds between writer impulses')
  parser.add_argument(
      '--payload_bytes', type=int, default=0,
      help='Size of random string padding per row (bytes). '
           'Increases row size for exercising multi-stream reads.')
  parser.add_argument(
      '--write_method', default='STREAMING_INSERTS',
      choices=['STREAMING_INSERTS', 'STORAGE_WRITE_API'],
      help='WriteToBigQuery method for writer branch')

  # Reader config
  parser.add_argument(
      '--poll_interval_sec', type=int, default=30,
      help='Seconds between CDC polls')
  parser.add_argument(
      '--buffer_sec', type=float, default=10.0,
      help='Safety buffer behind now() in seconds')
  parser.add_argument(
      '--temp_dataset', default='beam_ch_temp',
      help='Dataset for CDC temp tables')

  # Duration
  parser.add_argument(
      '--duration_sec', type=float, default=120.0,
      help='How long to run (seconds)')
  parser.add_argument(
      '--drain_polls', type=int, default=2,
      help='Extra poll cycles after writer stops for reader to drain. '
           'Dataflow needs more (8+) due to startup delay cascading backlog.')

  # Runner and Dataflow
  parser.add_argument(
      '--runner', default='PrismRunner',
      help='Pipeline runner')
  parser.add_argument('--region', default='us-central1', help='Dataflow region')
  parser.add_argument('--temp_location', default=None, help='GCS temp location')
  parser.add_argument(
      '--staging_location', default=None, help='GCS staging location')
  # parser.add_argument('--sdk_location', default=None, help='SDK tarball location')
  parser.add_argument(
      '--num_workers', type=int, default=1, help='Number of Dataflow workers')
  parser.add_argument(
      '--max_num_workers', type=int, default=4,
      help='Max number of Dataflow workers')
  parser.add_argument(
      '--machine_type', default='n1-standard-2', help='Worker machine type')
  parser.add_argument('--job_name', default=None, help='Dataflow job name')

  # Control
  parser.add_argument(
      '--keep_tables', action='store_true',
      help='Do not delete test tables after run')
  parser.add_argument(
      '--skip_validation', action='store_true',
      help='Skip SQL validation (just run the pipeline)')
  parser.add_argument(
      '--trace', action='store_true',
      help='Enable detailed CDC pipeline trace logging')

  args, pipeline_args = parser.parse_known_args()

  # Apply profile overrides (only for args not explicitly set on CLI)
  if args.profile:
    cli_text = ' '.join(sys.argv[1:])
    profile = PROFILES[args.profile]
    for key, value in profile.items():
      if f'--{key}' not in cli_text:
        setattr(args, key, value)
    _LOGGER.info('Applied profile "%s": %s', args.profile, profile)

  return args, pipeline_args


def main():
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s %(levelname)s %(name)s: %(message)s')

  args, pipeline_args = parse_args()


  # Generate unique table names for this run
  run_id = uuid.uuid4().hex[:8]
  source_table_name = f'ch_load_src_{run_id}'
  sink_table_name = f'ch_load_sink_{run_id}'
  source_fq = f'{args.project}.{args.dataset}.{source_table_name}'
  sink_fq = f'{args.project}.{args.dataset}.{sink_table_name}'
  # Beam uses project:dataset.table format
  source_beam = f'{args.project}:{args.dataset}.{source_table_name}'
  sink_beam = f'{args.project}:{args.dataset}.{sink_table_name}'

  _LOGGER.info('Run ID: %s', run_id)
  _LOGGER.info('Source table: %s', source_fq)
  _LOGGER.info('Sink table: %s', sink_fq)
  _LOGGER.info('Runner: %s', args.runner)
  _LOGGER.info(
      'Writer: %d rows every %.1fs (%s)',
      args.rows_per_impulse, args.insert_interval_sec, args.write_method)
  _LOGGER.info(
      'Reader: poll every %ds, buffer %.1fs',
      args.poll_interval_sec, args.buffer_sec)
  _LOGGER.info('Duration: %.0fs', args.duration_sec)

  # Create source table with change history enabled
  _LOGGER.info('Creating source table...')
  bq_wrapper = BigQueryWrapper()
  create_source_table(
      bq_wrapper, args.project, args.dataset, source_table_name)

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
      options.view_as(GoogleCloudOptions).staging_location = args.staging_location
    if args.job_name:
      options.view_as(GoogleCloudOptions).job_name = args.job_name
    options.view_as(WorkerOptions).num_workers = args.num_workers
    options.view_as(WorkerOptions).max_num_workers = args.max_num_workers
    options.view_as(WorkerOptions).machine_type = args.machine_type
    # if args.sdk_location:
    #   options.view_as(WorkerOptions).sdk_location = args.sdk_location

  # Run the pipeline
  # Writer and reader use different stop times: the reader needs extra time
  # after the writer finishes to drain remaining data.
  #
  # On Dataflow, startup delay (5-10 min) causes the first poll to cover a
  # huge backlog. Each BQ APPENDS() query for that backlog takes minutes,
  # creating a cascading delay. Extra drain polls account for this.
  start_time = time.time()
  writer_stop_time = start_time + args.duration_sec
  reader_stop_time = (writer_stop_time +
                      args.drain_polls * args.poll_interval_sec +
                      args.buffer_sec)

  _LOGGER.info(
      'Launching pipeline (start=%.1f, writer_stop=%.1f, reader_stop=%.1f)...',
      start_time, writer_stop_time, reader_stop_time)

  pipeline_start = time.time()
  with beam.Pipeline(options=options) as p:
    # Branch 1: Writer — insert data into source table
    _ = (
        p
        | 'WriteImpulse' >> PeriodicImpulse(
            start_timestamp=start_time,
            stop_timestamp=writer_stop_time,
            fire_interval=args.insert_interval_sec)
        | 'GenerateRows' >> beam.FlatMap(
            generate_rows,
            start_time=start_time,
            interval=args.insert_interval_sec,
            rows_per_impulse=args.rows_per_impulse,
            payload_bytes=args.payload_bytes)
        | 'WriteSource' >> beam.io.WriteToBigQuery(
            table=source_beam,
            method=args.write_method,
            schema=SOURCE_SCHEMA,
            create_disposition=beam.io.BigQueryDisposition.CREATE_NEVER))

    # Branch 2: Reader — read changes and write to sink
    # Reader runs longer than writer to drain remaining data.
    _ = (
        p
        | ReadBigQueryChangeHistory(
            table=source_beam,
            poll_interval_sec=args.poll_interval_sec,
            start_time=start_time,
            stop_time=reader_stop_time,
            change_function='APPENDS',
            buffer_sec=args.buffer_sec,
            project=args.project,
            temp_dataset=args.temp_dataset,
            trace=True)
        | 'CoerceDatetimes' >> beam.Map(_coerce_datetimes)
        | 'WriteSink' >> beam.io.WriteToBigQuery(
            table=sink_beam,
            method=args.write_method,
            schema=SINK_SCHEMA,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND))

  pipeline_end = time.time()
  duration_actual = pipeline_end - pipeline_start
  _LOGGER.info('Pipeline completed in %.1f seconds.', duration_actual)

  # Flush diagnostics: query source table immediately after pipeline ends
  # (before any settle wait) to see if Dataflow finished flushing all rows.
  flush_diag = {}
  if not args.skip_validation:
    _LOGGER.info('Querying source table immediately (no settle wait)...')
    immediate_sql = f'SELECT COUNT(*) FROM `{source_fq}`'
    immediate_rows = run_validation_query(bq_wrapper, args.project, immediate_sql)
    flush_diag['source_count_immediate'] = (
        int(_cell_val(immediate_rows[0].f[0])) if immediate_rows else 0)
    _LOGGER.info(
        'Immediate source count: %d', flush_diag['source_count_immediate'])

  # Validation
  validation = None
  if not args.skip_validation:
    settle_sec = args.buffer_sec + 30
    _LOGGER.info(
        'Waiting %.0fs for data to settle before validation...', settle_sec)
    time.sleep(settle_sec)

    _LOGGER.info('Running validation queries...')
    validation = validate_results(
        bq_wrapper, args.project, source_fq, sink_fq,
        start_time=start_time, reader_stop_time=reader_stop_time)
    validation['flush_diag'] = flush_diag

  # Report
  print_report(args, source_fq, sink_fq, validation, duration_actual)

  # Print gcloud commands for DataflowRunner
  if args.runner == 'DataflowRunner' and args.job_name:
    print()
    print('Dataflow commands:')
    print(f'  gcloud dataflow jobs describe {args.job_name} '
          f'--project={args.project} --region={args.region}')
    print(f'  gcloud dataflow metrics list {args.job_name} '
          f'--project={args.project} --region={args.region}')

  # Cleanup
  if not args.keep_tables:
    _LOGGER.info('Cleaning up test tables...')
    delete_table(bq_wrapper, args.project, args.dataset, source_table_name)
    delete_table(bq_wrapper, args.project, args.dataset, sink_table_name)
  else:
    _LOGGER.info(
        'Keeping tables (--keep_tables). Clean up manually:\n'
        '  bq rm -f %s.%s.%s\n'
        '  bq rm -f %s.%s.%s',
        args.project, args.dataset, source_table_name,
        args.project, args.dataset, sink_table_name)

  if validation and not validation['all_pass']:
    return 1
  return 0


if __name__ == '__main__':
  _LOGGER.setLevel(logging.INFO)
  exit(main())
