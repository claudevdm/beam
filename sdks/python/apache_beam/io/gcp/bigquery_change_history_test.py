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

"""Tests for BigQuery change history streaming source.

Unit tests (no GCP):
  - build_changes_query format
  - compute_ranges chunking
  - _table_key conversion
  - ReadBigQueryChangeHistory validation

Integration tests (real GCP, project=dataflow-twest, dataset=cdc):
  - Stage 3: cleanup DoFn deletes real temp tables
  - Stage 2: SDF reads real temp tables via Storage Read API
  - Stage 1: poll DoFn executes real APPENDS queries
  - End-to-end: write + read streaming pipeline
"""

import datetime
import logging
import secrets
import time
import unittest
import uuid

import apache_beam as beam
from apache_beam.io.gcp.bigquery_change_history import ReadBigQueryChangeHistory
from apache_beam.io.gcp.bigquery_change_history import _CleanupTempTablesFn
from apache_beam.io.gcp.bigquery_change_history import _PollChangeHistoryFn
from apache_beam.io.gcp.bigquery_change_history import _QueryResult
from apache_beam.io.gcp.bigquery_change_history import _ReadStorageStreamsSDF
from apache_beam.io.gcp.bigquery_change_history import _table_key
from apache_beam.io.gcp.bigquery_change_history import build_changes_query
from apache_beam.io.gcp.bigquery_change_history import compute_ranges
from apache_beam.io.gcp.bigquery_tools import BigQueryWrapper
from apache_beam.io.gcp.internal.clients import bigquery
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

try:
  from apitools.base.py.exceptions import HttpError
except ImportError:
  HttpError = None

_LOGGER = logging.getLogger(__name__)

# GCP test configuration
PROJECT = 'dataflow-twest'
DATASET = 'cdc'

# =============================================================================
# Unit Tests (no GCP dependencies)
# =============================================================================


class BuildChangesQueryTest(unittest.TestCase):
  """Tests for build_changes_query()."""
  def test_appends_query_format(self):
    # Use UTC-aware datetimes to avoid timezone offset issues
    ts_start = datetime.datetime(
        2025, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    ts_end = datetime.datetime(
        2025, 1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    sql = build_changes_query(
        'myproject.mydataset.mytable', ts_start, ts_end, 'APPENDS')
    self.assertIn('APPENDS', sql)
    self.assertIn('TABLE `myproject.mydataset.mytable`', sql)
    self.assertIn('2025-01-01T00:00:00', sql)
    self.assertIn('2025-01-01T01:00:00', sql)

  def test_changes_query_format(self):
    ts_start = datetime.datetime(
        2025, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    ts_end = datetime.datetime(
        2025, 6, 15, 18, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    sql = build_changes_query('proj.ds.tbl', ts_start, ts_end, 'CHANGES')
    self.assertIn('CHANGES', sql)
    self.assertIn('TABLE `proj.ds.tbl`', sql)

  def test_colon_normalized_to_dot(self):
    ts_start = datetime.datetime(
        2025, 1, 1, tzinfo=datetime.timezone.utc).timestamp()
    ts_end = datetime.datetime(
        2025, 1, 2, tzinfo=datetime.timezone.utc).timestamp()
    sql = build_changes_query(
        'myproject:mydataset.mytable', ts_start, ts_end, 'APPENDS')
    self.assertIn('TABLE `myproject.mydataset.mytable`', sql)
    # Verify colon in table ref is normalized (timestamps contain colons)
    table_part = sql.split('TABLE')[1].split(',')[0]
    self.assertNotIn(':', table_part)


class ComputeRangesTest(unittest.TestCase):
  """Tests for compute_ranges()."""
  def test_appends_single_range(self):
    """APPENDS has no chunking — returns single range even for multi-day."""
    start = 0.0
    end = 86400.0 * 5  # 5 days
    ranges = compute_ranges(start, end, 'APPENDS')
    self.assertEqual(len(ranges), 1)
    self.assertEqual(ranges[0], (start, end))

  def test_changes_single_day(self):
    """CHANGES within 1 day: single range."""
    start = 0.0
    end = 86400.0  # exactly 1 day
    ranges = compute_ranges(start, end, 'CHANGES')
    self.assertEqual(len(ranges), 1)
    self.assertEqual(ranges[0], (start, end))

  def test_changes_multi_day(self):
    """CHANGES spanning 3 days: should chunk into 3 ranges."""
    start = 0.0
    end = 86400.0 * 3  # 3 days
    ranges = compute_ranges(start, end, 'CHANGES')
    self.assertEqual(len(ranges), 3)
    # Verify no gaps
    for i in range(len(ranges) - 1):
      self.assertEqual(ranges[i][1], ranges[i + 1][0])
    self.assertEqual(ranges[0][0], start)
    self.assertEqual(ranges[-1][1], end)

  def test_changes_partial_day(self):
    """CHANGES spanning 1.5 days: should chunk into 2 ranges."""
    start = 0.0
    end = 86400.0 * 1.5
    ranges = compute_ranges(start, end, 'CHANGES')
    self.assertEqual(len(ranges), 2)
    self.assertEqual(ranges[0], (0.0, 86400.0))
    self.assertEqual(ranges[1], (86400.0, end))

  def test_zero_range(self):
    """end <= start: empty list."""
    self.assertEqual(compute_ranges(100.0, 100.0, 'CHANGES'), [])
    self.assertEqual(compute_ranges(100.0, 50.0, 'CHANGES'), [])
    self.assertEqual(compute_ranges(100.0, 100.0, 'APPENDS'), [])

  def test_exact_day_boundary(self):
    """Exactly 2 days: should produce 2 chunks."""
    start = 0.0
    end = 86400.0 * 2
    ranges = compute_ranges(start, end, 'CHANGES')
    self.assertEqual(len(ranges), 2)


class TableKeyTest(unittest.TestCase):
  """Tests for _table_key()."""
  def test_conversion(self):
    ref = bigquery.TableReference(
        projectId='proj', datasetId='ds', tableId='tbl')
    self.assertEqual(_table_key(ref), 'proj.ds.tbl')


class ValidationTest(unittest.TestCase):
  """Tests for ReadBigQueryChangeHistory validation."""
  def test_invalid_change_function(self):
    with self.assertRaises(ValueError):
      ReadBigQueryChangeHistory(table='p:d.t', change_function='INVALID')

  def test_invalid_poll_interval(self):
    with self.assertRaises(ValueError):
      ReadBigQueryChangeHistory(table='p:d.t', poll_interval_sec=0)
    with self.assertRaises(ValueError):
      ReadBigQueryChangeHistory(table='p:d.t', poll_interval_sec=-1)

  def test_default_buffer_changes(self):
    t = ReadBigQueryChangeHistory(table='p:d.t', change_function='CHANGES')
    self.assertEqual(t._buffer_sec, 600)

  def test_default_buffer_appends(self):
    t = ReadBigQueryChangeHistory(table='p:d.t', change_function='APPENDS')
    self.assertEqual(t._buffer_sec, 15)


# =============================================================================
# Integration Tests (real GCP)
# =============================================================================


class BigQueryChangeHistoryIntegrationBase(unittest.TestCase):
  """Base class for integration tests against real BigQuery.

  Uses project=dataflow-twest, dataset=cdc.
  Creates a unique temp dataset per test class for cleanup isolation.
  """
  @classmethod
  def setUpClass(cls):
    cls.project = PROJECT
    cls.dataset = DATASET
    cls.bq_wrapper = BigQueryWrapper()
    # Detect location from source dataset
    ds = cls.bq_wrapper.client.datasets.Get(
        bigquery.BigqueryDatasetsGetRequest(
            projectId=cls.project, datasetId=cls.dataset))
    cls.location = ds.location
    cls.temp_dataset = f'beam_ch_test_{int(time.time())}_{secrets.token_hex(3)}'
    cls.bq_wrapper.get_or_create_dataset(
        cls.project, cls.temp_dataset, location=cls.location)
    _LOGGER.info(
        'Created temp dataset %s:%s (location=%s)',
        cls.project,
        cls.temp_dataset,
        cls.location)

  @classmethod
  def tearDownClass(cls):
    try:
      request = bigquery.BigqueryDatasetsDeleteRequest(
          projectId=cls.project,
          datasetId=cls.temp_dataset,
          deleteContents=True)
      cls.bq_wrapper.client.datasets.Delete(request)
      _LOGGER.info('Deleted temp dataset %s', cls.temp_dataset)
    except Exception as e:
      _LOGGER.warning('Failed to clean up dataset %s: %s', cls.temp_dataset, e)

  @classmethod
  def _create_temp_table_with_data(cls, table_id, rows):
    """Create a table in the temp dataset and insert rows via streaming."""
    table_schema = bigquery.TableSchema()
    for field_name, field_type in [
        ('id', 'INTEGER'), ('name', 'STRING'), ('value', 'FLOAT')]:
      field = bigquery.TableFieldSchema()
      field.name = field_name
      field.type = field_type
      table_schema.fields.append(field)

    table = bigquery.Table(
        tableReference=bigquery.TableReference(
            projectId=cls.project, datasetId=cls.temp_dataset,
            tableId=table_id),
        schema=table_schema)
    request = bigquery.BigqueryTablesInsertRequest(
        projectId=cls.project, datasetId=cls.temp_dataset, table=table)
    cls.bq_wrapper.client.tables.Insert(request)

    # Wait for table to be visible
    cls.bq_wrapper.get_table(cls.project, cls.temp_dataset, table_id)

    if rows:
      cls.bq_wrapper.insert_rows(cls.project, cls.temp_dataset, table_id, rows)
      # Give streaming buffer time to flush
      time.sleep(5)

    return bigquery.TableReference(
        projectId=cls.project, datasetId=cls.temp_dataset, tableId=table_id)

  @classmethod
  def _create_change_history_table(cls, table_id, rows=None):
    """Create a table with enable_change_history via DDL."""
    ddl = (
        f'CREATE TABLE IF NOT EXISTS '
        f'`{cls.project}.{cls.dataset}.{table_id}` '
        f'(id INT64, name STRING, value FLOAT64) '
        f'OPTIONS (enable_change_history = true)')

    job_id = f'beam_ch_ddl_{uuid.uuid4().hex[:8]}'
    reference = bigquery.JobReference(jobId=job_id, projectId=cls.project)
    request = bigquery.BigqueryJobsInsertRequest(
        projectId=cls.project,
        job=bigquery.Job(
            configuration=bigquery.JobConfiguration(
                query=bigquery.JobConfigurationQuery(
                    query=ddl, useLegacySql=False)),
            jobReference=reference))
    response = cls.bq_wrapper._start_job(request)
    cls.bq_wrapper.wait_for_bq_job(response.jobReference, sleep_duration_sec=2)

    # Wait for table to be visible
    cls.bq_wrapper.get_table(cls.project, cls.dataset, table_id)

    if rows:
      cls.bq_wrapper.insert_rows(cls.project, cls.dataset, table_id, rows)
      time.sleep(5)

    return bigquery.TableReference(
        projectId=cls.project, datasetId=cls.dataset, tableId=table_id)

  @classmethod
  def _delete_table(cls, table_ref):
    """Delete a table, ignoring 404."""
    cls.bq_wrapper._delete_table(
        table_ref.projectId, table_ref.datasetId, table_ref.tableId)


class CleanupTempTablesFnTest(BigQueryChangeHistoryIntegrationBase):
  """Integration tests for _CleanupTempTablesFn against real BigQuery."""
  def test_single_complete_signal_deletes_table(self):
    """A single signal with streams_read == total deletes the temp table."""
    table_id = f'cleanup_test_{secrets.token_hex(4)}'
    table_ref = self._create_temp_table_with_data(
        table_id, [{
            'id': 1, 'name': 'a', 'value': 1.0
        }])
    table_key = _table_key(table_ref)

    # Feed cleanup signal: all 5 streams read out of 5
    with TestPipeline() as p:
      _ = (
          p
          | beam.Create([(table_key, (5, 5))])
          | beam.ParDo(_CleanupTempTablesFn()))

    # Verify table was deleted
    time.sleep(2)
    with self.assertRaises(Exception):
      self.bq_wrapper.get_table(self.project, self.temp_dataset, table_id)

  def test_partial_signals_then_complete(self):
    """Partial signals don't delete; final signal triggers cleanup."""
    table_id = f'cleanup_partial_{secrets.token_hex(4)}'
    table_ref = self._create_temp_table_with_data(
        table_id, [{
            'id': 1, 'name': 'a', 'value': 1.0
        }])
    table_key = _table_key(table_ref)

    # Feed two partial signals: 3/10 + 7/10 = 10/10
    with TestPipeline() as p:
      _ = (
          p
          | beam.Create([
              (table_key, (3, 10)),
              (table_key, (7, 10)),
          ])
          | beam.ParDo(_CleanupTempTablesFn()))

    time.sleep(2)
    with self.assertRaises(Exception):
      self.bq_wrapper.get_table(self.project, self.temp_dataset, table_id)

  def test_empty_result_triggers_immediately(self):
    """(0, 0) signal for empty table triggers cleanup."""
    table_id = f'cleanup_empty_{secrets.token_hex(4)}'
    table_ref = self._create_temp_table_with_data(table_id, [])
    table_key = _table_key(table_ref)

    with TestPipeline() as p:
      _ = (
          p
          | beam.Create([(table_key, (0, 0))])
          | beam.ParDo(_CleanupTempTablesFn()))

    time.sleep(2)
    with self.assertRaises(Exception):
      self.bq_wrapper.get_table(self.project, self.temp_dataset, table_id)


class ReadStorageStreamsSDFTest(BigQueryChangeHistoryIntegrationBase):
  """Integration tests for _ReadStorageStreamsSDF against real BigQuery."""
  def test_reads_rows_from_temp_table(self):
    """SDF reads rows from a real temp table via Storage Read API."""
    table_id = f'sdf_test_{secrets.token_hex(4)}'
    rows = [
        {
            'id': 1, 'name': 'alice', 'value': 10.0
        },
        {
            'id': 2, 'name': 'bob', 'value': 20.0
        },
        {
            'id': 3, 'name': 'charlie', 'value': 30.0
        },
    ]
    table_ref = self._create_temp_table_with_data(table_id, rows)

    query_result = _QueryResult(temp_table_ref=table_ref)

    with TestPipeline() as p:
      outputs = (
          p
          | beam.Create([query_result])
          | beam.ParDo(_ReadStorageStreamsSDF()).with_outputs(
              'cleanup', main='rows'))

      # Check that we get 3 rows
      row_count = (
          outputs['rows']
          | 'CountRows' >> beam.combiners.Count.Globally())
      assert_that(row_count, equal_to([3]), label='CheckRowCount')

  def test_cleanup_signal_emitted(self):
    """SDF emits cleanup signal with correct counts."""
    table_id = f'sdf_cleanup_{secrets.token_hex(4)}'
    rows = [{'id': 1, 'name': 'a', 'value': 1.0}]
    table_ref = self._create_temp_table_with_data(table_id, rows)

    query_result = _QueryResult(temp_table_ref=table_ref)

    with TestPipeline() as p:
      outputs = (
          p
          | beam.Create([query_result])
          | beam.ParDo(_ReadStorageStreamsSDF()).with_outputs(
              'cleanup', main='rows'))

      # Verify cleanup signal
      cleanup_table_keys = (
          outputs['cleanup']
          | 'ExtractKey' >> beam.Map(lambda x: x[0]))
      assert_that(
          cleanup_table_keys,
          equal_to([_table_key(table_ref)]),
          label='CheckCleanupKey')

  def test_empty_table(self):
    """Empty table produces 0 rows and cleanup signal."""
    table_id = f'sdf_empty_{secrets.token_hex(4)}'
    table_ref = self._create_temp_table_with_data(table_id, [])

    query_result = _QueryResult(temp_table_ref=table_ref)

    with TestPipeline() as p:
      outputs = (
          p
          | beam.Create([query_result])
          | beam.ParDo(_ReadStorageStreamsSDF()).with_outputs(
              'cleanup', main='rows'))

      row_count = (
          outputs['rows']
          | 'CountRows' >> beam.combiners.Count.Globally())
      assert_that(row_count, equal_to([0]), label='CheckZeroRows')


class PollChangeHistorySDFTest(BigQueryChangeHistoryIntegrationBase):
  """Integration tests for _PollChangeHistoryFn against real BigQuery."""
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    # Create a test table with APPENDS support and some data
    cls.test_table_id = f'poll_test_{secrets.token_hex(4)}'
    cls.test_table_ref = cls._create_change_history_table(
        cls.test_table_id,
        rows=[
            {
                'id': 1, 'name': 'row1', 'value': 1.0
            },
            {
                'id': 2, 'name': 'row2', 'value': 2.0
            },
        ])
    # Record time after insert for query range
    cls.insert_time = time.time()
    # Wait for streaming buffer + consistency
    time.sleep(10)

  @classmethod
  def tearDownClass(cls):
    cls._delete_table(cls.test_table_ref)
    super().tearDownClass()

  def test_poll_produces_query_result(self):
    """Triggering the poll SDF produces a _QueryResult with temp table."""
    table_str = f'{self.project}:{self.dataset}.{self.test_table_id}'
    # Use a time range covering our insert
    start_time = self.insert_time - 120  # 2 min before insert

    from apache_beam.io.gcp.bigquery_change_history import _PollConfig

    config = _PollConfig(
        table=table_str,
        project=self.project,
        change_function='APPENDS',
        buffer_sec=0,
        temp_dataset=self.temp_dataset,
        start_time=start_time,
        stop_time=time.time() + 5,
        poll_interval_sec=60)

    poll_sdf = _PollChangeHistoryFn(
        table=table_str,
        project=self.project,
        change_function='APPENDS',
        buffer_sec=0,
        temp_dataset=self.temp_dataset,
        start_time=start_time,
        stop_time=time.time() + 5,
        poll_interval_sec=60)

    with TestPipeline() as p:
      results = (p | beam.Create([config]) | beam.ParDo(poll_sdf))

      result_count = results | beam.combiners.Count.Globally()
      # Should produce at least 1 _QueryResult
      assert_that(
          result_count | beam.Map(lambda c: c >= 1),
          equal_to([True]),
          label='CheckAtLeastOneResult')


class EndToEndStreamingTest(BigQueryChangeHistoryIntegrationBase):
  """End-to-end test: all three stages wired together.

  Creates a change-history-enabled table, inserts rows, then exercises:
    Stage 1 (poll SDF) → Stage 2 (read SDF) → Stage 3 (cleanup)
  and verifies rows come through and temp tables are cleaned up.

  Uses beam.Create to feed a _PollConfig to the polling SDF. With
  stop_time set shortly in the future, the SDF polls once then stops.
  """
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.test_table_id = f'e2e_test_{secrets.token_hex(4)}'
    cls.test_table_ref = cls._create_change_history_table(
        cls.test_table_id,
        rows=[
            {
                'id': 1, 'name': 'alice', 'value': 10.0
            },
            {
                'id': 2, 'name': 'bob', 'value': 20.0
            },
            {
                'id': 3, 'name': 'charlie', 'value': 30.0
            },
        ])
    cls.insert_time = time.time()
    # Wait for streaming buffer + change history propagation
    _LOGGER.info('Waiting for streaming buffer to flush...')
    time.sleep(15)

  @classmethod
  def tearDownClass(cls):
    cls._delete_table(cls.test_table_ref)
    super().tearDownClass()

  def test_three_stages_wired_together(self):
    """All three stages wired together read inserted rows via APPENDS."""
    table_str = f'{self.project}:{self.dataset}.{self.test_table_id}'
    start_time = self.insert_time - 120  # 2 min before insert

    from apache_beam.io.gcp.bigquery_change_history import _PollConfig

    config = _PollConfig(
        table=table_str,
        project=self.project,
        change_function='APPENDS',
        buffer_sec=0,
        temp_dataset=self.temp_dataset,
        start_time=start_time,
        stop_time=time.time() + 5,
        poll_interval_sec=60)

    with TestPipeline() as p:
      # Stage 1: Poll SDF
      query_results = (
          p
          | beam.Create([config])
          | 'PollChangeHistory' >> beam.ParDo(
              _PollChangeHistoryFn(
                  table=table_str,
                  project=self.project,
                  change_function='APPENDS',
                  buffer_sec=0,
                  temp_dataset=self.temp_dataset,
                  start_time=start_time,
                  stop_time=time.time() + 5,
                  poll_interval_sec=60)))

      # Stage 2: Read via Storage Read API
      read_outputs = (
          query_results
          | 'ReadStorageStreams' >> beam.ParDo(
              _ReadStorageStreamsSDF()).with_outputs('cleanup', main='rows'))

      # Stage 3: Cleanup temp tables
      _ = (
          read_outputs['cleanup']
          | 'KeyByTable' >>
          beam.Map(lambda x: (x[0], (x[1], x[2]))).with_output_types(
              beam.typehints.Tuple[str, beam.typehints.Tuple[int, int]])
          | 'CleanupTempTables' >> beam.ParDo(_CleanupTempTablesFn()))

      # Verify rows
      row_count = (
          read_outputs['rows']
          | 'CountRows' >> beam.combiners.Count.Globally())
      assert_that(
          row_count | beam.Map(lambda c: c >= 3),
          equal_to([True]),
          label='CheckAtLeast3Rows')

      # Verify we got the expected IDs
      ids = (read_outputs['rows'] | 'ExtractIds' >> beam.Map(lambda r: r['id']))
      assert_that(
          ids | 'Distinct' >> beam.Distinct(),
          equal_to([1, 2, 3]),
          label='CheckIds')

  def test_public_api_ptransform(self):
    """ReadBigQueryChangeHistory PTransform with polling SDF."""
    table_str = f'{self.project}:{self.dataset}.{self.test_table_id}'
    start_time = self.insert_time - 120  # 2 min before insert
    stop_time = time.time() + 5  # Short run for test

    with TestPipeline() as p:
      rows = (
          p
          | ReadBigQueryChangeHistory(
              table=table_str,
              poll_interval_sec=60,
              start_time=start_time,
              stop_time=stop_time,
              change_function='APPENDS',
              buffer_sec=0,
              project=self.project,
              temp_dataset=self.temp_dataset))

      row_count = rows | beam.combiners.Count.Globally()
      assert_that(
          row_count | beam.Map(lambda c: c >= 3),
          equal_to([True]),
          label='CheckAtLeast3Rows')


# =============================================================================
# Data Insertion Utility
# =============================================================================


def insert_test_rows(project, dataset, table, n, bq_wrapper=None):
  """Insert n test rows into a BigQuery table.

  Args:
    project: GCP project ID.
    dataset: BigQuery dataset ID.
    table: BigQuery table ID.
    n: Number of rows to insert.
    bq_wrapper: Optional BigQueryWrapper instance (creates one if None).

  Returns:
    List of inserted row dicts.
  """
  if bq_wrapper is None:
    bq_wrapper = BigQueryWrapper()
  rows = [{'id': i, 'name': f'row_{i}', 'value': float(i)} for i in range(n)]
  bq_wrapper.insert_rows(project, dataset, table, rows)
  return rows


def create_change_history_table(project, dataset, table_id, bq_wrapper=None):
  """Create a table with enable_change_history via DDL.

  Args:
    project: GCP project ID.
    dataset: BigQuery dataset ID.
    table_id: Table name to create.
    bq_wrapper: Optional BigQueryWrapper instance.

  Returns:
    bigquery.TableReference for the created table.
  """
  if bq_wrapper is None:
    bq_wrapper = BigQueryWrapper()

  ddl = (
      f'CREATE TABLE IF NOT EXISTS '
      f'`{project}.{dataset}.{table_id}` '
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

  return bigquery.TableReference(
      projectId=project, datasetId=dataset, tableId=table_id)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  unittest.main()
