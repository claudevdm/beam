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

"""Streaming source for BigQuery change history (APPENDS/CHANGES functions).

This module provides ``ReadBigQueryChangeHistory``, a streaming PTransform
that continuously polls BigQuery APPENDS() or CHANGES() functions and emits
changed rows as an unbounded PCollection.

**Status: Experimental** — API may change without notice.

Usage::

    import apache_beam as beam
    from apache_beam.io.gcp.bigquery_change_history import ReadBigQueryChangeHistory

    with beam.Pipeline(options=pipeline_options) as p:
        changes = (
            p
            | ReadBigQueryChangeHistory(
                table='my-project:my_dataset.my_table',
                change_function='APPENDS',
                poll_interval_sec=60))

Architecture: Three-stage pipeline
  Stage 1: PeriodicImpulse + stateful DoFn. PeriodicImpulse fires on
           interval; stateful DoFn tracks last-queried timestamp in state,
           executes CHANGES/APPENDS query, writes results to temp table.
  Stage 2: SDF reads temp table via Storage Read API with dynamic splitting.
  Stage 3: Stateful DoFn tracks stream completion, deletes temp tables.
"""

import dataclasses
import datetime
import logging
import time
import uuid
from typing import Optional

import apache_beam as beam
from apache_beam.io.gcp import bigquery_tools
from apache_beam.io.gcp.internal.clients import bigquery
from apache_beam.io.restriction_trackers import OffsetRange
from apache_beam.io.restriction_trackers import OffsetRestrictionTracker
from apache_beam.io.watermark_estimators import ManualWatermarkEstimator
from apache_beam.metrics import Metrics
from apache_beam.transforms.core import WatermarkEstimatorProvider
from apache_beam.transforms.periodicsequence import PeriodicImpulse
from apache_beam.transforms.window import TimestampedValue
from apache_beam.utils.timestamp import MAX_TIMESTAMP
from apache_beam.utils.timestamp import Timestamp

try:
  from google.cloud import bigquery_storage_v1 as bq_storage
except ImportError:
  bq_storage = None  # type: ignore

_LOGGER = logging.getLogger(__name__)

__all__ = ['ReadBigQueryChangeHistory']

# Max time range for CHANGES() queries: 1 day in seconds.
_MAX_CHANGES_RANGE_SEC = 86400

# Side output tag for cleanup signals between Stage 2 and Stage 3.
_CLEANUP_TAG = 'cleanup'

# Default number of Storage Read API streams to request.
# Matches ReadFromBigQuery's MIN_SPLIT_COUNT to enable parallelism.
# The server may return fewer streams if the table is small.
_DEFAULT_MAX_STREAMS = 10

# =============================================================================
# Helpers and data classes
# =============================================================================


@dataclasses.dataclass
class _QueryResult:
  """Bridges Stage 1 (poll) to Stage 2 (read).

  After Stage 1 executes a CHANGES/APPENDS query, it emits a _QueryResult
  pointing to the temp table containing query results. Stage 2's SDF reads
  rows from that temp table via the Storage Read API.

  range_start/range_end define the change_timestamp window this query covers.
  Stage 2 uses range_start to set an initial watermark hold so the runner
  doesn't advance the watermark past the data's timestamps.
  """
  temp_table_ref: Optional[bigquery.TableReference] = None
  range_start: float = 0.0
  range_end: float = 0.0


class _StreamRestriction:
  """Restriction carrying BQ Storage stream names for cross-worker safety.

  Unlike a plain OffsetRange(0, N), this restriction is self-contained:
  each split carries the actual stream name strings so it can be processed
  on any worker without relying on a module-level cache.
  """
  __slots__ = ('stream_names', 'start', 'stop')

  def __init__(self, stream_names, start, stop):
    if start > stop:
      raise ValueError(
          'start must not be larger than stop. '
          'Received %d and %d respectively.' % (start, stop))
    self.stream_names = stream_names  # tuple of BQ stream name strings
    self.start = start
    self.stop = stop

  def __eq__(self, other):
    if not isinstance(other, _StreamRestriction):
      return False
    return (
        self.stream_names == other.stream_names and
        self.start == other.start and self.stop == other.stop)

  def __hash__(self):
    return hash((type(self), self.stream_names, self.start, self.stop))

  def __repr__(self):
    return (
        '_StreamRestriction(streams=%d, start=%d, stop=%d)' %
        (len(self.stream_names), self.start, self.stop))

  def split_at(self, pos):
    return (
        _StreamRestriction(self.stream_names, self.start, pos),
        _StreamRestriction(self.stream_names, pos, self.stop))

  def size(self):
    return self.stop - self.start


class _StreamRestrictionTracker(beam.io.iobase.RestrictionTracker):
  """Tracker for _StreamRestriction, delegating offset logic to
  OffsetRestrictionTracker."""
  def __init__(self, restriction):
    self._restriction = restriction
    self._offset_tracker = OffsetRestrictionTracker(
        OffsetRange(restriction.start, restriction.stop))

  def current_restriction(self):
    inner = self._offset_tracker.current_restriction()
    return _StreamRestriction(
        self._restriction.stream_names, inner.start, inner.stop)

  def try_claim(self, position):
    return self._offset_tracker.try_claim(position)

  def try_split(self, fraction_of_remainder):
    result = self._offset_tracker.try_split(fraction_of_remainder)
    if result is not None:
      primary, residual = result
      names = self._restriction.stream_names
      self._restriction = _StreamRestriction(names, primary.start, primary.stop)
      return (
          _StreamRestriction(names, primary.start, primary.stop),
          _StreamRestriction(names, residual.start, residual.stop))
    return None

  def check_done(self):
    self._offset_tracker.check_done()

  def current_progress(self):
    return self._offset_tracker.current_progress()

  def is_bounded(self):
    return True


def _table_key(table_ref):
  """Convert a TableReference to a 'project.dataset.table' string."""
  return f'{table_ref.projectId}.{table_ref.datasetId}.{table_ref.tableId}'


def build_changes_query(table, start_ts, end_ts, change_function):
  """Build a CHANGES() or APPENDS() SQL query.

  Args:
    table: Table name as 'project.dataset.table' or 'project:dataset.table'.
    start_ts: Start timestamp (float, seconds since epoch). Inclusive.
    end_ts: End timestamp (float, seconds since epoch). Exclusive.
    change_function: 'CHANGES' or 'APPENDS'.

  Returns:
    SQL string.
  """
  # Normalize 'project:dataset.table' to 'project.dataset.table'
  table = table.replace(':', '.')
  start_iso = datetime.datetime.fromtimestamp(
      start_ts, tz=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
  end_iso = datetime.datetime.fromtimestamp(
      end_ts, tz=datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
  # Pseudo-columns (_CHANGE_TYPE, _CHANGE_TIMESTAMP) can't be written to
  # destination tables with their original names. Rename them so they can
  # be persisted to the temp table for Storage Read API reading.
  pseudo_cols = '_CHANGE_TYPE, _CHANGE_TIMESTAMP'
  sql = (
      f"SELECT * EXCEPT({pseudo_cols}), "
      f"_CHANGE_TYPE AS change_type, "
      f"_CHANGE_TIMESTAMP AS change_timestamp "
      f"FROM {change_function}"
      f"(TABLE `{table}`, "
      f"TIMESTAMP '{start_iso}', "
      f"TIMESTAMP '{end_iso}')")
  return sql


def compute_ranges(start_ts, end_ts, change_function):
  """Split [start_ts, end_ts) into query-safe chunks.

  CHANGES() has a max 1-day range. APPENDS() has no limit.

  Args:
    start_ts: Start timestamp (float, seconds since epoch).
    end_ts: End timestamp (float, seconds since epoch).
    change_function: 'CHANGES' or 'APPENDS'.

  Returns:
    List of (start, end) float tuples. Empty if end_ts <= start_ts.
  """
  if end_ts <= start_ts:
    return []

  if change_function != 'CHANGES':
    return [(start_ts, end_ts)]

  # CHANGES: chunk into <=1-day ranges
  ranges = []
  current = start_ts
  while current < end_ts:
    chunk_end = min(current + _MAX_CHANGES_RANGE_SEC, end_ts)
    ranges.append((current, chunk_end))
    current = chunk_end
  return ranges


def _max_default_zero(values):
  """max() that returns 0 for empty iterables (needed for CombiningState)."""
  result = 0
  for v in values:
    if v > result:
      result = v
  return result


# =============================================================================
# Stage 1 — _PollChangeHistoryFn (Stateful DoFn)
# =============================================================================


class _PollChangeHistoryFn(beam.DoFn):
  """Stateful DoFn that polls BQ change history and emits _QueryResult elements.

  Receives (key, timestamp) elements from PeriodicImpulse + Map.
  Uses CombiningValueStateSpec to track the last queried end timestamp.
  The singleton key ensures only one instance processes at a time,
  serializing polls naturally without custom split prevention.

  On each invocation:
  1. Reads last_end_ts from state (or uses start_time if 0)
  2. Computes end_ts = now - buffer_sec
  3. If end_ts <= last_end_ts: skip (nothing new)
  4. Chunks into safe ranges (<=1 day for CHANGES)
  5. Executes each chunk as a BQ query writing to a unique temp table
  6. Yields _QueryResult per temp table
  7. Writes end_ts to state
  """
  LAST_END_TS = beam.transforms.userstate.CombiningValueStateSpec(
      'last_end_ts', _max_default_zero)

  def __init__(
      self,
      table,
      project,
      change_function,
      buffer_sec,
      temp_dataset,
      start_time,
      trace,
      location=None):
    self._table = table
    self._project = project
    self._change_function = change_function
    self._buffer_sec = buffer_sec
    self._temp_dataset = temp_dataset
    self._start_time = start_time
    self._trace = trace
    self._location = location

  def _log(self, msg, *args):
    if self._trace:
      _LOGGER.warning(msg, *args)

  def setup(self):
    self._log(
        '[Stage1-Poll] setup: creating BigQueryWrapper for table=%s, '
        'project=%s, temp_dataset=%s',
        self._table,
        self._project,
        self._temp_dataset)
    self._bq_wrapper = bigquery_tools.BigQueryWrapper()
    # Detect location from source table's dataset if not specified
    if self._location is None:
      table_normalized = self._table.replace(':', '.')
      parts = table_normalized.split('.')
      if len(parts) == 3:
        try:
          self._log(
              '[Stage1-Poll] Detecting location from dataset %s.%s',
              parts[0],
              parts[1])
          ds = self._bq_wrapper.client.datasets.Get(
              bigquery.BigqueryDatasetsGetRequest(
                  projectId=parts[0], datasetId=parts[1]))
          self._location = ds.location
          self._log(
              '[Stage1-Poll] Detected dataset location: %s', self._location)
        except Exception as e:
          _LOGGER.warning(
              '[Stage1-Poll] Could not detect dataset location: %s', e)
    self._log(
        '[Stage1-Poll] Creating/verifying temp dataset %s.%s (location=%s)',
        self._project,
        self._temp_dataset,
        self._location)
    self._bq_wrapper.get_or_create_dataset(
        self._project, self._temp_dataset, location=self._location)
    self._log(
        '[Stage1-Poll] setup complete: project=%s, temp_dataset=%s, '
        'table=%s, location=%s, change_function=%s, buffer_sec=%.1f, '
        'start_time=%.1f',
        self._project,
        self._temp_dataset,
        self._table,
        self._location,
        self._change_function,
        self._buffer_sec,
        self._start_time)

  def process(self, element, last_end_ts=beam.DoFn.StateParam(LAST_END_TS)):
    now = time.time()

    # Read state: last queried end timestamp
    start_ts = last_end_ts.read()
    if start_ts == 0:
      start_ts = self._start_time

    end_ts = now - self._buffer_sec

    self._log(
        '[Stage1-Poll] process: now=%.1f, '
        'start_ts=%.1f (%s), end_ts=%.1f (%s), buffer_sec=%.1f',
        now,
        start_ts,
        datetime.datetime.fromtimestamp(start_ts,
                                        tz=datetime.timezone.utc).isoformat(),
        end_ts,
        datetime.datetime.fromtimestamp(end_ts,
                                        tz=datetime.timezone.utc).isoformat(),
        self._buffer_sec)

    if end_ts <= start_ts:
      self._log(
          '[Stage1-Poll] No new data to poll: end_ts=%.1f <= '
          'start_ts=%.1f, skipping',
          end_ts,
          start_ts)
      return

    ranges = compute_ranges(start_ts, end_ts, self._change_function)
    self._log(
        '[Stage1-Poll] Polling %s: %d chunks covering [%s, %s)',
        self._table,
        len(ranges),
        datetime.datetime.fromtimestamp(start_ts,
                                        tz=datetime.timezone.utc).isoformat(),
        datetime.datetime.fromtimestamp(end_ts,
                                        tz=datetime.timezone.utc).isoformat())
    Metrics.counter('BigQueryChangeHistory', 'polls').inc()

    for chunk_idx, (chunk_start, chunk_end) in enumerate(ranges):
      self._log(
          '[Stage1-Poll] Executing chunk %d/%d: [%.1f, %.1f)',
          chunk_idx + 1,
          len(ranges),
          chunk_start,
          chunk_end)
      query_result = self._execute_query(chunk_start, chunk_end)
      self._log(
          '[Stage1-Poll] Chunk %d/%d produced _QueryResult: '
          'temp_table=%s',
          chunk_idx + 1,
          len(ranges),
          _table_key(query_result.temp_table_ref))
      # Emit with timestamp=range_start so the element holds the watermark
      # while buffered between Stage 1 and Stage 2.  Without this, the
      # watermark advances freely between polls and the SDF's watermark
      # estimator can't pull it back — only prevent further advancement.
      yield TimestampedValue(query_result, Timestamp(chunk_start))
      Metrics.counter('BigQueryChangeHistory', 'queries').inc()

    # Update state: record how far we've queried
    self._log(
        '[Stage1-Poll] Updating state: last_end_ts=%.1f (%s)',
        end_ts,
        datetime.datetime.fromtimestamp(end_ts,
                                        tz=datetime.timezone.utc).isoformat())
    last_end_ts.add(end_ts)

  def _execute_query(self, start_ts, end_ts):
    """Execute a CHANGES/APPENDS query and return _QueryResult."""
    sql = build_changes_query(
        self._table, start_ts, end_ts, self._change_function)
    temp_table_id = f'beam_ch_temp_{uuid.uuid4().hex[:8]}'
    job_id = f'beam_ch_{uuid.uuid4().hex[:12]}'

    self._log(
        '[Stage1-Poll] _execute_query: job_id=%s, temp_table=%s.%s, sql=%s',
        job_id,
        self._temp_dataset,
        temp_table_id,
        sql)

    temp_table_ref = bigquery.TableReference(
        projectId=self._project,
        datasetId=self._temp_dataset,
        tableId=temp_table_id)

    reference = bigquery.JobReference(
        jobId=job_id, projectId=self._project, location=self._location)

    request = bigquery.BigqueryJobsInsertRequest(
        projectId=self._project,
        job=bigquery.Job(
            configuration=bigquery.JobConfiguration(
                query=bigquery.JobConfigurationQuery(
                    query=sql,
                    useLegacySql=False,
                    destinationTable=temp_table_ref,
                    writeDisposition='WRITE_TRUNCATE',
                ),
            ),
            jobReference=reference))

    self._log('[Stage1-Poll] Submitting BQ job %s...', job_id)
    response = self._bq_wrapper._start_job(request)
    self._log(
        '[Stage1-Poll] BQ job %s submitted, waiting for completion...', job_id)
    self._bq_wrapper.wait_for_bq_job(
        response.jobReference, sleep_duration_sec=2)
    self._log(
        '[Stage1-Poll] BQ job %s DONE. Results in %s.%s',
        job_id,
        self._temp_dataset,
        temp_table_id)

    return _QueryResult(
        temp_table_ref=temp_table_ref, range_start=start_ts, range_end=end_ts)


class _CDCWatermarkEstimatorProvider(WatermarkEstimatorProvider):
  """WatermarkEstimatorProvider that initializes the hold from _QueryResult.

  Uses range_start from the element to set the initial watermark hold.
  This prevents the runner from advancing the watermark past the data's
  timestamps before any rows are emitted, reducing "late records" warnings.
  """
  def initial_estimator_state(self, element, restriction):
    if hasattr(element, 'range_start') and element.range_start > 0:
      ts = Timestamp(element.range_start)
      _LOGGER.warning(
          '[Watermark] initial_estimator_state: range_start=%.1f (%s), '
          'restriction=[%d,%d)',
          element.range_start,
          datetime.datetime.fromtimestamp(
              element.range_start, tz=datetime.timezone.utc).isoformat(),
          restriction.start,
          restriction.stop)
      return ts
    _LOGGER.warning(
        '[Watermark] initial_estimator_state: range_start not set, '
        'returning None (no hold)')
    return None

  def create_watermark_estimator(self, estimator_state):
    _LOGGER.warning(
        '[Watermark] create_watermark_estimator: state=%s', estimator_state)
    return ManualWatermarkEstimator(estimator_state)


# =============================================================================
# Stage 2 — _ReadStorageStreamsSDF
# =============================================================================


class _ReadStorageStreamsSDF(beam.DoFn,
                             beam.transforms.core.RestrictionProvider):
  """SDF that reads a temp table via BigQuery Storage Read API.

  The DoFn is its own RestrictionProvider (see core.py:220-222), which gives
  initial_restriction() access to self._create_read_session().

  Note on SDF lifecycle: the runner decomposes this SDF into three internal
  wrapper DoFns, each a separately deserialized copy:
    - Stage A (PairWithRestriction): calls initial_restriction() — no setup()
    - Stage B (SplitAndSizeRestrictions): calls split(), restriction_size()
    - Stage C (ProcessSizedElements): calls setup(), then process()
  Because initial_restriction() runs on a different copy than process(),
  _ensure_client() lazily creates a gRPC client on whichever copy needs one.
  The _StreamRestriction carries stream names directly so no shared state
  is needed between copies.

  Each element is a _QueryResult pointing to a temp table.

  Watermark: Uses ManualWatermarkEstimator so the watermark only advances
  as fast as the change_timestamp values we emit. Without this, the runner
  would advance the watermark based on processing time, causing all
  historical timestamps to be flagged as "late records."

  Emits:
    Main output: TimestampedValue(row_dict, change_timestamp)
    Side output (_CLEANUP_TAG): (table_key, streams_read, total_streams)
  """
  def __init__(self, trace=False):
    self._trace = trace
    self._storage_client = None

  def _log(self, msg, *args):
    if self._trace:
      _LOGGER.warning(msg, *args)

  def _ensure_client(self):
    """Lazily initialize the Storage client.

    Called from both setup() and initial_restriction() because the runner
    may invoke initial_restriction on the RestrictionProvider instance
    before setup() runs (or on a separately deserialized copy).
    """
    if self._storage_client is None:
      if bq_storage is None:
        raise ImportError(
            'google-cloud-bigquery-storage is required for '
            'ReadBigQueryChangeHistory. Install it with: '
            'pip install google-cloud-bigquery-storage')
      self._log('[Stage2-SDF] creating BigQueryReadClient')
      self._storage_client = bq_storage.BigQueryReadClient()

  def setup(self):
    self._ensure_client()

  # --- RestrictionProvider methods ---

  def initial_restriction(self, element):
    """Create ReadSession and return _StreamRestriction with stream names."""
    self._ensure_client()
    table_key = _table_key(element.temp_table_ref)
    session = self._create_read_session(element.temp_table_ref)
    stream_names = tuple(s.name for s in session.streams)
    self._log(
        '[Stage2-SDF] initial_restriction for %s: %d streams',
        table_key,
        len(stream_names))
    return _StreamRestriction(stream_names, 0, len(stream_names))

  def create_tracker(self, restriction):
    return _StreamRestrictionTracker(restriction)

  def restriction_size(self, element, restriction):
    return restriction.size()

  def split(self, element, restriction):
    """Yield one _StreamRestriction per stream for parallel distribution."""
    if restriction.size() <= 1:
      yield restriction
    else:
      for i in range(restriction.start, restriction.stop):
        yield _StreamRestriction(restriction.stream_names, i, i + 1)

  def is_bounded(self):
    return True

  # --- Process ---

  def process(
      self,
      element,
      restriction_tracker=beam.DoFn.RestrictionParam(),
      watermark_estimator=beam.DoFn.WatermarkEstimatorParam(
          _CDCWatermarkEstimatorProvider())):
    self._ensure_client()
    table_key = _table_key(element.temp_table_ref)

    restriction = restriction_tracker.current_restriction()
    stream_names = restriction.stream_names
    total_streams = len(stream_names)

    # Log watermark state at process() entry
    current_wm = watermark_estimator.current_watermark()
    self._log(
        '[Stage2-SDF] process() entry for %s: '
        'current_watermark=%s, range_start=%.1f (%s)',
        table_key,
        current_wm,
        element.range_start,
        datetime.datetime.fromtimestamp(
            element.range_start, tz=datetime.timezone.utc).isoformat()
        if element.range_start > 0 else 'N/A')

    streams_read = 0
    total_rows = 0
    first_wm_logged = False

    if total_streams == 0:
      self._log(
          '[Stage2-SDF] Empty table %s (0 streams), emitting cleanup',
          table_key)
    else:
      self._log(
          '[Stage2-SDF] Reading streams [%d, %d) of %d total for %s',
          restriction.start,
          restriction.stop,
          total_streams,
          table_key)

      for i in range(restriction.start, restriction.stop):
        if not restriction_tracker.try_claim(i):
          self._log(
              '[Stage2-SDF] try_claim(%d) FAILED for %s — '
              'runner split or checkpoint, breaking',
              i,
              table_key)
          break

        stream_name = stream_names[i]
        self._log(
            '[Stage2-SDF] try_claim(%d) succeeded — reading stream %s',
            i,
            stream_name)

        stream_rows = 0
        for row in self._read_stream(stream_name):
          ts = row.get('change_timestamp') or row.get('_CHANGE_TIMESTAMP')
          if ts is None:
            ts = Timestamp(0)
          elif isinstance(ts, datetime.datetime):
            ts = Timestamp.from_utc_datetime(ts)

          # Advance watermark to track progress. ManualWatermarkEstimator
          # requires monotonically increasing values, so only advance
          # when we see a timestamp beyond the current watermark.
          current_wm = watermark_estimator.current_watermark()
          if current_wm is None or ts > current_wm:
            if not first_wm_logged:
              self._log(
                  '[Watermark] First set_watermark for %s: '
                  'old=%s, new=%s, row_ts=%s',
                  table_key,
                  current_wm,
                  ts,
                  datetime.datetime.fromtimestamp(
                      ts.seconds(), tz=datetime.timezone.utc).isoformat()
                  if hasattr(ts, 'seconds') else str(ts))
              first_wm_logged = True
            watermark_estimator.set_watermark(ts)

          yield TimestampedValue(row, ts)
          stream_rows += 1
          total_rows += 1
          Metrics.counter('BigQueryChangeHistory', 'rows_emitted').inc()

        streams_read += 1
        self._log(
            '[Stage2-SDF] Finished reading stream %d for %s: %d rows',
            i,
            table_key,
            stream_rows)
        Metrics.counter('BigQueryChangeHistory', 'streams_read').inc()

    # Emit cleanup signal. Every split that reads at least one stream
    # reports how many it read. Empty tables (0 streams) also emit a
    # signal so Stage 3 can delete the temp table.
    if streams_read > 0 or total_streams == 0:
      self._log(
          '[Stage2-SDF] Emitting cleanup signal for %s: '
          'streams_read=%d, total_streams=%d, total_rows=%d',
          table_key,
          streams_read,
          total_streams,
          total_rows)
      yield beam.pvalue.TaggedOutput(
          _CLEANUP_TAG, (
              table_key,
              streams_read,
              total_streams,
          ))

  def _create_read_session(self, table_ref):
    """Create a BigQuery Storage ReadSession for the given table."""
    table_path = (
        f'projects/{table_ref.projectId}/'
        f'datasets/{table_ref.datasetId}/'
        f'tables/{table_ref.tableId}')

    requested_session = bq_storage.types.ReadSession()
    requested_session.table = table_path
    requested_session.data_format = bq_storage.types.DataFormat.ARROW
    requested_session.read_options \
        .arrow_serialization_options.buffer_compression = \
        bq_storage.types.ArrowSerializationOptions.CompressionCodec.LZ4_FRAME

    session = self._storage_client.create_read_session(
        parent=f'projects/{table_ref.projectId}',
        read_session=requested_session,
        max_stream_count=_DEFAULT_MAX_STREAMS)
    self._log(
        '[Stage2-SDF] _create_read_session: table=%s, %d streams',
        table_path,
        len(session.streams))
    return session

  def _read_stream(self, stream_name):
    """Read all rows from a single Storage API stream as dicts."""
    for row in self._storage_client.read_rows(stream_name).rows():
      yield dict((item[0], item[1].as_py()) for item in row.items())


# =============================================================================
# Stage 3 — _CleanupTempTablesFn
# =============================================================================


class _CleanupTempTablesFn(beam.DoFn):
  """Stateful DoFn that deletes temp tables after all streams are read.

  Receives cleanup signals from Stage 2 SDF as:
    (table_key, (streams_read_count, total_streams))

  Accumulates streams_read across all signals for the same table_key.
  When streams_read >= total, deletes the temp table. The >= (rather than ==)
  guards against duplicate delivery in at-least-once runners.
  """
  STREAMS_READ = beam.transforms.userstate.CombiningValueStateSpec(
      'streams_read', sum)
  TOTAL = beam.transforms.userstate.CombiningValueStateSpec(
      'total', _max_default_zero)

  def __init__(self, trace=False):
    self._trace = trace

  def _log(self, msg, *args):
    if self._trace:
      _LOGGER.warning(msg, *args)

  def setup(self):
    self._log('[Stage3-Cleanup] setup: creating BigQueryWrapper')
    self._bq_wrapper = bigquery_tools.BigQueryWrapper()

  def process(
      self,
      element,
      streams_read=beam.DoFn.StateParam(STREAMS_READ),
      total=beam.DoFn.StateParam(TOTAL)):
    table_key = element[0]
    split_count = element[1][0]
    total_streams = element[1][1]

    self._log(
        '[Stage3-Cleanup] Received cleanup signal for %s: '
        'split_count=%d, total_streams=%d',
        table_key,
        split_count,
        total_streams)

    streams_read.add(split_count)
    total.add(total_streams)

    current_read = streams_read.read()
    current_total = total.read()

    self._log(
        '[Stage3-Cleanup] State for %s: streams_read=%d/%d',
        table_key,
        current_read,
        current_total)

    if current_read >= current_total:
      parts = table_key.split('.')
      if len(parts) == 3:
        project, dataset, table = parts
        self._log(
            '[Stage3-Cleanup] All streams read — DELETING temp table %s',
            table_key)
        try:
          self._bq_wrapper._delete_table(project, dataset, table)
          self._log('[Stage3-Cleanup] Deleted temp table %s', table_key)
        except Exception as e:
          _LOGGER.warning(
              '[Stage3-Cleanup] Failed to delete temp table %s '
              '(may already be deleted): %s',
              table_key,
              e)
        Metrics.counter('BigQueryChangeHistory', 'temp_tables_deleted').inc()
    else:
      self._log(
          '[Stage3-Cleanup] Not yet complete for %s (%d/%d), '
          'waiting for more signals',
          table_key,
          current_read,
          current_total)


# =============================================================================
# Public API — ReadBigQueryChangeHistory
# =============================================================================


class ReadBigQueryChangeHistory(beam.PTransform):
  """Streaming source for BigQuery change history.

  Continuously polls BigQuery APPENDS() or CHANGES() functions and emits
  changed rows as an unbounded PCollection of dicts.

  Args:
    table: BigQuery table to read changes from.
        Format: 'project:dataset.table' or 'project.dataset.table'.
    poll_interval_sec: Seconds between polls. Default 60.
    start_time: Start reading from this timestamp (float, epoch seconds).
        Default: current time when pipeline starts.
    stop_time: Stop polling at this timestamp. Default: run forever.
    change_function: 'CHANGES' or 'APPENDS'. Default 'APPENDS'.
    buffer_sec: Safety buffer behind now(). Default: 600 for CHANGES,
        15 for APPENDS.
    project: GCP project ID. Default: from pipeline options.
    temp_dataset: Dataset for temp tables. Default 'beam_ch_temp'.
    trace: If True, emit detailed pipeline execution trace logs at
        WARNING level. Default False (silent).
  """
  def __init__(
      self,
      table,
      poll_interval_sec=60,
      start_time=None,
      stop_time=None,
      change_function='APPENDS',
      buffer_sec=None,
      project=None,
      temp_dataset='beam_ch_temp',
      trace=False):
    super().__init__()
    if change_function not in ('CHANGES', 'APPENDS'):
      raise ValueError(
          f"change_function must be 'CHANGES' or 'APPENDS', "
          f"got '{change_function}'")
    if poll_interval_sec <= 0:
      raise ValueError(
          f'poll_interval_sec must be positive, got {poll_interval_sec}')

    self._table = table
    self._poll_interval_sec = poll_interval_sec
    self._start_time = start_time
    self._stop_time = stop_time
    self._change_function = change_function
    self._buffer_sec = (
        buffer_sec if buffer_sec is not None else
        (600 if change_function == 'CHANGES' else 15))
    self._project = project
    self._temp_dataset = temp_dataset
    self._trace = trace

  def _log(self, msg, *args):
    if self._trace:
      _LOGGER.warning(msg, *args)

  def expand(self, pbegin):
    project = self._project
    if project is None:
      project = pbegin.pipeline.options.view_as(
          beam.options.pipeline_options.GoogleCloudOptions).project

    if project is None:
      raise ValueError(
          'project must be specified either in ReadBigQueryChangeHistory '
          'or in pipeline options (--project)')

    start_time = self._start_time or time.time()
    stop_time = self._stop_time or MAX_TIMESTAMP

    self._log(
        '[ReadBigQueryChangeHistory] expand: table=%s, project=%s, '
        'change_function=%s, poll_interval=%d sec, buffer=%d sec, '
        'temp_dataset=%s, start_time=%.1f, stop_time=%s',
        self._table,
        project,
        self._change_function,
        self._poll_interval_sec,
        self._buffer_sec,
        self._temp_dataset,
        start_time,
        self._stop_time)

    # Stage 1: PeriodicImpulse + stateful DoFn
    # PeriodicImpulse starts from "now" (not start_time) to avoid queuing
    # past impulses that would each trigger a tiny catch-up BQ query.
    # The stateful DoFn handles the full historical range [start_time, now)
    # in a single query on its first invocation via self._start_time.
    impulse_start = time.time()
    query_results = (
        pbegin
        | 'PollImpulse' >> PeriodicImpulse(
            start_timestamp=impulse_start,
            stop_timestamp=stop_time,
            fire_interval=self._poll_interval_sec)
        | 'KeyForState' >> beam.Map(lambda ts: ('__poll__', ts))
        | 'PollChangeHistory' >> beam.ParDo(
            _PollChangeHistoryFn(
                table=self._table,
                project=project,
                change_function=self._change_function,
                buffer_sec=self._buffer_sec,
                temp_dataset=self._temp_dataset,
                start_time=start_time,
                trace=self._trace)))

    # Stage 2: Read temp tables via Storage Read API
    read_outputs = (
        query_results
        | 'ReadStorageStreams' >> beam.ParDo(
            _ReadStorageStreamsSDF(trace=self._trace)).with_outputs(
                _CLEANUP_TAG, main='rows'))

    # Stage 3: Cleanup temp tables
    _ = (
        read_outputs[_CLEANUP_TAG]
        | 'KeyByTable' >>
        beam.Map(lambda x: (x[0], (x[1], x[2]))).with_output_types(
            beam.typehints.Tuple[str, beam.typehints.Tuple[int, int]])
        | 'CleanupTempTables' >> beam.ParDo(
            _CleanupTempTablesFn(trace=self._trace)))

    return read_outputs['rows']
