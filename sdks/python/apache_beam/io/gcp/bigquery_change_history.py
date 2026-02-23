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
                poll_interval_sec=30)
            | beam.Map(print))

Architecture: Three-stage pipeline
  Stage 1: Polling SDF with ManualWatermarkEstimator. Polls BQ on interval
           via defer_remainder(), executes CHANGES/APPENDS query, writes
           results to temp table, advances watermark to queried range end.
  Stage 2: SDF reads temp table via Storage Read API with dynamic splitting.
  Stage 3: Stateful DoFn tracks stream completion, deletes temp tables.

See docs/bigquery-change-history/ for design documentation.
"""

import dataclasses
import datetime
import logging
import sys
import time
import uuid
from typing import Any
from typing import Optional

import apache_beam as beam
from apache_beam.io.gcp import bigquery_tools
from apache_beam.io.gcp.internal.clients import bigquery
from apache_beam.io.restriction_trackers import OffsetRange
from apache_beam.io.restriction_trackers import OffsetRestrictionTracker
from apache_beam.io.watermark_estimators import ManualWatermarkEstimator
from apache_beam.metrics import Metrics
from apache_beam.transforms.window import TimestampedValue
from apache_beam.utils.timestamp import Timestamp

try:
  from google.cloud import bigquery_storage_v1 as bq_storage
except ImportError:
  bq_storage = None  # type: ignore

_LOGGER = logging.getLogger(__name__)

__all__ = ['ReadBigQueryChangeHistory']

# Max time range for CHANGES() queries: 1 day in seconds.
_MAX_CHANGES_RANGE_SEC = 86400

# Cleanup side output tag.
_CLEANUP_TAG = 'cleanup'

# =============================================================================
# Phase 1: Foundation — helpers and data classes
# =============================================================================


@dataclasses.dataclass
class _QueryResult:
  """Bridges Stage 1 (poll) to Stage 2 (read).

  After Stage 1 executes a CHANGES/APPENDS query, it emits a _QueryResult
  pointing to the temp table containing query results. Stage 2's SDF reads
  rows from that temp table via the Storage Read API.
  """
  temp_table_ref: Optional[bigquery.TableReference] = None
  total_streams: int = 0  # Set by SDF's initial_restriction from ReadSession
  range_end: float = 0.0  # End timestamp of this query range (for logging)
  session: Any = None  # Cached ReadSession (set in initial_restriction)


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
  # APPENDS returns: _CHANGE_TYPE, _CHANGE_TIMESTAMP
  # CHANGES returns: _CHANGE_TYPE, _CHANGE_TIMESTAMP
  pseudo_cols = '_CHANGE_TYPE, _CHANGE_TIMESTAMP'
  sql = (
      f"SELECT * EXCEPT({pseudo_cols}), "
      f"_CHANGE_TYPE AS change_type, "
      f"_CHANGE_TIMESTAMP AS change_timestamp "
      f"FROM {change_function}"
      f"(TABLE `{table}`, "
      f"TIMESTAMP '{start_iso}', "
      f"TIMESTAMP '{end_iso}')")
  _LOGGER.warning('[build_changes_query] Built SQL: %s', sql)
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
    _LOGGER.warning(
        '[compute_ranges] Empty range: end_ts=%.1f <= start_ts=%.1f',
        end_ts,
        start_ts)
    return []

  if change_function != 'CHANGES':
    _LOGGER.warning(
        '[compute_ranges] APPENDS: single range [%.1f, %.1f) (%.1f sec)',
        start_ts,
        end_ts,
        end_ts - start_ts)
    return [(start_ts, end_ts)]

  # CHANGES: chunk into <=1-day ranges
  ranges = []
  current = start_ts
  while current < end_ts:
    chunk_end = min(current + _MAX_CHANGES_RANGE_SEC, end_ts)
    ranges.append((current, chunk_end))
    current = chunk_end
  _LOGGER.warning(
      '[compute_ranges] CHANGES: %d chunks covering %.1f sec total',
      len(ranges),
      end_ts - start_ts)
  return ranges


# =============================================================================
# Phase 2: Stage 3 — _CleanupTempTablesFn
# =============================================================================


def _max_default_zero(values):
  """max() that returns 0 for empty iterables (needed for CombiningState)."""
  result = 0
  for v in values:
    if v > result:
      result = v
  return result


class _CleanupTempTablesFn(beam.DoFn):
  """Stateful DoFn that deletes temp tables after all streams are read.

  Receives cleanup signals from Stage 2 SDF as:
    (table_key, (streams_read_count, total_streams))

  Accumulates streams_read across all signals for the same table_key.
  When streams_read >= total, deletes the temp table.
  """
  STREAMS_READ = beam.transforms.userstate.CombiningValueStateSpec(
      'streams_read', sum)
  TOTAL = beam.transforms.userstate.CombiningValueStateSpec(
      'total', _max_default_zero)

  def setup(self):
    _LOGGER.warning('[Stage3-Cleanup] setup: creating BigQueryWrapper')
    self._bq_wrapper = bigquery_tools.BigQueryWrapper()

  def process(
      self,
      element,
      streams_read=beam.DoFn.StateParam(STREAMS_READ),
      total=beam.DoFn.StateParam(TOTAL)):
    table_key = element[0]
    split_count = element[1][0]
    total_streams = element[1][1]

    _LOGGER.warning(
        '[Stage3-Cleanup] Received cleanup signal for %s: '
        'split_count=%d, total_streams=%d',
        table_key,
        split_count,
        total_streams)

    streams_read.add(split_count)
    total.add(total_streams)

    current_read = streams_read.read()
    current_total = total.read()

    _LOGGER.warning(
        '[Stage3-Cleanup] State for %s: streams_read=%d/%d',
        table_key,
        current_read,
        current_total)

    if current_read >= current_total:
      parts = table_key.split('.')
      if len(parts) == 3:
        project, dataset, table = parts
        _LOGGER.warning(
            '[Stage3-Cleanup] All streams read — DELETING temp table %s',
            table_key)
        self._bq_wrapper._delete_table(project, dataset, table)
        _LOGGER.warning('[Stage3-Cleanup] Deleted temp table %s', table_key)
        Metrics.counter('BigQueryChangeHistory', 'temp_tables_deleted').inc()
    else:
      _LOGGER.warning(
          '[Stage3-Cleanup] Not yet complete for %s (%d/%d), '
          'waiting for more signals',
          table_key,
          current_read,
          current_total)


# =============================================================================
# Phase 3: Stage 2 — _ReadStorageStreamsSDF
# =============================================================================


class _ReadStorageStreamsRestrictionProvider(
    beam.transforms.core.RestrictionProvider):
  """RestrictionProvider for the Storage Read SDF.

  Creates a ReadSession for each _QueryResult element and returns an
  OffsetRange(0, num_streams) restriction. The runner can split this
  restriction to distribute streams across workers.
  """
  def __init__(self, max_streams=10):
    self._max_streams = max_streams

  def initial_restriction(self, element):
    # We can't create a ReadSession here (no storage_client on this object),
    # so we return a placeholder. The DoFn's process() creates the session.
    # We use OffsetRange(0, 1) as a minimum — the actual stream count
    # is determined at process() time.
    return OffsetRange(0, 1)

  def create_tracker(self, restriction):
    return OffsetRestrictionTracker(restriction)

  def restriction_size(self, element, restriction):
    return restriction.stop - restriction.start

  def split(self, element, restriction):
    # Don't do initial splitting — let the runner dynamically split
    yield restriction

  def is_bounded(self):
    return True


class _ReadStorageStreamsSDF(beam.DoFn):
  """SDF that reads a temp table via BigQuery Storage Read API.

  Each element is a _QueryResult pointing to a temp table. The DoFn creates
  a ReadSession to discover streams, then reads each stream yielding rows.

  Emits:
    Main output: TimestampedValue(row_dict, change_timestamp)
    Side output (_CLEANUP_TAG): (table_key, streams_read, total_streams)
  """
  CLEANUP_TAG = _CLEANUP_TAG

  def __init__(self, max_streams=10):
    self._max_streams = max_streams
    self._storage_client = None
    # Cache: table_key -> (session, total_streams)
    self._session_cache = {}

  def setup(self):
    if bq_storage is None:
      raise ImportError(
          'google-cloud-bigquery-storage is required for '
          'ReadBigQueryChangeHistory. Install it with: '
          'pip install google-cloud-bigquery-storage')
    _LOGGER.warning('[Stage2-SDF] setup: creating BigQueryReadClient')
    self._storage_client = bq_storage.BigQueryReadClient()

  def process(
      self,
      element,
      restriction_tracker=beam.DoFn.RestrictionParam(
          _ReadStorageStreamsRestrictionProvider())):
    # Create ReadSession (or use cache)
    table_key = _table_key(element.temp_table_ref)
    _LOGGER.warning(
        '[Stage2-SDF] process: received _QueryResult for temp table %s',
        table_key)

    if table_key in self._session_cache:
      session, total_streams = self._session_cache[table_key]
      _LOGGER.warning(
          '[Stage2-SDF] Using cached ReadSession for %s: %d streams',
          table_key,
          total_streams)
    else:
      _LOGGER.warning(
          '[Stage2-SDF] Creating ReadSession for %s '
          '(max_streams=%d)...',
          table_key,
          self._max_streams)
      session = self._create_read_session(element.temp_table_ref)
      total_streams = len(session.streams)
      self._session_cache[table_key] = (session, total_streams)
      _LOGGER.warning(
          '[Stage2-SDF] ReadSession created for %s: %d streams',
          table_key,
          total_streams)

    element.total_streams = total_streams

    streams_read = 0
    total_rows = 0

    if total_streams == 0:
      # Empty table — nothing to read, just emit cleanup
      _LOGGER.warning(
          '[Stage2-SDF] Empty table %s (0 streams), '
          'claiming offset 0 and emitting cleanup',
          table_key)
      restriction_tracker.try_claim(0)
    else:
      # Read streams according to restriction
      restriction = restriction_tracker.current_restriction()
      # Clamp to actual stream count
      stream_end = min(restriction.stop, total_streams)
      _LOGGER.warning(
          '[Stage2-SDF] Reading streams [%d, %d) of %d total for %s',
          restriction.start,
          stream_end,
          total_streams,
          table_key)

      for i in range(restriction.start, stream_end):
        if not restriction_tracker.try_claim(i):
          _LOGGER.warning(
              '[Stage2-SDF] try_claim(%d) FAILED for %s — '
              'runner split or checkpoint, breaking',
              i,
              table_key)
          break  # Runner split or checkpoint — residual handles rest

        stream_name = session.streams[i].name
        _LOGGER.warning(
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
          yield TimestampedValue(row, ts)
          stream_rows += 1
          total_rows += 1
          Metrics.counter('BigQueryChangeHistory', 'rows_emitted').inc()

        streams_read += 1
        _LOGGER.warning(
            '[Stage2-SDF] Finished reading stream %d for %s: %d rows',
            i,
            table_key,
            stream_rows)
        Metrics.counter('BigQueryChangeHistory', 'streams_read').inc()

    # Always emit cleanup signal — even after checkpoint/split
    if streams_read > 0 or total_streams == 0:
      _LOGGER.warning(
          '[Stage2-SDF] Emitting cleanup signal for %s: '
          'streams_read=%d, total_streams=%d, total_rows=%d',
          table_key,
          streams_read,
          total_streams,
          total_rows)
      yield beam.pvalue.TaggedOutput(
          self.CLEANUP_TAG, (
              table_key,
              streams_read,
              total_streams,
          ))

    # Clear cache entry after cleanup signal
    self._session_cache.pop(table_key, None)

  def _create_read_session(self, table_ref):
    """Create a BigQuery Storage ReadSession for the given table."""
    table_path = (
        f'projects/{table_ref.projectId}/'
        f'datasets/{table_ref.datasetId}/'
        f'tables/{table_ref.tableId}')
    _LOGGER.warning(
        '[Stage2-SDF] _create_read_session: table=%s, '
        'format=ARROW, compression=LZ4_FRAME',
        table_path)

    requested_session = bq_storage.types.ReadSession()
    requested_session.table = table_path
    requested_session.data_format = bq_storage.types.DataFormat.ARROW
    requested_session.read_options \
        .arrow_serialization_options.buffer_compression = \
        bq_storage.types.ArrowSerializationOptions.CompressionCodec.LZ4_FRAME

    session = self._storage_client.create_read_session(
        parent=f'projects/{table_ref.projectId}',
        read_session=requested_session,
        max_stream_count=self._max_streams)
    _LOGGER.warning(
        '[Stage2-SDF] _create_read_session: got session with %d streams',
        len(session.streams))
    return session

  def _read_stream(self, stream_name):
    """Read all rows from a single Storage API stream as dicts."""
    _LOGGER.warning(
        '[Stage2-SDF] _read_stream: starting read for %s', stream_name)
    row_iter = iter(self._storage_client.read_rows(stream_name).rows())
    row = next(row_iter, None)
    if row is None:
      _LOGGER.warning(
          '[Stage2-SDF] _read_stream: stream %s is empty', stream_name)
      return
    while row is not None:
      yield dict(map(lambda item: (item[0], item[1].as_py()), row.items()))
      row = next(row_iter, None)
    _LOGGER.warning(
        '[Stage2-SDF] _read_stream: finished reading %s', stream_name)


# =============================================================================
# Phase 4: Stage 1 — _PollChangeHistorySDF
# =============================================================================


@dataclasses.dataclass
class _PollConfig:
  """Configuration element for the polling SDF.

  A single _PollConfig element is created by expand() and fed to the SDF.
  The SDF loops forever (via defer_remainder), polling BQ on each invocation.
  """
  table: str = ''
  project: str = ''
  change_function: str = 'APPENDS'
  buffer_sec: float = 15.0
  temp_dataset: str = 'beam_ch_temp'
  start_time: float = 0.0
  stop_time: Optional[float] = None
  poll_interval_sec: int = 60


class _NonSplittableOffsetTracker(OffsetRestrictionTracker):
  """OffsetRestrictionTracker that refuses dynamic splitting.

  Stage 1 is a singleton poller — its watermark state determines what
  time range to query next. If the runner splits the restriction into
  parallel workers, each worker reads the same watermark and queries
  the same BQ time range, producing duplicate data.

  Allows try_split(0) for self-checkpointing (used by defer_remainder)
  but refuses try_split(fraction > 0) which the runner uses for
  dynamic parallel splitting.
  """
  def try_split(self, fraction_of_remainder):
    if fraction_of_remainder == 0:
      return super().try_split(0)
    return None


class _PollChangeHistoryRestrictionProvider(
    beam.transforms.core.RestrictionProvider):
  """RestrictionProvider for the polling SDF.

  Uses an unbounded OffsetRange — each offset is one poll iteration.
  The SDF defers after each poll, so only one offset is claimed per invocation.

  Key design choices:
  - Non-splittable tracker: prevents runner from creating parallel workers
    that would query the same BQ time range (watermark is shared state).
  - restriction_size returns 1 (not sys.maxsize) to avoid over-prioritizing
    this SDF over downstream stages.
  - split() yields the restriction unsplit to prevent initial splitting.
  """
  def initial_restriction(self, element):
    return OffsetRange(0, sys.maxsize)

  def create_tracker(self, restriction):
    return _NonSplittableOffsetTracker(restriction)

  def restriction_size(self, element, restriction):
    if restriction.start < restriction.stop:
      return 1
    return 0

  def split(self, element, restriction):
    # Don't split — this SDF is a singleton poller.
    yield restriction

  def truncate(self, element, restriction):
    # On drain, stop immediately
    return None

  def is_bounded(self):
    return False


class _PollChangeHistorySDF(beam.DoFn):
  """SDF that polls BQ change history and emits _QueryResult elements.

  Replaces the old PeriodicImpulse + stateful DoFn architecture.
  This SDF owns the polling loop via defer_remainder() and controls
  the output watermark via ManualWatermarkEstimator.

  On each invocation:
  1. Computes time range [watermark, now - buffer)
  2. Chunks into safe ranges (<=1 day for CHANGES)
  3. Executes each chunk as a BQ query writing to a unique temp table
  4. Yields _QueryResult per temp table
  5. Advances watermark to end of queried range
  6. Defers remainder to schedule next poll after poll_interval_sec
  """
  def __init__(
      self,
      table,
      project,
      change_function,
      buffer_sec,
      temp_dataset,
      start_time,
      stop_time=None,
      poll_interval_sec=60,
      location=None):
    self._table = table
    self._project = project
    self._change_function = change_function
    self._buffer_sec = buffer_sec
    self._temp_dataset = temp_dataset
    self._start_time = start_time
    self._stop_time = stop_time
    self._poll_interval_sec = poll_interval_sec
    self._location = location

  def setup(self):
    _LOGGER.warning(
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
          _LOGGER.warning(
              '[Stage1-Poll] Detecting location from dataset %s.%s',
              parts[0],
              parts[1])
          ds = self._bq_wrapper.client.datasets.Get(
              bigquery.BigqueryDatasetsGetRequest(
                  projectId=parts[0], datasetId=parts[1]))
          self._location = ds.location
          _LOGGER.warning(
              '[Stage1-Poll] Detected dataset location: %s', self._location)
        except Exception as e:
          _LOGGER.warning(
              '[Stage1-Poll] Could not detect dataset location: %s', e)
    _LOGGER.warning(
        '[Stage1-Poll] Creating/verifying temp dataset %s.%s (location=%s)',
        self._project,
        self._temp_dataset,
        self._location)
    self._bq_wrapper.get_or_create_dataset(
        self._project, self._temp_dataset, location=self._location)
    _LOGGER.warning(
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

  @beam.DoFn.unbounded_per_element()
  def process(
      self,
      element,
      restriction_tracker=beam.DoFn.RestrictionParam(
          _PollChangeHistoryRestrictionProvider()),
      watermark_estimator=beam.DoFn.WatermarkEstimatorParam(
          ManualWatermarkEstimator.default_provider())):
    poll_index = restriction_tracker.current_restriction().start
    now = time.time()

    # Check if we've passed stop_time
    if self._stop_time and now >= self._stop_time:
      _LOGGER.warning(
          '[Stage1-Poll] stop_time reached (%.1f >= %.1f), stopping',
          now,
          self._stop_time)
      return

    # Guard against premature re-invocation. DirectRunner doesn't honor
    # defer_remainder() resume timestamps and immediately re-invokes the
    # SDF. Without this check, every invocation runs a BQ query (~3s),
    # starving downstream stages. By checking the watermark, we know when
    # the last poll ended and can defer instantly (microseconds) until
    # enough time has passed — giving the runner a chance to schedule
    # Stage 2 between deferrals. This is the same pattern PeriodicImpulse
    # uses (check wall-clock time before try_claim, defer if too early).
    current_wm = watermark_estimator.current_watermark()
    if current_wm is not None:
      last_poll_end = current_wm.micros / 1e6
      next_allowed = last_poll_end + self._poll_interval_sec
      if now < next_allowed:
        restriction_tracker.defer_remainder(Timestamp.of(next_allowed))
        return

    _LOGGER.warning(
        '[Stage1-Poll] process: poll_index=%d, now=%.1f', poll_index, now)

    if not restriction_tracker.try_claim(poll_index):
      _LOGGER.warning(
          '[Stage1-Poll] try_claim(%d) failed, stopping', poll_index)
      return

    # Compute time range from watermark
    if current_wm is not None:
      start_ts = current_wm.micros / 1e6
    else:
      start_ts = self._start_time
    end_ts = now - self._buffer_sec

    _LOGGER.warning(
        '[Stage1-Poll] Time range: start_ts=%.1f (%s), end_ts=%.1f (%s), '
        'buffer_sec=%.1f',
        start_ts,
        datetime.datetime.fromtimestamp(start_ts,
                                        tz=datetime.timezone.utc).isoformat(),
        end_ts,
        datetime.datetime.fromtimestamp(end_ts,
                                        tz=datetime.timezone.utc).isoformat(),
        self._buffer_sec)

    if end_ts > start_ts:
      ranges = compute_ranges(start_ts, end_ts, self._change_function)
      _LOGGER.warning(
          '[Stage1-Poll] Polling %s: %d chunks covering [%s, %s)',
          self._table,
          len(ranges),
          datetime.datetime.fromtimestamp(start_ts,
                                          tz=datetime.timezone.utc).isoformat(),
          datetime.datetime.fromtimestamp(end_ts,
                                          tz=datetime.timezone.utc).isoformat())
      Metrics.counter('BigQueryChangeHistory', 'polls').inc()

      for chunk_idx, (chunk_start, chunk_end) in enumerate(ranges):
        _LOGGER.warning(
            '[Stage1-Poll] Executing chunk %d/%d: [%.1f, %.1f)',
            chunk_idx + 1,
            len(ranges),
            chunk_start,
            chunk_end)
        query_result = self._execute_query(chunk_start, chunk_end)
        if query_result is not None:
          _LOGGER.warning(
              '[Stage1-Poll] Chunk %d/%d produced _QueryResult: '
              'temp_table=%s',
              chunk_idx + 1,
              len(ranges),
              _table_key(query_result.temp_table_ref))
          yield query_result
        Metrics.counter('BigQueryChangeHistory', 'queries').inc()

      # Advance watermark to end of queried range
      _LOGGER.warning(
          '[Stage1-Poll] Advancing watermark to %.1f (%s)',
          end_ts,
          datetime.datetime.fromtimestamp(end_ts,
                                          tz=datetime.timezone.utc).isoformat())
      watermark_estimator.set_watermark(Timestamp.of(end_ts))
    else:
      _LOGGER.warning(
          '[Stage1-Poll] No new data to poll: end_ts=%.1f <= '
          'start_ts=%.1f, skipping',
          end_ts,
          start_ts)

    # Defer remainder — wake up after poll_interval_sec
    next_poll = now + self._poll_interval_sec
    _LOGGER.warning(
        '[Stage1-Poll] Deferring remainder: next poll at %.1f (%s)',
        next_poll,
        datetime.datetime.fromtimestamp(next_poll,
                                        tz=datetime.timezone.utc).isoformat())
    restriction_tracker.defer_remainder(Timestamp.of(next_poll))

  def _execute_query(self, start_ts, end_ts):
    """Execute a CHANGES/APPENDS query and return _QueryResult."""
    sql = build_changes_query(
        self._table, start_ts, end_ts, self._change_function)
    temp_table_id = f'beam_ch_temp_{uuid.uuid4().hex[:8]}'
    job_id = f'beam_ch_{uuid.uuid4().hex[:12]}'

    _LOGGER.warning(
        '[Stage1-Poll] _execute_query: job_id=%s, temp_table=%s.%s',
        job_id,
        self._temp_dataset,
        temp_table_id)

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

    _LOGGER.warning('[Stage1-Poll] Submitting BQ job %s...', job_id)
    response = self._bq_wrapper._start_job(request)
    _LOGGER.warning(
        '[Stage1-Poll] BQ job %s submitted, waiting for completion...', job_id)
    self._bq_wrapper.wait_for_bq_job(
        response.jobReference, sleep_duration_sec=2)
    _LOGGER.warning(
        '[Stage1-Poll] BQ job %s DONE. Results in %s.%s',
        job_id,
        self._temp_dataset,
        temp_table_id)

    return _QueryResult(
        temp_table_ref=temp_table_ref,
        total_streams=0,  # Set by Stage 2's ReadSession
        range_end=end_ts)


# =============================================================================
# Phase 5: Public API — ReadBigQueryChangeHistory
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
    max_streams: Max Storage Read API streams per read. Default 10.
    project: GCP project ID. Default: from pipeline options.
    temp_dataset: Dataset for temp tables. Default 'beam_ch_temp'.
  """
  def __init__(
      self,
      table,
      poll_interval_sec=60,
      start_time=None,
      stop_time=None,
      change_function='APPENDS',
      buffer_sec=None,
      max_streams=10,
      project=None,
      temp_dataset='beam_ch_temp'):
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
    self._max_streams = max_streams
    self._project = project
    self._temp_dataset = temp_dataset

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

    _LOGGER.warning(
        '[ReadBigQueryChangeHistory] expand: table=%s, project=%s, '
        'change_function=%s, poll_interval=%d sec, buffer=%d sec, '
        'max_streams=%d, temp_dataset=%s, start_time=%.1f, stop_time=%s',
        self._table,
        project,
        self._change_function,
        self._poll_interval_sec,
        self._buffer_sec,
        self._max_streams,
        self._temp_dataset,
        start_time,
        self._stop_time)

    # Stage 1: Polling SDF — owns the polling loop and watermark
    _LOGGER.warning(
        '[ReadBigQueryChangeHistory] Wiring Stage 1: '
        'Create config -> PollChangeHistory SDF')
    config = _PollConfig(
        table=self._table,
        project=project,
        change_function=self._change_function,
        buffer_sec=self._buffer_sec,
        temp_dataset=self._temp_dataset,
        start_time=start_time,
        stop_time=self._stop_time,
        poll_interval_sec=self._poll_interval_sec)

    query_results = (
        pbegin
        | 'CreateConfig' >> beam.Create([config])
        | 'PollChangeHistory' >> beam.ParDo(
            _PollChangeHistorySDF(
                table=self._table,
                project=project,
                change_function=self._change_function,
                buffer_sec=self._buffer_sec,
                temp_dataset=self._temp_dataset,
                start_time=start_time,
                stop_time=self._stop_time,
                poll_interval_sec=self._poll_interval_sec)))

    # Stage 2: Read temp tables via Storage Read API
    _LOGGER.warning(
        '[ReadBigQueryChangeHistory] Wiring Stage 2: '
        'ReadStorageStreams SDF with cleanup side output')
    read_outputs = (
        query_results
        | 'ReadStorageStreams' >> beam.ParDo(
            _ReadStorageStreamsSDF(max_streams=self._max_streams)).with_outputs(
                _CLEANUP_TAG, main='rows'))

    # Stage 3: Cleanup temp tables
    _LOGGER.warning(
        '[ReadBigQueryChangeHistory] Wiring Stage 3: '
        'KeyByTable -> CleanupTempTables')
    _ = (
        read_outputs[_CLEANUP_TAG]
        | 'KeyByTable' >>
        beam.Map(lambda x: (x[0], (x[1], x[2]))).with_output_types(
            beam.typehints.Tuple[str, beam.typehints.Tuple[int, int]])
        | 'CleanupTempTables' >> beam.ParDo(_CleanupTempTablesFn()))

    _LOGGER.warning('[ReadBigQueryChangeHistory] Pipeline wiring complete')
    return read_outputs['rows']
