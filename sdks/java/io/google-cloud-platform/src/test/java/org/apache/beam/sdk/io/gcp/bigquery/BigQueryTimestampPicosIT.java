/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.beam.sdk.io.gcp.bigquery;

import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;
import com.google.cloud.bigquery.storage.v1.DataFormat;
import java.security.SecureRandom;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.extensions.gcp.options.GcpOptions;
import org.apache.beam.sdk.io.gcp.testing.BigqueryClient;
import org.apache.beam.sdk.testing.PAssert;
import org.apache.beam.sdk.testing.TestPipeline;
import org.apache.beam.sdk.transforms.Create;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.vendor.guava.v32_1_2_jre.com.google.common.collect.ImmutableList;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for BigQuery TIMESTAMP with various precisions (picos, nanos, micros, millis).
 * Tests write using Storage Write API and read back using readTableRows with different format and
 * precision combinations.
 */
@RunWith(JUnit4.class)
public class BigQueryTimestampPicosIT {

  private static String project;
  private static final String DATASET_ID =
      "bq_timestamp_picos_it_" + System.currentTimeMillis() + "_" + new SecureRandom().nextInt(32);

  private static TestBigQueryOptions bqOptions;
  private static final BigqueryClient BQ_CLIENT = new BigqueryClient("BigQueryTimestampPicosIT");

  private static String tableSpec;
  private static final String TABLE_NAME = "timestamp_all_precisions";

  /**
   * Input rows with timestamp columns at different precisions. Each row contains: - ts_picos: full
   * picosecond precision (12 digits) in ISO format - ts_nanos: nanosecond precision (9 digits) in
   * UTC format (limited to int64 nanos range) - ts_micros: microsecond precision (6 digits) in UTC
   * format - ts_millis: millisecond precision (3 digits) in UTC format
   *
   * <p>Note: ts_nanos uses dates within int64 nanoseconds-since-epoch bounds (~1677-09-21 to
   * ~2262-04-11) since Avro/Arrow use int64 for nanos timestamps.
   */
  private static final List<TableRow> INPUT_ROWS =
      ImmutableList.of(
          new TableRow()
              .set("ts_picos", "2024-01-15T10:30:45.123456789012Z")
              .set("ts_nanos", "2024-01-15 10:30:45.123456789 UTC")
              .set("ts_micros", "2024-01-15 10:30:45.123456 UTC")
              .set("ts_millis", "2024-01-15 10:30:45.123 UTC"),
          new TableRow()
              .set("ts_picos", "2024-01-15T10:30:45.000000000001Z")
              .set("ts_nanos", "2024-01-15 10:30:45.000000001 UTC")
              .set("ts_micros", "2024-01-15 10:30:45.000001 UTC")
              .set("ts_millis", "2024-01-15 10:30:45.001 UTC"),
          new TableRow()
              .set("ts_picos", "0001-01-01T10:30:45.999999999999Z")
              .set("ts_nanos", "1677-09-21 00:12:43.145224192 UTC")
              .set("ts_micros", "0001-01-01 10:30:45.999999 UTC")
              .set("ts_millis", "0001-01-01 10:30:45.999 UTC"),
          new TableRow()
              .set("ts_picos", "1970-01-01T00:00:00.000000000001Z")
              .set("ts_nanos", "1970-01-01 00:00:00.000000001 UTC")
              .set("ts_micros", "1970-01-01 00:00:00.000001 UTC")
              .set("ts_millis", "1970-01-01 00:00:00.001 UTC"),
          new TableRow()
              .set("ts_picos", "9999-12-31T23:59:59.999999999999Z")
              .set("ts_nanos", "2262-04-11 23:47:16.854775807 UTC")
              .set("ts_micros", "9999-12-31 23:59:59.999999 UTC")
              .set("ts_millis", "9999-12-31 23:59:59.999 UTC"));

  // Expected values for each column when read at PICOS precision (ISO format, 12 digits)
  private static final List<String> EXPECTED_TS_PICOS_AT_PICOS =
      ImmutableList.of(
          "2024-01-15T10:30:45.123456789012Z",
          "2024-01-15T10:30:45.000000000001Z",
          "0001-01-01T10:30:45.999999999999Z",
          "1970-01-01T00:00:00.000000000001Z",
          "9999-12-31T23:59:59.999999999999Z");

  private static final List<String> EXPECTED_TS_NANOS_AT_PICOS =
      ImmutableList.of(
          "2024-01-15T10:30:45.123456789000Z",
          "2024-01-15T10:30:45.000000001000Z",
          "1677-09-21T00:12:43.145224192000Z",
          "1970-01-01T00:00:00.000000001000Z",
          "2262-04-11T23:47:16.854775807000Z");

  private static final List<String> EXPECTED_TS_MICROS_AT_PICOS =
      ImmutableList.of(
          "2024-01-15T10:30:45.123456000000Z",
          "2024-01-15T10:30:45.000001000000Z",
          "0001-01-01T10:30:45.999999000000Z",
          "1970-01-01T00:00:00.000001000000Z",
          "9999-12-31T23:59:59.999999000000Z");

  private static final List<String> EXPECTED_TS_MILLIS_AT_PICOS =
      ImmutableList.of(
          "2024-01-15T10:30:45.123000000000Z",
          "2024-01-15T10:30:45.001000000000Z",
          "0001-01-01T10:30:45.999000000000Z",
          "1970-01-01T00:00:00.001000000000Z",
          "9999-12-31T23:59:59.999000000000Z");

  // Expected values when read at NANOS precision (UTC format, 9 digits)
  private static final List<String> EXPECTED_TS_PICOS_AT_NANOS =
      ImmutableList.of(
          "2024-01-15 10:30:45.123456789 UTC",
          "2024-01-15 10:30:45.000000000 UTC",
          "0001-01-01 10:30:45.999999999 UTC",
          "1970-01-01 00:00:00.000000000 UTC",
          "9999-12-31 23:59:59.999999999 UTC");

  private static final List<String> EXPECTED_TS_NANOS_AT_NANOS =
      ImmutableList.of(
          "2024-01-15 10:30:45.123456789 UTC",
          "2024-01-15 10:30:45.000000001 UTC",
          "1677-09-21 00:12:43.145224192 UTC",
          "1970-01-01 00:00:00.000000001 UTC",
          "2262-04-11 23:47:16.854775807 UTC");

  private static final List<String> EXPECTED_TS_MICROS_AT_NANOS =
      ImmutableList.of(
          "2024-01-15 10:30:45.123456000 UTC",
          "2024-01-15 10:30:45.000001000 UTC",
          "0001-01-01 10:30:45.999999000 UTC",
          "1970-01-01 00:00:00.000001000 UTC",
          "9999-12-31 23:59:59.999999000 UTC");

  private static final List<String> EXPECTED_TS_MILLIS_AT_NANOS =
      ImmutableList.of(
          "2024-01-15 10:30:45.123000000 UTC",
          "2024-01-15 10:30:45.001000000 UTC",
          "0001-01-01 10:30:45.999000000 UTC",
          "1970-01-01 00:00:00.001000000 UTC",
          "9999-12-31 23:59:59.999000000 UTC");

  // Expected values when read at MICROS precision (UTC format, 6 digits)
  private static final List<String> EXPECTED_TS_PICOS_AT_MICROS =
      ImmutableList.of(
          "2024-01-15 10:30:45.123456 UTC",
          "2024-01-15 10:30:45.000000 UTC",
          "0001-01-01 10:30:45.999999 UTC",
          "1970-01-01 00:00:00.000000 UTC",
          "9999-12-31 23:59:59.999999 UTC");

  private static final List<String> EXPECTED_TS_NANOS_AT_MICROS =
      ImmutableList.of(
          "2024-01-15 10:30:45.123456 UTC",
          "2024-01-15 10:30:45.000000 UTC",
          "1677-09-21 00:12:43.145224 UTC",
          "1970-01-01 00:00:00.000000 UTC",
          "2262-04-11 23:47:16.854775 UTC");

  private static final List<String> EXPECTED_TS_MICROS_AT_MICROS =
      ImmutableList.of(
          "2024-01-15 10:30:45.123456 UTC",
          "2024-01-15 10:30:45.000001 UTC",
          "0001-01-01 10:30:45.999999 UTC",
          "1970-01-01 00:00:00.000001 UTC",
          "9999-12-31 23:59:59.999999 UTC");

  private static final List<String> EXPECTED_TS_MILLIS_AT_MICROS =
      ImmutableList.of(
          "2024-01-15 10:30:45.123000 UTC",
          "2024-01-15 10:30:45.001000 UTC",
          "0001-01-01 10:30:45.999000 UTC",
          "1970-01-01 00:00:00.001000 UTC",
          "9999-12-31 23:59:59.999000 UTC");

  // Expected values when read at MICROS precision with ARROW format (truncated to millis due to
  // known issue)
  private static final List<String> EXPECTED_TS_PICOS_AT_MICROS_ARROW =
      ImmutableList.of(
          "2024-01-15 10:30:45.123000 UTC",
          "2024-01-15 10:30:45.000000 UTC",
          "0001-01-01 10:30:45.999000 UTC",
          "1970-01-01 00:00:00.000000 UTC",
          "9999-12-31 23:59:59.999000 UTC");

  private static final List<String> EXPECTED_TS_NANOS_AT_MICROS_ARROW =
      ImmutableList.of(
          "2024-01-15 10:30:45.123000 UTC",
          "2024-01-15 10:30:45.000000 UTC",
          "1677-09-21 00:12:43.145000 UTC",
          "1970-01-01 00:00:00.000000 UTC",
          "2262-04-11 23:47:16.854000 UTC");

  private static final List<String> EXPECTED_TS_MICROS_AT_MICROS_ARROW =
      ImmutableList.of(
          "2024-01-15 10:30:45.123000 UTC",
          "2024-01-15 10:30:45.000000 UTC",
          "0001-01-01 10:30:45.999000 UTC",
          "1970-01-01 00:00:00.000000 UTC",
          "9999-12-31 23:59:59.999000 UTC");

  private static final List<String> EXPECTED_TS_MILLIS_AT_MICROS_ARROW =
      ImmutableList.of(
          "2024-01-15 10:30:45.123000 UTC",
          "2024-01-15 10:30:45.001000 UTC",
          "0001-01-01 10:30:45.999000 UTC",
          "1970-01-01 00:00:00.001000 UTC",
          "9999-12-31 23:59:59.999000 UTC");

  @BeforeClass
  public static void setup() throws Exception {
    bqOptions = TestPipeline.testingPipelineOptions().as(TestBigQueryOptions.class);
    project = bqOptions.as(GcpOptions.class).getProject();

    // Create dataset
    BQ_CLIENT.createNewDataset(project, DATASET_ID, null, "us-central1");

    tableSpec = String.format("%s:%s.%s", project, DATASET_ID, TABLE_NAME);

    // Write data once for all tests
    Pipeline writePipeline = Pipeline.create(bqOptions);
    writePipeline
        .apply("CreateInput", Create.of(INPUT_ROWS))
        .apply(
            "WriteToBQ",
            BigQueryIO.writeTableRows()
                .to(tableSpec)
                .withSchema(multiPrecisionSchema())
                .withMethod(BigQueryIO.Write.Method.STORAGE_WRITE_API)
                .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_IF_NEEDED)
                .withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_TRUNCATE));
    writePipeline.run().waitUntilFinish();
  }

  @AfterClass
  public static void cleanup() {
    BQ_CLIENT.deleteDataset(project, DATASET_ID);
  }

  /**
   * Schema with TIMESTAMP fields at different precisions: - ts_picos: 12 fractional digits
   * (picoseconds) - ts_nanos: 9 fractional digits (nanoseconds) - ts_micros: 6 fractional digits
   * (microseconds) - ts_millis: 3 fractional digits (milliseconds)
   */
  private static TableSchema multiPrecisionSchema() {
    return new TableSchema()
        .setFields(
            ImmutableList.of(
                new TableFieldSchema()
                    .setName("ts_picos")
                    .setType("TIMESTAMP")
                    .setTimestampPrecision(12L),
                new TableFieldSchema()
                    .setName("ts_nanos")
                    .setType("TIMESTAMP")
                    .setTimestampPrecision(9L),
                new TableFieldSchema()
                    .setName("ts_micros")
                    .setType("TIMESTAMP")
                    .setTimestampPrecision(6L),
                new TableFieldSchema()
                    .setName("ts_millis")
                    .setType("TIMESTAMP")
                    .setTimestampPrecision(3L)));
  }

  /** Builds expected TableRows from expected values for each column. */
  private List<TableRow> buildExpectedRows(
      List<String> tsPicos, List<String> tsNanos, List<String> tsMicros, List<String> tsMillis) {
    return java.util.stream.IntStream.range(0, tsPicos.size())
        .mapToObj(
            i ->
                new TableRow()
                    .set("ts_picos", tsPicos.get(i))
                    .set("ts_nanos", tsNanos.get(i))
                    .set("ts_micros", tsMicros.get(i))
                    .set("ts_millis", tsMillis.get(i)))
        .collect(Collectors.toList());
  }

  private void runReadTest(
      TimestampPrecision precision, DataFormat format, List<TableRow> expectedRows) {

    Pipeline readPipeline = Pipeline.create(bqOptions);

    PCollection<TableRow> readTableRows =
        readPipeline.apply(
            "ReadTableRows",
            BigQueryIO.readTableRows()
                .withMethod(BigQueryIO.TypedRead.Method.DIRECT_READ)
                .withFormat(format)
                .withDirectReadPicosTimestampPrecision(precision)
                .from(tableSpec));

    PCollection<TableRow> readTableRowsWithQuery =
        readPipeline.apply(
            "ReadTableRowsWithFromQuery",
            BigQueryIO.readTableRows()
                .withMethod(BigQueryIO.TypedRead.Method.DIRECT_READ)
                .fromQuery(String.format("SELECT * FROM %s.%s.%s", project, DATASET_ID, TABLE_NAME))
                .usingStandardSql()
                .withFormat(format)
                .withDirectReadPicosTimestampPrecision(precision));

    PAssert.that(readTableRows).containsInAnyOrder(expectedRows);
    PAssert.that(readTableRowsWithQuery).containsInAnyOrder(expectedRows);

    readPipeline.run().waitUntilFinish();
  }

  @Test
  public void testRead_Picos_Avro() {
    List<TableRow> expected =
        buildExpectedRows(
            EXPECTED_TS_PICOS_AT_PICOS,
            EXPECTED_TS_NANOS_AT_PICOS,
            EXPECTED_TS_MICROS_AT_PICOS,
            EXPECTED_TS_MILLIS_AT_PICOS);
    runReadTest(TimestampPrecision.PICOS, DataFormat.AVRO, expected);
  }

  @Test
  public void testRead_Picos_Arrow() {
    List<TableRow> expected =
        buildExpectedRows(
            EXPECTED_TS_PICOS_AT_PICOS,
            EXPECTED_TS_NANOS_AT_PICOS,
            EXPECTED_TS_MICROS_AT_PICOS,
            EXPECTED_TS_MILLIS_AT_PICOS);
    runReadTest(TimestampPrecision.PICOS, DataFormat.ARROW, expected);
  }

  @Test
  public void testRead_Nanos_Avro() {
    List<TableRow> expected =
        buildExpectedRows(
            EXPECTED_TS_PICOS_AT_NANOS,
            EXPECTED_TS_NANOS_AT_NANOS,
            EXPECTED_TS_MICROS_AT_NANOS,
            EXPECTED_TS_MILLIS_AT_NANOS);
    runReadTest(TimestampPrecision.NANOS, DataFormat.AVRO, expected);
  }

  @Test
  public void testRead_Nanos_Arrow() {
    List<TableRow> expected =
        buildExpectedRows(
            EXPECTED_TS_PICOS_AT_NANOS,
            EXPECTED_TS_NANOS_AT_NANOS,
            EXPECTED_TS_MICROS_AT_NANOS,
            EXPECTED_TS_MILLIS_AT_NANOS);
    runReadTest(TimestampPrecision.NANOS, DataFormat.ARROW, expected);
  }

  @Test
  public void testRead_Micros_Avro() {
    List<TableRow> expected =
        buildExpectedRows(
            EXPECTED_TS_PICOS_AT_MICROS,
            EXPECTED_TS_NANOS_AT_MICROS,
            EXPECTED_TS_MICROS_AT_MICROS,
            EXPECTED_TS_MILLIS_AT_MICROS);
    runReadTest(TimestampPrecision.MICROS, DataFormat.AVRO, expected);
  }

  @Test
  public void testRead_Micros_Arrow() {
    // Arrow micros has a known issue where values are truncated to milliseconds
    List<TableRow> expected =
        buildExpectedRows(
            EXPECTED_TS_PICOS_AT_MICROS_ARROW,
            EXPECTED_TS_NANOS_AT_MICROS_ARROW,
            EXPECTED_TS_MICROS_AT_MICROS_ARROW,
            EXPECTED_TS_MILLIS_AT_MICROS_ARROW);
    runReadTest(TimestampPrecision.MICROS, DataFormat.ARROW, expected);
  }
}
