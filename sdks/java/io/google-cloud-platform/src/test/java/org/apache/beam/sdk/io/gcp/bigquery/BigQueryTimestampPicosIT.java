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
 * Integration tests for BigQuery TIMESTAMP with picosecond precision (precision=12). Tests write
 * using Storage Write API and read back using readTableRows.
 */
@RunWith(JUnit4.class)
public class BigQueryTimestampPicosIT {

  private static String project;
  // private static final String DATASET_ID = "cvandermerwe_regional_test";
  private static final String DATASET_ID =
      "bq_timestamp_picos_it_" + System.currentTimeMillis() + "_" + new SecureRandom().nextInt(32);

  private static TestBigQueryOptions bqOptions;
  private static final BigqueryClient BQ_CLIENT = new BigqueryClient("BigQueryTimestampPicosIT");

  private static final List<TableRow> ISO_PICOS_TABLEROWS =
      ImmutableList.of(
          new TableRow().set("ts_picos", "2024-01-15T10:30:45.123456789012Z"),
          new TableRow().set("ts_picos", "2024-01-15T10:30:45.000000000001Z"),
          new TableRow().set("ts_picos", "0001-01-01T10:30:45.999999999999Z"),
          new TableRow().set("ts_picos", "1970-01-01T00:00:00.000000000001Z"),
          new TableRow().set("ts_picos", "9999-12-31T23:59:59.999999999999Z"));

  private static final List<TableRow> UTC_NANOS_TABLEROWS =
      ImmutableList.of(
          new TableRow().set("ts_picos", "2024-01-15 10:30:45.123456789 UTC"),
          new TableRow().set("ts_picos", "2024-01-15 10:30:45.011111111 UTC"),
          new TableRow().set("ts_picos", "2262-04-11 23:47:16.854775807 UTC"),
          new TableRow().set("ts_picos", "1970-01-01 00:00:00.001111111 UTC"),
          new TableRow().set("ts_picos", "1677-09-21 00:12:43.145224192 UTC"));

  private static final List<TableRow> UTC_MICROS_TABLEROWS =
      ImmutableList.of(
          new TableRow().set("ts_picos", "2024-01-15 10:30:45.123456 UTC"),
          new TableRow().set("ts_picos", "2024-01-15 10:30:45.011111 UTC"),
          new TableRow().set("ts_picos", "0001-01-01 10:30:45.999999 UTC"),
          new TableRow().set("ts_picos", "1970-01-01 00:00:00.001111 UTC"),
          new TableRow().set("ts_picos", "9999-12-31 23:59:59.999999 UTC"));

  private static final List<TableRow> UTC_MILLIS_TABLEROWS =
      ImmutableList.of(
          new TableRow().set("ts_picos", "2024-01-15 10:30:45.123 UTC"),
          new TableRow().set("ts_picos", "2024-01-15 10:30:45.011 UTC"),
          new TableRow().set("ts_picos", "0001-01-01 10:30:45.999 UTC"),
          new TableRow().set("ts_picos", "1970-01-01 00:00:00.001 UTC"),
          new TableRow().set("ts_picos", "9999-12-31 23:59:59.999 UTC"));

  @BeforeClass
  public static void setup() throws Exception {
    bqOptions = TestPipeline.testingPipelineOptions().as(TestBigQueryOptions.class);
    project = bqOptions.as(GcpOptions.class).getProject();
    // Create dataset for all test cases
    BQ_CLIENT.createNewDataset(project, DATASET_ID, null, "us-central1");
  }

  @AfterClass
  public static void cleanup() {
    BQ_CLIENT.deleteDataset(project, DATASET_ID);
  }

  /** Schema with a TIMESTAMP field having picosecond precision (12 fractional digits). */
  private TableSchema timestampPicosSchema() {
    return new TableSchema()
        .setFields(
            ImmutableList.of(
                new TableFieldSchema()
                    .setName("ts_picos")
                    .setType("TIMESTAMP")
                    .setTimestampPrecision(12L))); // Picosecond precision
  }

  private void runTimestampTest(
      TimestampPrecision precision,
      DataFormat format,
      List<TableRow> inputRows,
      List<TableRow> expectedRows) {

    String tableName = String.format("ts_%s_%s_%d", precision, format, System.currentTimeMillis());
    String tableSpec = String.format("%s:%s.%s", project, DATASET_ID, tableName);

    // Write (Always writes the full PICOS precision)
    Pipeline writePipeline = Pipeline.create(bqOptions);
    writePipeline
        .apply("CreateInput", Create.of(inputRows))
        .apply(
            "WriteToBQ",
            BigQueryIO.writeTableRows()
                .to(tableSpec)
                .withSchema(timestampPicosSchema())
                .withMethod(BigQueryIO.Write.Method.STORAGE_WRITE_API)
                .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_IF_NEEDED)
                .withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_TRUNCATE));
    writePipeline.run().waitUntilFinish();

    // Read & Verify
    Pipeline readPipeline = Pipeline.create(bqOptions);

    PCollection<TableRow> readTableRows =
        readPipeline.apply(
            "ReadTableRows",
            BigQueryIO.readTableRows()
                .withMethod(BigQueryIO.TypedRead.Method.DIRECT_READ)
                .withFormat(format)
                .withDirectReadPicosTimestampPrecision(precision)
                .from(tableSpec));
    PCollection<TableRow> readTableRowsWithSchema =
        readPipeline.apply(
            "ReadTableRowsWithSchema",
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
                .fromQuery(String.format("SELECT * FROM %s.%s.%s", project, DATASET_ID, tableName))
                .usingStandardSql()
                .withFormat(format)
                .withDirectReadPicosTimestampPrecision(precision));

    PAssert.that(readTableRows).containsInAnyOrder(expectedRows);
    PAssert.that(readTableRowsWithSchema).containsInAnyOrder(expectedRows);
    PAssert.that(readTableRowsWithQuery).containsInAnyOrder(expectedRows);

    readPipeline.run().waitUntilFinish();
  }

  @Test
  public void testPicos_Avro_roundTrip() {
    // Expect exact match of input (12 digits)
    runTimestampTest(
        TimestampPrecision.PICOS, DataFormat.AVRO, ISO_PICOS_TABLEROWS, ISO_PICOS_TABLEROWS);
  }

  @Test
  public void testPicos_Arrow_roundTrip() {
    runTimestampTest(
        TimestampPrecision.PICOS, DataFormat.ARROW, ISO_PICOS_TABLEROWS, ISO_PICOS_TABLEROWS);
  }

  @Test
  public void testNanos_Avro_roundTrip() {
    runTimestampTest(
        TimestampPrecision.NANOS, DataFormat.AVRO, UTC_NANOS_TABLEROWS, UTC_NANOS_TABLEROWS);
  }

  @Test
  public void testNanos_Arrow_roundTrip() {
    runTimestampTest(
        TimestampPrecision.NANOS, DataFormat.ARROW, UTC_NANOS_TABLEROWS, UTC_NANOS_TABLEROWS);
  }

  @Test
  public void testMicros_Avro_roundTrip() {
    runTimestampTest(
        TimestampPrecision.MICROS, DataFormat.AVRO, UTC_MICROS_TABLEROWS, UTC_MICROS_TABLEROWS);
  }

  @Test
  public void testMicros_Arrow_roundTrip() {
    runTimestampTest(
        TimestampPrecision.MICROS, DataFormat.ARROW, UTC_MICROS_TABLEROWS, UTC_MILLIS_TABLEROWS);
  }
}
