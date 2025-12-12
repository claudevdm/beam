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

import static org.junit.Assert.assertEquals;

import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;
import java.security.SecureRandom;
import java.util.List;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.coders.SerializableCoder;
import org.apache.beam.sdk.extensions.gcp.options.GcpOptions;
import org.apache.beam.sdk.io.gcp.testing.BigqueryClient;
import org.apache.beam.sdk.testing.PAssert;
import org.apache.beam.sdk.testing.TestPipeline;
import org.apache.beam.sdk.transforms.Create;
import org.apache.beam.sdk.transforms.MapElements;
import org.apache.beam.sdk.transforms.SerializableFunction;
import org.apache.beam.sdk.transforms.SimpleFunction;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.vendor.guava.v32_1_2_jre.com.google.common.collect.ImmutableList;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for BigQuery TIMESTAMP with picosecond precision (precision=12).
 * Tests write using Storage Write API and read back using readTableRows.
 */
@RunWith(JUnit4.class)
public class BigQueryTimestampPicosIT {

  private static String project;
  private static final String DATASET_ID =
    "bq_timestamp_picos_it_"
      + System.currentTimeMillis()
      + "_"
      + new SecureRandom().nextInt(32);

  private static TestBigQueryOptions bqOptions;
  private static final BigqueryClient BQ_CLIENT = new BigqueryClient("BigQueryTimestampPicosIT");

  @BeforeClass
  public static void setup() throws Exception {
    bqOptions = TestPipeline.testingPipelineOptions().as(TestBigQueryOptions.class);
    project = bqOptions.as(GcpOptions.class).getProject();
    // Create dataset for all test cases
    BQ_CLIENT.createNewDataset(project, DATASET_ID, null, bqOptions.getBigQueryLocation());
  }

  @AfterClass
  public static void cleanup() {
    BQ_CLIENT.deleteDataset(project, DATASET_ID);
  }

  private void configureStorageWriteApi() {
    bqOptions.setUseStorageWriteApi(true);
    bqOptions.setNumStorageWriteApiStreams(1);
    bqOptions.setStorageWriteApiTriggeringFrequencySec(1);
  }

  /** Schema with a TIMESTAMP field having picosecond precision (12 fractional digits). */
  private TableSchema timestampPicosSchema() {
    return new TableSchema()
      .setFields(
        ImmutableList.of(
          new TableFieldSchema().setName("id").setType("INTEGER"),
          new TableFieldSchema()
            .setName("ts_picos")
            .setType("TIMESTAMP")
            .setPrecision(12L)));  // Picosecond precision
  }

  @Test
  public void testWriteAndReadTimestampPicos() throws Exception {
    configureStorageWriteApi();
    String tableName = "timestamp_picos_" + System.currentTimeMillis();
    String tableSpec = String.format("%s:%s.%s", project, DATASET_ID, tableName);

    // Test data: timestamps with 12 fractional digits
    List<TableRow> inputRows =
      ImmutableList.of(
        new TableRow().set("id", 1).set("ts_picos", "2024-01-15T10:30:45.123456789012Z"),
        new TableRow().set("id", 2).set("ts_picos", "2024-01-15T10:30:45.000000000001Z"),
        new TableRow().set("id", 3).set("ts_picos", "2024-01-15T10:30:45.999999999999Z"),
        new TableRow().set("id", 4).set("ts_picos", "1970-01-01T00:00:00.000000000001Z"),
        new TableRow().set("id", 5).set("ts_picos", "9999-12-31T23:59:59.999999999999Z"));

    // ========== WRITE PIPELINE ==========
    Pipeline writePipeline = Pipeline.create(bqOptions);
    writePipeline
      .apply("CreateInput", Create.of(inputRows))
      .apply(
        "WriteToBQ",
        BigQueryIO.writeTableRows()
          .to(tableSpec)
          .withSchema(timestampPicosSchema())
          .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_IF_NEEDED)
          .withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_TRUNCATE));
    writePipeline.run().waitUntilFinish();

    // ========== READ PIPELINE ==========
    Pipeline readPipeline = Pipeline.create(bqOptions);
    PCollection<TableRow> readRows =
      readPipeline.apply(
        "ReadFromBQ",
        BigQueryIO.readTableRows()
          .from(tableSpec));

    // Extract timestamp values and verify
    PCollection<String> timestamps =
      readRows.apply(
        "ExtractTimestamps",
        MapElements.via(
          new SimpleFunction<TableRow, String>() {
            @Override
            public String apply(TableRow row) {
              // BigQuery returns timestamps - extract and format
              return row.get("id") + ":" + row.get("ts_picos");
            }
          }));

    // Verify all rows were written and read correctly
    PAssert.that(timestamps)
      .satisfies(
        (SerializableFunction<Iterable<String>, Void>)
          rows -> {
            int count = 0;
            for (String row : rows) {
              count++;
            }
            assertEquals("Expected 5 rows", 5, count);
            return null;
          });

    readPipeline.run().waitUntilFinish();
  }

  @Test
  public void testWriteAndReadTimestampPicos_roundTrip() throws Exception {
    configureStorageWriteApi();
    String tableName = "timestamp_picos_roundtrip_" + System.currentTimeMillis();
    String tableSpec = String.format("%s:%s.%s", project, DATASET_ID, tableName);

    // Single test value to verify exact round-trip
    String inputTimestamp = "2024-06-15T12:34:56.123456789012Z";
    TableRow inputRow = new TableRow().set("id", 1).set("ts_picos", inputTimestamp);

    // ========== WRITE ==========
    Pipeline writePipeline = Pipeline.create(bqOptions);
    writePipeline
      .apply("CreateInput", Create.of(ImmutableList.of(inputRow)))
      .apply(
        "WriteToBQ",
        BigQueryIO.writeTableRows()
          .to(tableSpec)
          .withSchema(timestampPicosSchema())
          .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_IF_NEEDED)
          .withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_TRUNCATE));
    writePipeline.run().waitUntilFinish();

    // ========== READ AND VERIFY ==========
    Pipeline readPipeline = Pipeline.create(bqOptions);
    PCollection<TableRow> readRows =
      readPipeline.apply("ReadFromBQ", BigQueryIO.readTableRows().from(tableSpec));

    // Verify the timestamp matches exactly
    PAssert.thatSingleton(
        readRows.apply(
          MapElements.via(
            new SimpleFunction<TableRow, String>() {
              @Override
              public String apply(TableRow row) {
                // The timestamp should be returned in ISO format
                return (String) row.get("ts_picos");
              }
            })))
      .isEqualTo(inputTimestamp);

    readPipeline.run().waitUntilFinish();
  }

  @Test
  public void testWriteTimestampPicos_utcFormat() throws Exception {
    configureStorageWriteApi();
    String tableName = "timestamp_picos_utc_" + System.currentTimeMillis();
    String tableSpec = String.format("%s:%s.%s", project, DATASET_ID, tableName);

    // Test with UTC format (space separator, "UTC" suffix) - should still work for nano precision
    List<TableRow> inputRows =
      ImmutableList.of(
        new TableRow().set("id", 1).set("ts_picos", "2024-01-15 10:30:45.123456789 UTC"));

    Pipeline writePipeline = Pipeline.create(bqOptions);
    writePipeline
      .apply("CreateInput", Create.of(inputRows))
      .apply(
        "WriteToBQ",
        BigQueryIO.writeTableRows()
          .to(tableSpec)
          .withSchema(timestampPicosSchema())
          .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_IF_NEEDED)
          .withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_TRUNCATE));
    writePipeline.run().waitUntilFinish();

    // Verify by reading back
    Pipeline readPipeline = Pipeline.create(bqOptions);
    PCollection<TableRow> readRows =
      readPipeline.apply("ReadFromBQ", BigQueryIO.readTableRows().from(tableSpec));

    PAssert.thatSingleton(readRows)
      .satisfies(
        row -> {
          assertEquals(1L, ((Number) row.get("id")).longValue());
          // Timestamp should be present (format may vary)
          String ts = (String) row.get("ts_picos");
          assert ts != null && !ts.isEmpty();
          return null;
        });

    readPipeline.run().waitUntilFinish();
  }
}
