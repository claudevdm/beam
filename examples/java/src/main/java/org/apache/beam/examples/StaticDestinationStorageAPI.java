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
package org.apache.beam.examples;

import com.google.api.services.bigquery.model.TableFieldSchema;
import com.google.api.services.bigquery.model.TableRow;
import com.google.api.services.bigquery.model.TableSchema;
import java.time.Instant;
import java.util.Arrays;
import java.util.List;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.io.gcp.bigquery.BigQueryIO;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.vendor.guava.v32_1_2_jre.com.google.common.collect.ImmutableList;

public class StaticDestinationStorageAPI {
  public static void main(String[] args) {
    Pipeline p = Pipeline.create(PipelineOptionsFactory.fromArgs(args).create());

    // Single schema for all records
    TableSchema schema =
        new TableSchema()
            .setFields(
                ImmutableList.of(
                    new TableFieldSchema()
                        .setName("ts")
                        .setType("TIMESTAMP")
                        .setMode("REQUIRED")
                        .setTimestampPrecision(12L)));
    // .setTimestampPrecision(12L)));

    // Create fake timestamp data - using epoch microseconds
    List<TableRow> rows =
        Arrays.asList(
            new TableRow().set("ts", Instant.parse("2024-01-15T10:30:01.123456789Z").toString()),
            new TableRow().set("ts", Instant.parse("2024-02-20T14:45:02.123456789Z").toString()),
            new TableRow().set("ts", Instant.parse("2024-03-25T08:15:03.123456789Z").toString()),
            new TableRow().set("ts", Instant.now().toString()));

    // p.apply("CreateRows", Create.of(rows))
    //     .apply(
    //         "WriteToBQ",
    //         BigQueryIO.writeTableRows()
    //             // Static destination - single table
    //             .to("poject_id.cvandermerwe_test.created")
    //             .withSchema(schema)
    //             .withMethod(BigQueryIO.Write.Method.STORAGE_WRITE_API) // Exactly-once semantics
    //             .withCreateDisposition(BigQueryIO.Write.CreateDisposition.CREATE_IF_NEEDED)
    //             .withWriteDisposition(BigQueryIO.Write.WriteDisposition.WRITE_APPEND));
    p.apply(
        "Read from BigQuery table",
        BigQueryIO.readTableRowsWithSchema()
            // BigQueryIO.readTableRows()
            .withMethod(BigQueryIO.TypedRead.Method.DIRECT_READ)
            .withFormat(com.google.cloud.bigquery.storage.v1.DataFormat.ARROW)
            .from("poject_id.cvandermerwe_test.created")
            .withTimestampPrecision(org.apache.beam.sdk.io.gcp.bigquery.TimestampPrecision.NANOS));

    p.run().waitUntilFinish();
  }
}
