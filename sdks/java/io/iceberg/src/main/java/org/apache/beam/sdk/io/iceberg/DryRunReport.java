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
package org.apache.beam.sdk.io.iceberg;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.apache.beam.sdk.io.iceberg.SchemaEvolutionConfig.IncompatibleSchemaHandling;
import org.apache.beam.sdk.schemas.Schema;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.windowing.BoundedWindow;
import org.apache.beam.sdk.values.Row;
import org.apache.beam.vendor.guava.v32_1_2_jre.com.google.common.hash.Hashing;
import org.apache.iceberg.SchemaParser;
import org.apache.iceberg.Table;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.exceptions.NoSuchTableException;
import org.checkerframework.checker.nullness.qual.MonotonicNonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.joda.time.Instant;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Dry run of the schema pre-pass: classifies each distinct file schema against the table and
 * reports what a real run would do, without committing or registering anything. One row per
 * distinct schema plus one summary row per window; the rendered table is also logged.
 */
class DryRunReport extends DoFn<List<CollectDistinctSchemas.SchemaGroup>, Row> {
  private static final Logger LOG = LoggerFactory.getLogger(DryRunReport.class);

  static final Schema REPORT_SCHEMA =
      Schema.builder()
          .addStringField("schema_key")
          .addStringField("schema")
          .addInt64Field("num_files")
          .addArrayField("changes", Schema.FieldType.STRING)
          .addBooleanField("allowed")
          .addStringField("reason")
          .addBooleanField("would_create_table")
          .addBooleanField("summary")
          .addDateTimeField("window_end")
          .build();

  private final IcebergCatalogConfig catalogConfig;
  private final String identifier;
  private final SchemaEvolutionConfig config;
  private final IncompatibleSchemaHandling handling;
  private transient @MonotonicNonNull Catalog catalog;

  DryRunReport(
      IcebergCatalogConfig catalogConfig,
      String identifier,
      SchemaEvolutionConfig config,
      IncompatibleSchemaHandling handling) {
    this.catalogConfig = catalogConfig;
    this.identifier = identifier;
    this.config = config;
    this.handling = handling;
  }

  @ProcessElement
  public void process(
      @Element List<CollectDistinctSchemas.SchemaGroup> schemas,
      BoundedWindow window,
      OutputReceiver<Row> out) {
    if (catalog == null) {
      catalog = catalogConfig.catalog();
    }
    TableIdentifier tableId = IcebergUtils.parseTableIdentifier(identifier);
    @Nullable Table table;
    try {
      table = catalog.loadTable(tableId);
    } catch (NoSuchTableException e) {
      table = null;
    }
    Instant windowEnd = window.maxTimestamp();

    long totalFiles = 0;
    long allowedFiles = 0;
    int allowedSchemas = 0;
    int incompatibleSchemas = 0;
    long incompatibleFiles = 0;
    StringBuilder rendered = new StringBuilder();
    for (CollectDistinctSchemas.SchemaGroup group : schemas) {
      String json = group.schemaJson;
      long files = group.files;
      totalFiles += files;
      List<String> changes;
      boolean allowed;
      String reason = "";
      if (table == null) {
        changes = createChanges(json);
        allowed = true;
      } else {
        SchemaDelta delta =
            SchemaDelta.classify(
                table,
                FileSchemas.markRequired(SchemaParser.fromJson(json), group.nullFreeColumns));
        changes = delta.descriptions();
        allowed = delta.allowedBy(config);
        reason = allowed ? "" : delta.disallowedReason(config);
      }
      if (allowed) {
        allowedSchemas++;
        allowedFiles += files;
      } else {
        incompatibleSchemas++;
        incompatibleFiles += files;
      }
      Row row =
          Row.withSchema(REPORT_SCHEMA)
              .addValues(
                  key(json), json, files, changes, allowed, reason, table == null, false, windowEnd)
              .build();
      out.output(row);
      rendered.append(
          String.format(
              "  %-8s %8d  allowed=%-5s %s %s%n", key(json), files, allowed, changes, reason));
    }

    String summary =
        String.format(
            "%d distinct schemas; %d allowed covering %d files; %d incompatible covering %d files",
            schemas.size(), allowedSchemas, allowedFiles, incompatibleSchemas, incompatibleFiles);
    String consequence = "";
    if (incompatibleSchemas > 0) {
      consequence =
          handling == IncompatibleSchemaHandling.FAIL_PIPELINE
              ? "a real run would fail before committing (" + handling + ")"
              : "a real run would route "
                  + incompatibleFiles
                  + " files to errors ("
                  + handling
                  + ")";
    }
    out.output(
        Row.withSchema(REPORT_SCHEMA)
            .addValues(
                "",
                "",
                totalFiles,
                Collections.singletonList(summary),
                incompatibleSchemas == 0,
                consequence,
                table == null,
                true,
                windowEnd)
            .build());
    LOG.info(
        "Dry run for {}{}: {}{}\n{}",
        identifier,
        table == null ? " (table would be created)" : "",
        summary,
        consequence.isEmpty() ? "" : "; " + consequence,
        rendered);
  }

  /** With no table, every column of the schema would be created (optional unless pinned). */
  private List<String> createChanges(String json) {
    org.apache.iceberg.Schema seed =
        CommitSchemaUnion.relaxUnpinned(SchemaParser.fromJson(json), config);
    List<String> changes = new ArrayList<>();
    for (org.apache.iceberg.types.Types.NestedField field : seed.columns()) {
      changes.add(
          "create "
              + (field.isOptional() ? "optional " : "required ")
              + field.name()
              + " "
              + field.type());
    }
    return changes;
  }

  static String key(String schemaJson) {
    return "s"
        + Hashing.murmur3_32_fixed().hashUnencodedChars(schemaJson).toString().substring(0, 6);
  }
}
