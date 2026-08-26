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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.beam.sdk.io.iceberg.AddFilesFuzz.FileSpec;
import org.apache.beam.sdk.io.iceberg.AddFilesFuzz.Scenario;
import org.apache.beam.sdk.io.iceberg.AddFilesFuzz.TableState;
import org.apache.beam.sdk.values.Row;
import org.apache.iceberg.BaseTable;
import org.apache.iceberg.DataFile;
import org.apache.iceberg.FileScanTask;
import org.apache.iceberg.PartitionSpec;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Snapshot;
import org.apache.iceberg.SortOrder;
import org.apache.iceberg.Table;
import org.apache.iceberg.TableProperties;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.data.parquet.GenericParquetReaders;
import org.apache.iceberg.hadoop.HadoopCatalog;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.mapping.MappedField;
import org.apache.iceberg.mapping.NameMapping;
import org.apache.iceberg.parquet.Parquet;
import org.apache.iceberg.types.TypeUtil;
import org.apache.iceberg.types.Types;
import org.checkerframework.checker.nullness.qual.Nullable;

/** The invariants every AddFiles run must satisfy, checked against the table and the outputs. */
final class AddFilesFuzzSignals {
  private AddFilesFuzzSignals() {}

  /** Table state captured before the run. */
  static final class Before {
    final boolean exists;
    final @Nullable String metadataLocation;
    final @Nullable Schema schema;
    final @Nullable PartitionSpec spec;
    final @Nullable SortOrder sortOrder;
    final Map<String, String> properties;
    final int snapshots;

    Before(Scenario scenario) {
      HadoopCatalog catalog = scenario.catalog();
      exists = catalog.tableExists(scenario.tableId);
      if (!exists) {
        metadataLocation = null;
        schema = null;
        spec = null;
        sortOrder = null;
        properties = new HashMap<>();
        snapshots = 0;
        return;
      }
      Table table = catalog.loadTable(scenario.tableId);
      metadataLocation = location(table);
      schema = table.schema();
      spec = table.spec();
      sortOrder = table.sortOrder();
      properties = new HashMap<>(table.properties());
      int count = 0;
      for (Snapshot ignored : table.snapshots()) {
        count++;
      }
      snapshots = count;
    }
  }

  /** What the run produced. */
  static final class Outcome {
    final @Nullable Exception failure;
    final List<Row> errors;
    final Map<String, Long> counters;

    Outcome(@Nullable Exception failure, List<Row> errors, Map<String, Long> counters) {
      this.failure = failure;
      this.errors = errors;
      this.counters = counters;
    }
  }

  static final class Violation extends AssertionError {
    Violation(String message) {
      super(message);
    }
  }

  @com.google.errorprone.annotations.FormatMethod
  private static void check(
      boolean condition,
      @com.google.errorprone.annotations.FormatString String message,
      Object... args) {
    if (!condition) {
      throw new Violation(String.format(message, args));
    }
  }

  static void verify(
      Scenario scenario, Before before, Outcome outcome, AddFilesFuzzExpectations.Result expected)
      throws IOException {
    HadoopCatalog catalog = scenario.catalog();
    boolean exists = catalog.tableExists(scenario.tableId);

    if (outcome.failure != null) {
      verifyFailure(scenario, before, outcome, catalog, exists);
      if (scenario.disturbance != AddFilesFuzz.Disturbance.TABLE_DROPPED_AFTER_COMMIT) {
        check(
            expected.pipelineFails, "pipeline failed but no file was expected to be incompatible");
      }
      return;
    }
    check(!expected.pipelineFails, "an incompatible file was expected to fail the pipeline");

    Map<String, FileSpec> specs = new LinkedHashMap<>();
    for (FileSpec file : scenario.files) {
      specs.put(file.path, file);
    }
    Set<String> errorPaths = new HashSet<>();
    for (Row row : outcome.errors) {
      String path = row.getString("file");
      check(errorPaths.add(path), "duplicate error row for %s", path);
      check(specs.containsKey(path), "error row for unknown path %s", path);
    }

    if (!exists) {
      // nothing registered, nothing created: every path must be an error
      check(
          errorPaths.size() == specs.size(),
          "table missing after the run but only %d of %d paths errored",
          errorPaths.size(),
          specs.size());
      return;
    }

    if (scenario.state != TableState.MISSING
        && scenario.disturbance == AddFilesFuzz.Disturbance.TABLE_DROPPED_AFTER_COMMIT) {
      // The table was dropped under the pipeline; registration recreated it from the first file
      // (today's create path) or failed. Only conservation is meaningful here.
      return;
    }
    Table table = catalog.loadTable(scenario.tableId);
    Map<String, DataFile> registered = new LinkedHashMap<>();
    Map<String, Integer> schemaIdByPath = new HashMap<>();
    for (Snapshot snapshot : table.snapshots()) {
      for (DataFile file : snapshot.addedDataFiles(table.io())) {
        String path = file.path().toString();
        check(!registered.containsKey(path), "path registered twice: %s", path);
        registered.put(path, file);
        schemaIdByPath.put(
            path, snapshot.schemaId() == null ? table.schema().schemaId() : snapshot.schemaId());
      }
    }
    // stats need the column-stats scan
    Map<String, DataFile> withStats = new HashMap<>();
    for (FileScanTask task : table.newScan().includeColumnStats().planFiles()) {
      withStats.put(task.file().path().toString(), task.file());
    }

    // 1. conservation
    for (String path : specs.keySet()) {
      boolean isRegistered = registered.containsKey(path);
      boolean isError = errorPaths.contains(path);
      check(isRegistered != isError, "path %s registered=%s error=%s", path, isRegistered, isError);
    }
    check(
        registered.size() + errorPaths.size() == specs.size(),
        "registered %d + errors %d != inputs %d",
        registered.size(),
        errorPaths.size(),
        specs.size());

    // 1b. per-file expectations from the specs
    for (Map.Entry<String, AddFilesFuzzExpectations.Expected> entry : expected.byPath.entrySet()) {
      String path = entry.getKey();
      boolean isRegistered = registered.containsKey(path);
      switch (entry.getValue()) {
        case REGISTER:
          check(
              isRegistered,
              "%s expected to register but errored: %s",
              path,
              errorMessage(outcome, path));
          break;
        case ERROR_UNREADABLE:
          check(!isRegistered, "%s (unreadable) was registered", path);
          break;
        case ERROR_INCOMPATIBLE:
          check(!isRegistered, "%s (incompatible) was registered", path);
          check(
              errorMessage(outcome, path).contains("does not cover the file"),
              "%s: expected an uncovered error, got: %s",
              path,
              errorMessage(outcome, path));
          break;
        case ERROR_PIN:
          check(!isRegistered, "%s (pin violated) was registered", path);
          check(
              errorMessage(outcome, path).contains("Pinned required column"),
              "%s: expected a pin error, got: %s",
              path,
              errorMessage(outcome, path));
          break;
        default:
          throw new IllegalStateException();
      }
    }

    // 1c. differential checks independent of the classifier
    boolean everythingAllowed =
        scenario.config.getOptions().size() == SchemaEvolutionOption.values().length
            && scenario.config.getRequiredColumns().isEmpty();
    if (everythingAllowed || !scenario.config.isEnabled()) {
      for (FileSpec spec : scenario.files) {
        boolean readable = spec.readable() && spec.kind != AddFilesFuzz.FileKind.AVRO_EXTENSION;
        if (readable) {
          check(
              registered.containsKey(spec.path),
              "%s should always register under this config: %s",
              spec.path,
              errorMessage(outcome, spec.path));
        }
      }
    }

    // 2. gate ordering: stats only for ids that existed at the file's snapshot
    for (Map.Entry<String, DataFile> entry : withStats.entrySet()) {
      Schema at = table.schemas().get(schemaIdByPath.get(entry.getKey()));
      check(at != null, "no schema for snapshot of %s", entry.getKey());
      Map<Integer, Long> nulls = entry.getValue().nullValueCounts();
      if (nulls != null) {
        for (Integer id : nulls.keySet()) {
          check(
              at.findField(id) != null,
              "stats for id %d absent from schema at snapshot for %s",
              id,
              entry.getKey());
        }
      }
    }

    // 3. policy: schema diff by id (an externally added column is not ours to judge)
    Schema after = table.schema();
    boolean external =
        scenario.disturbance == AddFilesFuzz.Disturbance.EXTERNAL_COLUMN_AFTER_COMMIT
            || scenario.disturbance == AddFilesFuzz.Disturbance.COMMIT_FAILS_ONCE;
    boolean fired =
        AddFilesFuzzTest.DisturbingCommitter.fired(scenario.warehouse, scenario.tableId.toString());
    if (external) {
      // the disturbance fires inside the schema commit; with nothing to commit it never runs
      if (fired) {
        check(after.findField(AddFilesFuzz.EXTERNAL_COLUMN) != null, "external column lost");
      }
      after = withoutExternal(after);
    }
    if (before.schema != null) {
      SchemaDelta delta = SchemaDelta.diff(before.schema, after);
      check(
          delta.conflict() == null,
          "schema changed in a way a union cannot produce: %s",
          delta.descriptions());
      for (SchemaDelta.Kind kind : delta.kinds()) {
        check(
            kind.option != null && scenario.config.allows(kind.option),
            "disallowed change applied: %s",
            delta.descriptions());
      }
      int maxBefore = 0;
      for (Types.NestedField field : TypeUtil.indexById(before.schema.asStruct()).values()) {
        maxBefore = Math.max(maxBefore, field.fieldId());
      }
      for (Map.Entry<Integer, Types.NestedField> entry :
          TypeUtil.indexById(after.asStruct()).entrySet()) {
        Types.NestedField old = TypeUtil.indexById(before.schema.asStruct()).get(entry.getKey());
        if (old == null) {
          check(
              entry.getKey() > maxBefore,
              "new field id %d not above previous max %d",
              entry.getKey(),
              maxBefore);
          check(entry.getValue().isOptional(), "new field %s is required", entry.getValue().name());
        } else {
          check(old.name().equals(entry.getValue().name()), "field %d renamed", entry.getKey());
        }
      }
    }
    for (String pinned : scenario.config.getRequiredColumns()) {
      Types.NestedField field = after.findField(pinned);
      if (field != null) {
        check(field.isRequired(), "pinned column %s is optional", pinned);
      }
    }
    for (String ignored : scenario.config.getIgnoredColumns()) {
      if (before.schema == null || before.schema.findField(ignored) == null) {
        check(after.findField(ignored) == null, "ignored column %s was added", ignored);
      }
    }
    for (String alias : scenario.config.getColumnAliases().keySet()) {
      check(after.findField(alias) == null, "alias %s became a column", alias);
    }

    // 4. spec, sort order, properties untouched
    if (before.spec != null) {
      check(before.spec.equals(table.spec()), "partition spec changed");
      check(before.sortOrder.equals(table.sortOrder()), "sort order changed");
      for (Map.Entry<String, String> property : before.properties.entrySet()) {
        if (!property.getKey().equals(TableProperties.DEFAULT_NAME_MAPPING)) {
          check(
              property.getValue().equals(table.properties().get(property.getKey())),
              "property %s changed",
              property.getKey());
        }
      }
    }

    // 5. mapping health
    @Nullable
    NameMapping mapping =
        NameMappingUtils.parseOrNull(table.properties().get(TableProperties.DEFAULT_NAME_MAPPING));
    boolean schemaChanged = before.schema == null || !before.schema.sameSchema(table.schema());
    if (!registered.isEmpty() || schemaChanged) {
      check(mapping != null, "no parseable name mapping after the run");
      check(
          NameMappingUtils.covers(mapping, after.asStruct()), "mapping does not cover the schema");
      check(
          NameMappingUtils.hasAliases(mapping, after, scenario.config.getColumnAliases()),
          "mapping lacks configured aliases");
      boolean mappingDestroyed =
          fired
              && scenario.disturbance == AddFilesFuzz.Disturbance.MAPPING_OVERWRITTEN_AFTER_COMMIT;
      if (scenario.state == TableState.CUSTOM_MAPPING && !mappingDestroyed) {
        @Nullable MappedField ident = mapping.find("ident");
        check(
            ident != null && ident.id() != null && ident.id() == after.findField("id").fieldId(),
            "custom name 'ident' lost");
      }
    }

    // 6. readability and stats per registered file
    for (Map.Entry<String, DataFile> entry : registered.entrySet()) {
      FileSpec spec = specs.get(entry.getKey());
      check(spec.readable(), "unreadable file %s was registered", spec);
      // Full readability is only promised when the table covers the file. With evolution off a
      // file may carry columns the table lacks; Iceberg's generic reader then NPEs on an unmapped
      // nested struct (BaseParquetReaders.FallbackReadBuilder), a finding recorded in the plan.
      if (scenario.config.isEnabled() || coveredByTable(spec, after)) {
        verifyReadable(table, mapping, spec, scenario);
      }
      DataFile stats = withStats.get(entry.getKey());
      check(stats != null, "registered file %s not planned", entry.getKey());
      Map<Integer, Long> nulls = stats.nullValueCounts();
      for (String column : spec.columns) {
        if (scenario.config.getIgnoredColumns().contains(column)) {
          continue;
        }
        Types.NestedField field = after.findField(column);
        if (field == null || !field.type().isPrimitiveType()) {
          continue;
        }
        if (spec.rows.isEmpty()) {
          continue;
        }
        check(
            nulls != null && nulls.containsKey(field.fieldId()),
            "no null count for %s on %s",
            column,
            spec.path);
      }
      // ignoring only prevents adding a column; one the table already has is read and counted
      for (String ignored : scenario.config.getIgnoredColumns()) {
        boolean preExisting = before.schema != null && before.schema.findField(ignored) != null;
        Types.NestedField field = after.findField(ignored);
        if (field != null && nulls != null && !preExisting) {
          check(
              !nulls.containsKey(field.fieldId()),
              "stats collected for ignored column %s",
              ignored);
        }
      }
    }

    // 7. metadata economy (disturbances add their own commits)
    int disturbanceVersions = fired ? 2 : 0;
    int versionsAdded =
        version(location(table))
            - (before.metadataLocation == null ? 0 : version(before.metadataLocation))
            - disturbanceVersions;
    boolean schemaOrMappingChanged =
        before.schema == null
            || !before.schema.sameSchema(after)
            || !java.util.Objects.equals(
                before.properties.get(TableProperties.DEFAULT_NAME_MAPPING),
                table.properties().get(TableProperties.DEFAULT_NAME_MAPPING));
    int expectedMax = (schemaOrMappingChanged ? 1 : 0) + (registered.isEmpty() ? 0 : 1);
    if (before.exists) {
      check(
          versionsAdded <= expectedMax,
          "%d metadata versions added, expected at most %d",
          versionsAdded,
          expectedMax);
    }

    // 8. counters
    if (scenario.config.isEnabled()) {
      Long read = outcome.counters.get(ReadFooterSchema.FILES_READ_COUNTER);
      check(read != null && read == specs.size(), "numFilesRead=%s, inputs=%d", read, specs.size());
    }
    Long errorFiles = outcome.counters.get("numErrorFiles");
    check(
        errorFiles == null || errorFiles == errorPaths.size(),
        "numErrorFiles=%s, error rows=%d",
        errorFiles,
        errorPaths.size());
  }

  private static void verifyFailure(
      Scenario scenario, Before before, Outcome outcome, HadoopCatalog catalog, boolean exists) {
    String message = String.valueOf(outcome.failure);
    for (Throwable t = outcome.failure; t != null; t = t.getCause()) {
      message += " | " + t.getMessage();
    }
    if (scenario.disturbance == AddFilesFuzz.Disturbance.TABLE_DROPPED_AFTER_COMMIT) {
      // the table vanished under the pipeline: an environmental failure, so the job fails
      // rather than routing files to errors
      return;
    }
    check(message.contains("Incompatible schemas"), "unexpected pipeline failure: %s", message);
    check(outcome.errors.isEmpty(), "error rows emitted although the pipeline failed");
    check(
        scenario.config.incompatibleSchemaHandling(true)
            == SchemaEvolutionConfig.IncompatibleSchemaHandling.FAIL_PIPELINE,
        "pipeline failed under ROUTE_TO_ERRORS");
    if (!before.exists) {
      check(!exists, "table created although the pipeline failed");
      return;
    }
    Table table = catalog.loadTable(scenario.tableId);
    check(
        location(table).equals(before.metadataLocation),
        "metadata changed although the pipeline failed");
  }

  /** Reads the registered file exactly as a reader would: table schema plus name mapping. */
  private static void verifyReadable(
      Table table, @Nullable NameMapping mapping, FileSpec spec, Scenario scenario)
      throws IOException {
    check(mapping != null, "no mapping to read %s", spec.path);
    Schema projection = table.schema();
    List<Record> rows = new ArrayList<>();
    try (CloseableIterable<Record> reader =
        Parquet.read(table.io().newInputFile(spec.path))
            .project(projection)
            .withNameMapping(mapping)
            .createReaderFunc(
                fileSchema -> GenericParquetReaders.buildReader(projection, fileSchema))
            .build()) {
      for (Record record : reader) {
        rows.add(record);
      }
    }
    check(
        rows.size() == spec.rows.size(),
        "%s: read %d rows, wrote %d",
        spec.path,
        rows.size(),
        spec.rows.size());
    for (String column : spec.columns) {
      if (scenario.config.getIgnoredColumns().contains(column)) {
        continue;
      }
      Types.NestedField field = projection.findField(column);
      if (field == null) {
        continue;
      }
      long nonNull = 0;
      for (Record record : rows) {
        if (record.getField(column) != null) {
          nonNull++;
        }
      }
      check(
          nonNull == spec.nonNullCount(column),
          "%s: column %s read %d non-null values, wrote %d (mapping or alias broken)",
          spec.path,
          column,
          nonNull,
          spec.nonNullCount(column));
    }
  }

  private static boolean coveredByTable(FileSpec spec, Schema table) {
    for (String column : spec.columns) {
      if (table.findField(column) == null) {
        return false;
      }
      if (column.equals("address")) {
        if (table.findField("address.city") == null || table.findField("address.zip") == null) {
          return false;
        }
      }
    }
    return true;
  }

  private static Schema withoutExternal(Schema schema) {
    List<Types.NestedField> kept = new ArrayList<>();
    for (Types.NestedField field : schema.columns()) {
      if (!field.name().equals(AddFilesFuzz.EXTERNAL_COLUMN)) {
        kept.add(field);
      }
    }
    return new Schema(kept);
  }

  private static String errorMessage(Outcome outcome, String path) {
    for (Row row : outcome.errors) {
      if (path.equals(row.getString("file"))) {
        return String.valueOf(row.getString("error"));
      }
    }
    return "(no error row)";
  }

  static String location(Table table) {
    return ((BaseTable) table).operations().current().metadataFileLocation();
  }

  static int version(String location) {
    String file = location.substring(location.lastIndexOf('/') + 2);
    return Integer.parseInt(file.substring(0, file.indexOf('.')));
  }
}
