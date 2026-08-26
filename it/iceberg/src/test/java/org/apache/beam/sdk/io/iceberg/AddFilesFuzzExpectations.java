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

import java.io.File;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import org.apache.beam.sdk.io.iceberg.AddFilesFuzz.FileKind;
import org.apache.beam.sdk.io.iceberg.AddFilesFuzz.FileSpec;
import org.apache.beam.sdk.io.iceberg.AddFilesFuzz.Scenario;
import org.apache.beam.sdk.io.iceberg.SchemaEvolutionConfig.IncompatibleSchemaHandling;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Table;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.hadoop.HadoopCatalog;
import org.apache.iceberg.types.Types;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * What each file should become, derived from its spec and the initial table, before the run. The
 * delta classification reuses the classifier (self-consistency); the adversarial and pin rules are
 * independent of the pipeline.
 */
final class AddFilesFuzzExpectations {
  private AddFilesFuzzExpectations() {}

  enum Expected {
    REGISTER,
    /** Footer unreadable, non-Parquet or missing: an error row whatever the config. */
    ERROR_UNREADABLE,
    /** Needs a change the config does not allow (ROUTE) or fails the run (FAIL). */
    ERROR_INCOMPATIBLE,
    /** A pinned column absent or with nulls. */
    ERROR_PIN
  }

  static final class Result {
    final Map<String, Expected> byPath = new LinkedHashMap<>();
    boolean pipelineFails;
  }

  /** Computes expectations; needs a scratch dir for a copy of the initial table. */
  static Result compute(Scenario scenario, @Nullable Schema initialTable, File scratch)
      throws IOException {
    Result result = new Result();
    SchemaEvolutionConfig config = scenario.config;
    boolean fail =
        config.incompatibleSchemaHandling(true) == IncompatibleSchemaHandling.FAIL_PIPELINE;

    @Nullable Table scratchTable = null;
    if (config.isEnabled() && initialTable != null) {
      HadoopCatalog catalog =
          new HadoopCatalog(new Configuration(), "file:" + scratch.getAbsolutePath());
      scratchTable =
          catalog.createTable(TableIdentifier.of("scratch", "t" + scenario.seed), initialTable);
    }

    for (FileSpec file : scenario.files) {
      if (!file.readable() || file.kind == FileKind.AVRO_EXTENSION) {
        result.byPath.put(file.path, Expected.ERROR_UNREADABLE);
        continue;
      }
      if (!config.isEnabled()) {
        result.byPath.put(file.path, Expected.REGISTER);
        continue;
      }
      Schema effective = FileSchemas.effective(ParquetFooters.read(file.path), config);
      boolean incompatible = false;
      if (scratchTable != null) {
        SchemaDelta delta = SchemaDelta.classify(scratchTable, effective);
        incompatible = !delta.allowedBy(config);
      }
      if (incompatible) {
        result.byPath.put(file.path, Expected.ERROR_INCOMPATIBLE);
        result.pipelineFails |= fail;
        continue;
      }
      if (pinViolated(file, effective, config)) {
        result.byPath.put(file.path, Expected.ERROR_PIN);
        continue;
      }
      result.byPath.put(file.path, Expected.REGISTER);
    }
    return result;
  }

  /** Independent of the pipeline: absent from the file, or any null in a pinned column. */
  private static boolean pinViolated(
      FileSpec file, Schema effective, SchemaEvolutionConfig config) {
    for (String pinned : config.getRequiredColumns()) {
      String top = pinned.contains(".") ? pinned.substring(0, pinned.indexOf('.')) : pinned;
      if (!file.columns.contains(top)) {
        return true;
      }
      Types.NestedField field = effective.findField(pinned);
      if (field == null) {
        return true;
      }
      if (file.rows.isEmpty()) {
        continue;
      }
      if (file.nonNullCount(top) < file.rows.size()) {
        return true;
      }
    }
    return false;
  }
}
