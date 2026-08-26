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

import com.google.auto.value.AutoValue;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashSet;
import java.util.Set;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * Schema evolution settings for {@link AddFiles}. With no options the table schema is never changed
 * and files register as on a plain AddFiles; every other setting requires at least one option.
 *
 * <pre>{@code
 * SchemaEvolutionConfig.builder()
 *     .setOptions(List.of(ALLOW_FIELD_ADDITION, ALLOW_FIELD_RELAXATION, ALLOW_TYPE_PROMOTION))
 *     .setRequiredColumns(List.of("id", "address.city"))   // never relaxed
 *     .setIncompatibleSchemaHandling(IncompatibleSchemaHandling.ROUTE_TO_ERRORS)
 *     .build();
 * }</pre>
 *
 * <p><b>Pins.</b> Required columns are pinned: never made optional whatever the options say, and
 * created required when this transform creates the table. A file that lacks a pinned column, has
 * nulls in it, or carries no null-count statistics for it is routed to the error output. Pins name
 * canonical (table) paths, dotted for nested fields.
 *
 * <p><b>Incompatible schemas.</b> A schema that needs a change the options do not allow, or that
 * conflicts with the table or with another file's schema. {@link IncompatibleSchemaHandling}
 * decides whether that fails the pipeline before any schema commit (the batch default) or skips the
 * schema so its files reach the error output (the streaming default). Files whose footer cannot be
 * read or converted always go to the error output and never fail the pipeline.
 */
@AutoValue
public abstract class SchemaEvolutionConfig implements Serializable {

  public enum IncompatibleSchemaHandling {
    /**
     * Fail the pipeline before committing any schema change, with a message listing every
     * incompatible schema, its reason and file count. The batch default.
     */
    FAIL_PIPELINE,
    /**
     * Skip the incompatible schema, commit the rest, and route its files to the error output with
     * the specific reason. The streaming default.
     */
    ROUTE_TO_ERRORS
  }

  public abstract Set<SchemaEvolutionOption> getOptions();

  /**
   * Canonical column paths (dotted for nested fields) that are never relaxed and are created
   * required; files that cannot prove they hold no nulls in them go to the error output.
   */
  public abstract Set<String> getRequiredColumns();

  public boolean isPinned(String columnPath) {
    return getRequiredColumns().contains(columnPath);
  }

  /**
   * Unset resolves by mode: {@code FAIL_PIPELINE} in batch, {@code ROUTE_TO_ERRORS} in streaming.
   */
  public abstract @Nullable IncompatibleSchemaHandling getIncompatibleSchemaHandling();

  public IncompatibleSchemaHandling incompatibleSchemaHandling(boolean bounded) {
    IncompatibleSchemaHandling handling = getIncompatibleSchemaHandling();
    if (handling != null) {
      return handling;
    }
    return bounded
        ? IncompatibleSchemaHandling.FAIL_PIPELINE
        : IncompatibleSchemaHandling.ROUTE_TO_ERRORS;
  }

  public boolean isEnabled() {
    return !getOptions().isEmpty();
  }

  public boolean allows(SchemaEvolutionOption option) {
    return getOptions().contains(option);
  }

  public static SchemaEvolutionConfig disabled() {
    return builder().build();
  }

  public static SchemaEvolutionConfig of(SchemaEvolutionOption... options) {
    return builder().setOptions(Arrays.asList(options)).build();
  }

  public static Builder builder() {
    return new AutoValue_SchemaEvolutionConfig.Builder()
        .setOptions(Collections.emptySet())
        .setRequiredColumns(Collections.emptySet());
  }

  @AutoValue.Builder
  public abstract static class Builder {
    abstract Builder setOptions(Set<SchemaEvolutionOption> options);

    public Builder setOptions(Iterable<SchemaEvolutionOption> options) {
      Set<SchemaEvolutionOption> copy = EnumSet.noneOf(SchemaEvolutionOption.class);
      for (SchemaEvolutionOption option : options) {
        copy.add(option);
      }
      return setOptions(Collections.unmodifiableSet(copy));
    }

    abstract Builder setRequiredColumns(Set<String> requiredColumns);

    public Builder setRequiredColumns(Iterable<String> requiredColumns) {
      Set<String> copy = new LinkedHashSet<>();
      for (String column : requiredColumns) {
        copy.add(column);
      }
      return setRequiredColumns(Collections.unmodifiableSet(copy));
    }

    public abstract Builder setIncompatibleSchemaHandling(
        @Nullable IncompatibleSchemaHandling handling);

    public abstract SchemaEvolutionConfig build();
  }
}
