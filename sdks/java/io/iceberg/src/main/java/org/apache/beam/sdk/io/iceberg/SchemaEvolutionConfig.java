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

import static org.apache.beam.vendor.guava.v32_1_2_jre.com.google.common.base.Preconditions.checkArgument;

import com.google.auto.value.AutoValue;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
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
 *     .withColumnAliases(Map.of("amt", "amount"))          // file name -> table name
 *     .setIgnoredColumns(List.of("debug_payload"))         // never added
 *     .setIncompatibleSchemaHandling(IncompatibleSchemaHandling.ROUTE_TO_ERRORS)
 *     .build();
 * }</pre>
 *
 * <p><b>Pins.</b> Required columns are pinned: never made optional whatever the options say, and
 * created required when this transform creates the table. A file that lacks a pinned column, has
 * nulls in it, or carries no null-count statistics for it is routed to the error output. Pins name
 * canonical (table) paths, dotted for nested fields.
 *
 * <p><b>Aliases.</b> Column aliases map a file column name to the table column it stands for (alias
 * to canonical, dotted paths for nested fields, leaf rename within the same parent). Aliased files
 * are classified, registered and read as if they carried the canonical name; the alias is added to
 * the table's name mapping so readers resolve it and stats are collected for the canonical column.
 * No self mapping, no chains, case sensitive; an alias cannot be pinned or ignored; a file carrying
 * both alias and canonical name is unreadable for schema purposes.
 *
 * <p><b>Ignored columns.</b> Dropped from file schemas before anything else looks at them, so they
 * are never added to the table; the file bytes are untouched. A column the table already has cannot
 * be ignored: readers resolve it through the name mapping and stats are collected for it.
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

  /** File column path to the table column path it stands for; see the class docs for the rules. */
  public abstract Map<String, String> getColumnAliases();

  /** Column paths dropped from every file schema, so never added to the table. */
  public abstract Set<String> getIgnoredColumns();

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
        .setRequiredColumns(Collections.emptySet())
        .setColumnAliases(Collections.emptyMap())
        .setIgnoredColumns(Collections.emptySet());
  }

  /**
   * Aliases, pins and ignores must not contradict each other: an alias maps to a real canonical
   * name (no self map, no chain, same parent), pins and ignores name canonical columns, and a
   * column is never both pinned and ignored.
   */
  private static void validate(Map<String, String> aliases, Set<String> pins, Set<String> ignored) {
    for (Map.Entry<String, String> alias : aliases.entrySet()) {
      String from = alias.getKey();
      String to = alias.getValue();
      checkArgument(!from.isEmpty() && !to.isEmpty(), "Empty column alias: '%s' -> '%s'", from, to);
      checkArgument(!from.equals(to), "Column alias '%s' maps to itself", from);
      checkArgument(
          !aliases.containsKey(to),
          "Column alias '%s' -> '%s' chains into another alias; map every alias directly to its"
              + " canonical name",
          from,
          to);
      checkArgument(
          parent(from).equals(parent(to)),
          "Column alias '%s' -> '%s' must rename a field within the same parent",
          from,
          to);
      checkArgument(
          !pins.contains(from), "Pinned column '%s' is an alias; pin the canonical name", from);
      checkArgument(
          !ignored.contains(from) && !ignored.contains(to),
          "Column alias '%s' -> '%s' names an ignored column",
          from,
          to);
    }
    for (String column : ignored) {
      checkArgument(!pins.contains(column), "Column '%s' is both pinned and ignored", column);
    }
  }

  private static String parent(String path) {
    int dot = path.lastIndexOf('.');
    return dot < 0 ? "" : path.substring(0, dot);
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

    abstract Builder setColumnAliases(Map<String, String> aliases);

    /** Alias to canonical column path. */
    public Builder withColumnAliases(Map<String, String> aliases) {
      return setColumnAliases(Collections.unmodifiableMap(new LinkedHashMap<>(aliases)));
    }

    abstract Builder setIgnoredColumns(Set<String> ignored);

    public Builder setIgnoredColumns(Iterable<String> ignored) {
      Set<String> copy = new LinkedHashSet<>();
      for (String column : ignored) {
        copy.add(column);
      }
      return setIgnoredColumns(Collections.unmodifiableSet(copy));
    }

    abstract SchemaEvolutionConfig autoBuild();

    public SchemaEvolutionConfig build() {
      SchemaEvolutionConfig config = autoBuild();
      validate(config.getColumnAliases(), config.getRequiredColumns(), config.getIgnoredColumns());
      return config;
    }
  }
}
