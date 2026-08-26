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

import static org.apache.beam.vendor.guava.v32_1_2_jre.com.google.common.base.Preconditions.checkState;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.apache.beam.sdk.io.iceberg.SchemaEvolutionConfig.IncompatibleSchemaHandling;
import org.apache.beam.sdk.util.BackOff;
import org.apache.beam.sdk.util.BackOffUtils;
import org.apache.beam.sdk.util.FluentBackoff;
import org.apache.beam.sdk.util.Sleeper;
import org.apache.iceberg.Schema;
import org.apache.iceberg.SchemaParser;
import org.apache.iceberg.Table;
import org.apache.iceberg.TableProperties;
import org.apache.iceberg.Transaction;
import org.apache.iceberg.UpdateSchema;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.exceptions.AlreadyExistsException;
import org.apache.iceberg.exceptions.CommitFailedException;
import org.apache.iceberg.exceptions.NoSuchTableException;
import org.apache.iceberg.exceptions.ValidationException;
import org.apache.iceberg.mapping.NameMapping;
import org.apache.iceberg.mapping.NameMappingParser;
import org.apache.iceberg.types.Type;
import org.apache.iceberg.types.TypeUtil;
import org.apache.iceberg.types.Types;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.joda.time.Duration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Applies the distinct file schemas of a window to the table in one transaction: fresh load,
 * classify each schema most common first, fold the allowed unions (plus explicit relaxations for
 * required columns absent from files) on a scratch transaction, stage the folded result as one
 * schema update, repair the name mapping, commit once. The fold keeps per-schema blame for
 * cross-schema conflicts while the table gains a single schema version per window; the scratch
 * transaction is never committed. Nothing is committed when nothing changes.
 *
 * <p>Incompatible schemas either fail the whole call before any commit ({@link
 * IncompatibleSchemaHandling#FAIL_PIPELINE}) or are skipped so their files reach the error output
 * at registration ({@link IncompatibleSchemaHandling#ROUTE_TO_ERRORS}).
 */
final class CommitSchemaUnion {
  private static final Logger LOG = LoggerFactory.getLogger(CommitSchemaUnion.class);

  static final int MAX_ATTEMPTS = 5;

  /** Returned when the table does not exist and there is no schema to create it from. */
  static final long NO_TABLE = -1L;

  /** How to create the table when it does not exist: from the union of the window's schemas. */
  static final class TableCreation implements Serializable {
    final @Nullable List<String> partitionFields;
    final @Nullable List<String> sortFields;
    final @Nullable Map<String, String> properties;

    TableCreation(
        @Nullable List<String> partitionFields,
        @Nullable List<String> sortFields,
        @Nullable Map<String, String> properties) {
      this.partitionFields = partitionFields;
      this.sortFields = sortFields;
      this.properties = properties;
    }
  }

  /** Injectable so tests can exercise the commit retry path. */
  interface Committer extends Serializable {
    void commit(Transaction txn);
  }

  static final Committer DEFAULT_COMMITTER = Transaction::commitTransaction;

  /** Thrown under FAIL_PIPELINE; the message lists every incompatible schema. */
  static final class IncompatibleSchemaException extends IllegalStateException {
    IncompatibleSchemaException(String message) {
      super(message);
    }
  }

  private static final class Incompatible {
    final String schemaJson;
    final long files;
    final String reason;

    Incompatible(String schemaJson, long files, String reason) {
      this.schemaJson = schemaJson;
      this.files = files;
      this.reason = reason;
    }

    @Override
    public String toString() {
      return files + " file(s) with schema " + truncate(schemaJson) + ": " + reason;
    }
  }

  /** Canonical JSON of a wide schema runs to hundreds of KB; the reason is what matters. */
  private static final int MAX_SCHEMA_JSON_CHARS = 1024;

  private static String truncate(String json) {
    if (json.length() <= MAX_SCHEMA_JSON_CHARS) {
      return json;
    }
    return json.substring(0, MAX_SCHEMA_JSON_CHARS)
        + "... ("
        + (json.length() - MAX_SCHEMA_JSON_CHARS)
        + " chars truncated)";
  }

  private CommitSchemaUnion() {}

  /**
   * Applies the schemas and returns the table's schema id after the call, or {@link #NO_TABLE} when
   * the table is missing and there is no schema to create it from.
   *
   * @param schemas the window's distinct schema groups, most common first
   */
  static long commit(
      Catalog catalog,
      TableIdentifier tableId,
      List<CollectDistinctSchemas.SchemaGroup> schemas,
      SchemaEvolutionConfig config,
      IncompatibleSchemaHandling handling,
      TableCreation creation,
      Committer committer) {
    // The catalog is already under contention when a retry fires; back off (jittered by
    // FluentBackoff) instead of piling on. Iceberg's own metadata retries (commit.retry.*)
    // sit below this loop.
    BackOff backoff =
        FluentBackoff.DEFAULT
            .withMaxRetries(MAX_ATTEMPTS - 1)
            .withInitialBackoff(Duration.millis(100))
            .withMaxBackoff(Duration.standardSeconds(2))
            .backoff();
    for (int attempt = 1; ; attempt++) {
      try {
        return commitOnce(catalog, tableId, schemas, config, handling, creation, committer);
      } catch (CommitFailedException | AlreadyExistsException e) {
        // a concurrent commit, or a create race: the next attempt loads the fresh state
        try {
          if (!BackOffUtils.next(Sleeper.DEFAULT, backoff)) {
            throw e;
          }
        } catch (InterruptedException interrupted) {
          Thread.currentThread().interrupt();
          throw e;
        }
        LOG.info(
            "Schema commit attempt {}/{} for {} failed; reloading and rebuilding",
            attempt,
            MAX_ATTEMPTS,
            tableId,
            e);
      }
    }
  }

  private static long commitOnce(
      Catalog catalog,
      TableIdentifier tableId,
      List<CollectDistinctSchemas.SchemaGroup> schemas,
      SchemaEvolutionConfig config,
      IncompatibleSchemaHandling handling,
      TableCreation creation,
      Committer committer) {
    Table table;
    try {
      table = catalog.loadTable(tableId);
    } catch (NoSuchTableException e) {
      return create(catalog, tableId, schemas, config, handling, creation, committer);
    }
    // Every transaction below must share this snapshot: classification, the fold and the replay
    // all reason about the same table state (newTransactionOn enforces it).
    Schema base = table.schema();
    for (String ignored : config.getIgnoredColumns()) {
      if (base.findField(ignored) != null) {
        LOG.warn(
            "Ignored column '{}' already exists in {}; it stays readable and keeps getting stats."
                + " Ignoring only prevents adding a column.",
            ignored,
            tableId);
      }
    }
    List<Incompatible> incompatible = new ArrayList<>();
    List<Accepted> accepted = new ArrayList<>();
    for (CollectDistinctSchemas.SchemaGroup group : schemas) {
      Schema fileSchema =
          FileSchemas.markRequired(
              SchemaParser.fromJson(group.getSchemaJson()), group.getNullFreeColumns());
      SchemaDelta delta = SchemaDelta.classify(table, fileSchema);
      if (delta.isEmpty()) {
        continue;
      }
      if (!delta.allowedBy(config)) {
        incompatible.add(
            new Incompatible(
                group.getSchemaJson(), group.getFiles(), delta.disallowedReason(config)));
        continue;
      }
      accepted.add(new Accepted(fileSchema, group.getSchemaJson(), group.getFiles(), delta));
    }

    Transaction scratch;
    while (true) {
      scratch = newTransactionOn(table, base, tableId);
      Accepted failed = stageAll(scratch, accepted, incompatible);
      if (failed == null) {
        break;
      }
      accepted.remove(failed);
    }
    boolean folded = !accepted.isEmpty();
    if (folded) {
      relaxNewRequiredFields(scratch, base);
    }

    Transaction txn = newTransactionOn(table, base, tableId);
    if (folded) {
      Schema merged = scratch.table().schema();
      // One union replays the fold's net effect (additions, promotions, relaxations) so the
      // table gains a single schema version instead of one per folded schema.
      txn.updateSchema().unionByNameWith(merged).commit();
      // toString of the args runs only on failure
      Schema foldResult = TypeUtil.assignIncreasingFreshIds(merged);
      Schema replayResult = TypeUtil.assignIncreasingFreshIds(txn.table().schema());
      checkState(
          replayResult.sameSchema(foldResult),
          "replaying the folded schema union for %s diverged from the fold; fold: %s replay: %s",
          tableId,
          foldResult,
          replayResult);
    }
    boolean staged = folded;
    staged |= stageNameMapping(txn, config);

    if (!incompatible.isEmpty()) {
      reportIncompatible(tableId, incompatible, handling, "no schema change was committed");
    }

    if (!staged) {
      LOG.info(
          "Table {} already covers all {} file schema(s); nothing to commit",
          tableId,
          schemas.size());
      return table.schema().schemaId();
    }
    committer.commit(txn);
    table.refresh();
    long acceptedFiles = 0;
    for (Accepted item : accepted) {
      acceptedFiles += item.files;
    }
    LOG.info(
        "Committed schema union for {}: {} schema(s) covering {} file(s), now at schema id {}",
        tableId,
        accepted.size(),
        acceptedFiles,
        table.schema().schemaId());
    return table.schema().schemaId();
  }

  /**
   * Creates the table from the union of the window's schemas, with every column optional at every
   * level so that one lucky file cannot impose required columns on the table. Pins do not shape the
   * created schema; they are enforced per file at registration.
   */
  private static long create(
      Catalog catalog,
      TableIdentifier tableId,
      List<CollectDistinctSchemas.SchemaGroup> schemas,
      SchemaEvolutionConfig config,
      IncompatibleSchemaHandling handling,
      TableCreation creation,
      Committer committer) {
    if (schemas.isEmpty()) {
      LOG.info("Table {} does not exist and no file schema was read; not creating it", tableId);
      return NO_TABLE;
    }
    // Null evidence is irrelevant here: creation relaxes every column anyway.
    Schema seed = SchemaParser.fromJson(schemas.get(0).getSchemaJson());
    List<Accepted> rest = new ArrayList<>();
    for (CollectDistinctSchemas.SchemaGroup group : schemas.subList(1, schemas.size())) {
      Schema fileSchema = SchemaParser.fromJson(group.getSchemaJson());
      rest.add(new Accepted(fileSchema, group.getSchemaJson(), group.getFiles(), null));
    }
    // The union of the schemas is folded on a scratch create transaction that is never
    // committed, then the real table is built from the folded result directly: it is born with
    // one schema version, and partition and sort fields resolve against the union rather than
    // the seed alone.
    List<Incompatible> incompatible = new ArrayList<>();
    Schema merged;
    while (true) {
      Transaction scratch = catalog.buildTable(tableId, seed).createTransaction();
      Accepted failed = stageAll(scratch, rest, incompatible);
      if (failed == null) {
        merged = scratch.table().schema();
        break;
      }
      rest.remove(failed);
    }
    if (!incompatible.isEmpty()) {
      reportIncompatible(tableId, incompatible, handling, "no table was created");
    }
    Schema created = relaxAll(merged);
    Map<String, String> properties =
        creation.properties == null ? new HashMap<>() : new HashMap<>(creation.properties);
    Transaction txn =
        catalog
            .buildTable(tableId, created)
            .withPartitionSpec(PartitionUtils.toPartitionSpec(creation.partitionFields, created))
            .withSortOrder(SortOrderUtils.toSortOrder(creation.sortFields, created))
            .withProperties(properties)
            .createTransaction();
    stageNameMapping(txn, config);
    committer.commit(txn);
    Table table = catalog.loadTable(tableId);
    LOG.info(
        "Created table {} from {} file schema(s), schema id {}",
        tableId,
        schemas.size() - incompatible.size(),
        table.schema().schemaId());
    return table.schema().schemaId();
  }

  /**
   * Every field optional at every level, list elements and map values included; map key subtrees
   * keep their declared shape (keys are required by definition). Pins do not shape new columns, and
   * nothing depends on a created table's schema yet.
   */
  static Schema relaxAll(Schema schema) {
    List<Types.NestedField> fields = new ArrayList<>();
    for (Types.NestedField field : schema.asStruct().fields()) {
      fields.add(relaxAll(field));
    }
    return new Schema(fields);
  }

  private static Types.NestedField relaxAll(Types.NestedField field) {
    return Types.NestedField.from(field)
        .ofType(relaxAllType(field.type()))
        .isOptional(true)
        .build();
  }

  private static Type relaxAllType(Type type) {
    if (type.isStructType()) {
      List<Types.NestedField> fields = new ArrayList<>();
      for (Types.NestedField field : type.asStructType().fields()) {
        fields.add(relaxAll(field));
      }
      return Types.StructType.of(fields);
    }
    if (type.isListType()) {
      Types.ListType list = type.asListType();
      return Types.ListType.ofOptional(list.elementId(), relaxAllType(list.elementType()));
    }
    if (type.isMapType()) {
      Types.MapType map = type.asMapType();
      return Types.MapType.ofOptional(
          map.keyId(), map.valueId(), map.keyType(), relaxAllType(map.valueType()));
    }
    return type;
  }

  private static void reportIncompatible(
      TableIdentifier tableId,
      List<Incompatible> incompatible,
      IncompatibleSchemaHandling handling,
      String consequence) {
    long files = 0;
    for (Incompatible item : incompatible) {
      files += item.files;
    }
    if (handling == IncompatibleSchemaHandling.FAIL_PIPELINE) {
      throw new IncompatibleSchemaException(
          "Incompatible schemas for "
              + tableId
              + " ("
              + incompatible.size()
              + " schema(s), "
              + files
              + " file(s)); "
              + consequence
              + ":\n  "
              + joinLines(incompatible));
    }
    LOG.warn(
        "Skipping {} incompatible schema(s) ({} file(s)) for {}; their files will be routed to"
            + " the error output:\n  {}",
        incompatible.size(),
        files,
        tableId,
        joinLines(incompatible));
  }

  private static final class Accepted {
    final Schema schema;
    final String json;
    final long files;

    /** Null on the create path: the seed table is empty, so there is nothing to relax. */
    final @Nullable SchemaDelta delta;

    Accepted(Schema schema, String json, long files, @Nullable SchemaDelta delta) {
      this.schema = schema;
      this.json = json;
      this.files = files;
      this.delta = delta;
    }
  }

  /**
   * Stages one union per accepted schema onto {@code txn}: a scratch transaction on the evolve path
   * (its per-schema versions stay in memory; only the folded result is ever committed), the create
   * transaction on the create path. A schema can conflict with another schema's additions, which
   * only surfaces while staging and poisons the transaction, so on a conflict the offender is
   * returned for the caller to drop and retry with a fresh transaction.
   */
  private static @Nullable Accepted stageAll(
      Transaction txn, List<Accepted> accepted, List<Incompatible> incompatible) {
    for (Accepted item : accepted) {
      // Both caught types carry staging conflicts: ValidationException from Schema
      // construction at apply ("multiple fields for name"), IllegalArgumentException from
      // SchemaUpdate preconditions ("Cannot change column type").
      try {
        stage(txn, item);
      } catch (ValidationException | IllegalArgumentException e) {
        incompatible.add(
            new Incompatible(
                item.json,
                item.files,
                "conflicts with another file schema in the same window: "
                    + AddFiles.errorMessage(e)));
        return item;
      }
    }
    return null;
  }

  /**
   * Iceberg refreshes the table on every {@code newTransaction()}, so a concurrent schema commit
   * can slip between two transactions here. Any drift from the snapshot the window classified
   * against is thrown as {@link CommitFailedException} so the commit-level retry reloads and
   * rebuilds, leaving the replay checkState as a pure bug detector.
   */
  private static Transaction newTransactionOn(Table table, Schema base, TableIdentifier tableId) {
    Transaction txn = table.newTransaction();
    if (!txn.table().schema().sameSchema(base)) {
      throw new CommitFailedException(
          "concurrent schema change on %s while staging the schema union", tableId);
    }
    return txn;
  }

  private static void stage(Transaction txn, Accepted item) {
    UpdateSchema update = txn.updateSchema().unionByNameWith(item.schema);
    if (item.delta != null) {
      for (String path : item.delta.absentRequiredPaths()) {
        update = update.makeColumnOptional(path);
      }
    }
    update.commit();
  }

  /**
   * New columns are optional at every level. The union adds top-level columns optional but keeps
   * the file's optionality below them, so one file's luck would otherwise impose required fields on
   * everyone. Pins do not shape new columns: they keep existing required columns from being relaxed
   * (SchemaDelta) and gate files at registration.
   */
  private static void relaxNewRequiredFields(Transaction txn, Schema before) {
    List<String> toRelax = newRequiredPaths(before, txn.table().schema());
    if (toRelax.isEmpty()) {
      return;
    }
    UpdateSchema update = txn.updateSchema();
    for (String path : toRelax) {
      update = update.makeColumnOptional(path);
    }
    update.commit();
  }

  /**
   * Paths of required fields that {@code after} has and {@code before} lacks, in schema order;
   * includes fields under lists and maps (a required list element or map value counts). Map key
   * subtrees are skipped: keys are required by definition and relaxing inside a struct key would
   * change key identity.
   */
  static List<String> newRequiredPaths(Schema before, Schema after) {
    Set<Integer> beforeIds = TypeUtil.indexById(before.asStruct()).keySet();
    List<String> paths = new ArrayList<>();
    collectNewRequired(after.asStruct(), "", beforeIds, paths);
    return paths;
  }

  private static void collectNewRequired(
      Type.NestedType type, String prefix, Set<Integer> beforeIds, List<String> paths) {
    for (Types.NestedField field : type.fields()) {
      if (type.isMapType() && field.fieldId() == type.asMapType().keyId()) {
        continue;
      }
      String path = prefix + field.name();
      if (!beforeIds.contains(field.fieldId()) && field.isRequired()) {
        paths.add(path);
      }
      if (field.type().isNestedType()) {
        collectNewRequired(field.type().asNestedType(), path + ".", beforeIds, paths);
      }
    }
  }

  /**
   * Regenerates the name mapping when absent, malformed or not covering the staged schema, and adds
   * the configured aliases. A missing alias alone is reason enough to commit: without it, aliased
   * files are unreadable.
   */
  private static boolean stageNameMapping(Transaction txn, SchemaEvolutionConfig config) {
    Schema schema = txn.table().schema();
    Map<String, String> aliases = config.getColumnAliases();
    @Nullable NameMapping existing =
        NameMappingUtils.parseOrNull(
            txn.table().properties().get(TableProperties.DEFAULT_NAME_MAPPING));
    NameMapping base;
    if (existing != null && NameMappingUtils.covers(existing, schema.asStruct())) {
      if (NameMappingUtils.hasAliases(existing, schema, aliases)) {
        return false;
      }
      base = existing;
    } else {
      base = NameMappingParser.fromJson(NameMappingUtils.regenerate(schema, existing));
    }
    NameMapping withAliases = NameMappingUtils.withAliases(base, schema, aliases);
    txn.updateProperties()
        .set(TableProperties.DEFAULT_NAME_MAPPING, NameMappingParser.toJson(withAliases))
        .commit();
    return true;
  }

  private static String joinLines(List<Incompatible> items) {
    List<String> lines = new ArrayList<>();
    for (Incompatible item : items) {
      lines.add(item.toString());
    }
    return String.join("\n  ", lines);
  }
}
