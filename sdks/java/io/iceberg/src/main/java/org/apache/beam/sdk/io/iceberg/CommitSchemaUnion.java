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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.apache.beam.sdk.io.iceberg.SchemaEvolutionConfig.IncompatibleSchemaHandling;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Applies the distinct file schemas of a window to the table in one transaction: fresh load,
 * classify each schema most common first, stage the allowed unions (plus explicit relaxations for
 * required columns absent from files), repair the name mapping, commit once. Nothing is committed
 * when nothing changes.
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
      return files + " file(s) with schema " + schemaJson + ": " + reason;
    }
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
    for (int attempt = 1; ; attempt++) {
      try {
        return commitOnce(catalog, tableId, schemas, config, handling, creation, committer);
      } catch (CommitFailedException | AlreadyExistsException e) {
        // a concurrent commit, or a create race: the next attempt loads the fresh state
        if (attempt >= MAX_ATTEMPTS) {
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
    for (String ignored : config.getIgnoredColumns()) {
      if (table.schema().findField(ignored) != null) {
        LOG.warn(
            "Ignored column '{}' already exists in {}; it stays readable and keeps getting stats."
                + " Ignoring only prevents adding a column.",
            ignored,
            tableId);
      }
    }
    List<Incompatible> incompatible = new ArrayList<>();
    List<Accepted> accepted = new ArrayList<>();
    long acceptedFiles = 0;
    for (CollectDistinctSchemas.SchemaGroup group : schemas) {
      Schema fileSchema =
          FileSchemas.markRequired(SchemaParser.fromJson(group.schemaJson), group.nullFreeColumns);
      SchemaDelta delta = SchemaDelta.classify(table, fileSchema);
      if (delta.isEmpty()) {
        continue;
      }
      if (!delta.allowedBy(config)) {
        incompatible.add(
            new Incompatible(group.schemaJson, group.files, delta.disallowedReason(config)));
        continue;
      }
      accepted.add(new Accepted(fileSchema, group.schemaJson, group.files, delta));
      acceptedFiles += group.files;
    }

    Transaction txn;
    while (true) {
      txn = table.newTransaction();
      Accepted failed = stageAll(txn, accepted, incompatible);
      if (failed == null) {
        break;
      }
      accepted.remove(failed);
    }
    boolean staged = !accepted.isEmpty();
    if (staged) {
      relaxNewRequiredFields(txn, table.schema(), config);
    }
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
    LOG.info(
        "Committed schema union for {}: {} schema(s) covering {} file(s), now at schema id {}",
        tableId,
        accepted.size(),
        acceptedFiles,
        table.schema().schemaId());
    return table.schema().schemaId();
  }

  /**
   * Creates the table from the most common schema, with every column not pinned as required made
   * optional so that one lucky file cannot impose required columns on the table, then unions the
   * remaining schemas in the same create transaction.
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
    // Null evidence is irrelevant here: creation relaxes every unpinned column anyway.
    Schema seed = relaxUnpinned(SchemaParser.fromJson(schemas.get(0).schemaJson), config);
    List<Accepted> rest = new ArrayList<>();
    for (CollectDistinctSchemas.SchemaGroup group : schemas.subList(1, schemas.size())) {
      Schema fileSchema = SchemaParser.fromJson(group.schemaJson);
      rest.add(new Accepted(fileSchema, group.schemaJson, group.files, null));
    }
    List<Incompatible> incompatible = new ArrayList<>();
    Transaction txn;
    while (true) {
      Map<String, String> properties =
          creation.properties == null ? new HashMap<>() : new HashMap<>(creation.properties);
      txn =
          catalog
              .buildTable(tableId, seed)
              .withPartitionSpec(PartitionUtils.toPartitionSpec(creation.partitionFields, seed))
              .withSortOrder(SortOrderUtils.toSortOrder(creation.sortFields, seed))
              .withProperties(properties)
              .createTransaction();
      Accepted failed = stageAll(txn, rest, incompatible);
      if (failed == null) {
        break;
      }
      rest.remove(failed);
    }
    if (!incompatible.isEmpty()) {
      reportIncompatible(tableId, incompatible, handling, "no table was created");
    }
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

  /** Every column optional except pinned ones, at every level (pins are dotted paths). */
  static Schema relaxUnpinned(Schema schema, SchemaEvolutionConfig config) {
    return new Schema(relaxUnpinned(schema.asStruct(), "", config).fields());
  }

  private static Types.StructType relaxUnpinned(
      Types.StructType struct, String prefix, SchemaEvolutionConfig config) {
    List<Types.NestedField> fields = new ArrayList<>();
    for (Types.NestedField field : struct.fields()) {
      String path = prefix + field.name();
      Type type = field.type();
      if (type.isStructType()) {
        type = relaxUnpinned(type.asStructType(), path + ".", config);
      }
      boolean optional = !config.isPinned(path);
      fields.add(Types.NestedField.from(field).ofType(type).isOptional(optional).build());
    }
    return Types.StructType.of(fields);
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
   * Stages one union per accepted schema. A schema can conflict with another schema's additions,
   * which only surfaces while staging and poisons the transaction, so on a conflict the offender
   * moves to {@code incompatible} and the transaction is rebuilt without it.
   */
  private static @Nullable Accepted stageAll(
      Transaction txn, List<Accepted> accepted, List<Incompatible> incompatible) {
    for (Accepted item : accepted) {
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
   * the file's optionality for fields inside a newly added struct, so one file's luck would
   * otherwise impose required nested columns on everyone. Pinned paths stay as they are.
   */
  private static void relaxNewRequiredFields(
      Transaction txn, Schema before, SchemaEvolutionConfig config) {
    Schema after = txn.table().schema();
    Map<Integer, Types.NestedField> beforeById = TypeUtil.indexById(before.asStruct());
    Map<Integer, String> pathById = TypeUtil.indexNameById(after.asStruct());
    List<String> toRelax = new ArrayList<>();
    for (Map.Entry<Integer, Types.NestedField> entry :
        TypeUtil.indexById(after.asStruct()).entrySet()) {
      Types.NestedField field = entry.getValue();
      String path = pathById.get(entry.getKey());
      if (path == null || beforeById.containsKey(entry.getKey()) || field.isOptional()) {
        continue;
      }
      boolean underContainer =
          path.contains(".element") || path.contains(".key") || path.contains(".value");
      if (!underContainer && !config.isPinned(path)) {
        toRelax.add(path);
      }
    }
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
   * Regenerates the name mapping when absent, malformed or not covering the staged schema, and adds
   * the configured aliases. A missing alias alone is reason enough to commit: without it, aliased
   * files are unreadable.
   */
  private static boolean stageNameMapping(Transaction txn, SchemaEvolutionConfig config) {
    Schema schema = txn.table().schema();
    Map<String, String> aliases = config.getColumnAliases();
    @Nullable
    NameMapping existing =
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
