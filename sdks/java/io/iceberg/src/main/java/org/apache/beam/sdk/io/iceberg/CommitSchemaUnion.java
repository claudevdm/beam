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
import org.apache.iceberg.exceptions.CommitFailedException;
import org.apache.iceberg.exceptions.ValidationException;
import org.apache.iceberg.mapping.NameMapping;
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
   * Applies the schemas and returns the table's schema id after the call.
   *
   * @param schemas the window's distinct schema groups, most common first
   */
  static long commit(
      Catalog catalog,
      TableIdentifier tableId,
      List<CollectDistinctSchemas.SchemaGroup> schemas,
      SchemaEvolutionConfig config,
      IncompatibleSchemaHandling handling,
      Committer committer) {
    for (int attempt = 1; ; attempt++) {
      try {
        return commitOnce(catalog, tableId, schemas, config, handling, committer);
      } catch (CommitFailedException e) {
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
      Committer committer) {
    Table table = catalog.loadTable(tableId);
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

    Transaction txn = stageAll(table, accepted, incompatible);
    boolean staged = !accepted.isEmpty();
    if (staged) {
      relaxNewRequiredFields(txn, table.schema(), config);
    }
    staged |= stageNameMapping(txn);

    if (!incompatible.isEmpty()) {
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
                + " file(s)); no schema change was committed:\n  "
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

  private static final class Accepted {
    final Schema schema;
    final String json;
    final long files;
    final SchemaDelta delta;

    Accepted(Schema schema, String json, long files, SchemaDelta delta) {
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
  private static Transaction stageAll(
      Table table, List<Accepted> accepted, List<Incompatible> incompatible) {
    while (true) {
      Transaction txn = table.newTransaction();
      Accepted failed = null;
      for (Accepted item : accepted) {
        try {
          stage(txn, item);
        } catch (ValidationException | IllegalArgumentException e) {
          failed = item;
          incompatible.add(
              new Incompatible(
                  item.json,
                  item.files,
                  "conflicts with another file schema in the same window: "
                      + AddFiles.errorMessage(e)));
          break;
        }
      }
      if (failed == null) {
        return txn;
      }
      accepted.remove(failed);
    }
  }

  private static void stage(Transaction txn, Accepted item) {
    UpdateSchema update = txn.updateSchema().unionByNameWith(item.schema);
    for (String path : item.delta.absentRequiredPaths()) {
      update = update.makeColumnOptional(path);
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

  /** Regenerates the name mapping when absent, malformed or not covering the staged schema. */
  private static boolean stageNameMapping(Transaction txn) {
    Schema schema = txn.table().schema();
    @Nullable
    NameMapping existing =
        NameMappingUtils.parseOrNull(
            txn.table().properties().get(TableProperties.DEFAULT_NAME_MAPPING));
    if (existing != null && NameMappingUtils.covers(existing, schema.asStruct())) {
      return false;
    }
    String regenerated = NameMappingUtils.regenerate(schema, existing);
    txn.updateProperties().set(TableProperties.DEFAULT_NAME_MAPPING, regenerated).commit();
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
