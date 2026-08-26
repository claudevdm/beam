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

import static org.apache.beam.sdk.util.Preconditions.checkStateNotNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Table;
import org.apache.iceberg.exceptions.ValidationException;
import org.apache.iceberg.types.Type;
import org.apache.iceberg.types.TypeUtil;
import org.apache.iceberg.types.Types;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * What {@code unionByNameWith(fileSchema)} would change on a table, without changing it. Computed
 * by diffing the union result against the table schema by field id: existing fields keep their ids
 * and additions get fresh ones, so the diff is exact and independent of column order.
 *
 * <p>The union ignores table columns absent from the file, but every row of such a file reads null
 * in them, so a required column absent from the file is also a relaxation. The commit side stages
 * those explicitly via {@link #absentRequiredPaths()}.
 */
final class SchemaDelta {

  enum Kind {
    FIELD_ADDITION(SchemaEvolutionOption.ALLOW_FIELD_ADDITION),
    FIELD_RELAXATION(SchemaEvolutionOption.ALLOW_FIELD_RELAXATION),
    TYPE_PROMOTION(SchemaEvolutionOption.ALLOW_TYPE_PROMOTION),
    /** The union is impossible (for example string vs int); never allowed. */
    CONFLICT(null);

    final @Nullable SchemaEvolutionOption option;

    Kind(@Nullable SchemaEvolutionOption option) {
      this.option = option;
    }

    boolean allowedBy(SchemaEvolutionConfig config) {
      return option != null && config.allows(option);
    }
  }

  private static final class Change {
    final Kind kind;
    /** Unquoted column path for the config lookup; empty for conflicts without a field. */
    final String path;

    final String description;
    /** A relaxation because the column is absent from the file, not declared optional. */
    final boolean absent;

    Change(Kind kind, String path, String description) {
      this(kind, path, description, false);
    }

    Change(Kind kind, String path, String description, boolean absent) {
      this.kind = kind;
      this.path = path;
      this.description = description;
      this.absent = absent;
    }

    boolean allowedBy(SchemaEvolutionConfig config) {
      if (kind == Kind.FIELD_RELAXATION && config.isPinned(path)) {
        return false;
      }
      return kind.allowedBy(config);
    }

    String disallowedReason(SchemaEvolutionConfig config) {
      if (kind == Kind.FIELD_RELAXATION && config.isPinned(path)) {
        return description + " (pinned as required)";
      }
      return description + " (needs " + kind.option + ")";
    }
  }

  private final List<Change> changes;

  private SchemaDelta(List<Change> changes) {
    this.changes = Collections.unmodifiableList(changes);
  }

  static SchemaDelta classify(Table table, Schema fileSchema) {
    Schema merged;
    try {
      merged = table.updateSchema().unionByNameWith(fileSchema).apply();
    } catch (ValidationException | IllegalArgumentException e) {
      // SchemaUpdate reports type conflicts through both exception types
      return conflict(e.getClass().getSimpleName() + ": " + AddFiles.errorMessage(e));
    }
    SchemaDelta delta = diff(table.schema(), merged);
    List<Change> changes = new ArrayList<>(delta.changes);
    findAbsentRequired(table.schema().asStruct(), fileSchema.asStruct(), "", changes);
    return new SchemaDelta(changes);
  }

  /**
   * Required table columns with no counterpart in the file, by name per level. Children are checked
   * only when their parent is present; an absent struct is the relaxation itself. Descends through
   * list elements and map values (paths use {@code element} and {@code value}, which
   * makeColumnOptional accepts); map keys are required by definition. FileSchemas tightening stops
   * at lists and maps for a different reason (ambiguous null counts); the two are independent.
   */
  private static void findAbsentRequired(
      Types.StructType tableStruct,
      Types.StructType fileStruct,
      String prefix,
      List<Change> changes) {
    for (Types.NestedField field : tableStruct.fields()) {
      String rawPath = prefix + field.name();
      Types.NestedField fileField = fileStruct.field(field.name());
      if (fileField == null) {
        if (field.isRequired()) {
          changes.add(
              new Change(
                  Kind.FIELD_RELAXATION,
                  rawPath,
                  "relax "
                      + prefix
                      + quoteIfDotted(field.name())
                      + " to optional (absent from file)",
                  true));
        }
        continue;
      }
      findAbsentRequiredInType(field.type(), fileField.type(), rawPath, changes);
    }
  }

  private static void findAbsentRequiredInType(
      Type tableType, Type fileType, String rawPath, List<Change> changes) {
    if (tableType.isStructType() && fileType.isStructType()) {
      findAbsentRequired(tableType.asStructType(), fileType.asStructType(), rawPath + ".", changes);
    } else if (tableType.isListType() && fileType.isListType()) {
      findAbsentRequiredInType(
          tableType.asListType().elementType(),
          fileType.asListType().elementType(),
          rawPath + ".element",
          changes);
    } else if (tableType.isMapType() && fileType.isMapType()) {
      findAbsentRequiredInType(
          tableType.asMapType().valueType(),
          fileType.asMapType().valueType(),
          rawPath + ".value",
          changes);
    }
  }

  /** Paths of required table columns absent from the file; the union alone does not relax them. */
  List<String> absentRequiredPaths() {
    List<String> paths = new ArrayList<>();
    for (Change change : changes) {
      if (change.absent) {
        paths.add(change.path);
      }
    }
    return paths;
  }

  private static SchemaDelta conflict(String message) {
    List<Change> changes = new ArrayList<>();
    changes.add(new Change(Kind.CONFLICT, "", message));
    return new SchemaDelta(changes);
  }

  /**
   * Changes from {@code before} to {@code after}, ordered by field path. Fields are matched by id;
   * paths only appear in messages (quoted when a name contains a dot). Anything a union by name
   * cannot produce is reported as a conflict so it is never applied unclassified.
   */
  static SchemaDelta diff(Schema before, Schema after) {
    Map<Integer, Types.NestedField> beforeById = TypeUtil.indexById(before.asStruct());
    Map<Integer, Types.NestedField> afterById = TypeUtil.indexById(after.asStruct());
    Map<Integer, Integer> parentById = TypeUtil.indexParents(after.asStruct());
    Map<Integer, String> rawPathById = TypeUtil.indexNameById(after.asStruct());
    Map<Integer, String> pathById =
        TypeUtil.indexQuotedNameById(after.asStruct(), SchemaDelta::quoteIfDotted);

    List<Integer> idsByPath = new ArrayList<>(afterById.keySet());
    idsByPath.sort(
        (a, b) ->
            checkStateNotNull(rawPathById.get(a)).compareTo(checkStateNotNull(rawPathById.get(b))));

    List<Change> changes = new ArrayList<>();
    for (Integer id : idsByPath) {
      String path = checkStateNotNull(pathById.get(id));
      String rawPath = checkStateNotNull(rawPathById.get(id));
      Types.NestedField newField = checkStateNotNull(afterById.get(id));
      Types.NestedField oldField = beforeById.get(id);
      if (oldField == null) {
        if (!hasAddedAncestor(id, parentById, beforeById)) {
          changes.add(
              new Change(
                  Kind.FIELD_ADDITION,
                  rawPath,
                  "add " + optionality(newField) + " " + path + " " + newField.type()));
        }
        continue;
      }
      compareField(path, rawPath, oldField, newField, changes);
    }

    Map<Integer, String> beforePathById =
        TypeUtil.indexQuotedNameById(before.asStruct(), SchemaDelta::quoteIfDotted);
    List<String> removed = new ArrayList<>();
    for (Integer id : beforeById.keySet()) {
      if (!afterById.containsKey(id)) {
        removed.add(checkStateNotNull(beforePathById.get(id)));
      }
    }
    Collections.sort(removed);
    for (String path : removed) {
      changes.add(new Change(Kind.CONFLICT, "", "field removed: " + path));
    }
    return new SchemaDelta(changes);
  }

  /**
   * Attribute by attribute: name, doc and defaults must be equal; required to optional is the
   * relaxation; primitive types must be equal or a promotion; nested types must stay the same kind,
   * their children are compared on their own ids.
   */
  private static void compareField(
      String path,
      String rawPath,
      Types.NestedField oldField,
      Types.NestedField newField,
      List<Change> changes) {
    if (!oldField.name().equals(newField.name())) {
      changes.add(
          new Change(
              Kind.CONFLICT,
              rawPath,
              "renamed " + path + " from " + oldField.name() + " to " + newField.name()));
    }
    if (!Objects.equals(oldField.doc(), newField.doc())) {
      // benign but unsupported: schema evolution has no option for doc updates
      changes.add(
          new Change(Kind.CONFLICT, rawPath, "doc changed on " + path + " (not supported)"));
    }
    if (!Objects.equals(oldField.initialDefault(), newField.initialDefault())
        || !Objects.equals(oldField.writeDefault(), newField.writeDefault())) {
      changes.add(
          new Change(Kind.CONFLICT, rawPath, "default changed on " + path + " (not supported)"));
    }
    if (oldField.isRequired() && newField.isOptional()) {
      changes.add(new Change(Kind.FIELD_RELAXATION, rawPath, "relax " + path + " to optional"));
    } else if (oldField.isOptional() && newField.isRequired()) {
      changes.add(new Change(Kind.CONFLICT, rawPath, "optionality tightened on " + path));
    }
    boolean oldPrimitive = oldField.type().isPrimitiveType();
    boolean newPrimitive = newField.type().isPrimitiveType();
    if (oldPrimitive && newPrimitive) {
      if (oldField.type().equals(newField.type())) {
        return;
      }
      if (TypeUtil.isPromotionAllowed(oldField.type(), newField.type().asPrimitiveType())) {
        changes.add(
            new Change(
                Kind.TYPE_PROMOTION,
                rawPath,
                "promote " + path + " " + oldField.type() + " to " + newField.type()));
      } else {
        changes.add(
            new Change(
                Kind.CONFLICT,
                rawPath,
                "type changed on "
                    + path
                    + " from "
                    + oldField.type()
                    + " to "
                    + newField.type()
                    + " (not a promotion)"));
      }
    } else if (oldPrimitive != newPrimitive
        || oldField.type().typeId() != newField.type().typeId()) {
      changes.add(
          new Change(
              Kind.CONFLICT,
              rawPath,
              "type changed on " + path + " from " + oldField.type() + " to " + newField.type()));
    }
  }

  /** A field added inside a newly added struct is reported once, as part of its ancestor. */
  private static boolean hasAddedAncestor(
      int id, Map<Integer, Integer> parentById, Map<Integer, Types.NestedField> beforeById) {
    Integer parent = parentById.get(id);
    while (parent != null) {
      if (!beforeById.containsKey(parent)) {
        return true;
      }
      parent = parentById.get(parent);
    }
    return false;
  }

  private static String quoteIfDotted(String name) {
    if (name.contains(".")) {
      return "`" + name + "`";
    }
    return name;
  }

  private static String optionality(Types.NestedField field) {
    return field.isOptional() ? "optional" : "required";
  }

  boolean isEmpty() {
    return changes.isEmpty();
  }

  Set<Kind> kinds() {
    Set<Kind> kinds = EnumSet.noneOf(Kind.class);
    for (Change change : changes) {
      kinds.add(change.kind);
    }
    return kinds;
  }

  List<String> descriptions() {
    List<String> descriptions = new ArrayList<>();
    for (Change change : changes) {
      descriptions.add(change.description);
    }
    return Collections.unmodifiableList(descriptions);
  }

  @Nullable
  String conflict() {
    for (Change change : changes) {
      if (change.kind == Kind.CONFLICT) {
        return change.description;
      }
    }
    return null;
  }

  boolean allowedBy(SchemaEvolutionConfig config) {
    for (Change change : changes) {
      if (!change.allowedBy(config)) {
        return false;
      }
    }
    return true;
  }

  /** Why {@link #allowedBy} is false; empty when it is true. */
  String disallowedReason(SchemaEvolutionConfig config) {
    List<String> conflicts = new ArrayList<>();
    for (Change change : changes) {
      if (change.kind == Kind.CONFLICT) {
        conflicts.add(change.description);
      }
    }
    if (!conflicts.isEmpty()) {
      return "file schema conflicts with the table schema: " + String.join("; ", conflicts);
    }
    List<String> disallowed = new ArrayList<>();
    for (Change change : changes) {
      if (!change.allowedBy(config)) {
        disallowed.add(change.disallowedReason(config));
      }
    }
    if (disallowed.isEmpty()) {
      return "";
    }
    return "file schema needs changes that are not allowed: " + String.join("; ", disallowed);
  }

  @Override
  public String toString() {
    return "SchemaDelta" + descriptions();
  }
}
