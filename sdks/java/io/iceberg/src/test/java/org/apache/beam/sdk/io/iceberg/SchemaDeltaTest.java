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

import static org.apache.iceberg.types.Types.NestedField.optional;
import static org.apache.iceberg.types.Types.NestedField.required;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.EnumSet;
import org.apache.beam.sdk.io.iceberg.SchemaDelta.Kind;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Table;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.types.Types;
import org.junit.ClassRule;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.rules.TestName;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class SchemaDeltaTest {
  @ClassRule public static final TemporaryFolder TEMPORARY_FOLDER = new TemporaryFolder();

  @Rule
  public transient TestDataWarehouse warehouse = new TestDataWarehouse(TEMPORARY_FOLDER, "default");

  @Rule public TestName testName = new TestName();

  private static final Schema TABLE =
      new Schema(
          required(1, "id", Types.LongType.get()),
          optional(2, "name", Types.StringType.get()),
          optional(3, "score", Types.FloatType.get()),
          optional(
              4,
              "address",
              Types.StructType.of(
                  required(5, "city", Types.StringType.get()),
                  optional(6, "zip", Types.IntegerType.get()))),
          optional(7, "tags", Types.ListType.ofOptional(8, Types.StringType.get())),
          optional(9, "amount", Types.DecimalType.of(9, 2)));

  private static final SchemaEvolutionConfig ALL =
      SchemaEvolutionConfig.of(SchemaEvolutionOption.values());

  private Table table;

  private SchemaDelta classify(Schema fileSchema) {
    return classify(TABLE, fileSchema);
  }

  private SchemaDelta classify(Schema tableSchema, Schema fileSchema) {
    if (table == null) {
      table =
          warehouse.createTable(
              TableIdentifier.of("default", testName.getMethodName()), tableSchema);
    }
    return SchemaDelta.classify(table, fileSchema);
  }

  // ---- empty deltas

  @Test
  public void testIdenticalSchemaIsEmpty() {
    assertTrue(classify(TABLE).isEmpty());
  }

  @Test
  public void testNarrowerFileTypeIsCovered() {
    Schema file = new Schema(required(1, "id", Types.IntegerType.get()));
    assertTrue(classify(file).isEmpty());
  }

  @Test
  public void testFileRequiredWhereTableOptionalIsCovered() {
    Schema file =
        new Schema(
            required(1, "name", Types.StringType.get()), required(2, "id", Types.LongType.get()));
    assertTrue(classify(file).isEmpty());
  }

  @Test
  public void testReorderedAndSubsetColumnsAreCovered() {
    Schema file =
        new Schema(
            optional(1, "name", Types.StringType.get()), required(2, "id", Types.LongType.get()));
    assertTrue(classify(file).isEmpty());
  }

  // ---- required columns absent from the file

  @Test
  public void testAbsentRequiredColumnIsRelaxation() {
    Schema file = new Schema(optional(1, "name", Types.StringType.get()));
    SchemaDelta delta = classify(file);
    assertEquals(EnumSet.of(Kind.FIELD_RELAXATION), delta.kinds());
    assertEquals(Arrays.asList("relax id to optional (absent from file)"), delta.descriptions());
    assertEquals(Arrays.asList("id"), delta.absentRequiredPaths());
    assertTrue(
        delta.allowedBy(SchemaEvolutionConfig.of(SchemaEvolutionOption.ALLOW_FIELD_RELAXATION)));
    assertFalse(
        delta.allowedBy(SchemaEvolutionConfig.of(SchemaEvolutionOption.ALLOW_FIELD_ADDITION)));
  }

  @Test
  public void testAbsentOptionalColumnIsNotAChange() {
    Schema file = new Schema(required(1, "id", Types.LongType.get()));
    assertTrue(classify(file).isEmpty());
  }

  @Test
  public void testAbsentNestedRequiredChildIsRelaxation() {
    Schema file =
        new Schema(
            required(1, "id", Types.LongType.get()),
            optional(
                2, "address", Types.StructType.of(optional(3, "zip", Types.IntegerType.get()))));
    SchemaDelta delta = classify(file);
    assertEquals(
        Arrays.asList("relax address.city to optional (absent from file)"), delta.descriptions());
    assertEquals(Arrays.asList("address.city"), delta.absentRequiredPaths());
  }

  @Test
  public void testAbsentRequiredStructIsOneRelaxation() {
    Schema tableSchema =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(
                2, "address", Types.StructType.of(required(3, "city", Types.StringType.get()))));
    Schema file = new Schema(required(1, "id", Types.LongType.get()));
    SchemaDelta delta = classify(tableSchema, file);
    assertEquals(
        Arrays.asList("relax address to optional (absent from file)"), delta.descriptions());
  }

  @Test
  public void testAbsentPinnedColumnIsRejected() {
    Schema file = new Schema(optional(1, "name", Types.StringType.get()));
    SchemaEvolutionConfig pinned =
        SchemaEvolutionConfig.builder()
            .setOptions(Arrays.asList(SchemaEvolutionOption.values()))
            .setRequiredColumns(Arrays.asList("id"))
            .build();
    SchemaDelta delta = classify(file);
    assertFalse(delta.allowedBy(pinned));
    assertEquals(
        "file schema needs changes that are not allowed: "
            + "relax id to optional (absent from file) (pinned as required)",
        delta.disallowedReason(pinned));
  }

  @Test
  public void testAbsentRequiredUnderListElementIsRelaxation() {
    Schema tableSchema =
        new Schema(
            required(1, "id", Types.LongType.get()),
            optional(
                2,
                "items",
                Types.ListType.ofOptional(
                    3, Types.StructType.of(required(4, "sku", Types.StringType.get())))));
    Schema file =
        new Schema(
            required(1, "id", Types.LongType.get()),
            optional(
                2,
                "items",
                Types.ListType.ofOptional(
                    3, Types.StructType.of(optional(4, "qty", Types.IntegerType.get())))));
    SchemaDelta delta = classify(tableSchema, file);
    assertEquals(
        Arrays.asList(
            "add optional items.element.qty int",
            "relax items.element.sku to optional (absent from file)"),
        delta.descriptions());
    assertEquals(Arrays.asList("items.element.sku"), delta.absentRequiredPaths());
  }

  @Test
  public void testAbsentRequiredUnderMapValueIsRelaxation() {
    Schema tableSchema =
        new Schema(
            required(1, "id", Types.LongType.get()),
            optional(
                2,
                "attrs",
                Types.MapType.ofOptional(
                    3,
                    4,
                    Types.StringType.get(),
                    Types.StructType.of(required(5, "v", Types.StringType.get())))));
    Schema file =
        new Schema(
            required(1, "id", Types.LongType.get()),
            optional(
                2,
                "attrs",
                Types.MapType.ofOptional(
                    3,
                    4,
                    Types.StringType.get(),
                    Types.StructType.of(optional(5, "w", Types.StringType.get())))));
    assertEquals(Arrays.asList("attrs.value.v"), classify(tableSchema, file).absentRequiredPaths());
  }

  @Test
  public void testIllegalPrimitiveChangeInDiffIsConflict() {
    Schema before = new Schema(required(1, "id", Types.LongType.get()));
    Schema after = new Schema(required(1, "id", Types.IntegerType.get()));
    SchemaDelta delta = SchemaDelta.diff(before, after);
    assertEquals(
        Arrays.asList("type changed on id from long to int (not a promotion)"),
        delta.descriptions());
    assertEquals(EnumSet.of(Kind.CONFLICT), delta.kinds());
  }

  // ---- additions

  @Test
  public void testTopLevelAddition() {
    Schema file =
        new Schema(
            optional(1, "email", Types.StringType.get()), required(2, "id", Types.LongType.get()));
    SchemaDelta delta = classify(file);
    assertEquals(EnumSet.of(Kind.FIELD_ADDITION), delta.kinds());
    assertEquals(Arrays.asList("add optional email string"), delta.descriptions());
  }

  /**
   * A tightened (required) new column is still added optional: tightening only avoids relaxations.
   */
  @Test
  public void testRequiredNewColumnIsAddedOptional() {
    Schema file =
        new Schema(
            required(1, "email", Types.StringType.get()), required(2, "id", Types.LongType.get()));
    assertEquals(Arrays.asList("add optional email string"), classify(file).descriptions());
  }

  @Test
  public void testNestedAddition() {
    Schema file =
        new Schema(
            required(3, "id", Types.LongType.get()),
            optional(
                1,
                "address",
                Types.StructType.of(
                    required(4, "city", Types.StringType.get()),
                    optional(2, "country", Types.StringType.get()))));
    SchemaDelta delta = classify(file);
    assertEquals(EnumSet.of(Kind.FIELD_ADDITION), delta.kinds());
    assertEquals(Arrays.asList("add optional address.country string"), delta.descriptions());
  }

  @Test
  public void testAddedStructIsReportedOnce() {
    Schema file =
        new Schema(
            required(9, "id", Types.LongType.get()),
            optional(
                1,
                "geo",
                Types.StructType.of(
                    optional(2, "lat", Types.DoubleType.get()),
                    optional(3, "lon", Types.DoubleType.get()))));
    SchemaDelta delta = classify(file);
    assertEquals(
        Arrays.asList("add optional geo struct<lat: optional double, lon: optional double>"),
        stripIds(delta));
  }

  // ---- relaxation

  @Test
  public void testTopLevelRelaxation() {
    Schema file = new Schema(optional(1, "id", Types.LongType.get()));
    SchemaDelta delta = classify(file);
    assertEquals(EnumSet.of(Kind.FIELD_RELAXATION), delta.kinds());
    assertEquals(Arrays.asList("relax id to optional"), delta.descriptions());
  }

  @Test
  public void testNestedRelaxation() {
    Schema file =
        new Schema(
            required(9, "id", Types.LongType.get()),
            optional(
                1, "address", Types.StructType.of(optional(2, "city", Types.StringType.get()))));
    SchemaDelta delta = classify(file);
    assertEquals(Arrays.asList("relax address.city to optional"), delta.descriptions());
  }

  // ---- promotion

  @Test
  public void testPromotions() {
    Schema file =
        new Schema(
            required(9, "id", Types.LongType.get()),
            optional(1, "score", Types.DoubleType.get()),
            optional(2, "amount", Types.DecimalType.of(18, 2)),
            optional(
                3,
                "address",
                Types.StructType.of(
                    required(5, "city", Types.StringType.get()),
                    optional(4, "zip", Types.LongType.get()))));
    SchemaDelta delta = classify(file);
    assertEquals(EnumSet.of(Kind.TYPE_PROMOTION), delta.kinds());
    assertEquals(
        Arrays.asList(
            "promote address.zip int to long",
            "promote amount decimal(9, 2) to decimal(18, 2)",
            "promote score float to double"),
        delta.descriptions());
  }

  // ---- conflicts

  @Test
  public void testTypeConflict() {
    Schema file = new Schema(optional(1, "name", Types.IntegerType.get()));
    SchemaDelta delta = classify(file);
    assertEquals(EnumSet.of(Kind.CONFLICT), delta.kinds());
    assertNotNull(delta.conflict());
    assertFalse(delta.allowedBy(ALL));
  }

  @Test
  public void testStructVersusPrimitiveConflict() {
    Schema file = new Schema(optional(1, "address", Types.StringType.get()));
    assertNotNull(classify(file).conflict());
  }

  @Test
  public void testNarrowingIsConflict() {
    Schema file = new Schema(required(1, "id", Types.IntegerType.get()));
    // narrower than the table: covered, not a change
    assertTrue(classify(file).isEmpty());
    SchemaDelta scaleChange =
        classify(new Schema(optional(1, "amount", Types.DecimalType.of(9, 4))));
    assertNotNull(scaleChange.toString(), scaleChange.conflict());
  }

  @Test
  public void testListElementConflict() {
    Schema file =
        new Schema(optional(1, "tags", Types.ListType.ofOptional(2, Types.LongType.get())));
    assertNotNull(classify(file).conflict());
  }

  // ---- combined and gating

  @Test
  public void testCombinedDeltaReportsEveryKind() {
    Schema file =
        new Schema(
            optional(1, "id", Types.LongType.get()),
            optional(2, "score", Types.DoubleType.get()),
            optional(3, "email", Types.StringType.get()));
    SchemaDelta delta = classify(file);
    assertEquals(
        EnumSet.of(Kind.FIELD_ADDITION, Kind.FIELD_RELAXATION, Kind.TYPE_PROMOTION), delta.kinds());
    assertNull(delta.conflict());
    assertTrue(delta.allowedBy(ALL));
    assertFalse(
        delta.allowedBy(
            SchemaEvolutionConfig.of(
                SchemaEvolutionOption.ALLOW_FIELD_ADDITION,
                SchemaEvolutionOption.ALLOW_TYPE_PROMOTION)));
    assertEquals(
        "file schema needs changes that are not allowed: "
            + "relax id to optional (needs ALLOW_FIELD_RELAXATION)",
        delta.disallowedReason(
            SchemaEvolutionConfig.of(
                SchemaEvolutionOption.ALLOW_FIELD_ADDITION,
                SchemaEvolutionOption.ALLOW_TYPE_PROMOTION)));
    assertEquals("", delta.disallowedReason(ALL));
  }

  @Test
  public void testPinnedColumnIsNeverRelaxed() {
    Schema file =
        new Schema(
            optional(1, "id", Types.LongType.get()),
            optional(
                2, "address", Types.StructType.of(optional(3, "city", Types.StringType.get()))));
    SchemaDelta delta = classify(file);
    assertEquals(EnumSet.of(Kind.FIELD_RELAXATION), delta.kinds());
    assertTrue(delta.allowedBy(ALL));
    SchemaEvolutionConfig pinned =
        SchemaEvolutionConfig.builder()
            .setOptions(Arrays.asList(SchemaEvolutionOption.values()))
            .setRequiredColumns(Arrays.asList("address.city"))
            .build();
    assertFalse(delta.allowedBy(pinned));
    assertEquals(
        "file schema needs changes that are not allowed: "
            + "relax address.city to optional (pinned as required)",
        delta.disallowedReason(pinned));
  }

  @Test
  public void testEmptyDeltaIsAllowedWithoutOptions() {
    assertTrue(classify(TABLE).allowedBy(SchemaEvolutionConfig.disabled()));
  }

  @Test
  public void testConflictReason() {
    Schema file = new Schema(optional(1, "name", Types.IntegerType.get()));
    String reason = classify(file).disallowedReason(ALL);
    assertTrue(reason, reason.startsWith("file schema conflicts with the table schema: "));
  }

  // ---- open decisions, pinned to current Iceberg behaviour

  @Test
  public void testDottedColumnNameConflicts() {
    Schema file = new Schema(optional(1, "address.zip", Types.IntegerType.get()));
    SchemaDelta delta = classify(file);
    assertEquals(delta.toString(), EnumSet.of(Kind.CONFLICT), delta.kinds());
  }

  @Test
  public void testEmptyColumnNameConflicts() {
    Schema file = new Schema(optional(1, "", Types.StringType.get()));
    SchemaDelta delta = classify(file);
    assertEquals(delta.toString(), EnumSet.of(Kind.CONFLICT), delta.kinds());
  }

  @Test
  public void testCaseOnlyDifferenceIsAnAddition() {
    Schema file =
        new Schema(
            optional(1, "NAME", Types.StringType.get()), required(2, "id", Types.LongType.get()));
    assertEquals(Arrays.asList("add optional NAME string"), classify(file).descriptions());
  }

  // ---- diff sanity checks, driven directly

  @Test
  public void testRemovedFieldIsConflict() {
    Schema before =
        new Schema(
            required(1, "id", Types.LongType.get()), optional(2, "x", Types.StringType.get()));
    Schema after = new Schema(required(1, "id", Types.LongType.get()));
    SchemaDelta delta = SchemaDelta.diff(before, after);
    assertEquals(EnumSet.of(Kind.CONFLICT), delta.kinds());
    assertFalse(delta.allowedBy(ALL));
  }

  @Test
  public void testTightenedOptionalityOnNestedFieldIsConflict() {
    Schema before =
        new Schema(
            optional(1, "a", Types.StructType.of(optional(2, "b", Types.IntegerType.get()))));
    Schema after =
        new Schema(
            required(1, "a", Types.StructType.of(optional(2, "b", Types.IntegerType.get()))));
    SchemaDelta delta = SchemaDelta.diff(before, after);
    assertEquals(Arrays.asList("optionality tightened on a"), delta.descriptions());
  }

  @Test
  public void testRelaxedFieldWithChangedDefaultIsStillConflict() {
    Schema before = new Schema(required(1, "id", Types.LongType.get()));
    Schema after =
        new Schema(
            Types.NestedField.optional("id")
                .withId(1)
                .ofType(Types.LongType.get())
                .withWriteDefault(org.apache.iceberg.expressions.Literal.of(7L))
                .build());
    SchemaDelta delta = SchemaDelta.diff(before, after);
    assertEquals(EnumSet.of(Kind.CONFLICT, Kind.FIELD_RELAXATION), delta.kinds());
  }

  @Test
  public void testMultipleConflictsAreAllReported() {
    Schema before =
        new Schema(
            required(1, "id", Types.LongType.get()),
            optional(
                2,
                "s",
                Types.StructType.of(
                    optional(3, "x", Types.IntegerType.get()),
                    optional(4, "y", Types.IntegerType.get()))));
    Schema after = new Schema(required(1, "id", Types.LongType.get()));
    String reason = SchemaDelta.diff(before, after).disallowedReason(ALL);
    assertEquals(
        "file schema conflicts with the table schema: field removed: s; field removed: s.x; field removed: s.y",
        reason);
  }

  @Test
  public void testDocChangeIsConflict() {
    Schema before = new Schema(required(1, "id", Types.LongType.get()));
    Schema after = new Schema(required(1, "id", Types.LongType.get(), "the id"));
    assertEquals(EnumSet.of(Kind.CONFLICT), SchemaDelta.diff(before, after).kinds());
  }

  /** Iceberg rejects a schema where a dotted name equals a nested path, so only quoting matters. */
  @Test
  public void testDottedNameIsQuotedAndDoesNotSwallowSiblings() {
    Schema before = new Schema(required(1, "id", Types.LongType.get()));
    Schema after =
        new Schema(
            required(1, "id", Types.LongType.get()),
            optional(2, "a.b", Types.StringType.get()),
            optional(3, "a", Types.StructType.of(optional(4, "c", Types.IntegerType.get()))));
    SchemaDelta delta = SchemaDelta.diff(before, after);
    assertEquals(
        Arrays.asList("add optional a struct<c: optional int>", "add optional `a.b` string"),
        stripIds(delta));
  }

  private static java.util.List<String> stripIds(SchemaDelta delta) {
    java.util.List<String> stripped = new java.util.ArrayList<>();
    for (String change : delta.descriptions()) {
      stripped.add(change.replaceAll("\\b\\d+: ", ""));
    }
    return stripped;
  }
}
