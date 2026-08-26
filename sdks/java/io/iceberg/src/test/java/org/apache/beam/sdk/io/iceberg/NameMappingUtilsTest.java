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

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.Map;
import org.apache.iceberg.Schema;
import org.apache.iceberg.mapping.MappingUtil;
import org.apache.iceberg.mapping.NameMapping;
import org.apache.iceberg.mapping.NameMappingParser;
import org.apache.iceberg.types.Types;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class NameMappingUtilsTest {

  private static final Schema FULL_SCHEMA =
      new Schema(
          Types.NestedField.required(1, "id", Types.IntegerType.get()),
          Types.NestedField.optional(
              2,
              "events",
              Types.ListType.ofOptional(
                  3,
                  Types.StructType.of(
                      Types.NestedField.optional(4, "a", Types.IntegerType.get()),
                      Types.NestedField.optional(5, "b", Types.StringType.get())))),
          Types.NestedField.optional(
              6,
              "attrs",
              Types.MapType.ofOptional(7, 8, Types.StringType.get(), Types.LongType.get())));

  /** Single-quoted JSON keeps test literals free of escape noise. */
  private static String json(String singleQuoted) {
    return singleQuoted.replace('\'', '"');
  }

  private static NameMapping mapping(String singleQuotedJson) {
    return NameMappingParser.fromJson(json(singleQuotedJson));
  }

  @Test
  public void testParseOrNullValidMapping() {
    String json = NameMappingParser.toJson(MappingUtil.create(FULL_SCHEMA));
    NameMapping parsed = NameMappingUtils.parseOrNull(json);
    assertNotNull(parsed);
    assertNotNull(parsed.find("events", "element", "b"));
  }

  @Test
  public void testParseOrNullAbsentAndGarbage() {
    assertNull(NameMappingUtils.parseOrNull(null));
    assertNull(NameMappingUtils.parseOrNull("not json at all"));
  }

  @Test
  public void testParseOrNullAmbiguousNames() {
    assertNull(
        NameMappingUtils.parseOrNull(
            json("[ {'field-id': 1, 'names': ['a']}, {'field-id': 2, 'names': ['a']} ]")));
  }

  @Test
  public void testCoversFullMapping() {
    assertTrue(NameMappingUtils.covers(MappingUtil.create(FULL_SCHEMA), FULL_SCHEMA.asStruct()));
  }

  /** Pins the exact mapping shape MappingUtil.create generates, and that covers accepts it. */
  @Test
  public void testGeneratedMappingMatchesExpectedShape() {
    NameMapping expected =
        mapping(
            "[ {'field-id': 1, 'names': ['id']},"
                + "  {'field-id': 2, 'names': ['events'], 'fields': ["
                + "    {'field-id': 3, 'names': ['element'], 'fields': ["
                + "      {'field-id': 4, 'names': ['a']},"
                + "      {'field-id': 5, 'names': ['b']} ]} ]},"
                + "  {'field-id': 6, 'names': ['attrs'], 'fields': ["
                + "    {'field-id': 7, 'names': ['key']},"
                + "    {'field-id': 8, 'names': ['value']} ]} ]");
    assertEquals(expected.asMappedFields(), MappingUtil.create(FULL_SCHEMA).asMappedFields());
    assertTrue(NameMappingUtils.covers(expected, FULL_SCHEMA.asStruct()));
  }

  @Test
  public void testCoversDetectsMissingTopLevelColumn() {
    Schema idOnly = new Schema(Types.NestedField.required(1, "id", Types.IntegerType.get()));
    assertFalse(NameMappingUtils.covers(MappingUtil.create(idOnly), FULL_SCHEMA.asStruct()));
  }

  @Test
  public void testCoversDescendsListsAndMaps() {
    // FULL_SCHEMA minus the nested "b": every top-level name still resolves, so only the
    // recursive walk can detect the gap.
    Schema staleView =
        new Schema(
            Types.NestedField.required(1, "id", Types.IntegerType.get()),
            Types.NestedField.optional(
                2,
                "events",
                Types.ListType.ofOptional(
                    3,
                    Types.StructType.of(
                        Types.NestedField.optional(4, "a", Types.IntegerType.get())))),
            Types.NestedField.optional(
                6,
                "attrs",
                Types.MapType.ofOptional(7, 8, Types.StringType.get(), Types.LongType.get())));
    NameMapping stale = MappingUtil.create(staleView);
    assertNotNull(stale.find("events"));
    assertNotNull(stale.find("events", "element", "a"));
    assertFalse(NameMappingUtils.covers(stale, FULL_SCHEMA.asStruct()));
  }

  @Test
  public void testCoversDescendsMapValueStructs() {
    Schema full =
        new Schema(
            Types.NestedField.optional(
                1,
                "attrs",
                Types.MapType.ofOptional(
                    2,
                    3,
                    Types.StringType.get(),
                    Types.StructType.of(
                        Types.NestedField.optional(4, "x", Types.IntegerType.get()),
                        Types.NestedField.optional(5, "y", Types.StringType.get())))));
    Schema staleView =
        new Schema(
            Types.NestedField.optional(
                1,
                "attrs",
                Types.MapType.ofOptional(
                    2,
                    3,
                    Types.StringType.get(),
                    Types.StructType.of(
                        Types.NestedField.optional(4, "x", Types.IntegerType.get())))));
    NameMapping stale = MappingUtil.create(staleView);
    assertNotNull(stale.find("attrs", "value", "x"));
    assertFalse(NameMappingUtils.covers(stale, full.asStruct()));
    assertTrue(NameMappingUtils.covers(MappingUtil.create(full), full.asStruct()));
  }

  @Test
  public void testCoversDescendsPlainStructs() {
    Schema full =
        new Schema(
            Types.NestedField.optional(
                1,
                "info",
                Types.StructType.of(
                    Types.NestedField.optional(2, "x", Types.IntegerType.get()),
                    Types.NestedField.optional(3, "y", Types.StringType.get()))));
    Schema staleView =
        new Schema(
            Types.NestedField.optional(
                1,
                "info",
                Types.StructType.of(Types.NestedField.optional(2, "x", Types.IntegerType.get()))));
    assertFalse(NameMappingUtils.covers(MappingUtil.create(staleView), full.asStruct()));
  }

  /** Even a list of primitives needs an "element" entry for its contents to resolve. */
  @Test
  public void testCoversRequiresElementForPrimitiveLists() {
    Schema listOfInts =
        new Schema(
            Types.NestedField.optional(
                1, "tags", Types.ListType.ofOptional(2, Types.IntegerType.get())));
    NameMapping withoutElement = mapping("[ {'field-id': 1, 'names': ['tags']} ]");
    assertFalse(NameMappingUtils.covers(withoutElement, listOfInts.asStruct()));

    NameMapping withElement =
        mapping(
            "[ {'field-id': 1, 'names': ['tags'], 'fields': ["
                + "  {'field-id': 2, 'names': ['element']} ]} ]");
    assertTrue(NameMappingUtils.covers(withElement, listOfInts.asStruct()));
  }

  @Test
  public void testCoversDeepComposition() {
    // list<map<string, struct<x>>>
    Schema deep =
        new Schema(
            Types.NestedField.optional(
                1,
                "rows",
                Types.ListType.ofOptional(
                    2,
                    Types.MapType.ofOptional(
                        3,
                        4,
                        Types.StringType.get(),
                        Types.StructType.of(
                            Types.NestedField.optional(5, "x", Types.IntegerType.get()))))));
    Schema staleView =
        new Schema(
            Types.NestedField.optional(
                1,
                "rows",
                Types.ListType.ofOptional(
                    2,
                    Types.MapType.ofOptional(
                        3,
                        4,
                        Types.StringType.get(),
                        Types.StructType.of(
                            Types.NestedField.optional(6, "w", Types.IntegerType.get()))))));
    assertTrue(NameMappingUtils.covers(MappingUtil.create(deep), deep.asStruct()));
    assertFalse(NameMappingUtils.covers(MappingUtil.create(staleView), deep.asStruct()));
  }

  /** A null-id entry means "maps to nothing": readers project nulls, so it is not coverage. */
  @Test
  public void testCoversRejectsNullIdEntry() {
    Schema schema = new Schema(Types.NestedField.required(7, "user_id", Types.IntegerType.get()));
    NameMapping tombstoned = mapping("[ {'names': ['user_id']} ]");
    assertFalse(NameMappingUtils.covers(tombstoned, schema.asStruct()));
  }

  /** A wrong-id binding silently reads the wrong column, worse than reading nulls. */
  @Test
  public void testCoversRejectsWrongIdBinding() {
    Schema schema =
        new Schema(
            Types.NestedField.optional(1, "a", Types.IntegerType.get()),
            Types.NestedField.optional(2, "b", Types.IntegerType.get()));
    NameMapping swapped =
        mapping("[ {'field-id': 2, 'names': ['a']}, {'field-id': 1, 'names': ['b']} ]");
    assertFalse(NameMappingUtils.covers(swapped, schema.asStruct()));
  }

  @Test
  public void testCoversRejectsWrongNestedIdBinding() {
    NameMapping wrongNestedId =
        mapping(
            "[ {'field-id': 1, 'names': ['id']},"
                + "  {'field-id': 2, 'names': ['events'], 'fields': ["
                + "    {'field-id': 3, 'names': ['element'], 'fields': ["
                + "      {'field-id': 9, 'names': ['a']},"
                + "      {'field-id': 5, 'names': ['b']} ]} ]},"
                + "  {'field-id': 6, 'names': ['attrs'], 'fields': ["
                + "    {'field-id': 7, 'names': ['key']},"
                + "    {'field-id': 8, 'names': ['value']} ]} ]");
    assertNotNull(wrongNestedId.find("events", "element", "a"));
    assertFalse(NameMappingUtils.covers(wrongNestedId, FULL_SCHEMA.asStruct()));
  }

  @Test
  public void testCoversEmptySchema() {
    Schema empty = new Schema();
    assertTrue(NameMappingUtils.covers(MappingUtil.create(empty), empty.asStruct()));
    NameMapping regenerated = NameMappingParser.fromJson(NameMappingUtils.regenerate(empty, null));
    assertTrue(NameMappingUtils.covers(regenerated, empty.asStruct()));
  }

  @Test
  public void testRegenerateWithoutExisting() {
    assertEquals(
        NameMappingParser.toJson(MappingUtil.create(FULL_SCHEMA)),
        NameMappingUtils.regenerate(FULL_SCHEMA, null));
  }

  @Test
  public void testRegeneratePreservesCustomNames() {
    Schema schema =
        new Schema(
            Types.NestedField.required(1, "id", Types.IntegerType.get()),
            Types.NestedField.optional(2, "name", Types.StringType.get()));
    NameMapping stale = mapping("[ {'field-id': 1, 'names': ['id', 'ident']} ]");

    NameMapping merged = NameMappingParser.fromJson(NameMappingUtils.regenerate(schema, stale));

    assertEquals(1, merged.find("ident").id().intValue());
    assertEquals(2, merged.find("name").id().intValue());
    assertTrue(NameMappingUtils.covers(merged, schema.asStruct()));
  }

  @Test
  public void testRegeneratePreservesNestedCustomNames() {
    NameMapping stale =
        mapping(
            "[ {'field-id': 1, 'names': ['id']},"
                + "  {'field-id': 2, 'names': ['events'], 'fields': ["
                + "    {'field-id': 3, 'names': ['element'], 'fields': ["
                + "      {'field-id': 4, 'names': ['a', 'legacy_a']} ]} ]} ]");

    NameMapping merged =
        NameMappingParser.fromJson(NameMappingUtils.regenerate(FULL_SCHEMA, stale));

    assertEquals(4, merged.find("events", "element", "legacy_a").id().intValue());
    assertEquals(5, merged.find("events", "element", "b").id().intValue());
    assertTrue(NameMappingUtils.covers(merged, FULL_SCHEMA.asStruct()));
  }

  /** A custom name colliding with a real column would make the mapping ambiguous; schema wins. */
  @Test
  public void testRegenerateDropsNameClaimedByRealColumn() {
    Schema schema =
        new Schema(
            Types.NestedField.optional(1, "amount", Types.DoubleType.get()),
            Types.NestedField.optional(2, "amt", Types.IntegerType.get()));
    NameMapping stale = mapping("[ {'field-id': 1, 'names': ['amount', 'amt']} ]");

    NameMapping merged = NameMappingParser.fromJson(NameMappingUtils.regenerate(schema, stale));

    assertEquals(2, merged.find("amt").id().intValue());
    assertEquals(1, merged.find("amount").id().intValue());
  }

  @Test
  public void testRegenerateDropsRemovedColumns() {
    Schema idOnly = new Schema(Types.NestedField.required(1, "id", Types.IntegerType.get()));
    NameMapping stale =
        mapping("[ {'field-id': 1, 'names': ['id']}, {'field-id': 9, 'names': ['ghost']} ]");

    NameMapping merged = NameMappingParser.fromJson(NameMappingUtils.regenerate(idOnly, stale));

    assertNull(merged.find("ghost"));
    assertEquals(1, merged.find("id").id().intValue());
  }

  /** The collision check applies per nesting level, not just at the top. */
  @Test
  public void testRegenerateDropsNestedCollidingName() {
    // The old entry for field 4 carries "b" as a custom name; ambiguous once the real "b"
    // (field 5) is generated alongside it.
    NameMapping stale =
        mapping(
            "[ {'field-id': 1, 'names': ['id']},"
                + "  {'field-id': 2, 'names': ['events'], 'fields': ["
                + "    {'field-id': 3, 'names': ['element'], 'fields': ["
                + "      {'field-id': 4, 'names': ['a', 'b']} ]} ]} ]");

    NameMapping merged =
        NameMappingParser.fromJson(NameMappingUtils.regenerate(FULL_SCHEMA, stale));

    assertEquals(5, merged.find("events", "element", "b").id().intValue());
    assertEquals(4, merged.find("events", "element", "a").id().intValue());
  }

  /**
   * Two old entries (parseable because they sat at different levels) donating the same name to one
   * level: first in field order wins, the other is dropped.
   */
  @Test
  public void testRegenerateSameNameDonatedTwice() {
    Schema schema =
        new Schema(
            Types.NestedField.optional(4, "a", Types.IntegerType.get()),
            Types.NestedField.optional(5, "b", Types.IntegerType.get()));
    NameMapping old =
        mapping(
            "[ {'field-id': 4, 'names': ['a', 'zz']},"
                + "  {'field-id': 9, 'names': ['wrap'], 'fields': ["
                + "    {'field-id': 5, 'names': ['b', 'zz']} ]} ]");

    NameMapping merged = NameMappingParser.fromJson(NameMappingUtils.regenerate(schema, old));

    assertEquals(4, merged.find("zz").id().intValue());
    assertEquals(5, merged.find("b").id().intValue());
  }

  /** The name-mapping spec allows entries without a field-id; they are skipped, not a crash. */
  @Test
  public void testRegenerateToleratesIdLessEntries() {
    Schema schema = new Schema(Types.NestedField.required(1, "id", Types.IntegerType.get()));
    NameMapping old =
        mapping("[ {'names': ['ghost']}, {'field-id': 1, 'names': ['id', 'ident']} ]");

    NameMapping merged = NameMappingParser.fromJson(NameMappingUtils.regenerate(schema, old));

    assertEquals(1, merged.find("ident").id().intValue());
    assertNull(merged.find("ghost"));
  }

  /**
   * Iceberg's own parser rejects duplicate ids, so a duplicate-id mapping self-resolves to "absent"
   * and can never reach {@code regenerate}.
   */
  @Test
  public void testDuplicateIdMappingIsUnparseable() {
    assertNull(
        NameMappingUtils.parseOrNull(
            json(
                "[ {'field-id': 4, 'names': ['a', 'x']},"
                    + "  {'field-id': 9, 'names': ['wrap'], 'fields': ["
                    + "    {'field-id': 4, 'names': ['a', 'y']} ]} ]")));
  }

  /** Files written before a column rename resolve through the carried-over old name. */
  @Test
  public void testRegenerateKeepsPreRenameName() {
    Schema renamed = new Schema(Types.NestedField.required(1, "new_name", Types.IntegerType.get()));
    NameMapping old = mapping("[ {'field-id': 1, 'names': ['old_name']} ]");

    NameMapping merged = NameMappingParser.fromJson(NameMappingUtils.regenerate(renamed, old));

    assertEquals(1, merged.find("new_name").id().intValue());
    assertEquals(1, merged.find("old_name").id().intValue());
  }

  // ---- aliases

  private static final Schema ALIAS_SCHEMA =
      new Schema(
          Types.NestedField.required(1, "id", Types.IntegerType.get()),
          Types.NestedField.optional(2, "amount", Types.LongType.get()),
          Types.NestedField.optional(
              3,
              "address",
              Types.StructType.of(
                  Types.NestedField.optional(4, "city", Types.StringType.get()),
                  Types.NestedField.optional(5, "zip", Types.IntegerType.get()))));

  private static Map<String, String> aliases(String... pairs) {
    Map<String, String> map = new java.util.LinkedHashMap<>();
    for (int i = 0; i < pairs.length; i += 2) {
      map.put(pairs[i], pairs[i + 1]);
    }
    return map;
  }

  private static NameMapping generated() {
    return NameMappingParser.fromJson(NameMappingUtils.regenerate(ALIAS_SCHEMA, null));
  }

  @Test
  public void testWithAliasesAddsTopLevelAndNestedNames() {
    NameMapping result =
        NameMappingUtils.withAliases(
            generated(), ALIAS_SCHEMA, aliases("amt", "amount", "address.zip_code", "address.zip"));
    NameMapping expected =
        mapping(
            "[ {'field-id': 1, 'names': ['id']},"
                + "  {'field-id': 2, 'names': ['amount', 'amt']},"
                + "  {'field-id': 3, 'names': ['address'], 'fields': ["
                + "     {'field-id': 4, 'names': ['city']},"
                + "     {'field-id': 5, 'names': ['zip', 'zip_code']} ]} ]");
    assertEquals(expected.asMappedFields(), result.asMappedFields());
    assertTrue(
        NameMappingUtils.hasAliases(
            result, ALIAS_SCHEMA, aliases("amt", "amount", "address.zip_code", "address.zip")));
    assertFalse(NameMappingUtils.hasAliases(generated(), ALIAS_SCHEMA, aliases("amt", "amount")));
  }

  @Test
  public void testTwoAliasesForOneColumn() {
    NameMapping result =
        NameMappingUtils.withAliases(
            generated(), ALIAS_SCHEMA, aliases("amt", "amount", "amnt", "amount"));
    assertEquals(2, result.find("amount").id().intValue());
    assertEquals(2, result.find("amt").id().intValue());
    assertEquals(2, result.find("amnt").id().intValue());
  }

  @Test
  public void testWithAliasesIsIdempotent() {
    Map<String, String> aliases = aliases("amt", "amount");
    NameMapping once = NameMappingUtils.withAliases(generated(), ALIAS_SCHEMA, aliases);
    NameMapping twice = NameMappingUtils.withAliases(once, ALIAS_SCHEMA, aliases);
    assertEquals(once.asMappedFields(), twice.asMappedFields());
  }

  @Test
  public void testAliasForAColumnNotInTheSchemaIsDeferred() {
    Map<String, String> aliases = aliases("eml", "email");
    NameMapping result = NameMappingUtils.withAliases(generated(), ALIAS_SCHEMA, aliases);
    assertEquals(generated().asMappedFields(), result.asMappedFields());
    assertTrue(
        "nothing applicable yet", NameMappingUtils.hasAliases(result, ALIAS_SCHEMA, aliases));
  }

  @Test
  public void testAliasShadowingARealColumnIsSkipped() {
    Map<String, String> aliases = aliases("id", "amount");
    NameMapping result = NameMappingUtils.withAliases(generated(), ALIAS_SCHEMA, aliases);
    assertEquals(generated().asMappedFields(), result.asMappedFields());
    assertEquals(1, result.find("id").id().intValue());
    assertTrue(NameMappingUtils.hasAliases(result, ALIAS_SCHEMA, aliases));
  }

  @Test
  public void testAliasAlreadyMappedToAnotherFieldIsSkipped() {
    NameMapping custom =
        mapping(
            "[ {'field-id': 1, 'names': ['id', 'amt']},"
                + "  {'field-id': 2, 'names': ['amount']},"
                + "  {'field-id': 3, 'names': ['address'], 'fields': ["
                + "     {'field-id': 4, 'names': ['city']},"
                + "     {'field-id': 5, 'names': ['zip']} ]} ]");
    Map<String, String> aliases = aliases("amt", "amount");
    NameMapping result = NameMappingUtils.withAliases(custom, ALIAS_SCHEMA, aliases);
    assertEquals(custom.asMappedFields(), result.asMappedFields());
    assertTrue(NameMappingUtils.hasAliases(result, ALIAS_SCHEMA, aliases));
  }

  @Test
  public void testAliasSurvivesRegeneration() {
    NameMapping aliased =
        NameMappingUtils.withAliases(generated(), ALIAS_SCHEMA, aliases("amt", "amount"));
    Schema wider =
        new Schema(
            Types.NestedField.required(1, "id", Types.IntegerType.get()),
            Types.NestedField.optional(2, "amount", Types.LongType.get()),
            Types.NestedField.optional(
                3,
                "address",
                Types.StructType.of(
                    Types.NestedField.optional(4, "city", Types.StringType.get()),
                    Types.NestedField.optional(5, "zip", Types.IntegerType.get()))),
            Types.NestedField.optional(6, "email", Types.StringType.get()));
    NameMapping regenerated =
        NameMappingParser.fromJson(NameMappingUtils.regenerate(wider, aliased));
    assertEquals(2, regenerated.find("amt").id().intValue());
    assertEquals(6, regenerated.find("email").id().intValue());
  }
}
