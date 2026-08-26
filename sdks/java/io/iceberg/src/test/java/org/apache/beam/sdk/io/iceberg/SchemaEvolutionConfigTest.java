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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.EnumSet;
import org.apache.beam.sdk.util.SerializableUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class SchemaEvolutionConfigTest {

  @Test
  public void testDisabledByDefault() {
    SchemaEvolutionConfig config = SchemaEvolutionConfig.builder().build();
    assertFalse(config.isEnabled());
    assertTrue(config.getOptions().isEmpty());
    assertEquals(SchemaEvolutionConfig.disabled(), config);
  }

  @Test
  public void testOfEnablesGivenOptions() {
    SchemaEvolutionConfig config =
        SchemaEvolutionConfig.of(
            SchemaEvolutionOption.ALLOW_FIELD_ADDITION,
            SchemaEvolutionOption.ALLOW_TYPE_PROMOTION,
            SchemaEvolutionOption.ALLOW_FIELD_ADDITION);
    assertTrue(config.isEnabled());
    assertEquals(
        EnumSet.of(
            SchemaEvolutionOption.ALLOW_FIELD_ADDITION, SchemaEvolutionOption.ALLOW_TYPE_PROMOTION),
        config.getOptions());
    assertTrue(config.allows(SchemaEvolutionOption.ALLOW_FIELD_ADDITION));
    assertFalse(config.allows(SchemaEvolutionOption.ALLOW_FIELD_RELAXATION));
  }

  @Test
  public void testOptionsAreImmutable() {
    SchemaEvolutionConfig config =
        SchemaEvolutionConfig.of(SchemaEvolutionOption.ALLOW_FIELD_ADDITION);
    assertThrows(
        UnsupportedOperationException.class,
        () -> config.getOptions().add(SchemaEvolutionOption.ALLOW_TYPE_PROMOTION));
  }

  @Test
  public void testBuilderCopiesInput() {
    java.util.List<SchemaEvolutionOption> input =
        new java.util.ArrayList<>(Arrays.asList(SchemaEvolutionOption.ALLOW_FIELD_RELAXATION));
    SchemaEvolutionConfig config = SchemaEvolutionConfig.builder().setOptions(input).build();
    input.add(SchemaEvolutionOption.ALLOW_FIELD_ADDITION);
    assertEquals(EnumSet.of(SchemaEvolutionOption.ALLOW_FIELD_RELAXATION), config.getOptions());
  }

  @Test
  public void testRequiredColumnsDefaultEmptyAndCopied() {
    assertTrue(SchemaEvolutionConfig.disabled().getRequiredColumns().isEmpty());
    java.util.List<String> input = new java.util.ArrayList<>(Arrays.asList("id", "address.city"));
    SchemaEvolutionConfig config =
        SchemaEvolutionConfig.builder().setRequiredColumns(input).build();
    input.add("name");
    assertTrue(config.isPinned("id"));
    assertTrue(config.isPinned("address.city"));
    assertFalse(config.isPinned("name"));
    assertThrows(UnsupportedOperationException.class, () -> config.getRequiredColumns().add("x"));
  }

  @Test
  public void testIncompatibleSchemaHandlingDefaultsByMode() {
    SchemaEvolutionConfig unset = SchemaEvolutionConfig.disabled();
    assertEquals(
        SchemaEvolutionConfig.IncompatibleSchemaHandling.FAIL_PIPELINE,
        unset.incompatibleSchemaHandling(true));
    assertEquals(
        SchemaEvolutionConfig.IncompatibleSchemaHandling.ROUTE_TO_ERRORS,
        unset.incompatibleSchemaHandling(false));
    SchemaEvolutionConfig forced =
        SchemaEvolutionConfig.builder()
            .setIncompatibleSchemaHandling(
                SchemaEvolutionConfig.IncompatibleSchemaHandling.ROUTE_TO_ERRORS)
            .build();
    assertEquals(
        SchemaEvolutionConfig.IncompatibleSchemaHandling.ROUTE_TO_ERRORS,
        forced.incompatibleSchemaHandling(true));
  }

  // ---- aliases, pins and ignores: every interaction

  private static SchemaEvolutionConfig.Builder enabled() {
    return SchemaEvolutionConfig.builder()
        .setOptions(Arrays.asList(SchemaEvolutionOption.ALLOW_FIELD_ADDITION));
  }

  private static java.util.Map<String, String> alias(String... pairs) {
    java.util.Map<String, String> map = new java.util.LinkedHashMap<>();
    for (int i = 0; i < pairs.length; i += 2) {
      map.put(pairs[i], pairs[i + 1]);
    }
    return map;
  }

  private static void assertRejected(String expectedMessagePart, Runnable build) {
    IllegalArgumentException e = assertThrows(IllegalArgumentException.class, build::run);
    assertTrue(e.getMessage(), e.getMessage().contains(expectedMessagePart));
  }

  @Test
  public void testAliasesAccepted() {
    SchemaEvolutionConfig config =
        enabled()
            .withColumnAliases(alias("amt", "amount", "address.zip_code", "address.zip"))
            .build();
    assertEquals("amount", config.getColumnAliases().get("amt"));
    assertEquals("address.zip", config.getColumnAliases().get("address.zip_code"));
  }

  @Test
  public void testTwoAliasesForOneCanonicalAccepted() {
    SchemaEvolutionConfig config =
        enabled().withColumnAliases(alias("amt", "amount", "amnt", "amount")).build();
    assertEquals(2, config.getColumnAliases().size());
  }

  @Test
  public void testAliasToItselfRejected() {
    assertRejected("maps to itself", () -> enabled().withColumnAliases(alias("a", "a")).build());
  }

  @Test
  public void testAliasChainRejected() {
    assertRejected("chains", () -> enabled().withColumnAliases(alias("a", "b", "b", "c")).build());
  }

  @Test
  public void testAliasAcrossParentsRejected() {
    assertRejected(
        "same parent",
        () -> enabled().withColumnAliases(alias("address.zip", "other.zip")).build());
    assertRejected(
        "same parent", () -> enabled().withColumnAliases(alias("zip", "address.zip")).build());
  }

  @Test
  public void testEmptyAliasRejected() {
    assertRejected("Empty", () -> enabled().withColumnAliases(alias("", "amount")).build());
    assertRejected("Empty", () -> enabled().withColumnAliases(alias("amt", "")).build());
  }

  @Test
  public void testPinnedAliasRejectedPinnedCanonicalAccepted() {
    assertRejected(
        "pin the canonical name",
        () ->
            enabled()
                .withColumnAliases(alias("amt", "amount"))
                .setRequiredColumns(Arrays.asList("amt"))
                .build());
    SchemaEvolutionConfig config =
        enabled()
            .withColumnAliases(alias("amt", "amount"))
            .setRequiredColumns(Arrays.asList("amount"))
            .build();
    assertTrue(config.isPinned("amount"));
  }

  @Test
  public void testIgnoredAliasOrCanonicalRejected() {
    assertRejected(
        "ignored",
        () ->
            enabled()
                .withColumnAliases(alias("amt", "amount"))
                .setIgnoredColumns(Arrays.asList("amt"))
                .build());
    assertRejected(
        "ignored",
        () ->
            enabled()
                .withColumnAliases(alias("amt", "amount"))
                .setIgnoredColumns(Arrays.asList("amount"))
                .build());
  }

  @Test
  public void testPinnedAndIgnoredRejected() {
    assertRejected(
        "both pinned and ignored",
        () ->
            enabled()
                .setRequiredColumns(Arrays.asList("id"))
                .setIgnoredColumns(Arrays.asList("id"))
                .build());
  }

  @Test
  public void testUnrelatedAliasPinAndIgnoreCoexist() {
    SchemaEvolutionConfig config =
        enabled()
            .withColumnAliases(alias("amt", "amount"))
            .setRequiredColumns(Arrays.asList("id"))
            .setIgnoredColumns(Arrays.asList("debug"))
            .build();
    assertTrue(config.isPinned("id"));
    assertTrue(config.getIgnoredColumns().contains("debug"));
    assertEquals("amount", config.getColumnAliases().get("amt"));
  }

  @Test
  public void testAliasesAndIgnoresAreImmutableCopies() {
    java.util.Map<String, String> aliases = alias("amt", "amount");
    SchemaEvolutionConfig config = enabled().withColumnAliases(aliases).build();
    aliases.put("x", "y");
    assertEquals(1, config.getColumnAliases().size());
    assertThrows(
        UnsupportedOperationException.class, () -> config.getColumnAliases().put("p", "q"));
    assertThrows(UnsupportedOperationException.class, () -> config.getIgnoredColumns().add("z"));
  }

  @Test
  public void testSerializableRoundTrip() {
    SchemaEvolutionConfig config =
        SchemaEvolutionConfig.of(
            SchemaEvolutionOption.ALLOW_FIELD_ADDITION,
            SchemaEvolutionOption.ALLOW_FIELD_RELAXATION);
    SchemaEvolutionConfig copy = SerializableUtils.ensureSerializable(config);
    assertEquals(config, copy);
    assertTrue(copy.allows(SchemaEvolutionOption.ALLOW_FIELD_RELAXATION));
  }
}
