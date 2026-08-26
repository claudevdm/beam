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
