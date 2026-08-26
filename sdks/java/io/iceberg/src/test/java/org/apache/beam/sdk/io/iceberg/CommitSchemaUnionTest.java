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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.apache.beam.sdk.io.iceberg.CommitSchemaUnion.Committer;
import org.apache.beam.sdk.io.iceberg.CommitSchemaUnion.IncompatibleSchemaException;
import org.apache.beam.sdk.io.iceberg.SchemaEvolutionConfig.IncompatibleSchemaHandling;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.BaseTable;
import org.apache.iceberg.Schema;
import org.apache.iceberg.SchemaParser;
import org.apache.iceberg.Table;
import org.apache.iceberg.TableProperties;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.exceptions.CommitFailedException;
import org.apache.iceberg.hadoop.HadoopCatalog;
import org.apache.iceberg.mapping.NameMapping;
import org.apache.iceberg.types.Types;
import org.junit.Before;
import org.junit.ClassRule;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.rules.TestName;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class CommitSchemaUnionTest {
  @ClassRule public static final TemporaryFolder TEMPORARY_FOLDER = new TemporaryFolder();

  @Rule
  public transient TestDataWarehouse warehouse = new TestDataWarehouse(TEMPORARY_FOLDER, "default");

  @Rule public TestName testName = new TestName();

  private static final Schema TABLE =
      new Schema(
          required(1, "id", Types.LongType.get()),
          optional(2, "name", Types.StringType.get()),
          optional(3, "score", Types.FloatType.get()),
          required(4, "region", Types.StringType.get()));

  private static final SchemaEvolutionConfig ALL =
      SchemaEvolutionConfig.of(SchemaEvolutionOption.values());
  private static final SchemaEvolutionConfig ADDITION_ONLY =
      SchemaEvolutionConfig.of(SchemaEvolutionOption.ALLOW_FIELD_ADDITION);

  private static final CommitSchemaUnion.TableCreation NO_CREATION =
      new CommitSchemaUnion.TableCreation(null, null, null);

  private HadoopCatalog catalog;
  private TableIdentifier tableId;

  @Before
  public void setUp() {
    catalog = new HadoopCatalog(new Configuration(), warehouse.location);
    tableId = TableIdentifier.of("default", testName.getMethodName());
    warehouse.createTable(tableId, TABLE);
  }

  private static String json(Schema schema) {
    return SchemaParser.toJson(FileSchemas.canonical(schema));
  }

  private static CollectDistinctSchemas.SchemaGroup files(Schema schema, long count) {
    return files(schema, count, Collections.emptyList());
  }

  private static CollectDistinctSchemas.SchemaGroup files(
      Schema schema, long count, List<String> nullFreeColumns) {
    return new CollectDistinctSchemas.SchemaGroup(json(schema), count, nullFreeColumns);
  }

  private long commit(
      SchemaEvolutionConfig config,
      IncompatibleSchemaHandling handling,
      CollectDistinctSchemas.SchemaGroup... schemas) {
    return CommitSchemaUnion.commit(
        catalog,
        tableId,
        Arrays.asList(schemas),
        config,
        handling,
        NO_CREATION,
        CommitSchemaUnion.DEFAULT_COMMITTER);
  }

  private Table load() {
    return catalog.loadTable(tableId);
  }

  private static String metadataLocation(Table table) {
    return ((BaseTable) table).operations().current().metadataFileLocation();
  }

  /** The vN in .../metadata/vN.metadata.json: one commit bumps it by exactly one. */
  private static int metadataVersion(Table table) {
    String location = metadataLocation(table);
    String file = location.substring(location.lastIndexOf('/') + 2);
    return Integer.parseInt(file.substring(0, file.indexOf('.')));
  }

  /** A healthy mapping, so no-op tests are not turned into commits by the mapping repair. */
  private void seedNameMapping() {
    Table table = load();
    table
        .updateProperties()
        .set(
            TableProperties.DEFAULT_NAME_MAPPING, NameMappingUtils.regenerate(table.schema(), null))
        .commit();
  }

  // ---- no-op

  @Test
  public void testCoveredSchemasCommitNothing() {
    seedNameMapping();
    String before = metadataLocation(load());
    Schema covered =
        new Schema(
            required(1, "id", Types.IntegerType.get()),
            required(2, "region", Types.StringType.get()));
    long schemaId = commit(ALL, IncompatibleSchemaHandling.FAIL_PIPELINE, files(covered, 3));
    Table table = load();
    assertEquals(table.schema().schemaId(), schemaId);
    assertEquals("no metadata written", before, metadataLocation(table));
  }

  // ---- changes applied

  @Test
  public void testAdditionPromotionAndRelaxationInOneCommit() {
    Schema file =
        new Schema(
            optional(1, "id", Types.LongType.get()),
            optional(2, "score", Types.DoubleType.get()),
            optional(3, "email", Types.StringType.get()),
            required(4, "region", Types.StringType.get()));
    long schemaId = commit(ALL, IncompatibleSchemaHandling.FAIL_PIPELINE, files(file, 2));
    Table table = load();
    assertEquals(table.schema().schemaId(), schemaId);
    assertTrue(table.schema().findField("id").isOptional());
    assertEquals(Types.DoubleType.get(), table.schema().findField("score").type());
    assertNotNull(table.schema().findField("email"));
    assertTrue(table.schema().findField("email").isOptional());
  }

  @Test
  public void testAbsentRequiredColumnIsRelaxedExplicitly() {
    Schema file = new Schema(required(1, "id", Types.LongType.get()));
    commit(ALL, IncompatibleSchemaHandling.FAIL_PIPELINE, files(file, 1));
    assertTrue(load().schema().findField("region").isOptional());
  }

  @Test
  public void testFieldsInsideAnAddedStructAreOptional() {
    Schema file =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            required(
                3,
                "address",
                Types.StructType.of(
                    required(4, "city", Types.StringType.get()),
                    required(5, "zip", Types.IntegerType.get()))));
    commit(ALL, IncompatibleSchemaHandling.FAIL_PIPELINE, files(file, 1));
    Table table = load();
    assertTrue(table.schema().findField("address").isOptional());
    assertTrue(table.schema().findField("address.city").isOptional());
    assertTrue(table.schema().findField("address.zip").isOptional());
  }

  @Test
  public void testPinnedFieldInsideAnAddedStructStaysRequired() {
    SchemaEvolutionConfig pinned =
        SchemaEvolutionConfig.builder()
            .setOptions(Arrays.asList(SchemaEvolutionOption.values()))
            .setRequiredColumns(Arrays.asList("address.city"))
            .build();
    Schema file =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            required(
                3, "address", Types.StructType.of(required(4, "city", Types.StringType.get()))));
    commit(pinned, IncompatibleSchemaHandling.FAIL_PIPELINE, files(file, 1));
    assertTrue(load().schema().findField("address.city").isRequired());
    assertTrue(load().schema().findField("address").isOptional());
  }

  /** A declared-optional column every file proved null-free does not relax the table. */
  @Test
  public void testNullFreeColumnIsNotRelaxed() {
    seedNameMapping();
    Schema file =
        new Schema(
            required(1, "id", Types.LongType.get()), optional(2, "region", Types.StringType.get()));
    int before = metadataVersion(load());
    commit(ALL, IncompatibleSchemaHandling.FAIL_PIPELINE, files(file, 3, Arrays.asList("region")));
    assertTrue(load().schema().findField("region").isRequired());
    assertEquals(before, metadataVersion(load()));
  }

  @Test
  public void testColumnWithoutNullFreeEvidenceStillRelaxes() {
    Schema file =
        new Schema(
            required(1, "id", Types.LongType.get()), optional(2, "region", Types.StringType.get()));
    commit(ALL, IncompatibleSchemaHandling.FAIL_PIPELINE, files(file, 3));
    assertTrue(load().schema().findField("region").isOptional());
  }

  @Test
  public void testMultipleSchemasMergeIntoOneSchemaVersion() {
    Schema a =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "a", Types.StringType.get()));
    Schema b =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "b", Types.LongType.get()));
    int before = metadataVersion(load());
    commit(ALL, IncompatibleSchemaHandling.FAIL_PIPELINE, files(a, 5), files(b, 1));
    Table table = load();
    assertNotNull(table.schema().findField("a"));
    assertNotNull(table.schema().findField("b"));
    assertEquals("one metadata commit", before + 1, metadataVersion(table));
  }

  @Test
  public void testFinalSchemaIsOrderIndependent() {
    Schema a =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "score", Types.DoubleType.get()));
    Schema b =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "extra", Types.StringType.get()));
    commit(ALL, IncompatibleSchemaHandling.FAIL_PIPELINE, files(a, 1), files(b, 2));
    Schema first = load().schema();

    TableIdentifier other = TableIdentifier.of("default", testName.getMethodName() + "_2");
    warehouse.createTable(other, TABLE);
    CommitSchemaUnion.commit(
        catalog,
        other,
        Arrays.asList(files(b, 2), files(a, 1)),
        ALL,
        IncompatibleSchemaHandling.FAIL_PIPELINE,
        NO_CREATION,
        CommitSchemaUnion.DEFAULT_COMMITTER);
    assertTrue(first.sameSchema(catalog.loadTable(other).schema()));
  }

  // ---- incompatible schemas

  @Test
  public void testFailPipelineCommitsNothingWhenAnySchemaIsIncompatible() {
    String before = metadataLocation(load());
    Schema good =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "email", Types.StringType.get()));
    Schema needsPromotion =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "score", Types.DoubleType.get()));
    IncompatibleSchemaException e =
        assertThrows(
            IncompatibleSchemaException.class,
            () ->
                commit(
                    ADDITION_ONLY,
                    IncompatibleSchemaHandling.FAIL_PIPELINE,
                    files(good, 9),
                    files(needsPromotion, 1)));
    assertTrue(e.getMessage(), e.getMessage().contains("1 schema(s), 1 file(s)"));
    assertTrue(e.getMessage(), e.getMessage().contains("promote score float to double"));
    assertEquals(before, metadataLocation(load()));
  }

  @Test
  public void testRouteToErrorsSkipsIncompatibleAndCommitsTheRest() {
    Schema good =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "email", Types.StringType.get()));
    Schema needsPromotion =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "score", Types.DoubleType.get()));
    commit(
        ADDITION_ONLY,
        IncompatibleSchemaHandling.ROUTE_TO_ERRORS,
        files(good, 9),
        files(needsPromotion, 1));
    Table table = load();
    assertNotNull(table.schema().findField("email"));
    assertEquals(Types.FloatType.get(), table.schema().findField("score").type());
  }

  @Test
  public void testPinnedRelaxationIsIncompatible() {
    SchemaEvolutionConfig pinned =
        SchemaEvolutionConfig.builder()
            .setOptions(Arrays.asList(SchemaEvolutionOption.values()))
            .setRequiredColumns(Arrays.asList("region"))
            .build();
    Schema file = new Schema(required(1, "id", Types.LongType.get()));
    IncompatibleSchemaException e =
        assertThrows(
            IncompatibleSchemaException.class,
            () -> commit(pinned, IncompatibleSchemaHandling.FAIL_PIPELINE, files(file, 4)));
    assertTrue(e.getMessage(), e.getMessage().contains("pinned as required"));
    assertTrue(load().schema().findField("region").isRequired());
  }

  @Test
  public void testMostCommonSchemaWinsAConflictBetweenFiles() {
    Schema asString =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "code", Types.StringType.get()));
    Schema asLong =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "code", Types.LongType.get()));
    commit(ALL, IncompatibleSchemaHandling.ROUTE_TO_ERRORS, files(asLong, 7), files(asString, 2));
    assertEquals(Types.LongType.get(), load().schema().findField("code").type());
  }

  @Test
  public void testConflictBetweenFilesFailsPipelineWithoutCommit() {
    String before = metadataLocation(load());
    Schema asString =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "code", Types.StringType.get()));
    Schema asLong =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "code", Types.LongType.get()));
    IncompatibleSchemaException e =
        assertThrows(
            IncompatibleSchemaException.class,
            () ->
                commit(
                    ALL,
                    IncompatibleSchemaHandling.FAIL_PIPELINE,
                    files(asLong, 7),
                    files(asString, 2)));
    assertTrue(e.getMessage(), e.getMessage().contains("conflicts with another file schema"));
    assertEquals(before, metadataLocation(load()));
  }

  // ---- name mapping

  @Test
  public void testStaleNameMappingIsRegeneratedForTheNewSchema() {
    Table table = load();
    table
        .updateProperties()
        .set(
            TableProperties.DEFAULT_NAME_MAPPING, NameMappingUtils.regenerate(table.schema(), null))
        .commit();
    Schema file =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "email", Types.StringType.get()));
    commit(ALL, IncompatibleSchemaHandling.FAIL_PIPELINE, files(file, 1));
    table = load();
    NameMapping mapping =
        NameMappingUtils.parseOrNull(table.properties().get(TableProperties.DEFAULT_NAME_MAPPING));
    assertNotNull(mapping);
    assertTrue(NameMappingUtils.covers(mapping, table.schema().asStruct()));
    assertNotNull(mapping.find("email"));
  }

  @Test
  public void testMissingNameMappingIsAddedEvenWithoutSchemaChanges() {
    assertFalse(load().properties().containsKey(TableProperties.DEFAULT_NAME_MAPPING));
    commit(ALL, IncompatibleSchemaHandling.FAIL_PIPELINE, files(TABLE, 1));
    Table table = load();
    NameMapping mapping =
        NameMappingUtils.parseOrNull(table.properties().get(TableProperties.DEFAULT_NAME_MAPPING));
    assertNotNull(mapping);
    assertTrue(NameMappingUtils.covers(mapping, table.schema().asStruct()));
  }

  // ---- aliases

  private static SchemaEvolutionConfig withAlias(String from, String to) {
    return SchemaEvolutionConfig.builder()
        .setOptions(Arrays.asList(SchemaEvolutionOption.values()))
        .withColumnAliases(java.util.Collections.singletonMap(from, to))
        .build();
  }

  @Test
  public void testAliasIsAddedToTheMappingEvenWithoutSchemaChanges() {
    seedNameMapping();
    int before = metadataVersion(load());
    commit(withAlias("nm", "name"), IncompatibleSchemaHandling.FAIL_PIPELINE, files(TABLE, 1));
    Table table = load();
    assertEquals("one commit for the mapping", before + 1, metadataVersion(table));
    NameMapping mapping =
        NameMappingUtils.parseOrNull(table.properties().get(TableProperties.DEFAULT_NAME_MAPPING));
    assertNotNull(mapping);
    assertEquals(mapping.find("name").id(), mapping.find("nm").id());
    // and nothing more to commit next time
    int after = metadataVersion(table);
    commit(withAlias("nm", "name"), IncompatibleSchemaHandling.FAIL_PIPELINE, files(TABLE, 1));
    assertEquals(after, metadataVersion(load()));
  }

  @Test
  public void testAliasAppliedTogetherWithASchemaChange() {
    Schema file =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "email", Types.StringType.get()));
    commit(withAlias("nm", "name"), IncompatibleSchemaHandling.FAIL_PIPELINE, files(file, 1));
    Table table = load();
    NameMapping mapping =
        NameMappingUtils.parseOrNull(table.properties().get(TableProperties.DEFAULT_NAME_MAPPING));
    assertNotNull(mapping);
    assertNotNull(mapping.find("email"));
    assertEquals(mapping.find("name").id(), mapping.find("nm").id());
  }

  @Test
  public void testAliasForAColumnAddedInTheSameCommit() {
    Schema file =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "email", Types.StringType.get()));
    commit(withAlias("eml", "email"), IncompatibleSchemaHandling.FAIL_PIPELINE, files(file, 1));
    NameMapping mapping =
        NameMappingUtils.parseOrNull(load().properties().get(TableProperties.DEFAULT_NAME_MAPPING));
    assertNotNull(mapping);
    assertEquals(mapping.find("email").id(), mapping.find("eml").id());
  }

  // ---- retry

  @Test
  public void testCommitFailedOnceIsRetriedAgainstFreshState() {
    AtomicInteger attempts = new AtomicInteger();
    Committer flakyThenExternalChange =
        txn -> {
          if (attempts.incrementAndGet() == 1) {
            // someone else adds a column between our load and commit
            load().updateSchema().addColumn("external", Types.StringType.get()).commit();
            throw new CommitFailedException("simulated concurrent commit");
          }
          txn.commitTransaction();
        };
    Schema file =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "email", Types.StringType.get()));
    CommitSchemaUnion.commit(
        catalog,
        tableId,
        Arrays.asList(files(file, 1)),
        ALL,
        IncompatibleSchemaHandling.FAIL_PIPELINE,
        NO_CREATION,
        flakyThenExternalChange);
    Table table = load();
    assertEquals(2, attempts.get());
    assertNotNull(table.schema().findField("external"));
    assertNotNull(table.schema().findField("email"));
  }

  @Test
  public void testPersistentCommitFailurePropagates() {
    AtomicInteger attempts = new AtomicInteger();
    Committer alwaysFails =
        txn -> {
          attempts.incrementAndGet();
          throw new CommitFailedException("always");
        };
    Schema file =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "email", Types.StringType.get()));
    assertThrows(
        CommitFailedException.class,
        () ->
            CommitSchemaUnion.commit(
                catalog,
                tableId,
                Arrays.asList(files(file, 1)),
                ALL,
                IncompatibleSchemaHandling.FAIL_PIPELINE,
                NO_CREATION,
                alwaysFails));
    assertEquals(CommitSchemaUnion.MAX_ATTEMPTS, attempts.get());
  }

  // ---- create path

  private TableIdentifier missing() {
    return TableIdentifier.of("default", testName.getMethodName() + "_new");
  }

  private long commitTo(
      TableIdentifier id,
      SchemaEvolutionConfig config,
      IncompatibleSchemaHandling handling,
      CommitSchemaUnion.TableCreation creation,
      CollectDistinctSchemas.SchemaGroup... schemas) {
    return CommitSchemaUnion.commit(
        catalog,
        id,
        Arrays.asList(schemas),
        config,
        handling,
        creation,
        CommitSchemaUnion.DEFAULT_COMMITTER);
  }

  @Test
  public void testMissingTableIsCreatedFromTheUnion() {
    TableIdentifier id = missing();
    Schema seed =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "email", Types.StringType.get()));
    Schema other =
        new Schema(
            required(1, "id", Types.LongType.get()), optional(2, "extra", Types.LongType.get()));
    CommitSchemaUnion.TableCreation creation =
        new CommitSchemaUnion.TableCreation(
            Arrays.asList("region"), null, java.util.Collections.singletonMap("k", "v"));
    long schemaId =
        commitTo(
            id,
            ALL,
            IncompatibleSchemaHandling.FAIL_PIPELINE,
            creation,
            files(seed, 5),
            files(other, 1));
    Table table = catalog.loadTable(id);
    assertEquals(table.schema().schemaId(), schemaId);
    assertNotNull(table.schema().findField("email"));
    assertNotNull(table.schema().findField("extra"));
    assertTrue("seed columns become optional", table.schema().findField("id").isOptional());
    assertTrue(table.schema().findField("region").isOptional());
    assertEquals("region", table.spec().fields().get(0).name());
    assertEquals("v", table.properties().get("k"));
    assertNotNull(table.properties().get(TableProperties.DEFAULT_NAME_MAPPING));
  }

  @Test
  public void testPinnedColumnsStayRequiredOnCreate() {
    TableIdentifier id = missing();
    SchemaEvolutionConfig pinned =
        SchemaEvolutionConfig.builder()
            .setOptions(Arrays.asList(SchemaEvolutionOption.values()))
            .setRequiredColumns(Arrays.asList("id"))
            .build();
    Schema seed =
        new Schema(
            required(1, "id", Types.LongType.get()), required(2, "region", Types.StringType.get()));
    commitTo(id, pinned, IncompatibleSchemaHandling.FAIL_PIPELINE, NO_CREATION, files(seed, 1));
    Table table = catalog.loadTable(id);
    assertTrue(table.schema().findField("id").isRequired());
    assertTrue(table.schema().findField("region").isOptional());
  }

  @Test
  public void testMissingTableWithoutSchemasIsNotCreated() {
    TableIdentifier id = missing();
    long result =
        CommitSchemaUnion.commit(
            catalog,
            id,
            new ArrayList<>(),
            ALL,
            IncompatibleSchemaHandling.FAIL_PIPELINE,
            NO_CREATION,
            CommitSchemaUnion.DEFAULT_COMMITTER);
    assertEquals(CommitSchemaUnion.NO_TABLE, result);
    assertFalse(catalog.tableExists(id));
  }

  @Test
  public void testConflictOnCreateFailsWithoutCreating() {
    TableIdentifier id = missing();
    Schema asString =
        new Schema(
            required(1, "id", Types.LongType.get()), optional(2, "code", Types.StringType.get()));
    Schema asLong =
        new Schema(
            required(1, "id", Types.LongType.get()), optional(2, "code", Types.LongType.get()));
    assertThrows(
        IncompatibleSchemaException.class,
        () ->
            commitTo(
                id,
                ALL,
                IncompatibleSchemaHandling.FAIL_PIPELINE,
                NO_CREATION,
                files(asLong, 3),
                files(asString, 1)));
    assertFalse(catalog.tableExists(id));
  }

  @Test
  public void testConflictOnCreateRoutesLoserAndCreates() {
    TableIdentifier id = missing();
    Schema asString =
        new Schema(
            required(1, "id", Types.LongType.get()), optional(2, "code", Types.StringType.get()));
    Schema asLong =
        new Schema(
            required(1, "id", Types.LongType.get()), optional(2, "code", Types.LongType.get()));
    commitTo(
        id,
        ALL,
        IncompatibleSchemaHandling.ROUTE_TO_ERRORS,
        NO_CREATION,
        files(asLong, 3),
        files(asString, 1));
    assertEquals(Types.LongType.get(), catalog.loadTable(id).schema().findField("code").type());
  }

  @Test
  public void testCreateRaceFallsBackToEvolvingTheExistingTable() {
    TableIdentifier id = missing();
    AtomicInteger attempts = new AtomicInteger();
    Committer raced =
        txn -> {
          if (attempts.incrementAndGet() == 1) {
            // someone else creates the table first
            warehouse.createTable(id, TABLE);
            throw new org.apache.iceberg.exceptions.AlreadyExistsException("raced");
          }
          txn.commitTransaction();
        };
    Schema file =
        new Schema(
            required(1, "id", Types.LongType.get()),
            required(2, "region", Types.StringType.get()),
            optional(3, "email", Types.StringType.get()));
    CommitSchemaUnion.commit(
        catalog,
        id,
        Arrays.asList(files(file, 1)),
        ALL,
        IncompatibleSchemaHandling.FAIL_PIPELINE,
        NO_CREATION,
        raced);
    Table table = catalog.loadTable(id);
    assertEquals(2, attempts.get());
    assertNotNull(table.schema().findField("email"));
    assertTrue(
        "evolved, not recreated: name from TABLE is still there",
        table.schema().findField("name") != null);
  }

  @Test
  public void testEmptyInputCommitsNothing() {
    seedNameMapping();
    String before = metadataLocation(load());
    List<CollectDistinctSchemas.SchemaGroup> none = new ArrayList<>();
    CommitSchemaUnion.commit(
        catalog,
        tableId,
        none,
        ALL,
        IncompatibleSchemaHandling.FAIL_PIPELINE,
        NO_CREATION,
        CommitSchemaUnion.DEFAULT_COMMITTER);
    assertEquals(before, metadataLocation(load()));
  }
}
