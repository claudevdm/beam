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

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.PipelineResult;
import org.apache.beam.sdk.io.iceberg.AddFilesFuzz.Scenario;
import org.apache.beam.sdk.io.iceberg.AddFilesFuzzSignals.Before;
import org.apache.beam.sdk.io.iceberg.AddFilesFuzzSignals.Outcome;
import org.apache.beam.sdk.metrics.MetricQueryResults;
import org.apache.beam.sdk.metrics.MetricResult;
import org.apache.beam.sdk.metrics.MetricsFilter;
import org.apache.beam.sdk.options.PipelineOptionsFactory;
import org.apache.beam.sdk.transforms.Create;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.values.PCollectionRowTuple;
import org.apache.beam.sdk.values.Row;
import org.apache.beam.vendor.guava.v32_1_2_jre.com.google.common.collect.ImmutableMap;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Seeded end-to-end fuzzing of AddFiles on the Direct runner: every seed builds a table state, a
 * random set of Parquet files (all option subsets, pins, aliases, ignored columns, adversarial
 * bytes) and a mid-run disturbance, runs AddFiles in batch, shuffled batch and streaming, and
 * checks the invariants in {@link AddFilesFuzzExpectations}. See
 * docs/icebergio-kb/AddFilesFuzzPlan.md for the invariant list and the findings so far.
 *
 * <p><b>How to run.</b> Through the Gradle task, which forwards {@code -D} properties to the test
 * JVM (the plain {@code test} task does not):
 *
 * <pre>
 * ./gradlew :it:iceberg:AddFilesFuzz -Dseeds=100          # seeds 1..100
 * ./gradlew :it:iceberg:AddFilesFuzz -Dseeds=50 -DseedStart=101
 * ./gradlew :it:iceberg:AddFilesFuzz -Dseed=37             # replay one seed
 * </pre>
 *
 * <p><b>Requirements.</b> None beyond a local JVM: tables live in a HadoopCatalog under a temporary
 * folder and every pipeline runs on the Direct runner. About one second per seed. Properties:
 * {@code seeds} (count, default 20), {@code seedStart} (default 1), {@code seed} (replay one). A
 * failure names the seed, the mode and the invariant; replay it with {@code -Dseed}.
 */
@RunWith(JUnit4.class)
public class AddFilesFuzzTest {
  private static final Logger LOG = LoggerFactory.getLogger(AddFilesFuzzTest.class);

  @Rule public final TemporaryFolder tmp = new TemporaryFolder();

  private static final Map<String, List<Row>> ERRORS = new ConcurrentHashMap<>();

  /** Applies the scenario's disturbance around the schema commit. */
  static class DisturbingCommitter implements CommitSchemaUnion.Committer {
    private final AddFilesFuzz.Disturbance disturbance;
    private final String warehouse;
    private final String table;
    private static final java.util.Set<String> FAILED_ONCE = ConcurrentHashMap.newKeySet();
    private static final java.util.Set<String> FIRED = ConcurrentHashMap.newKeySet();

    /** Whether the disturbance for this table ran at all (the pre-pass commits only if needed). */
    static boolean fired(String warehouse, String table) {
      return FIRED.contains(warehouse + table);
    }

    DisturbingCommitter(AddFilesFuzz.Disturbance disturbance, String warehouse, String table) {
      this.disturbance = disturbance;
      this.warehouse = warehouse;
      this.table = table;
    }

    private org.apache.iceberg.Table load() {
      return new org.apache.iceberg.hadoop.HadoopCatalog(
              new org.apache.hadoop.conf.Configuration(), warehouse)
          .loadTable(org.apache.iceberg.catalog.TableIdentifier.parse(table));
    }

    @Override
    public void commit(org.apache.iceberg.Transaction txn) {
      if (disturbance != AddFilesFuzz.Disturbance.NONE) {
        FIRED.add(warehouse + table);
      }
      switch (disturbance) {
        case COMMIT_FAILS_ONCE:
          if (FAILED_ONCE.add(warehouse + table)) {
            load()
                .updateSchema()
                .addColumn(
                    AddFilesFuzz.EXTERNAL_COLUMN, org.apache.iceberg.types.Types.StringType.get())
                .commit();
            throw new org.apache.iceberg.exceptions.CommitFailedException(
                "simulated concurrent commit");
          }
          txn.commitTransaction();
          return;
        case EXTERNAL_COLUMN_AFTER_COMMIT:
          txn.commitTransaction();
          load()
              .updateSchema()
              .addColumn(
                  AddFilesFuzz.EXTERNAL_COLUMN, org.apache.iceberg.types.Types.StringType.get())
              .commit();
          return;
        case MAPPING_OVERWRITTEN_AFTER_COMMIT:
          txn.commitTransaction();
          load()
              .updateProperties()
              .set(org.apache.iceberg.TableProperties.DEFAULT_NAME_MAPPING, "{garbage")
              .commit();
          return;
        case TABLE_DROPPED_AFTER_COMMIT:
          txn.commitTransaction();
          new org.apache.iceberg.hadoop.HadoopCatalog(
                  new org.apache.hadoop.conf.Configuration(), warehouse)
              .dropTable(org.apache.iceberg.catalog.TableIdentifier.parse(table));
          return;
        default:
          txn.commitTransaction();
      }
    }
  }

  /** Collects error rows in-process; the Direct runner runs in this JVM. */
  static class CollectErrors extends DoFn<Row, Void> {
    private final String key;

    CollectErrors(String key) {
      this.key = key;
    }

    @ProcessElement
    public void process(@Element Row row) {
      ERRORS.computeIfAbsent(key, k -> new CopyOnWriteArrayList<>()).add(row);
    }
  }

  @Test
  public void fuzz() throws Exception {
    // Gradle's test task does not forward -D properties; environment variables pass through.
    String single = setting("seed", "FUZZ_SEED", null);
    long start = Long.parseLong(setting("seedStart", "FUZZ_SEED_START", "1"));
    int count = Integer.parseInt(setting("seeds", "FUZZ_SEEDS", "20"));
    List<Long> seeds = new ArrayList<>();
    if (single != null) {
      seeds.add(Long.parseLong(single));
    } else {
      for (long s = start; s < start + count; s++) {
        seeds.add(s);
      }
    }
    List<String> failures = new ArrayList<>();
    int failed = 0;
    for (long seed : seeds) {
      try {
        runScenario(seed);
      } catch (Throwable t) {
        failed++;
        Throwable root = t;
        while (root.getCause() != null) {
          root = root.getCause();
        }
        StringBuilder frames = new StringBuilder();
        int shown = 0;
        for (StackTraceElement frame : root.getStackTrace()) {
          if (frame.getClassName().contains("iceberg")
              || frame.getClassName().contains("parquet")) {
            frames.append("    at ").append(frame).append('\n');
            if (++shown == 12) {
              break;
            }
          }
        }
        String report = "seed " + seed + ": " + t + "\n" + frames + describe(seed);
        LOG.error("{}", report, t);
        failures.add(report);
        if (failures.size() >= 5) {
          break;
        }
      }
    }
    LOG.info("fuzz: {} seeds, {} failed", seeds.size(), failed);
    if (!failures.isEmpty()) {
      throw new AssertionError(failed + " scenario(s) failed:\n" + String.join("\n", failures));
    }
  }

  private static @Nullable String setting(String property, String env, @Nullable String fallback) {
    String value = System.getProperty(property);
    if (value == null) {
      value = System.getenv(env);
    }
    return value != null ? value : fallback;
  }

  private String describe(long seed) {
    try {
      return AddFilesFuzz.generate(seed, tmp.newFolder("describe-" + seed)).toString();
    } catch (Exception e) {
      return "(could not regenerate: " + e + ")";
    }
  }

  /** What a run leaves behind, compared across orderings and modes. */
  static final class RunSummary {
    final @Nullable String schemaJson;
    final java.util.Set<String> registered = new java.util.TreeSet<>();
    final java.util.Set<String> errors = new java.util.TreeSet<>();
    final boolean failed;

    RunSummary(Scenario scenario, @Nullable Exception failure, List<Row> errorRows) {
      failed = failure != null;
      org.apache.iceberg.hadoop.HadoopCatalog catalog = scenario.catalog();
      if (catalog.tableExists(scenario.tableId)) {
        org.apache.iceberg.Table table = catalog.loadTable(scenario.tableId);
        // order-insensitive: the create path lays columns out alphabetically, the evolve path
        // appends in file order, so the same set of columns can come out in a different order
        schemaJson = org.apache.iceberg.SchemaParser.toJson(FileSchemas.canonical(table.schema()));
        for (org.apache.iceberg.Snapshot snapshot : table.snapshots()) {
          for (org.apache.iceberg.DataFile file : snapshot.addedDataFiles(table.io())) {
            registered.add(new File(file.path().toString()).getName());
          }
        }
      } else {
        schemaJson = null;
      }
      for (Row row : errorRows) {
        errors.add(new File(String.valueOf(row.getString("file"))).getName());
      }
    }

    void assertSameAs(RunSummary other, String what) {
      if (!java.util.Objects.equals(schemaJson, other.schemaJson)) {
        throw new AssertionError(
            what + ": final schema differs\n" + schemaJson + "\n" + other.schemaJson);
      }
      if (!registered.equals(other.registered)
          || !errors.equals(other.errors)
          || failed != other.failed) {
        throw new AssertionError(
            what
                + ": outcome differs: registered "
                + registered
                + " vs "
                + other.registered
                + ", errors "
                + errors
                + " vs "
                + other.errors
                + ", failed "
                + failed
                + " vs "
                + other.failed);
      }
    }
  }

  private void runScenario(long seed) throws Exception {
    RunSummary batch = runOnce(seed, "s", false, false, true);
    if (AddFilesFuzz.generate(seed, tmp.newFolder("probe-a-" + seed)).disturbance
        == AddFilesFuzz.Disturbance.TABLE_DROPPED_AFTER_COMMIT) {
      return;
    }

    // determinism: same scenario, shuffled input order
    RunSummary shuffled = runOnce(seed, "shuffled", false, true, false);
    if (AddFilesFuzz.generate(seed, tmp.newFolder("probe-b-" + seed)).disturbance
        != AddFilesFuzz.Disturbance.TABLE_DROPPED_AFTER_COMMIT) {
      batch.assertSameAs(shuffled, "shuffled input order");
    }

    // streaming parity: same scenario through TestStream; the incompatible default differs
    Scenario probe = AddFilesFuzz.generate(seed, tmp.newFolder("probe-c-" + seed));
    boolean handlingSet = probe.config.getIncompatibleSchemaHandling() != null;
    boolean parityMeaningful =
        probe.disturbance != AddFilesFuzz.Disturbance.TABLE_DROPPED_AFTER_COMMIT;
    if (parityMeaningful && (handlingSet || !batch.failed)) {
      RunSummary streaming = runOnce(seed, "streaming", true, false, false);
      if (handlingSet || !streaming.failed) {
        batch.assertSameAs(streaming, "streaming vs batch");
      }
    }
  }

  private RunSummary runOnce(
      long seed, String label, boolean streaming, boolean shuffle, boolean verify)
      throws Exception {
    File dir = tmp.newFolder(label + "-" + seed);
    Scenario scenario = AddFilesFuzz.generate(seed, dir);
    if (verify) {
      LOG.info("running {}", scenario);
    }
    Before before = new Before(scenario);

    IcebergCatalogConfig catalogConfig =
        IcebergCatalogConfig.builder()
            .setCatalogProperties(
                ImmutableMap.of("type", "hadoop", "warehouse", scenario.warehouse))
            .build();
    String key = label + "-" + seed;
    ERRORS.remove(key);
    Pipeline pipeline = Pipeline.create(PipelineOptionsFactory.create());
    List<String> paths = new ArrayList<>(scenario.paths());
    if (shuffle) {
      java.util.Collections.shuffle(paths, new java.util.Random(seed));
    }
    org.apache.beam.sdk.values.PCollection<String> input;
    if (streaming) {
      org.apache.beam.sdk.testing.TestStream.Builder<String> builder =
          org.apache.beam.sdk.testing.TestStream.create(
              org.apache.beam.sdk.coders.StringUtf8Coder.of());
      for (String path : paths) {
        builder =
            builder
                .addElements(
                    org.apache.beam.sdk.values.TimestampedValue.of(
                        path, new org.joda.time.Instant(0)))
                // keep every element in one pre-pass window: parity with batch only holds then
                .advanceProcessingTime(org.joda.time.Duration.millis(100));
      }
      input = pipeline.apply(builder.advanceWatermarkToInfinity());
    } else {
      input = pipeline.apply(Create.of(paths));
    }
    AddFiles addFiles =
        new AddFiles(
            catalogConfig,
            scenario.tableId.toString(),
            null,
            null,
            null,
            null,
            streaming ? 10 : null,
            streaming ? org.joda.time.Duration.standardSeconds(5) : null,
            scenario.config);
    addFiles.withSchemaCommitter(
        new DisturbingCommitter(
            scenario.disturbance, scenario.warehouse, scenario.tableId.toString()));
    PCollectionRowTuple output = input.apply(addFiles);
    output.get("errors").apply(ParDo.of(new CollectErrors(key)));

    Exception failure = null;
    Map<String, Long> counters = new HashMap<>();
    try {
      PipelineResult result = pipeline.run();
      result.waitUntilFinish();
      counters = counters(result);
    } catch (Exception e) {
      failure = e;
    }
    List<Row> errors = new ArrayList<>(ERRORS.getOrDefault(key, new ArrayList<>()));
    if (verify) {
      AddFilesFuzzExpectations.Result expected =
          AddFilesFuzzExpectations.compute(
              scenario, before.schema, tmp.newFolder("scratch-" + seed));
      AddFilesFuzzSignals.verify(
          scenario, before, new Outcome(failure, errors, counters), expected);
    }
    return new RunSummary(scenario, failure, errors);
  }

  private static Map<String, Long> counters(PipelineResult result) {
    Map<String, Long> counters = new HashMap<>();
    MetricQueryResults metrics = result.metrics().queryMetrics(MetricsFilter.builder().build());
    for (MetricResult<Long> counter : metrics.getCounters()) {
      counters.merge(counter.getName().getName(), counter.getAttempted(), Long::sum);
    }
    return counters;
  }
}
