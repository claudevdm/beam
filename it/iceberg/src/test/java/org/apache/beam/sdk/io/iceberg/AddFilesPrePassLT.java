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

import static org.apache.beam.it.truthmatchers.PipelineAsserts.assertThatResult;
import static org.apache.beam.sdk.util.Preconditions.checkStateNotNull;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.File;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.apache.beam.it.common.PipelineLauncher;
import org.apache.beam.it.common.PipelineOperator;
import org.apache.beam.it.common.TestProperties;
import org.apache.beam.it.common.dataflow.DefaultPipelineLauncher.PipelineMetricsType;
import org.apache.beam.it.common.dataflow.IOLoadTestBase;
import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.PipelineResult;
import org.apache.beam.sdk.coders.StringUtf8Coder;
import org.apache.beam.sdk.io.FileIO;
import org.apache.beam.sdk.io.FileSystems;
import org.apache.beam.sdk.io.TextIO;
import org.apache.beam.sdk.io.fs.MatchResult;
import org.apache.beam.sdk.io.iceberg.PrePassCorpus.CorpusIndex;
import org.apache.beam.sdk.io.iceberg.PrePassCorpus.FileSpec;
import org.apache.beam.sdk.io.iceberg.PrePassCorpus.Outcome;
import org.apache.beam.sdk.metrics.MetricNameFilter;
import org.apache.beam.sdk.metrics.MetricQueryResults;
import org.apache.beam.sdk.metrics.MetricResult;
import org.apache.beam.sdk.metrics.MetricsFilter;
import org.apache.beam.sdk.testing.TestPipeline;
import org.apache.beam.sdk.transforms.Combine;
import org.apache.beam.sdk.transforms.Create;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.MapElements;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.transforms.Reshuffle;
import org.apache.beam.sdk.values.KV;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.sdk.values.TypeDescriptors;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Load and coverage test for the AddFiles schema pre-pass: the read side ({@link ReadFooterSchema}
 * followed by {@link CollectDistinctSchemas}) or, for the {@code -addfiles} configs, the full
 * {@link AddFiles} with every schema evolution option into a BigLake table.
 *
 * <p><b>How to run.</b> Through the Gradle task, which forwards {@code -D} properties to the test
 * JVM (the plain {@code test} task does not):
 *
 * <pre>
 * ./gradlew :it:iceberg:AddFilesPrePassPerformanceTest \
 *     -DtestConfig=dataflow-10k \
 *     -Dproject=PROJECT_ID -Dregion=REGION -DartifactBucket=BUCKET_NAME
 * </pre>
 *
 * <p><b>Requirements.</b>
 *
 * <ul>
 *   <li>{@code direct-1k} and {@code coverage}: nothing beyond a local JVM; the corpus is written
 *       under {@code java.io.tmpdir}.
 *   <li>Dataflow configs: a GCP project with the Dataflow API enabled, a region, and a GCS bucket
 *       ({@code artifactBucket}) that holds the corpus under {@code prepass-bench/} and the staged
 *       pipeline. Application default credentials with permission to run Dataflow jobs and to read
 *       and write the bucket.
 *   <li>{@code -addfiles} configs: additionally a BigLake Iceberg REST catalog in end-user mode
 *       (the caller's own IAM on the warehouse bucket; a vended-credentials catalog rejects this
 *       test's requests) reachable at {@code
 *       https://biglake.googleapis.com/iceberg/v1/restcatalog}. The warehouse defaults to {@code
 *       gs://BUCKET_NAME} and can be set with {@code -DbiglakeWarehouse=gs://WAREHOUSE_BUCKET}.
 *       Tables are created under a {@code prepass_lt_*} namespace and are not deleted. The corpus
 *       must be on GCS for these configs: the table's GCSFileIO cannot read local files.
 *   <li>Machine type and worker counts are never set: the numbers reflect Dataflow defaults and
 *       autoscaling. Do not add them.
 * </ul>
 *
 * <p>Properties: {@code testConfig} (direct-1k, coverage, dataflow-10k, dataflow-100k, dataflow-1m
 * and the -addfiles variants), {@code numFiles}, {@code corpusDir}, {@code seed}, {@code
 * threadPoolSize}, {@code maxInFlightTasks}, {@code globInput} (default true except for coverage),
 * {@code generateOnDataflow} (default true above {@link #DATAFLOW_GENERATION_THRESHOLD} files),
 * {@code goldenDir} with {@code recordGolden} (opt-in drift check across library upgrades: record
 * before, compare after; nothing is tracked).
 *
 * <p>A corpus is generated once per (config, seed, numFiles) and reused by path. The JVM writes
 * about 37 files/s to GCS (16 threads), fine up to 100k (about 45 min once); above that a one-off
 * Dataflow pipeline writes the files and one index shard per range. Approximate size: 10 KB per
 * file, so 1M files is 10 GB.
 */
@RunWith(JUnit4.class)
public final class AddFilesPrePassLT extends IOLoadTestBase {
  private static final Logger LOG = LoggerFactory.getLogger(AddFilesPrePassLT.class);

  private static final class Config {
    final String name;
    final String runner;
    final int numFiles;
    final boolean coverage;
    final int timeoutMinutes;
    /**
     * Run the full AddFiles with schema evolution into a BigLake table instead of the read side.
     */
    final boolean addFiles;

    Config(String name, String runner, int numFiles, boolean coverage, int timeoutMinutes) {
      this(name, runner, numFiles, coverage, timeoutMinutes, false);
    }

    Config(
        String name,
        String runner,
        int numFiles,
        boolean coverage,
        int timeoutMinutes,
        boolean addFiles) {
      this.name = name;
      this.runner = runner;
      this.numFiles = numFiles;
      this.coverage = coverage;
      this.timeoutMinutes = timeoutMinutes;
      this.addFiles = addFiles;
    }
  }

  private static final Map<String, Config> CONFIGS = new TreeMap<>();

  static {
    CONFIGS.put("direct-1k", new Config("direct-1k", "DirectRunner", 1_000, false, 20));
    CONFIGS.put("coverage", new Config("coverage", "DirectRunner", 0, true, 20));
    CONFIGS.put("dataflow-10k", new Config("dataflow-10k", "DataflowRunner", 10_000, false, 60));
    CONFIGS.put(
        "dataflow-addfiles-10k",
        new Config("dataflow-addfiles-10k", "DataflowRunner", 10_000, false, 60, true));
    CONFIGS.put(
        "direct-addfiles-1k",
        new Config("direct-addfiles-1k", "DirectRunner", 1_000, false, 20, true));
    CONFIGS.put(
        "dataflow-100k", new Config("dataflow-100k", "DataflowRunner", 100_000, false, 120));
    CONFIGS.put(
        "dataflow-addfiles-100k",
        new Config("dataflow-addfiles-100k", "DataflowRunner", 100_000, false, 120, true));
    CONFIGS.put("dataflow-1m", new Config("dataflow-1m", "DataflowRunner", 1_000_000, false, 360));
    CONFIGS.put(
        "dataflow-addfiles-1m",
        new Config("dataflow-addfiles-1m", "DataflowRunner", 1_000_000, false, 360, true));
  }

  /** Above this many files the corpus is generated by a Dataflow pipeline, not the test JVM. */
  static final int DATAFLOW_GENERATION_THRESHOLD = 50_000;

  /** Files per generator element; also the index shard size. */
  private static final int GENERATION_RANGE = 1_000;

  private static final double APPROX_BYTES_PER_FILE = 10_000;

  @Rule public final TestPipeline pipeline = TestPipeline.create();

  private Config config;
  private long seed;
  private String corpusDir;
  private String resultDir;
  private CorpusIndex index;
  private double wallSeconds;
  /** Feed paths from a file pattern like users do; the index feeds paths that may not exist. */
  private boolean globInput;

  private @Nullable PipelineResult directResult;

  @Before
  public void setUpPrePass() throws Exception {
    String name = property("testConfig", "direct-1k");
    Config selected = CONFIGS.get(name);
    if (selected == null) {
      throw new IllegalArgumentException(
          "Unknown testConfig " + name + "; known: " + CONFIGS.keySet());
    }
    int numFiles = Integer.parseInt(property("numFiles", String.valueOf(selected.numFiles)));
    config =
        new Config(
            selected.name,
            selected.runner,
            numFiles,
            selected.coverage,
            selected.timeoutMinutes,
            selected.addFiles);
    seed = Long.parseLong(property("seed", "42"));

    String defaultCorpusDir;
    // a BigLake table's GCSFileIO cannot read local files: AddFiles configs always use GCS
    if (config.runner.equals("DirectRunner") && !config.addFiles) {
      defaultCorpusDir =
          new File(System.getProperty("java.io.tmpdir"), "prepass-bench").getAbsolutePath();
    } else {
      defaultCorpusDir = "gs://" + TestProperties.artifactBucket() + "/prepass-bench";
    }
    String root = property("corpusDir", defaultCorpusDir);
    String corpusName = config.name.replace("-addfiles", "");
    corpusDir = root + "/" + corpusName + "/" + seed + "-" + config.numFiles;
    resultDir = corpusDir + "/results/" + testName;
    index = loadOrGenerateCorpus();
    globInput = Boolean.parseBoolean(property("globInput", String.valueOf(!config.coverage)));
    if (globInput) {
      index = existingOnly(index);
    }
  }

  @Test
  public void testPrePass() throws Exception {
    if (config.addFiles) {
      testAddFiles();
      return;
    }
    buildPipeline();
    if (config.runner.equals("DirectRunner")) {
      // The it launcher cannot read counters outside its own namespace from a Direct run, so run
      // the pipeline here and read ReadFooterSchema's counters from the result.
      long started = System.nanoTime();
      directResult = pipeline.run();
      directResult.waitUntilFinish();
      wallSeconds = (System.nanoTime() - started) / 1e9;
      assertEquals(PipelineResult.State.DONE, directResult.getState());
      verifyCounts(null);
      LOG.info("config={} files={} wallSeconds={}", config.name, index.entries.size(), wallSeconds);
      return;
    }
    // Launch to done, including the operator's 15 second poll granularity: an upper bound.
    long started = System.nanoTime();
    PipelineLauncher.LaunchInfo info = launch();
    PipelineOperator.Result result =
        pipelineOperator.waitUntilDone(
            createConfig(info, Duration.ofMinutes(config.timeoutMinutes)));
    wallSeconds = (System.nanoTime() - started) / 1e9;
    assertThatResult(result).isLaunchFinished();
    assertEquals(
        PipelineLauncher.JobState.DONE,
        pipelineLauncher.getJobStatus(project, region, info.jobId()));

    verifyCounts(info);
    report(info);
  }

  // ---- full AddFiles into a BigLake table (step 5 of the fuzz plan, cost of the whole path)

  private Map<String, String> biglakeProps() {
    String warehouse = property("biglakeWarehouse", "gs://" + TestProperties.artifactBucket());
    Map<String, String> props = new TreeMap<>();
    props.put("type", "rest");
    props.put("uri", "https://biglake.googleapis.com/iceberg/v1/restcatalog");
    props.put("warehouse", warehouse);
    props.put("header.x-goog-user-project", project);
    props.put("rest.auth.type", "google");
    props.put("io-impl", "org.apache.iceberg.gcp.gcs.GCSFileIO");
    return props;
  }

  private void testAddFiles() throws Exception {
    String namespace = "prepass_lt_" + testName.replaceAll("[^A-Za-z0-9]", "_").toLowerCase();
    String tableName = "t" + System.currentTimeMillis();
    org.apache.iceberg.rest.RESTCatalog catalog = new org.apache.iceberg.rest.RESTCatalog();
    catalog.initialize("lt", biglakeProps());
    org.apache.iceberg.catalog.Namespace ns = org.apache.iceberg.catalog.Namespace.of(namespace);
    if (!catalog.namespaceExists(ns)) {
      catalog.createNamespace(ns);
    }
    org.apache.iceberg.catalog.TableIdentifier tableId =
        org.apache.iceberg.catalog.TableIdentifier.of(ns, tableName);

    SchemaEvolutionConfig evolution = SchemaEvolutionConfig.of(SchemaEvolutionOption.values());
    PCollection<String> paths =
        pipeline
            .apply("MatchFiles", FileIO.match().filepattern(corpusDir + "/files/*"))
            .apply(
                "ToPath",
                MapElements.into(TypeDescriptors.strings())
                    .via(metadata -> metadata.resourceId().toString()));
    org.apache.beam.sdk.values.PCollectionRowTuple out =
        paths.apply(
            new AddFiles(
                IcebergCatalogConfig.builder().setCatalogProperties(biglakeProps()).build(),
                namespace + "." + tableName,
                null,
                null,
                null,
                null,
                null,
                null,
                evolution));
    out.get("errors").apply("CountErrors", ParDo.of(new CountingFn<>("addfiles-errors")));

    long started = System.nanoTime();
    PipelineLauncher.LaunchInfo info = launch();
    PipelineOperator.Result result =
        pipelineOperator.waitUntilDone(
            createConfig(info, Duration.ofMinutes(config.timeoutMinutes)));
    wallSeconds = (System.nanoTime() - started) / 1e9;
    assertThatResult(result).isLaunchFinished();
    assertEquals(
        PipelineLauncher.JobState.DONE,
        pipelineLauncher.getJobStatus(project, region, info.jobId()));

    // verification against the corpus index
    int expectedRegistered = 0;
    int expectedErrors = 0;
    for (CorpusIndex.Entry entry : index.entries) {
      if (entry.expected == Outcome.SCHEMA && entry.writeFailure.isEmpty()) {
        expectedRegistered++;
      } else {
        expectedErrors++;
      }
    }
    org.apache.iceberg.Table table = catalog.loadTable(tableId);
    int registered = 0;
    java.util.Set<String> seen = new java.util.HashSet<>();
    for (org.apache.iceberg.Snapshot snapshot : table.snapshots()) {
      for (org.apache.iceberg.DataFile file : snapshot.addedDataFiles(table.io())) {
        assertTrue("registered twice: " + file.path(), seen.add(file.path().toString()));
        registered++;
      }
    }
    LOG.info(
        "registered={} expectedRegistered={} expectedErrors={} columns={}",
        registered,
        expectedRegistered,
        expectedErrors,
        table.schema().columns().size());
    assertEquals("registered files", expectedRegistered, registered);
    assertTrue("union has the 200-column variant", table.schema().findField("col_199") != null);
    assertTrue("union has the nested variant", table.schema().findField("address.city") != null);
    double errors = metric(info, "addfiles-errors");
    assertEquals("error rows", (double) expectedErrors, errors, 0.5);
    org.apache.iceberg.mapping.NameMapping mapping =
        NameMappingUtils.parseOrNull(
            table.properties().get(org.apache.iceberg.TableProperties.DEFAULT_NAME_MAPPING));
    assertTrue(
        "mapping covers the schema",
        mapping != null && NameMappingUtils.covers(mapping, table.schema().asStruct()));

    report(info);
    catalog.dropTable(tableId);
  }

  // ---- corpus

  private CorpusIndex loadOrGenerateCorpus() throws Exception {
    String indexGlob = corpusDir + "/corpus-index*.json";
    if (exists(indexGlob)) {
      LOG.info("Reusing corpus at {}", corpusDir);
      return PrePassCorpus.readIndexShards(indexGlob);
    }
    if (exists(corpusDir + "/files/*")) {
      throw new IllegalStateException(
          "Corpus at "
              + corpusDir
              + " has files but no index: a previous generation did not finish. Delete the"
              + " prefix before rerunning.");
    }
    boolean onDataflow =
        Boolean.parseBoolean(
            property(
                "generateOnDataflow",
                String.valueOf(
                    !config.coverage && config.numFiles > DATAFLOW_GENERATION_THRESHOLD)));
    if (onDataflow) {
      LOG.info(
          "Generating {} files (about {} MB) into {} with a Dataflow pipeline in {}",
          config.numFiles,
          Math.round(config.numFiles * APPROX_BYTES_PER_FILE / 1e6),
          corpusDir,
          region);
      generateCorpusOnDataflow();
      return PrePassCorpus.readIndexShards(indexGlob);
    }
    List<FileSpec> specs;
    if (config.coverage) {
      specs = PrePassCorpus.coverageSpecs(seed);
    } else {
      specs = PrePassCorpus.scaleSpecs(config.numFiles, seed);
    }
    LOG.info(
        "Generating {} files (about {} MB) into {} from this JVM",
        specs.size(),
        Math.round(specs.size() * APPROX_BYTES_PER_FILE / 1e6),
        corpusDir);
    CorpusIndex generated = new CorpusIndex();
    generated.config = config.name;
    generated.seed = seed;
    ExecutorService executor = Executors.newFixedThreadPool(16);
    try {
      List<Future<CorpusIndex.Entry>> futures = new ArrayList<>();
      for (FileSpec spec : specs) {
        futures.add(executor.submit(() -> PrePassCorpusWriters.write(spec, corpusDir + "/files")));
      }
      for (Future<CorpusIndex.Entry> future : futures) {
        generated.entries.add(future.get());
      }
    } finally {
      executor.shutdownNow();
    }
    PrePassCorpus.writeIndex(generated, corpusDir + "/corpus-index.json");
    return generated;
  }

  /**
   * One-off pipeline: ranges of spec indexes, reshuffled so every worker gets some, each written by
   * {@link GenerateCorpusFn} together with its own index shard. Default workers and autoscaling.
   */
  private void generateCorpusOnDataflow() throws IOException {
    List<KV<Integer, Integer>> ranges = new ArrayList<>();
    for (int from = 0; from < config.numFiles; from += GENERATION_RANGE) {
      ranges.add(KV.of(from, Math.min(from + GENERATION_RANGE, config.numFiles)));
    }
    Pipeline generator = Pipeline.create();
    generator
        .apply("Ranges", Create.of(ranges))
        .apply("Spread", Reshuffle.viaRandomKey())
        .apply(
            "WriteCorpus",
            ParDo.of(new GenerateCorpusFn(config.name, config.numFiles, seed, corpusDir)));
    PipelineLauncher.LaunchConfig.Builder options =
        PipelineLauncher.LaunchConfig.builder("addfiles-prepass-corpus-" + config.name)
            .setSdk(PipelineLauncher.Sdk.JAVA)
            .setPipeline(generator)
            .addParameter("runner", "DataflowRunner");
    long started = System.nanoTime();
    PipelineLauncher.LaunchInfo info = pipelineLauncher.launch(project, region, options.build());
    PipelineOperator.Result result =
        pipelineOperator.waitUntilDone(
            createConfig(info, Duration.ofMinutes(config.timeoutMinutes)));
    assertThatResult(result).isLaunchFinished();
    assertEquals(
        "corpus generation job",
        PipelineLauncher.JobState.DONE,
        pipelineLauncher.getJobStatus(project, region, info.jobId()));
    LOG.info(
        "Generated {} files in {} s (job {})",
        config.numFiles,
        Math.round((System.nanoTime() - started) / 1e9),
        info.jobId());
  }

  /**
   * Writes the files of one index range and its index shard. The spec list is deterministic in
   * (numFiles, seed) and cheap (specs share their column lists), so each worker builds it once.
   */
  static final class GenerateCorpusFn extends DoFn<KV<Integer, Integer>, Long> {
    private static final Map<String, List<FileSpec>> SPECS = new java.util.HashMap<>();

    private final String configName;
    private final int numFiles;
    private final long seed;
    private final String corpusDir;

    GenerateCorpusFn(String configName, int numFiles, long seed, String corpusDir) {
      this.configName = configName;
      this.numFiles = numFiles;
      this.seed = seed;
      this.corpusDir = corpusDir;
    }

    private List<FileSpec> specs() {
      String key = numFiles + "/" + seed;
      synchronized (SPECS) {
        List<FileSpec> cached = SPECS.get(key);
        if (cached == null) {
          cached = PrePassCorpus.scaleSpecs(numFiles, seed);
          SPECS.put(key, cached);
        }
        return cached;
      }
    }

    @ProcessElement
    public void process(@Element KV<Integer, Integer> range, OutputReceiver<Long> out)
        throws IOException {
      int from = range.getKey();
      int to = range.getValue();
      CorpusIndex shard = new CorpusIndex();
      shard.config = configName;
      shard.seed = seed;
      List<FileSpec> specs = specs();
      for (int i = from; i < to; i++) {
        shard.entries.add(PrePassCorpusWriters.write(specs.get(i), corpusDir + "/files"));
      }
      // Fixed-width names keep shards in index order when the reader sorts them.
      PrePassCorpus.writeIndex(shard, String.format("%s/corpus-index-%09d.json", corpusDir, from));
      out.output((long) (to - from));
    }
  }

  /** Under glob input only files that exist are matched, so keep the entries the glob will see. */
  private CorpusIndex existingOnly(CorpusIndex full) throws IOException {
    Set<String> listed = new HashSet<>();
    for (MatchResult.Metadata metadata : FileSystems.match(corpusDir + "/files/*").metadata()) {
      listed.add(metadata.resourceId().toString());
    }
    CorpusIndex kept = new CorpusIndex();
    kept.config = full.config;
    kept.seed = full.seed;
    for (CorpusIndex.Entry entry : full.entries) {
      if (listed.contains(entry.path)) {
        kept.entries.add(entry);
      }
    }
    LOG.info("Glob input: {} of {} index entries exist", kept.entries.size(), full.entries.size());
    return kept;
  }

  // ---- pipeline

  private void buildPipeline() {
    int threadPoolSize =
        Integer.parseInt(
            property("threadPoolSize", String.valueOf(ReadFooterSchema.DEFAULT_THREAD_POOL_SIZE)));
    int maxInFlight =
        Integer.parseInt(
            property(
                "maxInFlightTasks", String.valueOf(ReadFooterSchema.DEFAULT_MAX_IN_FLIGHT_TASKS)));
    PCollection<String> paths;
    if (globInput) {
      // How users feed AddFiles: match a pattern, keep the path strings.
      paths =
          pipeline
              .apply("MatchFiles", FileIO.match().filepattern(corpusDir + "/files/*"))
              .apply(
                  "ToPath",
                  MapElements.into(TypeDescriptors.strings())
                      .via(metadata -> metadata.resourceId().toString()));
    } else {
      paths = pipeline.apply("Paths", Create.of(index.paths()).withCoder(StringUtf8Coder.of()));
    }
    paths
        .apply(
            "ReadFooterSchema",
            ParDo.of(
                new ReadFooterSchema(
                    SchemaEvolutionConfig.disabled(), threadPoolSize, maxInFlight)))
        .setCoder(CollectDistinctSchemas.groupCoder())
        .apply("CollectDistinctSchemas", Combine.globally(new CollectDistinctSchemas()))
        .apply("ToJson", MapElements.into(TypeDescriptors.strings()).via(AddFilesPrePassLT::toJson))
        .apply(
            "WriteResult",
            TextIO.write().to(resultDir + "/result").withSuffix(".json").withoutSharding());
  }

  private PipelineLauncher.LaunchInfo launch() throws IOException {
    // Dataflow runs use default machine type and autoscaling on purpose: the numbers should
    // reflect what a user gets out of the box.
    PipelineLauncher.LaunchConfig.Builder options =
        PipelineLauncher.LaunchConfig.builder("addfiles-prepass-" + config.name)
            .setSdk(PipelineLauncher.Sdk.JAVA)
            .setPipeline(pipeline)
            .addParameter("runner", config.runner);
    return pipelineLauncher.launch(project, region, options.build());
  }

  private static String toJson(List<CollectDistinctSchemas.SchemaGroup> schemas) {
    JsonArray array = new JsonArray();
    for (CollectDistinctSchemas.SchemaGroup schema : schemas) {
      JsonObject entry = new JsonObject();
      entry.addProperty("count", schema.files);
      entry.addProperty("schema", schema.schemaJson);
      array.add(entry);
    }
    return array.toString();
  }

  // ---- verification

  private void verifyCounts(PipelineLauncher.@Nullable LaunchInfo info) throws IOException {
    List<KV<String, Long>> actual = readResult();
    double inputs = metric(info, ReadFooterSchema.FILES_READ_COUNTER);
    double schemas = metric(info, ReadFooterSchema.SCHEMAS_EMITTED_COUNTER);
    double errors = metric(info, ReadFooterSchema.FOOTER_READ_ERRORS_COUNTER);
    assertEquals("files read", index.entries.size(), inputs, 0.5);
    for (int i = 1; i < actual.size(); i++) {
      assertTrue("most common first", actual.get(i - 1).getValue() >= actual.get(i).getValue());
    }

    if (config.coverage) {
      // Whether Iceberg accepts a shape is what coverage discovers, so the expectation is the
      // same read-and-canonicalize path run in-process (checked against golden files below).
      Map<String, String> inProcess = readAllInProcess();
      Map<String, Long> expectedBySchema = new TreeMap<>();
      for (String result : inProcess.values()) {
        if (!result.startsWith(ERROR_PREFIX)) {
          Long existing = expectedBySchema.get(result);
          expectedBySchema.put(result, existing == null ? 1L : existing + 1);
        }
      }
      List<KV<String, Long>> expected = new ArrayList<>();
      for (Map.Entry<String, Long> entry : expectedBySchema.entrySet()) {
        expected.add(KV.of(entry.getKey(), entry.getValue()));
      }
      LOG.info(
          "inputs={} schemas={} distinct={} inProcessSchemas={} inProcessDistinct={}",
          inputs,
          schemas,
          actual.size(),
          inProcess.size() - errorCount(inProcess),
          expected.size());
      assertEquals(
          "schema count", (double) (inProcess.size() - errorCount(inProcess)), schemas, 0.5);
      assertEquals("footer errors", (double) errorCount(inProcess), errors, 0.5);
      assertEquals("distinct schemas", sortedBySchema(expected), sortedBySchema(actual));
      verifyGolden(inProcess);
      writeReviewTable(inProcess);
      return;
    }

    Map<String, Long> expectedByKey = new TreeMap<>();
    long expectedNothing = 0;
    long expectedErrors = 0;
    for (CorpusIndex.Entry entry : index.entries) {
      if (!entry.writeFailure.isEmpty()) {
        expectedErrors++;
      } else if (entry.expected == Outcome.SCHEMA) {
        Long existing = expectedByKey.get(entry.schemaKey);
        expectedByKey.put(entry.schemaKey, existing == null ? 1L : existing + 1);
      } else if (entry.expected == Outcome.NOTHING) {
        expectedNothing++;
      } else {
        expectedErrors++;
      }
    }
    // Two variants can canonicalize to the same schema for some seeds, so the expectation is
    // keyed by canonical JSON, not by variant key.
    List<KV<String, Long>> expectedSchemasList = expectedScaleSchemas(expectedByKey);
    List<Long> expectedCounts = new ArrayList<>();
    for (KV<String, Long> schema : expectedSchemasList) {
      expectedCounts.add(schema.getValue());
    }
    Collections.sort(expectedCounts);
    long expectedSchemas = 0;
    for (long count : expectedCounts) {
      expectedSchemas += count;
    }
    List<Long> actualCounts = new ArrayList<>();
    for (KV<String, Long> schema : actual) {
      actualCounts.add(schema.getValue());
    }
    Collections.sort(actualCounts);
    LOG.info(
        "inputs={} schemas={} distinct={} expectedSchemas={} expectedNothing={} expectedErrors={}",
        inputs,
        schemas,
        actual.size(),
        expectedSchemas,
        expectedNothing,
        expectedErrors);
    assertEquals("schema count", (double) expectedSchemas, schemas, 0.5);
    assertEquals("footer errors", (double) expectedErrors, errors, 0.5);
    assertEquals("distinct schema counts", expectedCounts, actualCounts);
    assertEquals("distinct schemas", sortedBySchema(expectedSchemasList), sortedBySchema(actual));
  }

  private static final String ERROR_PREFIX = "ERROR: ";

  /**
   * -DreviewTable=PATH: one tab-separated row per spec (id, Parquet schema, Iceberg struct or
   * error) for reviewing what the converter does with every shape.
   */
  private void writeReviewTable(Map<String, String> inProcess) throws IOException {
    String path = property("reviewTable", "");
    if (path.isEmpty()) {
      return;
    }
    Map<String, FileSpec> specsById = new TreeMap<>();
    for (FileSpec spec : PrePassCorpus.coverageSpecs(seed)) {
      specsById.put(spec.id, spec);
    }
    StringBuilder table = new StringBuilder();
    for (Map.Entry<String, String> entry : inProcess.entrySet()) {
      FileSpec spec = specsById.get(entry.getKey());
      String parquet = spec == null ? "?" : spec.toMessageType().toString().replaceAll("\\s+", " ");
      String result = entry.getValue();
      if (!result.startsWith(ERROR_PREFIX)) {
        result = org.apache.iceberg.SchemaParser.fromJson(result).asStruct().toString();
      }
      table.append(entry.getKey()).append('\t').append(parquet).append('\t').append(result);
      table.append('\n');
    }
    Files.write(new File(path).toPath(), table.toString().getBytes(StandardCharsets.UTF_8));
    LOG.info("Review table with {} rows written to {}", inProcess.size(), path);
  }

  /**
   * Expected output content: one file per schema key read in-process; keys whose files canonicalize
   * to the same schema are merged.
   */
  private List<KV<String, Long>> expectedScaleSchemas(Map<String, Long> countsByKey)
      throws IOException {
    Map<String, String> jsonByKey = new TreeMap<>();
    for (CorpusIndex.Entry entry : index.entries) {
      boolean readable = entry.expected == Outcome.SCHEMA && entry.writeFailure.isEmpty();
      if (readable && !jsonByKey.containsKey(entry.schemaKey)) {
        jsonByKey.put(
            entry.schemaKey,
            FileSchemas.canonicalJson(
                ParquetFooters.read(entry.path), SchemaEvolutionConfig.disabled()));
      }
    }
    Map<String, Long> countsBySchema = new TreeMap<>();
    for (Map.Entry<String, Long> count : countsByKey.entrySet()) {
      String json = checkStateNotNull(jsonByKey.get(count.getKey()));
      Long existing = countsBySchema.get(json);
      countsBySchema.put(json, existing == null ? count.getValue() : existing + count.getValue());
    }
    List<KV<String, Long>> expected = new ArrayList<>();
    for (Map.Entry<String, Long> count : countsBySchema.entrySet()) {
      expected.add(KV.of(count.getKey(), count.getValue()));
    }
    return expected;
  }

  /**
   * The DoFn's read-and-canonicalize path per spec, in-process: canonical JSON or an ERROR line.
   */
  private Map<String, String> readAllInProcess() {
    Map<String, String> results = new TreeMap<>();
    for (CorpusIndex.Entry entry : index.entries) {
      if (entry.expected == Outcome.NOTHING) {
        continue;
      }
      String result;
      try {
        result =
            FileSchemas.canonicalJson(
                ParquetFooters.read(entry.path), SchemaEvolutionConfig.disabled());
      } catch (Exception e) {
        result =
            ERROR_PREFIX + e.getClass().getSimpleName() + ": " + stable(AddFiles.errorMessage(e));
      }
      results.put(entry.specId, result);
    }
    return results;
  }

  /** Strips object hashes and corpus paths so error goldens are stable across runs. */
  private String stable(String message) {
    return message.replaceAll("@[0-9a-f]+", "@HASH").replace(corpusDir, "<corpus>");
  }

  private static int errorCount(Map<String, String> results) {
    int errors = 0;
    for (String result : results.values()) {
      if (result.startsWith(ERROR_PREFIX)) {
        errors++;
      }
    }
    return errors;
  }

  private static List<KV<String, Long>> sortedBySchema(List<KV<String, Long>> schemas) {
    List<KV<String, Long>> sorted = new ArrayList<>(schemas);
    sorted.sort((a, b) -> a.getKey().compareTo(b.getKey()));
    return sorted;
  }

  /**
   * Optionally diffs the in-process results against golden files (-DgoldenDir, recorded with
   * -DrecordGolden=true; nothing is tracked). Also lists specs the spec author expected to yield a
   * schema but the converter rejects: those are findings, not failures.
   */
  private void verifyGolden(Map<String, String> inProcess) throws IOException {
    // Goldens are opt-in: nothing is tracked. To check drift across a library upgrade, record
    // before (-DgoldenDir=DIR -DrecordGolden=true) and compare after (-DgoldenDir=DIR).
    boolean record = Boolean.parseBoolean(property("recordGolden", "false"));
    String goldenPath = property("goldenDir", "");
    @Nullable File goldenDir = goldenPath.isEmpty() ? null : new File(goldenPath);
    List<String> diffs = new ArrayList<>();
    List<String> missing = new ArrayList<>();
    List<String> rejected = new ArrayList<>();
    for (CorpusIndex.Entry entry : index.entries) {
      String actual = inProcess.get(entry.specId);
      if (actual == null) {
        continue;
      }
      boolean error = actual.startsWith(ERROR_PREFIX);
      if (entry.expected == Outcome.FOOTER_ERROR) {
        assertTrue(entry.specId + " should fail but read " + actual, error);
      } else if (error) {
        rejected.add(entry.specId + ": " + actual);
      }
      if (goldenDir == null) {
        continue;
      }
      File golden = new File(goldenDir, entry.specId.replace('/', '_') + ".json");
      if (record) {
        golden.getParentFile().mkdirs();
        Files.write(golden.toPath(), actual.getBytes(StandardCharsets.UTF_8));
      } else if (!golden.exists()) {
        missing.add(entry.specId);
      } else {
        String expected = new String(Files.readAllBytes(golden.toPath()), StandardCharsets.UTF_8);
        if (!expected.equals(actual)) {
          diffs.add(entry.specId + "\n  expected: " + expected + "\n  actual:   " + actual);
        }
      }
    }
    LOG.info("Specs expected to yield a schema that the converter rejects ({}):", rejected.size());
    for (String line : rejected) {
      LOG.info("  {}", line);
    }
    if (!missing.isEmpty()) {
      LOG.warn(
          "{} specs have no golden file (run with -DrecordGolden=true): {}",
          missing.size(),
          missing);
    }
    assertTrue("golden diffs:\n" + String.join("\n", diffs), diffs.isEmpty());
  }

  private List<KV<String, Long>> readResult() throws IOException {
    String json = readText(resultDir + "/result.json");
    List<KV<String, Long>> schemas = new ArrayList<>();
    for (com.google.gson.JsonElement element : JsonParser.parseString(json).getAsJsonArray()) {
      JsonObject entry = element.getAsJsonObject();
      schemas.add(KV.of(entry.get("schema").getAsString(), entry.get("count").getAsLong()));
    }
    return schemas;
  }

  // ---- reporting

  private void report(PipelineLauncher.LaunchInfo info) {
    Map<String, Double> metrics = new TreeMap<>();
    int files = index.entries.size();
    metrics.put("WallSecondsUpperBound", wallSeconds);
    metrics.put("FilesPerWallSecondLowerBound", files / wallSeconds);
    try {
      MetricsConfiguration metricsConfig =
          MetricsConfiguration.builder()
              .setInputPCollection("ToPath.out0")
              .setInputPCollectionV2("ToPath/Map/ParMultiDo(Anonymous).out0")
              .setOutputPCollection("ReadFooterSchema.out0")
              .setOutputPCollectionV2("ReadFooterSchema/ParMultiDo(ReadFooterSchema).out0")
              .build();
      metrics.putAll(getMetrics(info, metricsConfig));
    } catch (Exception e) {
      LOG.warn("Could not collect runner metrics", e);
    }
    Double vcpuSeconds = metrics.get("TotalVcpuTime");
    Double cost = metrics.get("EstimatedCost");
    if (vcpuSeconds != null && vcpuSeconds > 0) {
      metrics.put("FilesPerVcpuSecond", files / vcpuSeconds);
    }
    if (cost != null) {
      metrics.put("CostPer1MFiles", cost / files * 1_000_000);
    }
    StringBuilder table = new StringBuilder();
    table.append("\nconfig=").append(config.name).append(" files=").append(files).append('\n');
    for (Map.Entry<String, Double> metric : new TreeMap<>(metrics).entrySet()) {
      table.append(String.format("  %-32s %,.4f%n", metric.getKey(), metric.getValue()));
    }
    LOG.info("{}", table);
    try {
      exportMetricsToBigQuery(info, metrics);
    } catch (Exception e) {
      LOG.warn("Could not export metrics", e);
    }
  }

  // ---- helpers

  private double metric(PipelineLauncher.@Nullable LaunchInfo info, String counter)
      throws IOException {
    if (info == null) {
      MetricQueryResults metrics =
          checkStateNotNull(directResult)
              .metrics()
              .queryMetrics(
                  MetricsFilter.builder()
                      .addNameFilter(MetricNameFilter.named(ReadFooterSchema.class, counter))
                      .build());
      double total = 0;
      for (MetricResult<Long> result : metrics.getCounters()) {
        total += result.getAttempted();
      }
      return total;
    }
    String name = counter;
    if (counter.equals("addfiles-errors")) {
      name = getBeamMetricsName(PipelineMetricsType.COUNTER, counter);
    }
    Double value = pipelineLauncher.getMetric(project, region, info.jobId(), name);
    return value == null ? 0 : value;
  }

  private static String property(String name, String defaultValue) {
    return TestProperties.getProperty(name, defaultValue, TestProperties.Type.PROPERTY);
  }

  private static boolean exists(String path) throws IOException {
    MatchResult match = FileSystems.match(path);
    return match.status() == MatchResult.Status.OK && !match.metadata().isEmpty();
  }

  private static String readText(String path) throws IOException {
    MatchResult.Metadata metadata = FileSystems.matchSingleFileSpec(path);
    try (ReadableByteChannel channel = FileSystems.open(metadata.resourceId())) {
      return new String(Channels.newInputStream(channel).readAllBytes(), StandardCharsets.UTF_8);
    }
  }
}
