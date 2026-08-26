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
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.beam.sdk.io.iceberg.PrePassCorpus.Adversarial;
import org.apache.beam.sdk.io.iceberg.PrePassCorpus.CorpusIndex;
import org.apache.beam.sdk.io.iceberg.PrePassCorpus.FileSpec;
import org.apache.beam.sdk.io.iceberg.PrePassCorpus.Outcome;
import org.apache.beam.sdk.io.iceberg.PrePassCorpus.Producer;
import org.apache.iceberg.FileFormat;
import org.apache.parquet.hadoop.metadata.ParquetMetadata;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Checks that every corpus writer produces what its spec claims. Local and fast; runs with the
 * module's unit tests: {@code ./gradlew :it:iceberg:test --tests '*PrePassCorpusTest*'}.
 */
@RunWith(JUnit4.class)
public class PrePassCorpusTest {
  @Rule public final TemporaryFolder tmp = new TemporaryFolder();

  private static final long SEED = 42;

  @Test
  public void testCoverageSpecsWriteAsClaimed() throws IOException {
    String dir = tmp.getRoot().getAbsolutePath();
    List<String> writeFailures = new ArrayList<>();
    Set<String> ids = new HashSet<>();
    for (FileSpec spec : PrePassCorpus.coverageSpecs(SEED)) {
      assertTrue("duplicate spec id " + spec.id, ids.add(spec.id));
      CorpusIndex.Entry entry = PrePassCorpusWriters.write(spec, dir);
      if (!entry.writeFailure.isEmpty()) {
        writeFailures.add(spec.id + ": " + entry.writeFailure);
        continue;
      }
      check(spec, entry);
    }
    System.out.println("Specs the producer itself refused (" + writeFailures.size() + "):");
    for (String failure : writeFailures) {
      System.out.println("  " + failure);
    }
  }

  private static void check(FileSpec spec, CorpusIndex.Entry entry) throws IOException {
    switch (spec.expectedOutcome()) {
      case SCHEMA:
        ParquetMetadata footer = ParquetFooters.read(entry.path);
        assertEquals(FileFormat.PARQUET, AddFiles.inferFormat(entry.path));
        if (spec.producer == Producer.PARQUET_MR) {
          assertEquals(spec.id, spec.toMessageType(), footer.getFileMetaData().getSchema());
        }
        break;
      case NOTHING:
        assertTrue(spec.id, contributesNothing(entry.path));
        break;
      case FOOTER_ERROR:
        try {
          ParquetFooters.read(entry.path);
          fail(spec.id + " was readable but is expected to fail");
        } catch (Exception expected) {
          // the pre-pass counts this as a footer error
        }
        break;
      default:
        throw new IllegalStateException();
    }
  }

  /** Non-Parquet and unknown extensions both contribute nothing. */
  private static boolean contributesNothing(String path) {
    try {
      return AddFiles.inferFormat(path) != FileFormat.PARQUET;
    } catch (AddFiles.UnknownFormatException e) {
      return true;
    }
  }

  @Test
  public void testAdversarialKindsAllCovered() {
    Set<Adversarial> seen = new HashSet<>();
    for (FileSpec spec : PrePassCorpus.adversarialSpecs()) {
      seen.add(spec.adversarial);
    }
    assertEquals(Adversarial.values().length - 1, seen.size());
  }

  @Test
  public void testScaleSpecsShape() {
    List<FileSpec> specs = PrePassCorpus.scaleSpecs(1000, SEED);
    assertEquals(1000, specs.size());
    Set<String> keys = new HashSet<>();
    int adversarial = 0;
    for (FileSpec spec : specs) {
      keys.add(spec.schemaKey);
      if (spec.adversarial != Adversarial.NONE) {
        adversarial++;
      }
    }
    assertEquals(7, keys.size());
    assertEquals(10, adversarial);
    assertEquals(ids(specs), ids(PrePassCorpus.scaleSpecs(1000, SEED)));
  }

  private static List<String> ids(List<FileSpec> specs) {
    List<String> ids = new ArrayList<>();
    for (FileSpec spec : specs) {
      ids.add(spec.id);
    }
    return ids;
  }

  @Test
  public void testIndexRoundTrip() throws IOException {
    CorpusIndex index = new CorpusIndex();
    index.config = "direct-1k";
    index.seed = SEED;
    CorpusIndex.Entry entry = new CorpusIndex.Entry();
    entry.specId = "scale/v0/1";
    entry.path = "/tmp/x.parquet";
    entry.schemaKey = "scale/v0";
    entry.expected = Outcome.SCHEMA;
    index.entries.add(entry);
    String path = new File(tmp.getRoot(), "corpus-index.json").getAbsolutePath();
    PrePassCorpus.writeIndex(index, path);
    CorpusIndex read = PrePassCorpus.readIndex(path);
    assertEquals("direct-1k", read.config);
    assertEquals(SEED, read.seed);
    assertEquals(1, read.entries.size());
    assertEquals("scale/v0/1", read.entries.get(0).specId);
    assertEquals(Outcome.SCHEMA, read.entries.get(0).expected);
  }
}
