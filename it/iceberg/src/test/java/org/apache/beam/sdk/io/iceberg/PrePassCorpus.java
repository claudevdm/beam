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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.stream.JsonReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Serializable;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.apache.beam.sdk.io.FileSystems;
import org.apache.beam.sdk.io.fs.MatchResult;
import org.apache.beam.sdk.io.fs.ResourceId;
import org.apache.beam.sdk.util.MimeTypes;
import org.apache.parquet.schema.LogicalTypeAnnotation;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType;
import org.apache.parquet.schema.PrimitiveType.PrimitiveTypeName;
import org.apache.parquet.schema.Type;
import org.apache.parquet.schema.Type.Repetition;
import org.apache.parquet.schema.Types;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * Specs for the benchmark corpus: what each file looks like, how it is produced, and what the
 * pre-pass is expected to do with it. Ground truth is derived from specs, never from reading files
 * back. Writers live in {@link PrePassCorpusWriters}.
 */
final class PrePassCorpus {
  private PrePassCorpus() {}

  enum Outcome {
    SCHEMA,
    NOTHING,
    FOOTER_ERROR
  }

  enum Producer {
    PARQUET_MR,
    ICEBERG,
    PARQUET_AVRO
  }

  /** Post-processing applied after the normal file is written; each has a known outcome. */
  enum Adversarial {
    NONE(Outcome.SCHEMA, "parquet"),
    ZERO_BYTES(Outcome.FOOTER_ERROR, "parquet"),
    TRUNCATED_HALF(Outcome.FOOTER_ERROR, "parquet"),
    TRUNCATED_TAIL8(Outcome.FOOTER_ERROR, "parquet"),
    TRUNCATED_LAST_BYTE(Outcome.FOOTER_ERROR, "parquet"),
    GARBAGE_TEXT(Outcome.FOOTER_ERROR, "parquet"),
    ORC_MAGIC(Outcome.FOOTER_ERROR, "parquet"),
    AVRO_MAGIC(Outcome.FOOTER_ERROR, "parquet"),
    GZIPPED(Outcome.FOOTER_ERROR, "parquet"),
    MISSING_PATH(Outcome.FOOTER_ERROR, "parquet"),
    DIRECTORY_PATH(Outcome.FOOTER_ERROR, "parquet"),
    PARQUET_GZ_EXTENSION(Outcome.NOTHING, "parquet.gz"),
    ORC_EXTENSION(Outcome.NOTHING, "orc"),
    AVRO_EXTENSION(Outcome.NOTHING, "avro"),
    NO_EXTENSION(Outcome.NOTHING, ""),
    UPPERCASE_EXTENSION(Outcome.SCHEMA, "PARQUET"),
    PQT_EXTENSION(Outcome.SCHEMA, "pqt");

    final Outcome outcome;
    final String extension;

    Adversarial(Outcome outcome, String extension) {
      this.outcome = outcome;
      this.extension = extension;
    }
  }

  /** Parquet physical type plus logical annotation. */
  enum LeafType {
    BOOLEAN,
    INT32,
    INT8,
    INT16,
    UINT8,
    UINT16,
    UINT32,
    DATE,
    TIME_MILLIS,
    DECIMAL_INT32,
    INT64,
    UINT64,
    TIME_MICROS,
    TIME_NANOS,
    TIMESTAMP_MILLIS_UTC,
    TIMESTAMP_MILLIS_LOCAL,
    TIMESTAMP_MICROS_UTC,
    TIMESTAMP_MICROS_LOCAL,
    TIMESTAMP_NANOS_UTC,
    TIMESTAMP_NANOS_LOCAL,
    DECIMAL_INT64,
    INT96,
    FLOAT,
    DOUBLE,
    BINARY,
    STRING,
    ENUM,
    JSON,
    BSON,
    DECIMAL_BINARY,
    FIXED16,
    DECIMAL_FIXED,
    UUID,
    INTERVAL,
    FLOAT16;

    PrimitiveType toParquet(String name, Repetition repetition) {
      switch (this) {
        case BOOLEAN:
          return primitive(PrimitiveTypeName.BOOLEAN, name, repetition, null);
        case INT32:
          return primitive(PrimitiveTypeName.INT32, name, repetition, null);
        case INT8:
          return primitive(PrimitiveTypeName.INT32, name, repetition, intType(8, true));
        case INT16:
          return primitive(PrimitiveTypeName.INT32, name, repetition, intType(16, true));
        case UINT8:
          return primitive(PrimitiveTypeName.INT32, name, repetition, intType(8, false));
        case UINT16:
          return primitive(PrimitiveTypeName.INT32, name, repetition, intType(16, false));
        case UINT32:
          return primitive(PrimitiveTypeName.INT32, name, repetition, intType(32, false));
        case DATE:
          return primitive(
              PrimitiveTypeName.INT32, name, repetition, LogicalTypeAnnotation.dateType());
        case TIME_MILLIS:
          return primitive(PrimitiveTypeName.INT32, name, repetition, time(TimeUnit.MILLIS));
        case DECIMAL_INT32:
          return primitive(PrimitiveTypeName.INT32, name, repetition, decimal(9, 2));
        case INT64:
          return primitive(PrimitiveTypeName.INT64, name, repetition, null);
        case UINT64:
          return primitive(PrimitiveTypeName.INT64, name, repetition, intType(64, false));
        case TIME_MICROS:
          return primitive(PrimitiveTypeName.INT64, name, repetition, time(TimeUnit.MICROS));
        case TIME_NANOS:
          return primitive(PrimitiveTypeName.INT64, name, repetition, time(TimeUnit.NANOS));
        case TIMESTAMP_MILLIS_UTC:
          return primitive(
              PrimitiveTypeName.INT64, name, repetition, timestamp(TimeUnit.MILLIS, true));
        case TIMESTAMP_MILLIS_LOCAL:
          return primitive(
              PrimitiveTypeName.INT64, name, repetition, timestamp(TimeUnit.MILLIS, false));
        case TIMESTAMP_MICROS_UTC:
          return primitive(
              PrimitiveTypeName.INT64, name, repetition, timestamp(TimeUnit.MICROS, true));
        case TIMESTAMP_MICROS_LOCAL:
          return primitive(
              PrimitiveTypeName.INT64, name, repetition, timestamp(TimeUnit.MICROS, false));
        case TIMESTAMP_NANOS_UTC:
          return primitive(
              PrimitiveTypeName.INT64, name, repetition, timestamp(TimeUnit.NANOS, true));
        case TIMESTAMP_NANOS_LOCAL:
          return primitive(
              PrimitiveTypeName.INT64, name, repetition, timestamp(TimeUnit.NANOS, false));
        case DECIMAL_INT64:
          return primitive(PrimitiveTypeName.INT64, name, repetition, decimal(18, 4));
        case INT96:
          return primitive(PrimitiveTypeName.INT96, name, repetition, null);
        case FLOAT:
          return primitive(PrimitiveTypeName.FLOAT, name, repetition, null);
        case DOUBLE:
          return primitive(PrimitiveTypeName.DOUBLE, name, repetition, null);
        case BINARY:
          return primitive(PrimitiveTypeName.BINARY, name, repetition, null);
        case STRING:
          return primitive(
              PrimitiveTypeName.BINARY, name, repetition, LogicalTypeAnnotation.stringType());
        case ENUM:
          return primitive(
              PrimitiveTypeName.BINARY, name, repetition, LogicalTypeAnnotation.enumType());
        case JSON:
          return primitive(
              PrimitiveTypeName.BINARY, name, repetition, LogicalTypeAnnotation.jsonType());
        case BSON:
          return primitive(
              PrimitiveTypeName.BINARY, name, repetition, LogicalTypeAnnotation.bsonType());
        case DECIMAL_BINARY:
          return primitive(PrimitiveTypeName.BINARY, name, repetition, decimal(38, 10));
        case FIXED16:
          return fixed(16, name, repetition, null);
        case DECIMAL_FIXED:
          return fixed(16, name, repetition, decimal(38, 10));
        case UUID:
          return fixed(16, name, repetition, LogicalTypeAnnotation.uuidType());
        case INTERVAL:
          return fixed(
              12,
              name,
              repetition,
              LogicalTypeAnnotation.IntervalLogicalTypeAnnotation.getInstance());
        case FLOAT16:
          return fixed(2, name, repetition, LogicalTypeAnnotation.float16Type());
        default:
          throw new IllegalArgumentException(name());
      }
    }

    private static PrimitiveType primitive(
        PrimitiveTypeName physical,
        String name,
        Repetition repetition,
        @Nullable LogicalTypeAnnotation logical) {
      Types.PrimitiveBuilder<PrimitiveType> builder = Types.primitive(physical, repetition);
      if (logical != null) {
        builder = builder.as(logical);
      }
      return builder.named(name);
    }

    private static PrimitiveType fixed(
        int length, String name, Repetition repetition, @Nullable LogicalTypeAnnotation logical) {
      Types.PrimitiveBuilder<PrimitiveType> builder =
          Types.primitive(PrimitiveTypeName.FIXED_LEN_BYTE_ARRAY, repetition).length(length);
      if (logical != null) {
        builder = builder.as(logical);
      }
      return builder.named(name);
    }

    private static LogicalTypeAnnotation intType(int bits, boolean signed) {
      return LogicalTypeAnnotation.intType(bits, signed);
    }

    private static LogicalTypeAnnotation decimal(int precision, int scale) {
      return LogicalTypeAnnotation.decimalType(scale, precision);
    }

    private static LogicalTypeAnnotation time(TimeUnit unit) {
      return LogicalTypeAnnotation.timeType(true, unit.parquet);
    }

    private static LogicalTypeAnnotation timestamp(TimeUnit unit, boolean utc) {
      return LogicalTypeAnnotation.timestampType(utc, unit.parquet);
    }
  }

  private enum TimeUnit {
    MILLIS(LogicalTypeAnnotation.TimeUnit.MILLIS),
    MICROS(LogicalTypeAnnotation.TimeUnit.MICROS),
    NANOS(LogicalTypeAnnotation.TimeUnit.NANOS);

    final LogicalTypeAnnotation.TimeUnit parquet;

    TimeUnit(LogicalTypeAnnotation.TimeUnit parquet) {
      this.parquet = parquet;
    }
  }

  enum GroupKind {
    STRUCT,
    /** Standard three-level list: group(LIST) > repeated group "list" > element. */
    LIST3,
    /** Legacy two-level list: group(LIST) > repeated element. */
    LIST2,
    /** A repeated field with no annotation at all. */
    REPEATED_LEAF,
    /** Standard map: group(MAP) > repeated group "key_value" > key, value. */
    MAP,
    /** Legacy map: group(MAP_KEY_VALUE) > repeated group "map" > key, value. */
    MAP_LEGACY
  }

  abstract static class Node implements Serializable {
    final String name;
    final Repetition repetition;

    Node(String name, Repetition repetition) {
      this.name = name;
      this.repetition = repetition;
    }

    abstract Type toParquet();
  }

  static final class Leaf extends Node {
    final LeafType type;

    Leaf(String name, Repetition repetition, LeafType type) {
      super(name, repetition);
      this.type = type;
    }

    @Override
    Type toParquet() {
      return type.toParquet(name, repetition);
    }
  }

  /**
   * STRUCT takes any children. LIST3 and LIST2 take one child, the element, whose name is used
   * (element / item / array). REPEATED_LEAF takes one leaf. MAP and MAP_LEGACY take two children,
   * key then value.
   */
  static final class Group extends Node {
    final GroupKind kind;
    final List<Node> children;

    Group(String name, Repetition repetition, GroupKind kind, List<Node> children) {
      super(name, repetition);
      this.kind = kind;
      this.children = children;
    }

    @Override
    Type toParquet() {
      switch (kind) {
        case STRUCT:
          return Types.buildGroup(repetition).addFields(childTypes()).named(name);
        case LIST3:
          return Types.buildGroup(repetition)
              .as(LogicalTypeAnnotation.listType())
              .addField(Types.repeatedGroup().addField(children.get(0).toParquet()).named("list"))
              .named(name);
        case LIST2:
          return Types.buildGroup(repetition)
              .as(LogicalTypeAnnotation.listType())
              .addField(repeated(children.get(0)))
              .named(name);
        case REPEATED_LEAF:
          return repeated(children.get(0));
        case MAP:
          return Types.buildGroup(repetition)
              .as(LogicalTypeAnnotation.mapType())
              .addField(
                  Types.repeatedGroup()
                      .addField(children.get(0).toParquet())
                      .addField(children.get(1).toParquet())
                      .named("key_value"))
              .named(name);
        case MAP_LEGACY:
          return Types.buildGroup(repetition)
              .as(LogicalTypeAnnotation.MapKeyValueTypeAnnotation.getInstance())
              .addField(
                  Types.repeatedGroup()
                      .addField(children.get(0).toParquet())
                      .addField(children.get(1).toParquet())
                      .named("map"))
              .named(name);
        default:
          throw new IllegalArgumentException(kind.name());
      }
    }

    private Type[] childTypes() {
      Type[] types = new Type[children.size()];
      for (int i = 0; i < children.size(); i++) {
        types[i] = children.get(i).toParquet();
      }
      return types;
    }

    private static Type repeated(Node node) {
      if (node instanceof Leaf) {
        Leaf leaf = (Leaf) node;
        return leaf.type.toParquet(leaf.name, Repetition.REPEATED);
      }
      Group group = (Group) node;
      return new Group(group.name, Repetition.REPEATED, group.kind, group.children).toParquet();
    }
  }

  static final class FooterSpec implements Serializable {
    final int rows;
    final int rowGroups;
    final boolean stats;
    /** Fraction of null values per optional column: 0, 0.5 or 1. */
    final double nullFraction;

    FooterSpec(int rows, int rowGroups, boolean stats, double nullFraction) {
      this.rows = rows;
      this.rowGroups = rowGroups;
      this.stats = stats;
      this.nullFraction = nullFraction;
    }

    static final FooterSpec DEFAULT = new FooterSpec(10, 1, true, 0);
  }

  static final class FileSpec implements Serializable {
    final String id;
    final Producer producer;
    final List<Node> columns;
    final FooterSpec footer;
    final Adversarial adversarial;
    /** Files sharing a key are expected to produce the same canonical schema. */
    final String schemaKey;

    FileSpec(
        String id,
        Producer producer,
        List<Node> columns,
        FooterSpec footer,
        Adversarial adversarial,
        String schemaKey) {
      this.id = id;
      this.producer = producer;
      this.columns = columns;
      this.footer = footer;
      this.adversarial = adversarial;
      this.schemaKey = schemaKey;
    }

    static FileSpec of(String id, List<Node> columns) {
      return new FileSpec(
          id, Producer.PARQUET_MR, columns, FooterSpec.DEFAULT, Adversarial.NONE, id);
    }

    FileSpec withAdversarial(Adversarial adversarial) {
      return new FileSpec(id, producer, columns, footer, adversarial, schemaKey);
    }

    FileSpec withFooter(FooterSpec footer) {
      return new FileSpec(id, producer, columns, footer, adversarial, schemaKey);
    }

    FileSpec withProducer(Producer producer) {
      return new FileSpec(id, producer, columns, footer, adversarial, schemaKey);
    }

    Outcome expectedOutcome() {
      return adversarial.outcome;
    }

    MessageType toMessageType() {
      Types.MessageTypeBuilder builder = Types.buildMessage();
      for (Node column : columns) {
        builder.addField(column.toParquet());
      }
      return builder.named("root");
    }

    String fileName() {
      String base = id.replace('/', '_').replaceAll("[^A-Za-z0-9_.-]", "_");
      if (adversarial.extension.isEmpty()) {
        return base;
      }
      return base + "." + adversarial.extension;
    }
  }

  // ---- enumerators

  static Leaf leaf(String name, Repetition repetition, LeafType type) {
    return new Leaf(name, repetition, type);
  }

  static Group struct(String name, Repetition repetition, Node... children) {
    return new Group(name, repetition, GroupKind.STRUCT, Arrays.asList(children));
  }

  static Group list(String name, Repetition repetition, GroupKind kind, Node element) {
    return new Group(name, repetition, kind, Collections.singletonList(element));
  }

  static Group map(String name, Repetition repetition, GroupKind kind, Node key, Node value) {
    return new Group(name, repetition, kind, Arrays.asList(key, value));
  }

  private static Leaf stringLeaf(String name, Repetition repetition) {
    return leaf(name, repetition, LeafType.STRING);
  }

  /** Dimension 1 and the leaf half of dimension 2: every leaf type in every repetition. */
  static List<FileSpec> leafSpecs() {
    List<FileSpec> specs = new ArrayList<>();
    for (LeafType type : LeafType.values()) {
      for (Repetition repetition : Repetition.values()) {
        String id = "leaf/" + type.name().toLowerCase() + "/" + repetition.name().toLowerCase();
        specs.add(FileSpec.of(id, Collections.singletonList(leaf("col", repetition, type))));
      }
    }
    return specs;
  }

  /** Dimension 2, groups: every container kind, element and value optionality, element names. */
  static List<FileSpec> groupSpecs() {
    List<FileSpec> specs = new ArrayList<>();
    Repetition[] outer = {Repetition.REQUIRED, Repetition.OPTIONAL};
    Repetition[] inner = {Repetition.REQUIRED, Repetition.OPTIONAL};
    for (Repetition rep : outer) {
      String r = rep.name().toLowerCase();
      for (Repetition elementRep : inner) {
        String e = elementRep.name().toLowerCase();
        for (String elementName : new String[] {"element", "item", "array"}) {
          specs.add(
              spec(
                  "group/list3/" + r + "/" + e + "/" + elementName,
                  list("col", rep, GroupKind.LIST3, stringLeaf(elementName, elementRep))));
        }
        specs.add(
            spec(
                "group/list2/" + r + "/" + e,
                list("col", rep, GroupKind.LIST2, stringLeaf("array", elementRep))));
        specs.add(
            spec(
                "group/map/" + r + "/" + e,
                map(
                    "col",
                    rep,
                    GroupKind.MAP,
                    stringLeaf("key", Repetition.REQUIRED),
                    stringLeaf("value", elementRep))));
        specs.add(
            spec(
                "group/map_legacy/" + r + "/" + e,
                map(
                    "col",
                    rep,
                    GroupKind.MAP_LEGACY,
                    stringLeaf("key", Repetition.REQUIRED),
                    stringLeaf("value", elementRep))));
      }
      specs.add(
          spec(
              "group/struct/" + r,
              struct(
                  "col",
                  rep,
                  stringLeaf("a", Repetition.OPTIONAL),
                  leaf("b", Repetition.REQUIRED, LeafType.INT64))));
      specs.add(spec("group/struct_empty/" + r, struct("col", rep)));
    }
    specs.add(
        spec(
            "group/repeated_leaf",
            list(
                "col",
                Repetition.REPEATED,
                GroupKind.REPEATED_LEAF,
                stringLeaf("col", Repetition.REPEATED))));
    return specs;
  }

  /** Dimension 3 at depth 2: every container inside every container. */
  static List<FileSpec> pairwiseSpecs() {
    List<FileSpec> specs = new ArrayList<>();
    GroupKind[] kinds = {GroupKind.STRUCT, GroupKind.LIST3, GroupKind.LIST2, GroupKind.MAP};
    for (GroupKind outer : kinds) {
      for (GroupKind innerKind : kinds) {
        Node inner = container("inner", innerKind, stringLeaf("leaf", Repetition.OPTIONAL));
        specs.add(
            spec(
                "pair/" + outer.name().toLowerCase() + "-" + innerKind.name().toLowerCase(),
                container("col", outer, inner)));
      }
    }
    return specs;
  }

  /** Dimension 3 beyond depth 2: seeded random trees. */
  static List<FileSpec> randomTreeSpecs(long seed, int count, int maxDepth) {
    List<FileSpec> specs = new ArrayList<>();
    Random random = new Random(seed);
    for (int i = 0; i < count; i++) {
      int depth = 3 + random.nextInt(maxDepth - 2);
      List<Node> columns = new ArrayList<>();
      int width = 1 + random.nextInt(4);
      for (int c = 0; c < width; c++) {
        columns.add(randomNode(random, "c" + c, depth));
      }
      specs.add(FileSpec.of("random/" + i, columns));
    }
    return specs;
  }

  private static Node randomNode(Random random, String name, int depth) {
    Repetition repetition = random.nextBoolean() ? Repetition.OPTIONAL : Repetition.REQUIRED;
    if (depth == 0 || random.nextInt(4) == 0) {
      LeafType[] types = LeafType.values();
      return leaf(name, repetition, types[random.nextInt(types.length)]);
    }
    switch (random.nextInt(4)) {
      case 0:
        List<Node> children = new ArrayList<>();
        int width = 1 + random.nextInt(3);
        for (int c = 0; c < width; c++) {
          children.add(randomNode(random, name + "_" + c, depth - 1));
        }
        return new Group(name, repetition, GroupKind.STRUCT, children);
      case 1:
        return list(name, repetition, GroupKind.LIST3, randomNode(random, "element", depth - 1));
      case 2:
        // legacy 2-level lists only ever held leaves or structs; pair/list2-map covers the rest
        Node element = randomNode(random, "array", depth - 1);
        if (element instanceof Group && ((Group) element).kind != GroupKind.STRUCT) {
          element = stringLeaf("array", Repetition.REQUIRED);
        }
        return list(name, repetition, GroupKind.LIST2, element);
      default:
        return map(
            name,
            repetition,
            GroupKind.MAP,
            stringLeaf("key", Repetition.REQUIRED),
            randomNode(random, "value", depth - 1));
    }
  }

  /** Dimension 4: column names. */
  static List<FileSpec> nameSpecs() {
    List<FileSpec> specs = new ArrayList<>();
    specs.add(spec("name/dotted", stringLeaf("a.b", Repetition.OPTIONAL)));
    specs.add(spec("name/spaced", stringLeaf("a b", Repetition.OPTIONAL)));
    specs.add(spec("name/unicode", stringLeaf("éè中文", Repetition.OPTIONAL)));
    specs.add(spec("name/long", stringLeaf(repeat("x", 1000), Repetition.OPTIONAL)));
    specs.add(spec("name/empty", stringLeaf("", Repetition.OPTIONAL)));
    specs.add(
        FileSpec.of(
            "name/case_pair",
            Arrays.asList(
                stringLeaf("a", Repetition.OPTIONAL), stringLeaf("A", Repetition.OPTIONAL))));
    specs.add(
        FileSpec.of(
            "name/duplicate",
            Arrays.asList(
                stringLeaf("a", Repetition.OPTIONAL), stringLeaf("a", Repetition.OPTIONAL))));
    for (String reserved : new String[] {"element", "key", "value", "list", "key_value"}) {
      specs.add(spec("name/reserved_" + reserved, stringLeaf(reserved, Repetition.OPTIONAL)));
    }
    return specs;
  }

  /** Dimension 6: footer content, over a fixed mixed schema. */
  static List<FileSpec> footerSpecs() {
    List<FileSpec> specs = new ArrayList<>();
    specs.add(
        FileSpec.of("footer/rows0", mixedColumns()).withFooter(new FooterSpec(0, 1, true, 0)));
    specs.add(
        FileSpec.of("footer/rows1", mixedColumns()).withFooter(new FooterSpec(1, 1, true, 0)));
    specs.add(
        FileSpec.of("footer/rowgroups200", mixedColumns())
            .withFooter(new FooterSpec(1000, 200, true, 0)));
    specs.add(
        FileSpec.of("footer/nostats", mixedColumns()).withFooter(new FooterSpec(10, 1, false, 0)));
    specs.add(
        FileSpec.of("footer/nulls_some", mixedColumns())
            .withFooter(new FooterSpec(10, 1, true, 0.5)));
    specs.add(
        FileSpec.of("footer/nulls_all", mixedColumns()).withFooter(new FooterSpec(10, 1, true, 1)));
    specs.add(FileSpec.of("footer/iceberg_ids", mixedColumns()).withProducer(Producer.ICEBERG));
    specs.add(
        FileSpec.of("footer/parquet_avro", mixedColumns()).withProducer(Producer.PARQUET_AVRO));
    specs.add(
        FileSpec.of("footer/wide5000", wideColumns(5000))
            .withFooter(new FooterSpec(200, 200, true, 0)));
    return specs;
  }

  /** Every adversarial kind over the mixed schema. */
  static List<FileSpec> adversarialSpecs() {
    List<FileSpec> specs = new ArrayList<>();
    for (Adversarial adversarial : Adversarial.values()) {
      if (adversarial == Adversarial.NONE) {
        continue;
      }
      specs.add(
          FileSpec.of("adversarial/" + adversarial.name().toLowerCase(), mixedColumns())
              .withAdversarial(adversarial));
    }
    return specs;
  }

  static List<FileSpec> coverageSpecs(long seed) {
    List<FileSpec> specs = new ArrayList<>();
    specs.addAll(leafSpecs());
    specs.addAll(groupSpecs());
    specs.addAll(pairwiseSpecs());
    specs.addAll(randomTreeSpecs(seed, 200, 20));
    specs.addAll(nameSpecs());
    specs.addAll(footerSpecs());
    specs.addAll(adversarialSpecs());
    return specs;
  }

  /**
   * Scale corpus: a few realistic schema variants with Zipf prevalence, plus one percent
   * adversarial files. Files of one variant share a schema key so dedup counts are checked exactly.
   */
  static List<FileSpec> scaleSpecs(int numFiles, long seed) {
    List<List<Node>> variants = new ArrayList<>();
    List<String> keys = new ArrayList<>();
    variants.add(realisticColumns(20, false, false, false));
    keys.add("scale/base");
    // same columns in another order: canonicalizes to the base schema
    variants.add(permuted(realisticColumns(20, false, false, false), seed));
    keys.add("scale/base");
    variants.add(realisticColumns(20, true, false, false));
    keys.add("scale/all_required");
    variants.add(realisticColumns(20, false, true, false));
    keys.add("scale/wide_ints");
    variants.add(realisticColumns(20, false, false, true));
    keys.add("scale/wide_floats");
    variants.add(realisticColumns(21, false, false, false));
    keys.add("scale/extra_column");
    variants.add(realisticColumns(200, false, false, false));
    keys.add("scale/wide200");
    variants.add(nestedRealisticColumns());
    keys.add("scale/nested");
    Adversarial[] adversarials = {
      Adversarial.GARBAGE_TEXT, Adversarial.ZERO_BYTES, Adversarial.MISSING_PATH
    };
    Random random = new Random(seed);
    List<FileSpec> specs = new ArrayList<>();
    for (int i = 0; i < numFiles; i++) {
      int variant = zipf(random, variants.size());
      FileSpec spec =
          new FileSpec(
              "scale/v" + variant + "/" + i,
              Producer.PARQUET_MR,
              variants.get(variant),
              FooterSpec.DEFAULT,
              Adversarial.NONE,
              keys.get(variant));
      if (i % 100 == 99) {
        spec = spec.withAdversarial(adversarials[(i / 100) % adversarials.length]);
      }
      specs.add(spec);
    }
    return specs;
  }

  private static int zipf(Random random, int n) {
    double total = 0;
    for (int i = 1; i <= n; i++) {
      total += 1.0 / i;
    }
    double target = random.nextDouble() * total;
    double running = 0;
    for (int i = 1; i <= n; i++) {
      running += 1.0 / i;
      if (target <= running) {
        return i - 1;
      }
    }
    return n - 1;
  }

  private static List<Node> realisticColumns(
      int width, boolean allRequired, boolean widenInts, boolean widenFloats) {
    LeafType[] cycle = {
      LeafType.INT64,
      LeafType.STRING,
      LeafType.DOUBLE,
      LeafType.BOOLEAN,
      LeafType.INT32,
      LeafType.FLOAT,
      LeafType.TIMESTAMP_MICROS_UTC,
      LeafType.DECIMAL_INT64
    };
    List<Node> columns = new ArrayList<>();
    for (int i = 0; i < width; i++) {
      LeafType type = cycle[i % cycle.length];
      if (widenInts && type == LeafType.INT32) {
        type = LeafType.INT64;
      }
      if (widenFloats && type == LeafType.FLOAT) {
        type = LeafType.DOUBLE;
      }
      Repetition repetition = allRequired || i == 0 ? Repetition.REQUIRED : Repetition.OPTIONAL;
      columns.add(leaf("col_" + i, repetition, type));
    }
    return columns;
  }

  private static List<Node> nestedRealisticColumns() {
    return Arrays.asList(
        leaf("id", Repetition.REQUIRED, LeafType.INT64),
        struct(
            "address",
            Repetition.OPTIONAL,
            stringLeaf("city", Repetition.OPTIONAL),
            leaf("zip", Repetition.OPTIONAL, LeafType.INT32)),
        list(
            "tags",
            Repetition.OPTIONAL,
            GroupKind.LIST3,
            stringLeaf("element", Repetition.OPTIONAL)),
        map(
            "attributes",
            Repetition.OPTIONAL,
            GroupKind.MAP,
            stringLeaf("key", Repetition.REQUIRED),
            leaf("value", Repetition.OPTIONAL, LeafType.DOUBLE)));
  }

  private static List<Node> permuted(List<Node> columns, long seed) {
    List<Node> shuffled = new ArrayList<>(columns);
    Collections.shuffle(shuffled, new Random(seed));
    return shuffled;
  }

  private static List<Node> mixedColumns() {
    return Arrays.asList(
        leaf("id", Repetition.REQUIRED, LeafType.INT64),
        stringLeaf("name", Repetition.OPTIONAL),
        leaf("score", Repetition.OPTIONAL, LeafType.DOUBLE),
        struct(
            "address",
            Repetition.OPTIONAL,
            stringLeaf("city", Repetition.OPTIONAL),
            leaf("zip", Repetition.OPTIONAL, LeafType.INT32)),
        list(
            "tags",
            Repetition.OPTIONAL,
            GroupKind.LIST3,
            stringLeaf("element", Repetition.OPTIONAL)));
  }

  private static List<Node> wideColumns(int width) {
    List<Node> columns = new ArrayList<>();
    for (int i = 0; i < width; i++) {
      columns.add(
          leaf("col_" + i, Repetition.OPTIONAL, i % 2 == 0 ? LeafType.INT64 : LeafType.STRING));
    }
    return columns;
  }

  private static Node container(String name, GroupKind kind, Node child) {
    switch (kind) {
      case STRUCT:
        return struct(name, Repetition.OPTIONAL, child);
      case LIST3:
        return list(name, Repetition.OPTIONAL, GroupKind.LIST3, renamed(child, "element"));
      case LIST2:
        return list(name, Repetition.OPTIONAL, GroupKind.LIST2, renamed(child, "array"));
      case MAP:
        return map(
            name,
            Repetition.OPTIONAL,
            GroupKind.MAP,
            stringLeaf("key", Repetition.REQUIRED),
            renamed(child, "value"));
      default:
        throw new IllegalArgumentException(kind.name());
    }
  }

  private static Node renamed(Node node, String name) {
    if (node instanceof Leaf) {
      Leaf leaf = (Leaf) node;
      return new Leaf(name, leaf.repetition, leaf.type);
    }
    Group group = (Group) node;
    return new Group(name, group.repetition, group.kind, group.children);
  }

  private static FileSpec spec(String id, Node column) {
    return FileSpec.of(id, Collections.singletonList(column));
  }

  private static String repeat(String s, int n) {
    StringBuilder builder = new StringBuilder();
    for (int i = 0; i < n; i++) {
      builder.append(s);
    }
    return builder.toString();
  }

  // ---- index

  /**
   * Test-only ground truth written next to the corpus as corpus-index.json (not an Iceberg index).
   */
  static final class CorpusIndex {
    String config = "";
    long seed;
    List<Entry> entries = new ArrayList<>();

    static final class Entry {
      String specId = "";
      String path = "";
      String schemaKey = "";
      Outcome expected = Outcome.SCHEMA;
      /** Set when the producer itself could not write the file; the path then does not exist. */
      String writeFailure = "";
    }

    List<String> paths() {
      List<String> paths = new ArrayList<>();
      for (Entry entry : entries) {
        paths.add(entry.path);
      }
      return paths;
    }
  }

  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  static void writeIndex(CorpusIndex index, String path) throws IOException {
    ResourceId resource = FileSystems.matchNewResource(path, false);
    byte[] bytes = GSON.toJson(index).getBytes(StandardCharsets.UTF_8);
    try (WritableByteChannel channel = FileSystems.create(resource, MimeTypes.TEXT)) {
      channel.write(ByteBuffer.wrap(bytes));
    }
  }

  static CorpusIndex readIndex(String path) throws IOException {
    MatchResult.Metadata metadata = FileSystems.matchSingleFileSpec(path);
    return readShard(metadata);
  }

  /**
   * Reads every index shard matching {@code glob} into one index. A JVM-generated corpus has a
   * single shard; a Dataflow-generated one has one shard per range. Streams each shard so a
   * million-entry index never exists as one JSON string.
   */
  static CorpusIndex readIndexShards(String glob) throws IOException {
    List<MatchResult.Metadata> shards = new ArrayList<>(FileSystems.match(glob).metadata());
    if (shards.isEmpty()) {
      throw new IOException("No corpus index matches " + glob);
    }
    shards.sort((a, b) -> a.resourceId().toString().compareTo(b.resourceId().toString()));
    CorpusIndex merged = new CorpusIndex();
    for (MatchResult.Metadata shard : shards) {
      CorpusIndex part = readShard(shard);
      merged.config = part.config;
      merged.seed = part.seed;
      merged.entries.addAll(part.entries);
    }
    return merged;
  }

  private static CorpusIndex readShard(MatchResult.Metadata metadata) throws IOException {
    try (ReadableByteChannel channel = FileSystems.open(metadata.resourceId());
        JsonReader reader =
            new JsonReader(
                new InputStreamReader(Channels.newInputStream(channel), StandardCharsets.UTF_8))) {
      return GSON.fromJson(reader, CorpusIndex.class);
    }
  }
}
