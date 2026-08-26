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
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.apache.avro.SchemaBuilder;
import org.apache.avro.generic.GenericData;
import org.apache.beam.sdk.io.iceberg.SchemaEvolutionConfig.IncompatibleSchemaHandling;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.iceberg.Schema;
import org.apache.iceberg.Table;
import org.apache.iceberg.TableProperties;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.hadoop.HadoopCatalog;
import org.apache.iceberg.mapping.MappingUtil;
import org.apache.iceberg.mapping.NameMappingParser;
import org.apache.iceberg.types.Type;
import org.apache.iceberg.types.Types;
import org.apache.parquet.avro.AvroParquetWriter;
import org.apache.parquet.hadoop.ParquetWriter;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Seeded scenario generator for end-to-end AddFiles fuzzing. See AddFilesFuzzPlan.md. */
final class AddFilesFuzz {
  private AddFilesFuzz() {}

  enum TableState {
    MISSING,
    NARROW,
    EXACT,
    WIDER,
    REQUIRED_SOME,
    ZERO_COLUMNS,
    CUSTOM_MAPPING,
    MALFORMED_MAPPING
  }

  /** Something happening to the table during the run, injected through the schema committer. */
  enum Disturbance {
    NONE,
    /** Another writer adds a column right after our schema commit lands. */
    EXTERNAL_COLUMN_AFTER_COMMIT,
    /** Our first commit fails because another writer committed first; the retry must succeed. */
    COMMIT_FAILS_ONCE,
    /** The mapping property is overwritten with garbage after our commit. */
    MAPPING_OVERWRITTEN_AFTER_COMMIT,
    /** The table is dropped after our commit; registration must not corrupt anything. */
    TABLE_DROPPED_AFTER_COMMIT
  }

  static final String EXTERNAL_COLUMN = "external_col";

  enum FileKind {
    NORMAL,
    ZERO_ROWS,
    GARBAGE,
    TRUNCATED,
    MISSING,
    ZERO_BYTES,
    AVRO_EXTENSION
  }

  /** The column pool, canonical names. Nested struct exercises paths. */
  static final List<String> POOL = Arrays.asList("id", "name", "age", "score", "flag", "address");

  static final class FileSpec {
    final String path;
    final FileKind kind;
    /** Canonical column names present in the file, in file order. */
    final List<String> columns;
    /** Canonical name to name written in the file (aliases). */
    final Map<String, String> writtenNames;
    /** Type variant per column: "int" or "long" for age, "float" or "double" for score. */
    final Map<String, String> variants;
    /** Canonical column to whether it is declared optional in the file. */
    final Map<String, Boolean> optional;
    /** Rows as canonical column to value (null for a null). */
    final List<Map<String, @Nullable Object>> rows;

    FileSpec(
        String path,
        FileKind kind,
        List<String> columns,
        Map<String, String> writtenNames,
        Map<String, String> variants,
        Map<String, Boolean> optional,
        List<Map<String, @Nullable Object>> rows) {
      this.path = path;
      this.kind = kind;
      this.columns = columns;
      this.writtenNames = writtenNames;
      this.variants = variants;
      this.optional = optional;
      this.rows = rows;
    }

    boolean readable() {
      return kind == FileKind.NORMAL || kind == FileKind.ZERO_ROWS;
    }

    long nonNullCount(String column) {
      long count = 0;
      for (Map<String, @Nullable Object> row : rows) {
        if (row.get(column) != null) {
          count++;
        }
      }
      return count;
    }

    @Override
    public String toString() {
      return kind
          + " "
          + path
          + " columns="
          + columns
          + " written="
          + writtenNames
          + " variants="
          + variants
          + " optional="
          + optional
          + " rows="
          + rows.size();
    }
  }

  static final class Scenario {
    final long seed;
    final TableState state;
    final SchemaEvolutionConfig config;
    final List<FileSpec> files;
    final TableIdentifier tableId;
    final String warehouse;
    final @Nullable String initialMappingJson;
    final Disturbance disturbance;

    Scenario(
        long seed,
        TableState state,
        SchemaEvolutionConfig config,
        List<FileSpec> files,
        TableIdentifier tableId,
        String warehouse,
        @Nullable String initialMappingJson,
        Disturbance disturbance) {
      this.seed = seed;
      this.state = state;
      this.config = config;
      this.files = files;
      this.tableId = tableId;
      this.warehouse = warehouse;
      this.initialMappingJson = initialMappingJson;
      this.disturbance = disturbance;
    }

    List<String> paths() {
      List<String> paths = new ArrayList<>();
      for (FileSpec file : files) {
        paths.add(file.path);
      }
      return paths;
    }

    HadoopCatalog catalog() {
      return new HadoopCatalog(new Configuration(), warehouse);
    }

    @Override
    public String toString() {
      StringBuilder sb =
          new StringBuilder()
              .append("seed=")
              .append(seed)
              .append(" state=")
              .append(state)
              .append(" options=")
              .append(config.getOptions())
              .append(" pins=")
              .append(config.getRequiredColumns())
              .append(" aliases=")
              .append(config.getColumnAliases())
              .append(" ignored=")
              .append(config.getIgnoredColumns())
              .append(" handling=")
              .append(config.getIncompatibleSchemaHandling())
              .append(" initialMapping=")
              .append(initialMappingJson)
              .append(" disturbance=")
              .append(disturbance)
              .append('\n');
      for (FileSpec file : files) {
        sb.append("  ").append(file).append('\n');
      }
      return sb.toString();
    }
  }

  // ---- generation

  static Scenario generate(long seed, File dir) throws IOException {
    Random rnd = new Random(seed);
    TableState state = TableState.values()[rnd.nextInt(TableState.values().length)];

    // config
    Set<SchemaEvolutionOption> options = EnumSet.noneOf(SchemaEvolutionOption.class);
    for (SchemaEvolutionOption option : SchemaEvolutionOption.values()) {
      if (rnd.nextBoolean()) {
        options.add(option);
      }
    }
    if (options.isEmpty() && rnd.nextInt(4) != 0) {
      options.add(SchemaEvolutionOption.ALLOW_FIELD_ADDITION);
    }
    boolean enabled = !options.isEmpty();
    Set<String> pins = new LinkedHashSet<>();
    Map<String, String> aliases = new LinkedHashMap<>();
    Set<String> ignored = new LinkedHashSet<>();
    @Nullable IncompatibleSchemaHandling handling = null;
    if (enabled) {
      if (state == TableState.REQUIRED_SOME && rnd.nextBoolean()) {
        pins.add(rnd.nextBoolean() ? "id" : "name");
      }
      if (rnd.nextInt(3) != 0) {
        String canonical = pick(rnd, Arrays.asList("name", "age", "score", "address.zip"));
        if (!pins.contains(canonical)) {
          aliases.put(aliasName(canonical), canonical);
        }
      }
      if (rnd.nextInt(3) == 0) {
        String candidate = pick(rnd, Arrays.asList("flag", "score", "address.city"));
        boolean clash = pins.contains(candidate) || aliases.containsValue(candidate);
        if (!clash) {
          ignored.add(candidate);
        }
      }
      int h = rnd.nextInt(3);
      handling =
          h == 0
              ? null
              : h == 1
                  ? IncompatibleSchemaHandling.FAIL_PIPELINE
                  : IncompatibleSchemaHandling.ROUTE_TO_ERRORS;
    }
    SchemaEvolutionConfig config =
        SchemaEvolutionConfig.builder()
            .setOptions(options)
            .setRequiredColumns(pins)
            .withColumnAliases(aliases)
            .setIgnoredColumns(ignored)
            .setIncompatibleSchemaHandling(handling)
            .build();

    // table
    String warehouse = "file:" + new File(dir, "warehouse").getAbsolutePath();
    TableIdentifier tableId = TableIdentifier.of("default", "t" + seed);
    @Nullable String initialMappingJson = createTable(state, tableId, warehouse);

    // files
    File filesDir = new File(dir, "files");
    filesDir.mkdirs();
    int numFiles = 1 + rnd.nextInt(6);
    List<FileSpec> files = new ArrayList<>();
    for (int i = 0; i < numFiles; i++) {
      files.add(generateFile(rnd, i, filesDir, aliases));
    }
    Disturbance disturbance = Disturbance.NONE;
    if (enabled && rnd.nextInt(3) == 0) {
      Disturbance[] all = Disturbance.values();
      disturbance = all[1 + rnd.nextInt(all.length - 1)];
    }
    return new Scenario(
        seed, state, config, files, tableId, warehouse, initialMappingJson, disturbance);
  }

  private static String aliasName(String canonical) {
    int dot = canonical.lastIndexOf('.');
    return canonical.substring(0, dot + 1) + canonical.substring(dot + 1) + "_alias";
  }

  private static <T> T pick(Random rnd, List<T> from) {
    return from.get(rnd.nextInt(from.size()));
  }

  static final Types.StructType ADDRESS =
      Types.StructType.of(
          Types.NestedField.optional(101, "city", Types.StringType.get()),
          Types.NestedField.optional(102, "zip", Types.IntegerType.get()));

  private static Schema tableSchema(TableState state) {
    List<Types.NestedField> fields = new ArrayList<>();
    switch (state) {
      case NARROW:
        fields.add(Types.NestedField.optional(1, "id", Types.LongType.get()));
        fields.add(Types.NestedField.optional(2, "name", Types.StringType.get()));
        break;
      case REQUIRED_SOME:
        fields.add(Types.NestedField.required(1, "id", Types.LongType.get()));
        fields.add(Types.NestedField.required(2, "name", Types.StringType.get()));
        fields.add(Types.NestedField.optional(3, "age", Types.IntegerType.get()));
        fields.add(Types.NestedField.optional(4, "score", Types.DoubleType.get()));
        fields.add(Types.NestedField.optional(5, "flag", Types.BooleanType.get()));
        fields.add(Types.NestedField.optional(6, "address", ADDRESS));
        break;
      case ZERO_COLUMNS:
        break;
      case WIDER:
        fields.add(Types.NestedField.optional(1, "id", Types.LongType.get()));
        fields.add(Types.NestedField.optional(2, "name", Types.StringType.get()));
        fields.add(Types.NestedField.optional(3, "age", Types.IntegerType.get()));
        fields.add(Types.NestedField.optional(4, "score", Types.DoubleType.get()));
        fields.add(Types.NestedField.optional(5, "flag", Types.BooleanType.get()));
        fields.add(Types.NestedField.optional(6, "address", ADDRESS));
        fields.add(Types.NestedField.optional(7, "extra", Types.StringType.get()));
        break;
      default:
        fields.add(Types.NestedField.optional(1, "id", Types.LongType.get()));
        fields.add(Types.NestedField.optional(2, "name", Types.StringType.get()));
        fields.add(Types.NestedField.optional(3, "age", Types.IntegerType.get()));
        fields.add(Types.NestedField.optional(4, "score", Types.DoubleType.get()));
        fields.add(Types.NestedField.optional(5, "flag", Types.BooleanType.get()));
        fields.add(Types.NestedField.optional(6, "address", ADDRESS));
    }
    return new Schema(fields);
  }

  private static @Nullable String createTable(
      TableState state, TableIdentifier tableId, String warehouse) {
    if (state == TableState.MISSING) {
      return null;
    }
    HadoopCatalog catalog = new HadoopCatalog(new Configuration(), warehouse);
    Table table = catalog.createTable(tableId, tableSchema(state));
    @Nullable String mapping = null;
    if (state == TableState.CUSTOM_MAPPING) {
      String generated = NameMappingParser.toJson(MappingUtil.create(table.schema()));
      mapping = generated.replace("\"names\" : [ \"id\" ]", "\"names\" : [ \"id\", \"ident\" ]");
      if (mapping.equals(generated)) {
        mapping = generated.replace("\"names\":[\"id\"]", "\"names\":[\"id\",\"ident\"]");
      }
    } else if (state == TableState.MALFORMED_MAPPING) {
      mapping = "{not json";
    }
    if (mapping != null) {
      table.updateProperties().set(TableProperties.DEFAULT_NAME_MAPPING, mapping).commit();
    }
    return mapping;
  }

  private static FileSpec generateFile(Random rnd, int index, File dir, Map<String, String> aliases)
      throws IOException {
    FileKind kind = FileKind.NORMAL;
    int roll = rnd.nextInt(20);
    if (roll < 1) {
      kind = FileKind.ZERO_ROWS;
    } else if (roll < 2) {
      kind = FileKind.GARBAGE;
    } else if (roll < 3) {
      kind = FileKind.TRUNCATED;
    } else if (roll < 4) {
      kind = FileKind.MISSING;
    } else if (roll < 5) {
      kind = FileKind.ZERO_BYTES;
    } else if (roll < 6) {
      kind = FileKind.AVRO_EXTENSION;
    }

    List<String> columns = new ArrayList<>();
    for (String column : POOL) {
      if (rnd.nextInt(4) != 0) {
        columns.add(column);
      }
    }
    if (columns.isEmpty()) {
      columns.add("id");
    }
    Map<String, String> variants = new HashMap<>();
    variants.put("age", rnd.nextBoolean() ? "int" : "long");
    variants.put("score", rnd.nextBoolean() ? "float" : "double");
    Map<String, Boolean> optional = new HashMap<>();
    for (String column : columns) {
      optional.put(column, rnd.nextInt(3) != 0);
    }
    Map<String, String> written = new LinkedHashMap<>();
    for (Map.Entry<String, String> alias : aliases.entrySet()) {
      String canonical = alias.getValue();
      String top =
          canonical.contains(".") ? canonical.substring(0, canonical.indexOf('.')) : canonical;
      if (columns.contains(top) && rnd.nextInt(3) != 0) {
        String aliasLeaf = alias.getKey().substring(alias.getKey().lastIndexOf('.') + 1);
        written.put(canonical, aliasLeaf);
      }
    }

    int rows = kind == FileKind.ZERO_ROWS ? 0 : 1 + rnd.nextInt(5);
    List<Map<String, @Nullable Object>> data = new ArrayList<>();
    for (int r = 0; r < rows; r++) {
      Map<String, @Nullable Object> row = new LinkedHashMap<>();
      for (String column : columns) {
        boolean isNull = optional.get(column) && rnd.nextInt(3) == 0;
        row.put(column, isNull ? null : value(column, r, rnd));
      }
      data.add(row);
    }

    String extension = kind == FileKind.AVRO_EXTENSION ? ".avro" : ".parquet";
    File file = new File(dir, "f" + index + extension);
    FileSpec spec =
        new FileSpec(file.getAbsolutePath(), kind, columns, written, variants, optional, data);
    write(spec, file);
    return spec;
  }

  private static Object value(String column, int r, Random rnd) {
    switch (column) {
      case "id":
        return (long) (r + rnd.nextInt(1000));
      case "name":
        return "n" + r;
      case "age":
        return r + 20;
      case "score":
        return r * 1.5;
      case "flag":
        return r % 2 == 0;
      case "address":
        Map<String, Object> address = new LinkedHashMap<>();
        address.put("city", "c" + r);
        address.put("zip", 1000 + r);
        return address;
      default:
        throw new IllegalArgumentException(column);
    }
  }

  // ---- writing (parquet-avro: no embedded field ids, like most producers)

  private static void write(FileSpec spec, File file) throws IOException {
    if (spec.kind == FileKind.MISSING) {
      return;
    }
    if (spec.kind == FileKind.ZERO_BYTES) {
      Files.write(file.toPath(), new byte[0]);
      return;
    }
    if (spec.kind == FileKind.GARBAGE) {
      Files.write(file.toPath(), "not a parquet file".getBytes(StandardCharsets.UTF_8));
      return;
    }
    File parquet = spec.kind == FileKind.TRUNCATED ? new File(file.getPath() + ".full") : file;
    org.apache.avro.Schema avro = avroSchema(spec);
    try (ParquetWriter<Object> writer =
        AvroParquetWriter.builder(new Path(parquet.getAbsolutePath())).withSchema(avro).build()) {
      for (Map<String, @Nullable Object> row : spec.rows) {
        writer.write(record(avro, spec, row));
      }
    }
    if (spec.kind == FileKind.TRUNCATED) {
      byte[] bytes = Files.readAllBytes(parquet.toPath());
      Files.write(file.toPath(), Arrays.copyOf(bytes, bytes.length / 2));
      parquet.delete();
    }
  }

  private static org.apache.avro.Schema avroSchema(FileSpec spec) {
    SchemaBuilder.FieldAssembler<org.apache.avro.Schema> fields =
        SchemaBuilder.record("r").fields();
    for (String column : spec.columns) {
      String name = spec.writtenNames.getOrDefault(column, column);
      boolean optional = spec.optional.get(column);
      org.apache.avro.Schema type = avroType(column, spec);
      if (optional) {
        fields =
            fields.name(name).type().unionOf().nullType().and().type(type).endUnion().noDefault();
      } else {
        fields = fields.name(name).type(type).noDefault();
      }
    }
    return fields.endRecord();
  }

  private static org.apache.avro.Schema avroType(String column, FileSpec spec) {
    switch (column) {
      case "id":
        return org.apache.avro.Schema.create(org.apache.avro.Schema.Type.LONG);
      case "name":
        return org.apache.avro.Schema.create(org.apache.avro.Schema.Type.STRING);
      case "age":
        return org.apache.avro.Schema.create(
            spec.variants.get("age").equals("long")
                ? org.apache.avro.Schema.Type.LONG
                : org.apache.avro.Schema.Type.INT);
      case "score":
        return org.apache.avro.Schema.create(
            spec.variants.get("score").equals("double")
                ? org.apache.avro.Schema.Type.DOUBLE
                : org.apache.avro.Schema.Type.FLOAT);
      case "flag":
        return org.apache.avro.Schema.create(org.apache.avro.Schema.Type.BOOLEAN);
      case "address":
        String city = spec.writtenNames.getOrDefault("address.city", "city");
        String zip = spec.writtenNames.getOrDefault("address.zip", "zip");
        return SchemaBuilder.record("address")
            .fields()
            .optionalString(city)
            .optionalInt(zip)
            .endRecord();
      default:
        throw new IllegalArgumentException(column);
    }
  }

  private static GenericData.Record record(
      org.apache.avro.Schema avro, FileSpec spec, Map<String, @Nullable Object> row) {
    GenericData.Record record = new GenericData.Record(avro);
    for (String column : spec.columns) {
      String name = spec.writtenNames.getOrDefault(column, column);
      @Nullable Object value = row.get(column);
      if (value == null) {
        continue;
      }
      switch (column) {
        case "age":
          record.put(
              name,
              spec.variants.get("age").equals("long") ? ((Integer) value).longValue() : value);
          break;
        case "score":
          record.put(
              name,
              spec.variants.get("score").equals("float") ? ((Double) value).floatValue() : value);
          break;
        case "address":
          org.apache.avro.Schema addressSchema = nonNull(avro.getField(name).schema());
          GenericData.Record address = new GenericData.Record(addressSchema);
          @SuppressWarnings("unchecked")
          Map<String, Object> values = (Map<String, Object>) value;
          address.put(spec.writtenNames.getOrDefault("address.city", "city"), values.get("city"));
          address.put(spec.writtenNames.getOrDefault("address.zip", "zip"), values.get("zip"));
          record.put(name, address);
          break;
        default:
          record.put(name, value);
      }
    }
    return record;
  }

  private static org.apache.avro.Schema nonNull(org.apache.avro.Schema schema) {
    if (schema.getType() == org.apache.avro.Schema.Type.UNION) {
      for (org.apache.avro.Schema branch : schema.getTypes()) {
        if (branch.getType() != org.apache.avro.Schema.Type.NULL) {
          return branch;
        }
      }
    }
    return schema;
  }

  /** Iceberg type of a canonical column as the file declares it. */
  static Type fileType(String column, FileSpec spec) {
    switch (column) {
      case "id":
        return Types.LongType.get();
      case "name":
        return Types.StringType.get();
      case "age":
        return spec.variants.get("age").equals("long")
            ? Types.LongType.get()
            : Types.IntegerType.get();
      case "score":
        return spec.variants.get("score").equals("double")
            ? Types.DoubleType.get()
            : Types.FloatType.get();
      case "flag":
        return Types.BooleanType.get();
      case "address":
        return ADDRESS;
      default:
        throw new IllegalArgumentException(column);
    }
  }
}
