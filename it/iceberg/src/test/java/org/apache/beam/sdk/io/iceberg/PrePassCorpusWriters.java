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

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.zip.GZIPOutputStream;
import org.apache.avro.generic.GenericData;
import org.apache.beam.sdk.io.FileSystems;
import org.apache.beam.sdk.io.fs.ResourceId;
import org.apache.beam.sdk.util.MimeTypes;
import org.apache.hadoop.fs.Path;
import org.apache.iceberg.PartitionSpec;
import org.apache.iceberg.Schema;
import org.apache.iceberg.data.GenericRecord;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.data.parquet.GenericParquetWriter;
import org.apache.iceberg.io.DataWriter;
import org.apache.iceberg.parquet.Parquet;
import org.apache.iceberg.parquet.ParquetSchemaUtil;
import org.apache.iceberg.types.Types;
import org.apache.parquet.avro.AvroParquetWriter;
import org.apache.parquet.avro.AvroSchemaConverter;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.io.api.Binary;
import org.apache.parquet.schema.GroupType;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType;
import org.apache.parquet.schema.Type;

/** Writes one corpus file per {@link PrePassCorpus.FileSpec}. */
final class PrePassCorpusWriters {
  private PrePassCorpusWriters() {}

  /** Writes the spec's file under {@code targetDir} and returns its index entry. */
  static PrePassCorpus.CorpusIndex.Entry write(PrePassCorpus.FileSpec spec, String targetDir)
      throws IOException {
    PrePassCorpus.CorpusIndex.Entry entry = new PrePassCorpus.CorpusIndex.Entry();
    entry.specId = spec.id;
    entry.schemaKey = spec.schemaKey;
    entry.expected = spec.expectedOutcome();
    entry.path = join(targetDir, spec.fileName());

    if (spec.adversarial == PrePassCorpus.Adversarial.MISSING_PATH) {
      return entry;
    }
    if (spec.adversarial == PrePassCorpus.Adversarial.DIRECTORY_PATH) {
      writeBytes(join(entry.path, "marker"), new byte[0]);
      return entry;
    }
    byte[] bytes;
    try {
      bytes = produce(spec);
    } catch (Exception e) {
      entry.writeFailure = e.toString();
      return entry;
    }
    writeBytes(entry.path, corrupt(bytes, spec.adversarial));
    return entry;
  }

  private static byte[] produce(PrePassCorpus.FileSpec spec) throws IOException {
    File local = File.createTempFile("corpus", ".parquet");
    local.delete();
    try {
      MessageType messageType = spec.toMessageType();
      switch (spec.producer) {
        case PARQUET_MR:
          writeParquetMr(local, messageType, spec.footer);
          break;
        case ICEBERG:
          writeIceberg(local, messageType, spec.footer);
          break;
        case PARQUET_AVRO:
          writeParquetAvro(local, messageType, spec.footer);
          break;
        default:
          throw new IllegalArgumentException(spec.producer.name());
      }
      return Files.readAllBytes(local.toPath());
    } finally {
      local.delete();
    }
  }

  // ---- parquet-mr

  private static void writeParquetMr(
      File file, MessageType messageType, PrePassCorpus.FooterSpec footer) throws IOException {
    ExampleParquetWriter.Builder builder =
        ExampleParquetWriter.builder(new Path(file.getAbsolutePath()))
            .withType(messageType)
            .withStatisticsEnabled(footer.stats);
    if (footer.rowGroups > 1) {
      int rowsPerGroup = Math.max(1, footer.rows / footer.rowGroups);
      builder =
          builder
              .withRowGroupSize(1L)
              .withMinRowCountForPageSizeCheck(rowsPerGroup)
              .withMaxRowCountForPageSizeCheck(rowsPerGroup);
    }
    Random random = new Random(0);
    SimpleGroupFactory factory = new SimpleGroupFactory(messageType);
    try (ParquetWriter<Group> writer = builder.build()) {
      for (int row = 0; row < footer.rows; row++) {
        Group group = factory.newGroup();
        fill(group, messageType, row, footer.nullFraction, random);
        writer.write(group);
      }
    }
  }

  private static void fill(
      Group group, GroupType type, int row, double nullFraction, Random random) {
    for (Type field : type.getFields()) {
      int copies = 1;
      if (field.isRepetition(Type.Repetition.REPEATED)) {
        copies = 2;
      } else if (field.isRepetition(Type.Repetition.OPTIONAL)
          && random.nextDouble() < nullFraction) {
        continue;
      }
      for (int i = 0; i < copies; i++) {
        if (field.isPrimitive()) {
          addPrimitive(group, field.asPrimitiveType(), row + i);
        } else {
          fill(group.addGroup(field.getName()), field.asGroupType(), row + i, nullFraction, random);
        }
      }
    }
  }

  private static void addPrimitive(Group group, PrimitiveType type, int value) {
    String name = type.getName();
    switch (type.getPrimitiveTypeName()) {
      case BOOLEAN:
        group.add(name, value % 2 == 0);
        break;
      case INT32:
        group.add(name, value);
        break;
      case INT64:
        group.add(name, (long) value);
        break;
      case INT96:
        group.add(name, Binary.fromConstantByteArray(new byte[12]));
        break;
      case FLOAT:
        group.add(name, (float) value);
        break;
      case DOUBLE:
        group.add(name, (double) value);
        break;
      case BINARY:
        group.add(name, Binary.fromString("v" + value));
        break;
      case FIXED_LEN_BYTE_ARRAY:
        group.add(name, Binary.fromConstantByteArray(new byte[type.getTypeLength()]));
        break;
      default:
        throw new IllegalArgumentException(type.toString());
    }
  }

  // ---- iceberg (embedded field ids)

  private static void writeIceberg(
      File file, MessageType messageType, PrePassCorpus.FooterSpec footer) throws IOException {
    Schema schema = ParquetSchemaUtil.convert(messageType);
    DataWriter<Record> writer =
        Parquet.writeData(org.apache.iceberg.Files.localOutput(file))
            .schema(schema)
            .withSpec(PartitionSpec.unpartitioned())
            .createWriterFunc(GenericParquetWriter::create)
            .build();
    try {
      for (int row = 0; row < footer.rows; row++) {
        writer.write(icebergRecord(schema.asStruct(), row));
      }
    } finally {
      writer.close();
    }
  }

  private static Record icebergRecord(Types.StructType struct, int row) {
    Record record = GenericRecord.create(struct);
    for (Types.NestedField field : struct.fields()) {
      record.setField(field.name(), icebergValue(field.type(), row));
    }
    return record;
  }

  private static Object icebergValue(org.apache.iceberg.types.Type type, int row) {
    if (type.isStructType()) {
      return icebergRecord(type.asStructType(), row);
    }
    if (type.isListType()) {
      return Arrays.asList(icebergValue(type.asListType().elementType(), row));
    }
    if (type.isMapType()) {
      Map<Object, Object> map = new HashMap<>();
      map.put(
          icebergValue(type.asMapType().keyType(), row),
          icebergValue(type.asMapType().valueType(), row));
      return map;
    }
    switch (type.typeId()) {
      case BOOLEAN:
        return row % 2 == 0;
      case INTEGER:
        return row;
      case LONG:
        return (long) row;
      case FLOAT:
        return (float) row;
      case DOUBLE:
        return (double) row;
      case STRING:
        return "v" + row;
      default:
        throw new IllegalArgumentException("Unsupported in the iceberg producer: " + type);
    }
  }

  // ---- parquet-avro

  private static void writeParquetAvro(
      File file, MessageType messageType, PrePassCorpus.FooterSpec footer) throws IOException {
    org.apache.avro.Schema avroSchema = new AvroSchemaConverter().convert(messageType);
    try (ParquetWriter<Object> writer =
        AvroParquetWriter.builder(new Path(file.getAbsolutePath()))
            .withSchema(avroSchema)
            .build()) {
      for (int row = 0; row < footer.rows; row++) {
        writer.write(avroRecord(avroSchema, row));
      }
    }
  }

  private static GenericData.Record avroRecord(org.apache.avro.Schema schema, int row) {
    GenericData.Record record = new GenericData.Record(schema);
    for (org.apache.avro.Schema.Field field : schema.getFields()) {
      record.put(field.name(), avroValue(field.schema(), row));
    }
    return record;
  }

  private static Object avroValue(org.apache.avro.Schema schema, int row) {
    switch (schema.getType()) {
      case RECORD:
        return avroRecord(schema, row);
      case UNION:
        for (org.apache.avro.Schema branch : schema.getTypes()) {
          if (branch.getType() != org.apache.avro.Schema.Type.NULL) {
            return avroValue(branch, row);
          }
        }
        return null;
      case ARRAY:
        List<Object> list = new ArrayList<>();
        list.add(avroValue(schema.getElementType(), row));
        return list;
      case MAP:
        Map<String, Object> map = new HashMap<>();
        map.put("k" + row, avroValue(schema.getValueType(), row));
        return map;
      case BOOLEAN:
        return row % 2 == 0;
      case INT:
        return row;
      case LONG:
        return (long) row;
      case FLOAT:
        return (float) row;
      case DOUBLE:
        return (double) row;
      case STRING:
        return "v" + row;
      case BYTES:
        return ByteBuffer.wrap(new byte[] {(byte) row});
      case FIXED:
        return new GenericData.Fixed(schema, new byte[schema.getFixedSize()]);
      default:
        throw new IllegalArgumentException("Unsupported in the avro producer: " + schema);
    }
  }

  // ---- adversarial post-processing

  static byte[] corrupt(byte[] bytes, PrePassCorpus.Adversarial adversarial) throws IOException {
    switch (adversarial) {
      case ZERO_BYTES:
        return new byte[0];
      case TRUNCATED_HALF:
        return Arrays.copyOf(bytes, bytes.length / 2);
      case TRUNCATED_TAIL8:
        return Arrays.copyOf(bytes, Math.max(0, bytes.length - 8));
      case TRUNCATED_LAST_BYTE:
        return Arrays.copyOf(bytes, Math.max(0, bytes.length - 1));
      case GARBAGE_TEXT:
        return "this is not a parquet file".getBytes(StandardCharsets.UTF_8);
      case ORC_MAGIC:
        return withMagic("ORC", bytes);
      case AVRO_MAGIC:
        return withMagic("Obj", bytes);
      case GZIPPED:
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (GZIPOutputStream gzip = new GZIPOutputStream(out)) {
          gzip.write(bytes);
        }
        return out.toByteArray();
      default:
        return bytes;
    }
  }

  /** Replaces the leading PAR1 magic and drops the trailing one, so only the new magic remains. */
  private static byte[] withMagic(String magic, byte[] bytes) {
    byte[] head = magic.getBytes(StandardCharsets.ISO_8859_1);
    byte[] body = Arrays.copyOfRange(bytes, 4, Math.max(4, bytes.length - 4));
    byte[] result = new byte[head.length + body.length];
    System.arraycopy(head, 0, result, 0, head.length);
    System.arraycopy(body, 0, result, head.length, body.length);
    return result;
  }

  // ---- output

  private static void writeBytes(String path, byte[] bytes) throws IOException {
    ResourceId resource = FileSystems.matchNewResource(path, false);
    try (WritableByteChannel channel = FileSystems.create(resource, MimeTypes.BINARY)) {
      channel.write(ByteBuffer.wrap(bytes));
    }
  }

  private static String join(String dir, String name) {
    if (dir.endsWith("/")) {
      return dir + name;
    }
    return dir + "/" + name;
  }
}
