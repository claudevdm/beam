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

import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.apache.hadoop.fs.Path;
import org.apache.iceberg.Schema;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.parquet.ParquetSchemaUtil;
import org.apache.iceberg.types.Type;
import org.apache.iceberg.types.Types.NestedField;
import org.apache.iceberg.types.Types.StructType;
import org.apache.parquet.example.data.Group;
import org.apache.parquet.example.data.simple.SimpleGroupFactory;
import org.apache.parquet.hadoop.ParquetWriter;
import org.apache.parquet.hadoop.example.ExampleParquetWriter;
import org.apache.parquet.schema.GroupType;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.PrimitiveType;

/**
 * Writes Iceberg {@link Record}s as Parquet without embedded field ids, the shape real zero-copy
 * files have. The Parquet structure is exactly what Iceberg's writer would produce (three-level
 * lists with {@code element}, maps with {@code key_value}) minus the ids, so schemas read back
 * identically; files written with ids do not drive schema evolution and must match the table's ids
 * exactly to register.
 */
final class IdLessParquet {
  private IdLessParquet() {}

  static String write(String path, Schema schema, Record... records) throws IOException {
    MessageType withIds = ParquetSchemaUtil.convert(schema, "root");
    MessageType idLess = stripIds(withIds);
    SimpleGroupFactory factory = new SimpleGroupFactory(idLess);
    try (ParquetWriter<Group> writer =
        ExampleParquetWriter.builder(new Path(path)).withType(idLess).build()) {
      for (Record record : records) {
        Group group = factory.newGroup();
        fillStruct(group, schema.asStruct(), record);
        writer.write(group);
      }
    }
    return path;
  }

  private static MessageType stripIds(MessageType type) {
    org.apache.parquet.schema.Types.MessageTypeBuilder builder =
        org.apache.parquet.schema.Types.buildMessage();
    for (org.apache.parquet.schema.Type field : type.getFields()) {
      builder.addField(stripIds(field));
    }
    return builder.named(type.getName());
  }

  private static org.apache.parquet.schema.Type stripIds(org.apache.parquet.schema.Type type) {
    if (type.isPrimitive()) {
      PrimitiveType primitive = type.asPrimitiveType();
      org.apache.parquet.schema.Types.PrimitiveBuilder<PrimitiveType> builder =
          org.apache.parquet.schema.Types.primitive(
              primitive.getPrimitiveTypeName(), primitive.getRepetition());
      if (primitive.getPrimitiveTypeName()
          == PrimitiveType.PrimitiveTypeName.FIXED_LEN_BYTE_ARRAY) {
        builder = builder.length(primitive.getTypeLength());
      }
      if (primitive.getLogicalTypeAnnotation() != null) {
        builder = builder.as(primitive.getLogicalTypeAnnotation());
      }
      return builder.named(primitive.getName());
    }
    GroupType group = type.asGroupType();
    org.apache.parquet.schema.Types.GroupBuilder<GroupType> builder =
        org.apache.parquet.schema.Types.buildGroup(group.getRepetition());
    if (group.getLogicalTypeAnnotation() != null) {
      builder = builder.as(group.getLogicalTypeAnnotation());
    }
    for (org.apache.parquet.schema.Type field : group.getFields()) {
      builder = builder.addField(stripIds(field));
    }
    return builder.named(group.getName());
  }

  private static void fillStruct(Group group, StructType struct, Record record) {
    for (int i = 0; i < struct.fields().size(); i++) {
      NestedField field = struct.fields().get(i);
      Object value = record.get(i);
      if (value == null) {
        continue;
      }
      fillValue(group, field.name(), field.type(), value);
    }
  }

  @SuppressWarnings("unchecked")
  private static void fillValue(Group group, String name, Type type, Object value) {
    if (type.isStructType()) {
      fillStruct(group.addGroup(name), type.asStructType(), (Record) value);
      return;
    }
    if (type.isListType()) {
      Group listGroup = group.addGroup(name);
      for (Object element : (List<Object>) value) {
        Group item = listGroup.addGroup("list");
        if (element != null) {
          fillValue(item, "element", type.asListType().elementType(), element);
        }
      }
      return;
    }
    if (type.isMapType()) {
      Group mapGroup = group.addGroup(name);
      for (Map.Entry<Object, Object> entry : ((Map<Object, Object>) value).entrySet()) {
        Group keyValue = mapGroup.addGroup("key_value");
        fillValue(keyValue, "key", type.asMapType().keyType(), entry.getKey());
        if (entry.getValue() != null) {
          fillValue(keyValue, "value", type.asMapType().valueType(), entry.getValue());
        }
      }
      return;
    }
    if (value instanceof Integer) {
      group.add(name, (Integer) value);
    } else if (value instanceof Long) {
      group.add(name, (Long) value);
    } else if (value instanceof Double) {
      group.add(name, (Double) value);
    } else if (value instanceof Float) {
      group.add(name, (Float) value);
    } else if (value instanceof Boolean) {
      group.add(name, (Boolean) value);
    } else if (value instanceof CharSequence) {
      group.add(name, value.toString());
    } else {
      throw new IllegalArgumentException(
          "IdLessParquet does not know how to write " + value.getClass() + " for column " + name);
    }
  }
}
