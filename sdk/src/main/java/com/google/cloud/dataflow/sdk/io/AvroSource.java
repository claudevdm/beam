/*
 * Copyright (C) 2015 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */

package com.google.cloud.dataflow.sdk.io;

import com.google.cloud.dataflow.sdk.annotations.Experimental;
import com.google.cloud.dataflow.sdk.coders.AvroCoder;
import com.google.cloud.dataflow.sdk.options.PipelineOptions;
import com.google.cloud.dataflow.sdk.util.IOChannelUtils;
import com.google.cloud.dataflow.sdk.values.PCollection;

import org.apache.avro.Schema;
import org.apache.avro.file.DataFileConstants;
import org.apache.avro.generic.GenericRecord;
import org.apache.avro.io.BinaryDecoder;
import org.apache.avro.io.DatumReader;
import org.apache.avro.io.DecoderFactory;
import org.apache.avro.reflect.ReflectData;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.compress.compressors.snappy.SnappyCompressorInputStream;
import org.apache.commons.compress.compressors.xz.XZCompressorInputStream;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PushbackInputStream;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.util.Arrays;
import java.util.zip.Inflater;
import java.util.zip.InflaterInputStream;

// JAVADOCSTYLE OFF
/**
 * A {@code FileBasedSource} for reading Avro-format files.
 *
 * <p>To read a {@link PCollection} of objects from one or more Avro files, use
 * {@link AvroSource#from} to specify the path(s) of the files to read. The {@code AvroSource} that
 * is returned will read objects of type {@code GenericRecord} with the schema(s) that were written
 * at file creation. To further configure the {@code AvroSource} to read with a user-defined schema,
 * or to return records of a type other than {@code GenericRecord}, use
 * {@link AvroSource#withSchema(Schema)} (using a {@code Schema} object),
 * {@link AvroSource#withSchema(String)} (using a JSON schema), or
 * {@link AvroSource#withSchema(Class)} (to return objects of the class specified).
 *
 * <p>An {@code AvroSource} can be read from using the {@link Read} transform. For example:
 *
 * <pre>
 * {@code
 * AvroSource<T> source = AvroSource.from(file.toPath()).withSchema(MyType.class);
 * PCollection<MyType> records = Read.from(mySource);
 * }
 * </pre>
 *
 * <p>The {@link AvroSource#readFromFileWithClass(String, Class)} method is a convenience method
 * that returns a read transform. For example:
 *
 * <pre>
 * {@code
 * PCollection<MyType> records = AvroSource.readFromFileWithClass(file.toPath(), MyType.class));
 * }
 * </pre>
 *
 * <p>This class's implementation is based on the <a
 * href="https://avro.apache.org/docs/1.7.7/spec.html">Avro 1.7.7</a> specification and implements
 * parsing of some parts of Avro Object Container Files. The rationale for doing so is that the Avro
 * API does not provide efficient ways of computing the precise offsets of blocks within a file,
 * which is necessary to support dynamic work rebalancing. However, whenever it is possible to use
 * the Avro API in a way that supports maintaining precise offsets, this class uses the Avro API.
 *
 * <p>Avro Object Container files store records in blocks. Each block contains a collection of
 * records. Blocks may be encoded (e.g., with bzip2, deflate, snappy, etc.). Blocks are delineated
 * from one another by a 16-byte sync marker.
 *
 * <p>An {@code AvroSource} for a subrange of a single file contains records in the blocks such that
 * the start offset of the block is greater than or equal to the start offset of the source and less
 * than the end offset of the source.
 *
 * <p>To use XZ-encoded Avro files, please include an explicit dependency on {@code xz-1.5.jar},
 * which has been marked as optional in the Maven {@code sdk/pom.xml} for Google Cloud Dataflow:
 * <pre>
 * {@code
 * <dependency>
 *   <groupId>org.tukaani</groupId>
 *   <artifactId>xz</artifactId>
 *   <version>1.5</version>
 * </dependency>
 * }
 * </pre>
 *
 * @param <T> The type of records to be read from the source.
 *
 * <p><h3>Permissions</h3>
 * Permission requirements depend on the
 * {@link com.google.cloud.dataflow.sdk.runners.PipelineRunner PipelineRunner} that is
 * used to execute the Dataflow job. Please refer to the documentation of corresponding
 * {@code PipelineRunner}s for more details.
 */
// JAVADOCSTYLE ON
@Experimental(Experimental.Kind.SOURCE_SINK)
public class AvroSource<T> extends BlockBasedSource<T> {
  private static final long serialVersionUID = 0;

  // Default minimum bundle size (chosen as two default-size Avro blocks to attempt to
  // ensure that every source has at least one block of records).
  // The default sync interval is 64k.
  static final long DEFAULT_MIN_BUNDLE_SIZE = 2 * DataFileConstants.DEFAULT_SYNC_INTERVAL;

  // The JSON schema used to encode records.
  private final String schema;

  // The type of the records contained in the file.
  private final Class<T> type;

  // The following metadata fields are not user-configurable. They are extracted from the object
  // container file header upon subsource creation.

  // The codec used to encode the blocks in the Avro file. String value drawn from those in
  // https://avro.apache.org/docs/1.7.7/api/java/org/apache/avro/file/CodecFactory.html
  private final String codec;

  // The object container file's 16-byte sync marker.
  private final byte[] syncMarker;

  // Default output coder, lazily initialized.
  private transient AvroCoder<T> coder = null;

  /**
   * Creates a {@code Read} transform that will read from an {@code AvroSource} that is configured
   * to read records of the given type from a file pattern.
   */
  public static <T> Read.Bounded<T> readFromFileWithClass(String filePattern, Class<T> clazz) {
    return Read.from(new AvroSource<T>(filePattern, DEFAULT_MIN_BUNDLE_SIZE,
        ReflectData.get().getSchema(clazz).toString(), clazz, null, null));
  }

  /**
   * Creates an {@code AvroSource} that reads from the given file name or pattern ("glob"). The
   * returned source can be further configured by calling {@code withSchema} to return a type other
   * than {@code GenericRecord}.
   */
  public static AvroSource<GenericRecord> from(String fileNameOrPattern) {
    return new AvroSource<GenericRecord>(
        fileNameOrPattern, DEFAULT_MIN_BUNDLE_SIZE, null, GenericRecord.class, null, null);
  }

  /**
   * Returns an {@code AvroSource} that's like this one but reads files containing records that
   * conform to the given schema.
   */
  public AvroSource<GenericRecord> withSchema(String schema) {
    return new AvroSource<>(
        getFileOrPatternSpec(), getMinBundleSize(), schema, GenericRecord.class, codec, syncMarker);
  }

  /**
   * Returns an {@code AvroSource} that's like this one but reads files containing records that
   * conform to the given schema.
   */
  public AvroSource<GenericRecord> withSchema(Schema schema) {
    return new AvroSource<>(getFileOrPatternSpec(), getMinBundleSize(), schema.toString(),
        GenericRecord.class, codec, syncMarker);
  }

  /**
   * Returns an {@code AvroSource} that's like this one but reads files containing records of the
   * type of the given class.
   */
  public <X> AvroSource<X> withSchema(Class<X> clazz) {
    return new AvroSource<X>(getFileOrPatternSpec(), getMinBundleSize(),
        ReflectData.get().getSchema(clazz).toString(), clazz, codec, syncMarker);
  }

  /**
   * Returns an {@code AvroSource} that's like this one but uses the supplied minimum bundle size.
   * Refer to {@link ByteOffsetBasedSource} for a description of {@code minBundleSize} and its use.
   */
  public AvroSource<T> withMinBundleSize(long minBundleSize) {
    return new AvroSource<T>(
        getFileOrPatternSpec(), minBundleSize, schema, type, codec, syncMarker);
  }

  private AvroSource(String fileNameOrPattern, long minBundleSize, String schema, Class<T> type,
      String codec, byte[] syncMarker) {
    super(fileNameOrPattern, minBundleSize);
    this.schema = schema;
    this.codec = codec;
    this.syncMarker = syncMarker;
    this.type = type;
  }

  private AvroSource(String fileName, long minBundleSize, long startOffset, long endOffset,
      String schema, Class<T> type, String codec, byte[] syncMarker) {
    super(fileName, minBundleSize, startOffset, endOffset);
    this.schema = schema;
    this.codec = codec;
    this.syncMarker = syncMarker;
    this.type = type;
  }

  @Override
  public void validate() {
    // AvroSource objects do not need to be configured with more than a file pattern. Overridden to
    // make this explicit.
    super.validate();
  }

  /**
   * Avro file metadata. Visible for testing.
   */
  static class Metadata {
    byte[] syncMarker;
    String codec;
    String schema;

    public Metadata(byte[] syncMarker, String codec, String schema) {
      this.syncMarker = syncMarker;
      this.codec = codec;
      this.schema = schema;
    }
  }

  /**
   * Reads the {@code Metadata} from the header of an Avro file. Throws an IOException if the file
   * is an invalid format.
   *
   * <p>This method parses the header of an Avro
   * <a href="https://avro.apache.org/docs/1.7.7/spec.html#Object+Container+Files">
   * Object Container File</a>.
   */
  static Metadata readMetadataFromFile(String fileName) throws IOException {
    String codec = null;
    String schema = null;
    byte[] syncMarker;
    try (InputStream stream =
        Channels.newInputStream(IOChannelUtils.getFactory(fileName).open(fileName))) {
      BinaryDecoder decoder = DecoderFactory.get().binaryDecoder(stream, null);

      // The header of an object container file begins with a four-byte magic number, followed
      // by the file metadata (including the schema and codec), encoded as a map. Finally, the
      // header ends with the file's 16-byte sync marker.
      // See https://avro.apache.org/docs/1.7.7/spec.html#Object+Container+Files for details on
      // the encoding of container files.

      // Read the magic number.
      byte[] magic = new byte[DataFileConstants.MAGIC.length];
      decoder.readFixed(magic);
      if (!Arrays.equals(magic, DataFileConstants.MAGIC)) {
        throw new IOException("Missing Avro file signature: " + fileName);
      }

      // Read the metadata to find the codec and schema.
      ByteBuffer valueBuffer = ByteBuffer.allocate(512);
      long numRecords = decoder.readMapStart();
      while (numRecords > 0) {
        for (long recordIndex = 0; recordIndex < numRecords; recordIndex++) {
          String key = decoder.readString();
          // readBytes() clears the buffer and returns a buffer where:
          // - position is the start of the bytes read
          // - limit is the end of the bytes read
          valueBuffer = decoder.readBytes(valueBuffer);
          byte[] bytes = new byte[valueBuffer.remaining()];
          valueBuffer.get(bytes);
          if (key.equals(DataFileConstants.CODEC)) {
            codec = new String(bytes, "UTF-8");
          } else if (key.equals(DataFileConstants.SCHEMA)) {
            schema = new String(bytes, "UTF-8");
          }
        }
        numRecords = decoder.mapNext();
      }
      if (codec == null) {
        codec = DataFileConstants.NULL_CODEC;
      }

      // Finally, read the sync marker.
      syncMarker = new byte[DataFileConstants.SYNC_SIZE];
      decoder.readFixed(syncMarker);
    }
    return new Metadata(syncMarker, codec, schema);
  }

  @Override
  public AvroSource<T> createForSubrangeOfFile(String fileName, long start, long end) {
    byte[] syncMarker = this.syncMarker;
    String codec = this.codec;
    String schema = this.schema;
    // codec and syncMarker are initially null when the source is created, as they differ
    // across input files and must be read from the file. Here, when we are creating a source
    // for a subrange of a file, we can initialize these values. When the resulting AvroSource
    // is further split, they do not need to be read again.
    if (codec == null || syncMarker == null) {
      Metadata metadata;
      try {
        metadata = readMetadataFromFile(fileName);
      } catch (IOException e) {
        throw new RuntimeException("Error reading metadata from file " + fileName, e);
      }
      codec = metadata.codec;
      syncMarker = metadata.syncMarker;
      // If the source was created with a null schema, use the schema that we read from the file's
      // metadata.
      if (schema == null) {
        schema = metadata.schema;
      }
    }
    return new AvroSource<T>(
        fileName, getMinBundleSize(), start, end, schema, type, codec, syncMarker);
  }

  @Override
  public AvroReader<T> createSingleFileReader(PipelineOptions options) {
    return new AvroReader<T>(this);
  }

  @Override
  public boolean producesSortedKeys(PipelineOptions options) throws Exception {
    return false;
  }

  @Override
  public AvroCoder<T> getDefaultOutputCoder() {
    if (coder == null) {
      Schema.Parser parser = new Schema.Parser();
      coder = AvroCoder.of(type, parser.parse(schema));
    }
    return coder;
  }

  public String getSchema() {
    return schema;
  }

  private byte[] getSyncMarker() {
    return syncMarker;
  }

  private String getCodec() {
    return codec;
  }

  /**
   * A {@link BlockBasedSource.Block} of Avro records. Visible for testing.
   * @param <T> The type of records stored in the block.
   */
  @Experimental(Experimental.Kind.SOURCE_SINK)
  static class AvroBlock<T> extends Block<T> {
    // The number of records in the block.
    private final long numRecords;

    // The current record in the block.
    private T currentRecord;

    // The index of the current record in the block.
    private long currentRecordIndex = 0;

    // A DatumReader to read records from the block.
    private final DatumReader<T> reader;

    // A BinaryDecoder used by the reader to decode records.
    private final BinaryDecoder decoder;

    /**
     * Decodes a byte array as an InputStream. The byte array may be compressed using some
     * codec. Reads from the returned stream will result in decompressed bytes.
     *
     * <p>This supports the same codecs as Avro's {@code CodecFactory}, namely those defined in
     * <a
     * href="https://avro.apache.org/docs/1.7.7/api/java/org/apache/avro/file/DataFileConstants.html">
     * {@code DataFileConstants}</a>.
     * <ul>
     * <li>"snappy" : Google's Snappy compression
     * <li>"deflate" : deflate compression
     * <li>"bzip2" : Bzip2 compression
     * <li>"xz" : xz compression
     * <li>"null" (the string, not the value): Uncompressed data
     * </ul>
     */
    private static InputStream decodeAsInputStream(byte[] data, String codec) throws IOException {
      ByteArrayInputStream byteStream = new ByteArrayInputStream(data);
      switch (codec) {
        case DataFileConstants.SNAPPY_CODEC:
          return new SnappyCompressorInputStream(byteStream);
        case DataFileConstants.DEFLATE_CODEC:
          // nowrap == true: Do not expect ZLIB header or checksum, as Avro does not write them.
          Inflater inflater = new Inflater(true);
          return new InflaterInputStream(byteStream, inflater);
        case DataFileConstants.XZ_CODEC:
          return new XZCompressorInputStream(byteStream);
        case DataFileConstants.BZIP2_CODEC:
          return new BZip2CompressorInputStream(byteStream);
        case DataFileConstants.NULL_CODEC:
          return byteStream;
        default:
          throw new IllegalArgumentException("Unsupported codec: " + codec);
      }
    }

    AvroBlock(byte[] data, long numRecords, AvroSource<T> source) throws IOException {
      this.numRecords = numRecords;
      this.reader = source.getDefaultOutputCoder().createDatumReader();
      this.decoder =
          DecoderFactory.get().binaryDecoder(decodeAsInputStream(data, source.getCodec()), null);
    }

    @Override
    public T getCurrentRecord() {
      return currentRecord;
    }

    @Override
    public boolean readNextRecord() throws IOException {
      if (currentRecordIndex >= numRecords) {
        return false;
      }
      currentRecord = reader.read(null, decoder);
      currentRecordIndex++;
      return true;
    }

    @Override
    public double getFractionOfBlockConsumed() {
      return ((double) currentRecordIndex) / numRecords;
    }
  }

  /**
   * A {@link BlockBasedSource.BlockBasedReader} for reading blocks from Avro files.
   *
   * <p>An Avro Object Container File consists of a header followed by a 16-bit sync marker
   * and then a sequence of blocks, where each block begins with two encoded longs representing
   * the total number of records in the block and the block's size in bytes, followed by the
   * block's (optionally-encoded) records. Each block is terminated by a 16-bit sync marker.
   *
   * <p>Here, we consider the sync marker that precedes a block to be its offset, as this allows
   * a reader that begins reading at that offset to detect the sync marker and the beginning of
   * the block.
   *
   * @param <T> The type of records contained in the block.
   */
  @Experimental(Experimental.Kind.SOURCE_SINK)
  public static class AvroReader<T> extends BlockBasedReader<T> {
    // The current block.
    private AvroBlock<T> currentBlock;

    // Offset of the block.
    private long currentBlockOffset = 0;

    // Size of the current block.
    private long currentBlockSizeBytes = 0;

    // Current offset within the stream.
    private long currentOffset = 0;

    // Stream used to read from the underlying file.
    // A pushback stream is used to restore bytes buffered during seeking/decoding.
    private PushbackInputStream stream;

    // Small buffer for reading encoded values from the stream.
    // The maximum size of an encoded long is 10 bytes, and this buffer will be used to read two.
    private final byte[] readBuffer = new byte[20];

    // Decoder to decode binary-encoded values from the buffer.
    private BinaryDecoder decoder;

    public AvroReader(AvroSource<T> source) {
      super(source);
    }

    @Override
    public AvroSource<T> getCurrentSource() {
      return (AvroSource<T>) super.getCurrentSource();
    }

    @Override
    public boolean readNextBlock() throws IOException {
      // The next block in the file is after the first sync marker that can be read starting from
      // the current offset. First, we seek past the next sync marker, if it exists. After a sync
      // marker is the start of a block. A block begins with the number of records contained in
      // the block, encoded as a long, followed by the size of the block in bytes, encoded as a
      // long. The currentOffset after this method should be last byte after this block, and the
      // currentBlockOffset should be the start of the sync marker before this block.

      // Seek to the next sync marker, if one exists.
      currentOffset += advancePastNextSyncMarker(stream, getCurrentSource().getSyncMarker());

      // The offset of the current block includes its preceding sync marker.
      currentBlockOffset = currentOffset - getCurrentSource().getSyncMarker().length;

      // Read a small buffer to parse the block header.
      // We cannot use a BinaryDecoder to do this directly from the stream because a BinaryDecoder
      // internally buffers data and we only want to read as many bytes from the stream as the size
      // of the header. Though BinaryDecoder#InputStream returns an input stream that is aware of
      // its internal buffering, we would have to re-wrap this input stream to seek for the next
      // block in the file.
      int read = stream.read(readBuffer);
      // We reached the last sync marker in the file.
      if (read <= 0) {
        return false;
      }
      decoder = DecoderFactory.get().binaryDecoder(readBuffer, decoder);
      long numRecords = decoder.readLong();
      long blockSize = decoder.readLong();

      // The decoder buffers data internally, but since we know the size of the stream the
      // decoder has constructed from the readBuffer, the number of bytes available in the
      // input stream is equal to the number of unconsumed bytes.
      int headerSize = readBuffer.length - decoder.inputStream().available();
      stream.unread(readBuffer, headerSize, read - headerSize);

      // Create the current block by reading blockSize bytes. Block sizes permitted by the Avro
      // specification are [32, 2^30], so this narrowing is ok.
      byte[] data = new byte[(int) blockSize];
      stream.read(data);
      currentBlock = new AvroBlock<>(data, numRecords, getCurrentSource());
      currentBlockSizeBytes = blockSize;

      // Update current offset with the number of bytes we read to get the next block.
      currentOffset += headerSize + blockSize;
      return true;
    }

    @Override
    public AvroBlock<T> getCurrentBlock() {
      return currentBlock;
    }

    @Override
    public long getCurrentBlockOffset() {
      return currentBlockOffset;
    }

    @Override
    public long getCurrentBlockSize() {
      return currentBlockSizeBytes;
    }

    /**
     * Creates a {@code PushbackInputStream} that has a large enough pushback buffer to be able
     * to push back the syncBuffer and the readBuffer.
     */
    private PushbackInputStream createStream(ReadableByteChannel channel) {
      return new PushbackInputStream(
          Channels.newInputStream(channel),
          getCurrentSource().getSyncMarker().length + readBuffer.length);
    }

    /**
     * Starts reading from the provided channel. Assumes that the channel is already seeked to
     * the source's start offset.
     */
    @Override
    protected void startReading(ReadableByteChannel channel) throws IOException {
      stream = createStream(channel);
      currentOffset = getCurrentSource().getStartOffset();
    }

    /**
     * Advances to the first byte after the next occurrence of the sync marker in the
     * stream when reading from the current offset. Returns the number of bytes consumed
     * from the stream. Note that this method requires a PushbackInputStream with a buffer
     * at least as big as the marker it is seeking for.
     */
    static long advancePastNextSyncMarker(PushbackInputStream stream, byte[] syncMarker)
        throws IOException {
      Seeker seeker = new Seeker(syncMarker);
      byte[] syncBuffer = new byte[syncMarker.length];
      long totalBytesConsumed = 0;
      // Seek until either a sync marker is found or we reach the end of the file.
      int mark = -1; // Position of the last byte in the sync marker.
      int read; // Number of bytes read.
      do {
        read = stream.read(syncBuffer);
        if (read >= 0) {
          mark = seeker.find(syncBuffer, read);
          // Update the currentOffset with the number of bytes read.
          totalBytesConsumed += read;
        }
      } while (mark < 0 && read > 0);

      // If the sync marker was found, unread block data and update the current offsets.
      if (mark >= 0) {
        // The current offset after this call should be just past the sync marker, so we should
        // unread the remaining buffer contents and update the currentOffset accordingly.
        stream.unread(syncBuffer, mark + 1, read - (mark + 1));
        totalBytesConsumed = totalBytesConsumed - (read - (mark + 1));
      }
      return totalBytesConsumed;
    }

    /**
     * A {@code Seeker} looks for a given marker within a byte buffer. Uses naive string matching
     * with a sliding window, as sync markers are small and random.
     */
    static class Seeker {
      // The marker to search for.
      private byte[] marker;

      // Buffer used for the sliding window.
      private byte[] searchBuffer;

      // Number of bytes available to be matched in the buffer.
      private int available = 0;

      /**
       * Create a {@code Seeker} that looks for the given marker.
       */
      public Seeker(byte[] marker) {
        this.marker = marker;
        this.searchBuffer = new byte[marker.length];
      }

      /**
       * Find the marker in the byte buffer. Returns the index of the end of the marker in the
       * buffer. If the marker is not found, returns -1.
       *
       * <p>State is maintained between calls. If the marker was partially matched, a subsequent
       * call to find will resume matching the marker.
       *
       * @param buffer
       * @return the index of the end of the marker within the buffer, or -1 if the buffer was not
       * found.
       */
      public int find(byte[] buffer, int length) {
        for (int i = 0; i < length; i++) {
          System.arraycopy(searchBuffer, 1, searchBuffer, 0, searchBuffer.length - 1);
          searchBuffer[searchBuffer.length - 1] = buffer[i];
          available = Math.min(available + 1, searchBuffer.length);
          if (ByteBuffer.wrap(searchBuffer, searchBuffer.length - available, available)
                  .equals(ByteBuffer.wrap(marker))) {
            available = 0;
            return i;
          }
        }
        return -1;
      }
    }
  }
}
