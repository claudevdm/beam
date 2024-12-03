from typing import Optional, List, Dict, Any
import apache_beam as beam
from apache_beam.ml.rag.storage.base import VectorDatabaseConfig
from apache_beam.ml.rag.types import Embedding

class BigQueryVectorWriterConfig(VectorDatabaseConfig):
    """Configuration for writing vectors to BigQuery using managed transforms."""
    
    def __init__(
        self,
        *,  # Force keyword arguments
        id_column: str = "id",
        embedding_column: str = "embedding",
        content_column: str = "content",
        metadata_columns: Optional[List[str]] = None,
        write_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize BigQuery writer config.
        
        Args:
            embedding_column: Column name for embedding vector
            content_column: Column name for chunk content
            metadata_columns: List of metadata fields to write (None for all)
            write_config: BigQuery write configuration dict. Must include 'table'.
                Other options like create_disposition, write_disposition can be
                specified here.
        """
        if not write_config or 'table' not in write_config:
            raise ValueError("write_config must be provided with 'table' specified")
        
        self.id_column = id_column
        self.embedding_column = embedding_column
        self.content_column = content_column
        self.metadata_columns = metadata_columns
        self.write_config = write_config

    def create_write_transform(self) -> beam.PTransform:
        """Creates transform to write to BigQuery."""
        return _WriteToBigQueryVectorDatabase(self)

class _WriteToBigQueryVectorDatabase(beam.PTransform):
    """Implementation of BigQuery vector database write."""
    
    def __init__(self, config: BigQueryVectorWriterConfig):
        self.config = config

    def expand(self, pcoll: beam.PCollection[Embedding]):

        return (
            pcoll
            | "Convert to Rows" >> beam.Select(
                id=lambda x: str(x.id),
                embedding=lambda x: [float(v) for v in x.dense_embedding],
                content=lambda x: str(x.content.text),
                metadata=lambda x: {str(k): str(v) for k, v in x.metadata.items()}
            )
            | "Write to BigQuery" >> beam.managed.Write(
                beam.managed.BIGQUERY,
                config=self.config.write_config
            )
        )

# java -jar sdks/java/io/google-cloud-platform/expansion-service/build/libs/beam-sdks-java-io-google-cloud-platform-expansion-service-2.62.0-SNAPSHOT.jar 8888