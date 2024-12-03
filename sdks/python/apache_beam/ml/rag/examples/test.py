import apache_beam as beam

import tempfile

from typing import Any, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from apache_beam.ml.transforms.base import MLTransform
from apache_beam.ml.rag.chunking.langchain import LangChainChunkingProvider
from apache_beam.ml.rag.embedding.huggingface import HuggingfaceTextEmbeddings
from apache_beam.ml.rag.storage.base import VectorDatabaseWriteTransform
from apache_beam.ml.rag.storage.bigquery import BigQueryVectorWriterConfig
from transformers import AutoTokenizer
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.ml.rag.types import Embedding, Chunk, ChunkContent
from apache_beam.ml.rag.enrichment.bigquery_vector_search import BigQueryVectorSearchEnrichmentHandler, BigQueryVectorSearchParameters
from apache_beam.transforms.enrichment import Enrichment


embedder = HuggingfaceTextEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=512,
    chunk_overlap=52,
)

def join_fn(left: Embedding, right: Dict[str, Any]) -> Embedding:
    left.metadata['enrichment_data'] = right
    return left


def run_pipeline():
    with beam.Pipeline(options=PipelineOptions([
        '--runner=DirectRunner',
        '--temp_location=gs://cvandermerwe/managed',
        '--expansion_service_port=8888'
        ])) as p:

        # Ingestion
        _ = (p
             | beam.Create([{
                    'content': 'This is a simple test document. It has multiple sentences. '
                            'We will use it to test basic splitting. '*20,
                    'source': 'simple.txt',
                    'language': 'en'
                    },
                    {
                    'content': ('The patient arrived at 2 p.m. yesterday. '
                            'Initial assessment was completed. '
                            'Lab results showed normal ranges. '
                            'Follow-up scheduled for next week.'*10),
                    'source': 'medical.txt',
                    'language': 'en'
                    }
                ]
              )
              | MLTransform(write_artifact_location=tempfile.mkdtemp())
                .with_transform(
                    LangChainChunkingProvider(
                        text_splitter=splitter,
                        document_field="content",
                        metadata_fields=["source", "language"]
                    )
                )
                .with_transform(embedder)
              | VectorDatabaseWriteTransform(BigQueryVectorWriterConfig(
                    write_config={
                        "table": "dataflow-twest:claude_test.rag_test",
                        "create_disposition": "CREATE_IF_NEEDED",
                        "write_disposition": "WRITE_TRUNCATE",
                    }
                ))
            )
        
        # Enrichment
        _ = (p
             | beam.Create([
                 Chunk(
                     id="simple_query",
                     content=ChunkContent(text="This is a simple test document."),
                     metadata={"language": "en"}
                 ),
                 Chunk(
                     id="medical_query",
                     content=ChunkContent(text="When did the patient arrive?"),
                     metadata={"language": "en"}
                ),
                ]
             )
             | MLTransform(write_artifact_location=tempfile.mkdtemp())
                .with_transform(embedder)
             | Enrichment(
                  BigQueryVectorSearchEnrichmentHandler(
                   project="dataflow-twest",
                   vector_search_parameters=BigQueryVectorSearchParameters(
                        table_name='dataflow-twest.claude_test.rag_test',
                        embedding_column='embedding',
                        columns=['metadata', 'content'],
                        neighbor_count=3,
                        metadata_restriction_template=(
                            "check_metadata(metadata, 'language','{language}')"
                        )
                    )
                  ),
                  join_fn=join_fn,
                  use_custom_types=True
                )
              |beam.Map(print)
            )
    

if __name__ == '__main__':
    run_pipeline()