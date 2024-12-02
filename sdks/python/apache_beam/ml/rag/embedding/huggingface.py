"""RAG-specific embedding implementations using HuggingFace models."""

from typing import Optional
import apache_beam as beam
from apache_beam.ml.transforms.embeddings.huggingface import (
    SentenceTransformer,
    _SentenceTransformerModelHandler
)
from apache_beam.ml.transforms.base import EmbeddingsManager, _TextEmbeddingHandler
from apache_beam.ml.inference.base import RunInference
from apache_beam.ml.rag.embedding.base import create_rag_adapter

class HuggingfaceTextEmbeddings(EmbeddingsManager):
    """SentenceTransformer embeddings for RAG pipeline.
    
    Extends EmbeddingsManager to work with RAG-specific types:
    - Input: Chunk objects containing text to embed
    - Output: Embedding objects containing vector representations
    
    The adapter automatically:
    - Extracts text from Chunk.content.text
    - Preserves Chunk.id in Embedding.id
    - Copies Chunk.metadata to Embedding.metadata
    - Converts model output to Embedding.dense_embedding
    """
    def __init__(
        self,
        model_name: str,
        *,
        max_seq_length: Optional[int] = None,
        **kwargs
    ):
        """Initialize RAG embeddings.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            max_seq_length: Maximum sequence length for the model
            **kwargs: Additional arguments passed to parent
        """
        rag_adapter = create_rag_adapter()
        super().__init__(type_adapter=rag_adapter, **kwargs)
        self.type_adapter = rag_adapter
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.model_class = SentenceTransformer

    def get_model_handler(self):
        """Returns model handler configured with RAG adapter."""
        return _SentenceTransformerModelHandler(
            model_class=self.model_class,
            max_seq_length=self.max_seq_length,
            model_name=self.model_name,
            load_model_args=self.load_model_args,
            min_batch_size=self.min_batch_size,
            max_batch_size=self.max_batch_size,
            large_model=self.large_model
        )

    def get_ptransform_for_processing(self, **kwargs) -> beam.PTransform:
        """Returns PTransform that uses the RAG adapter."""
        return RunInference(
            model_handler=_TextEmbeddingHandler(self),
            inference_args=self.inference_args
        )