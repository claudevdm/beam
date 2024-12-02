from apache_beam.ml.transforms.base import  EmbeddingTypeAdapter
from apache_beam.ml.rag.types import Embedding

def create_rag_adapter() -> EmbeddingTypeAdapter:
    """Creates adapter for converting between Chunk and Embedding types.
    
    The adapter:
    - Extracts text from Chunk.content.text for embedding
    - Creates Embedding objects from model output
    - Preserves Chunk.id and metadata in Embedding
    - Sets sparse_embedding to None (dense embeddings only)
    
    Returns:
        EmbeddingTypeAdapter configured for RAG pipeline types
    """
    return EmbeddingTypeAdapter(
        input_fn=lambda chunks: [chunk.content.text for chunk in chunks],
        output_fn=lambda chunks, embeddings: [
            Embedding(
                id=chunk.id,
                dense_embedding=embeddings,
                sparse_embedding=None,
                metadata=chunk.metadata,
                content=chunk.content
            )
            for chunk, embeddings in zip(chunks, embeddings)
        ]
    )