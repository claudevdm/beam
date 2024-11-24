"""Core types for RAG pipelines.

This module contains the core dataclasses used throughout the RAG pipeline
implementation, including Chunk and Embedding types that define the data
contracts between different stages of the pipeline.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

@dataclass
class Chunk:
    """Represents a chunk of text with metadata.
    
    Attributes:
        id: Unique identifier for the chunk
        index: Index of this chunk within the original document
        text: The actual text content of the chunk
        metadata: Additional metadata about the chunk (e.g., source document, summary)
    """
    id: str
    index: int
    text: str
    metadata: Dict[str, Any]

@dataclass
class Embedding:
    """Represents vector embeddings with associated metadata.
    
    Attributes:
        id: Unique identifier for the embedding
        dense_embedding: Dense vector representation
        sparse_embedding: Optional sparse vector representation for hybrid search
        metadata: Additional metadata about the embedding
    """
    id: str
    dense_embedding: Optional[List[float]]
    sparse_embedding: Optional[Tuple[List[int], List[float]]]  # For hybrid search
    metadata: Dict[str, Any]
