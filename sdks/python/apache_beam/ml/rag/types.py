"""Core types for RAG pipelines.

This module contains the core dataclasses used throughout the RAG pipeline
implementation, including Chunk and Embedding types that define the data
contracts between different stages of the pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

@dataclass
class ChunkContent:
    """Container for content to be embedded.
    """
    text: Optional[str] = None
    image_data: Optional[bytes] = None

@dataclass
class Chunk:
    """Represents a chunk of text with metadata.
    
    Attributes:
        text: The actual content of the chunk
        id: Unique identifier for the chunk
        index: Index of this chunk within the original document
        metadata: Additional metadata about the chunk (e.g., source document, summary)
    """
    content: ChunkContent
    id: Optional[str] = None
    index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

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
    dense_embedding: Optional[List[float]] = None
    sparse_embedding: Optional[Tuple[List[int], List[float]]] = None  # For hybrid search
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: Optional[ChunkContent] = None
