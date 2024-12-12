#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from apache_beam.ml.transforms.base import EmbeddingTypeAdapter
from apache_beam.ml.rag.types import Embedding, Chunk
from typing import List
from collections.abc import Sequence


def create_rag_adapter() -> EmbeddingTypeAdapter[Chunk, Embedding]:
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
      input_fn=_extract_chunk_text, output_fn=_output_fn)


def _extract_chunk_text(chunks: Sequence[Chunk]) -> List[str]:
  """Extract text from chunks for embedding."""
  chunk_texts = []
  for chunk in chunks:
    if not chunk.content.text:
      raise ValueError("Expected chunk text content.")
    chunk_texts.append(chunk.content.text)
  return chunk_texts


def _output_fn(chunks: Sequence[Chunk],
               embeddings: Sequence[List[float]]) -> List[Embedding]:
  """Create Embeddings from chunks and embedding vectors."""
  return [
      Embedding(
          id=chunk.id,
          dense_embedding=embedding,
          sparse_embedding=None,
          metadata=chunk.metadata,
          content=chunk.content) for chunk,
      embedding in zip(chunks, embeddings)
  ]
