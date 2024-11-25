import apache_beam as beam
from apache_beam.ml.transforms.base import MLTransformProvider, MLTransform
from apache_beam.ml.rag.types import Chunk, ChunkContent
from typing import List, Optional
from collections.abc import Callable
import abc
import uuid
import functools

ChunkIdFn = Callable[[Chunk], str]

def create_random_id(chunk: Chunk):
    return str(uuid.uuid4())

def assign_chunk_id(chunk_id_fn: ChunkIdFn, chunk: Chunk):
   chunk.id = chunk_id_fn(chunk)
   return chunk

class ChunkingTransformProvider(MLTransformProvider):
    def __init__(self,
                chunk_id_fn: Optional[ChunkIdFn] = None):
       self.assign_chunk_id_fn = functools.partial(
          assign_chunk_id,
          chunk_id_fn if chunk_id_fn is not None else create_random_id
        )

    @abc.abstractmethod
    def get_text_splitter_transform(self) -> beam.DoFn:
      "Return DoFn emits splits for given content."
      NotImplementedError

    def get_ptransform_for_processing(self, **kwargs) -> beam.PTransform:
      return (
          "Split document" >> self.get_text_splitter_transform().with_output_types(Chunk)
          | "Assign chunk id" >> beam.Map(self.assign_chunk_id_fn)
      )