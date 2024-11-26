import apache_beam as beam
from langchain.text_splitter import TextSplitter
from apache_beam.ml.rag.chunking.base import ChunkingTransformProvider, ChunkIdFn
from apache_beam.ml.rag.types import Chunk, ChunkContent
from typing import List, Optional

class LangChainChunkingProvider(ChunkingTransformProvider):
    def __init__(
        self,
        text_splitter: TextSplitter,
        document_field: str,
        metadata_fields: List[str] = [],
        chunk_id_fn: Optional[ChunkIdFn] = None
    ):
      if not isinstance(text_splitter, TextSplitter):
        raise TypeError("text_splitter must be a LangChain TextSplitter")
      if not document_field:
        raise ValueError("document_field cannot be empty")
      super().__init__(chunk_id_fn)
      self.text_splitter = text_splitter
      self.document_field = document_field
      self.metadata_fields = metadata_fields

    def get_text_splitter_transform(self) -> beam.DoFn:
       return "Langchain text split" >> beam.ParDo(
          LangChainTextSplitter(
            text_splitter=self.text_splitter,
            document_field=self.document_field,
            metadata_fields=self.metadata_fields   
          )
        )

class LangChainTextSplitter(beam.DoFn):
    def __init__(
      self,
      text_splitter: TextSplitter,
      document_field: str,
      metadata_fields: List[str]
    ):
      self.text_splitter = text_splitter
      self.document_field = document_field
      self.metadata_fields = metadata_fields

    def process(self, element):
      text_chunks = self.text_splitter.split_text(element[self.document_field])
      metadata = {field: element[field] for field in self.metadata_fields}
      for i, text_chunk in enumerate(text_chunks):
         yield Chunk(content=ChunkContent(text=text_chunk), index=i, metadata=metadata)