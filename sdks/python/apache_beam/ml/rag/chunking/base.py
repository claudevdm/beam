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
#

import abc
import functools
from collections.abc import Callable
from typing import Any, Dict, Optional

import apache_beam as beam
from apache_beam.ml.rag.types import Chunk
from apache_beam.ml.transforms.base import MLTransformProvider

ChunkIdFn = Callable[[Chunk], str]


def assign_chunk_id(chunk_id_fn: ChunkIdFn, chunk: Chunk):
  chunk.id = chunk_id_fn(chunk)
  return chunk


class ChunkingTransformProvider(MLTransformProvider):
  def __init__(self, chunk_id_fn: Optional[ChunkIdFn] = None):
    self.assign_chunk_id_fn = functools.partial(
        assign_chunk_id, chunk_id_fn) if chunk_id_fn is not None else None

  @abc.abstractmethod
  def get_text_splitter_transform(
      self
  ) -> beam.PTransform[beam.PCollection[Dict[str, Any]],
                       beam.PCollection[Chunk]]:
    """Creates transforms that emits splits for given content."""
    raise NotImplementedError(
        "Subclasses must implement get_text_splitter_transform")

  def get_ptransform_for_processing(
      self, **kwargs
  ) -> beam.PTransform[beam.PCollection[Dict[str, Any]],
                       beam.PCollection[Chunk]]:
    """Creates transform for processing documents into chunks."""
    ptransform = (
        "Split document" >>
        self.get_text_splitter_transform().with_output_types(Chunk))
    if self.assign_chunk_id_fn:
      ptransform = (
          ptransform | "Assign chunk id" >> beam.Map(
              self.assign_chunk_id_fn).with_output_types(Chunk))
    return ptransform
