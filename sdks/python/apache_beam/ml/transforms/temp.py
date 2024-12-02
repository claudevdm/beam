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

import abc
import collections
import logging
import os
import tempfile
import uuid
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import Generic
from typing import Optional
from typing import TypeVar
from typing import Union
from typing import List

import jsonpickle
import numpy as np

from dataclasses import dataclass
from collections.abc import Callable

import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.metrics.metric import Metrics
from apache_beam.ml.inference.base import ModelHandler
from apache_beam.ml.inference.base import ModelT
from apache_beam.ml.inference.base import RunInferenceDLQ
from apache_beam.options.pipeline_options import PipelineOptions

_LOGGER = logging.getLogger(__name__)
_ATTRIBUTE_FILE_NAME = 'attributes.json'

__all__ = [
    'MLTransform',
    'ProcessHandler',
    'MLTransformProvider',
    'BaseOperation',
    'EmbeddingsManager'
]

TransformedDatasetT = TypeVar('TransformedDatasetT')
TransformedMetadataT = TypeVar('TransformedMetadataT')

# Input/Output types to the MLTransform.
MLTransformOutputT = TypeVar('MLTransformOutputT')
ExampleT = TypeVar('ExampleT')

# Input to the apply() method of BaseOperation.
OperationInputT = TypeVar('OperationInputT')
# Output of the apply() method of BaseOperation.
OperationOutputT = TypeVar('OperationOutputT')


def _convert_list_of_dicts_to_dict_of_lists(
    list_of_dicts: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
  keys_to_element_list = collections.defaultdict(list)
  input_keys = list_of_dicts[0].keys()
  for d in list_of_dicts:
    if set(d.keys()) != set(input_keys):
      extra_keys = set(d.keys()) - set(input_keys) if len(
          d.keys()) > len(input_keys) else set(input_keys) - set(d.keys())
      raise RuntimeError(
          f'All the dicts in the input data should have the same keys. '
          f'Got: {extra_keys} instead.')
    for key, value in d.items():
      keys_to_element_list[key].append(value)
  return keys_to_element_list


def _convert_dict_of_lists_to_lists_of_dict(
    dict_of_lists: dict[str, list[Any]]) -> list[dict[str, Any]]:
  batch_length = len(next(iter(dict_of_lists.values())))
  result: list[dict[str, Any]] = [{} for _ in range(batch_length)]
  # all the values in the dict_of_lists should have same length
  for key, values in dict_of_lists.items():
    assert len(values) == batch_length, (
        "This function expects all the values "
        "in the dict_of_lists to have same length."
        )
    for i in range(len(values)):
      result[i][key] = values[i]
  return result


def _map_errors_to_beam_row(element, cls_name=None):
  row_elements = {
      'element': element[0],
      'msg': str(element[1][1]),
      'stack': str(element[1][2]),
  }
  if cls_name is not None:
    row_elements['transform_name'] = cls_name
  return beam.Row(**row_elements)


class ArtifactMode(object):
  PRODUCE = 'produce'
  CONSUME = 'consume'


class MLTransformProvider:
  """
  Data processing transforms that are intended to be used with MLTransform
  should subclass MLTransformProvider and implement
  get_ptransform_for_processing().

  get_ptransform_for_processing() method should return a PTransform that can be
  used to process the data.

  """
  @abc.abstractmethod
  def get_ptransform_for_processing(self, **kwargs) -> beam.PTransform:
    """
    Returns a PTransform that can be used to process the data.
    """

  def get_counter(self):
    """
    Returns the counter name for the data processing transform.
    """
    counter_name = self.__class__.__name__
    return Metrics.counter(MLTransform, f'BeamML_{counter_name}')


class BaseOperation(Generic[OperationInputT, OperationOutputT],
                    MLTransformProvider,
                    abc.ABC):
  def __init__(self, columns: list[str]) -> None:
    """
    Base Opertation class data processing transformations.
    Args:
      columns: List of column names to apply the transformation.
    """
    self.columns = columns

  @abc.abstractmethod
  def apply_transform(self, data: OperationInputT,
                      output_column_name: str) -> dict[str, OperationOutputT]:
    """
    Define any processing logic in the apply_transform() method.
    processing logics are applied on inputs and returns a transformed
    output.
    Args:
      inputs: input data.
    """

  def __call__(self, data: OperationInputT,
               output_column_name: str) -> dict[str, OperationOutputT]:
    """
    This method is called when the instance of the class is called.
    This method will invoke the apply() method of the class.
    """
    transformed_data = self.apply_transform(data, output_column_name)
    return transformed_data


class ProcessHandler(
    beam.PTransform[beam.PCollection[ExampleT],
                    Union[beam.PCollection[MLTransformOutputT],
                          tuple[beam.PCollection[MLTransformOutputT],
                                beam.PCollection[beam.Row]]]],
    abc.ABC):
  """
  Only for internal use. No backwards compatibility guarantees.
  """
  @abc.abstractmethod
  def append_transform(self, transform: BaseOperation):
    """
    Append transforms to the ProcessHandler.
    """
InputT = TypeVar('InputT')  # e.g., Chunk
OutputT = TypeVar('OutputT')  # e.g., Embedding

@dataclass
class EmbeddingTypeAdapter:
    """Adapts input types to text for embedding and converts output embeddings.
    
    Args:
        input_fn: Function to extract text for embedding from input type
        output_fn: Function to create output type from input and embeddings
    """
    input_fn: Callable[[List[InputT]], List[str]]
    output_fn: Callable[[List[InputT], List[Any]], List[OutputT]]


# TODO:https://github.com/apache/beam/issues/29356
#  Add support for inference_fn
class EmbeddingsManager(MLTransformProvider):
  def __init__(
      self,
      *,
      columns: list[str] = None,
      type_adapter: Optional[EmbeddingTypeAdapter] = None,
      # common args for all ModelHandlers.
      load_model_args: Optional[dict[str, Any]] = None,
      min_batch_size: Optional[int] = None,
      max_batch_size: Optional[int] = None,
      large_model: bool = False,
      **kwargs):
    if columns is not None and type_adapter is not None:
        raise ValueError(
            "Cannot specify both 'columns' and 'type_adapter'. "
            "Use either columns for dict processing or type_adapter "
            "for custom types."
        )
    elif columns is None and type_adapter is None:
      raise ValueError(
                "Must provide either 'columns' or 'type_adapter'."
            )
    elif columns is not None:
        self.columns = columns
    elif type_adapter is None:
        self.type_adapter = type_adapter

    self.load_model_args = load_model_args or {}
    self.min_batch_size = min_batch_size
    self.max_batch_size = max_batch_size
    self.large_model = large_model
    self.columns = columns
    self.inference_args = kwargs.pop('inference_args', {})

    if kwargs:
      _LOGGER.warning("Ignoring the following arguments: %s", kwargs.keys())

  # TODO:https://github.com/apache/beam/pull/29564 add set_model_handler method
  @abc.abstractmethod
  def get_model_handler(self) -> ModelHandler:
    """
    Return framework specific model handler.
    """

  def get_columns_to_apply(self):
    return self.columns



class _EmbeddingHandler(ModelHandler):
  """
  A ModelHandler intended to be work on list[dict[str, Any]] inputs.

  The inputs to the model handler are expected to be a list of dicts.

  For example, if the original mode is used with RunInference to take a
  PCollection[E] to a PCollection[P], this ModelHandler would take a
  PCollection[dict[str, E]] to a PCollection[dict[str, P]].

  _EmbeddingHandler will accept an EmbeddingsManager instance, which
  contains the details of the model to be loaded and the inference_fn to be
  used. The purpose of _EmbeddingHandler is to generate embeddings for
  general inputs using the EmbeddingsManager instance.

  This is an internal class and offers no backwards compatibility guarantees.

  Args:
    embeddings_manager: An EmbeddingsManager instance.
  """
  def __init__(self, embeddings_manager: EmbeddingsManager):
    self.embedding_config = embeddings_manager
    self._underlying = self.embedding_config.get_model_handler()
    self.columns = self.embedding_config.get_columns_to_apply()

  def load_model(self):
    model = self._underlying.load_model()
    return model

  def _validate_column_data(self, batch):
    pass

  def _validate_batch(self, batch: Sequence[dict[str, Any]]):
    if not batch:
      return TypeError("Expected batch to not be None")
    # if self.embedding_config.type_adapter:
    #   return True 
    if not isinstance(batch[0], dict):
      raise TypeError(
          'Expected data to be dicts, got '
          f'{type(batch[0])} instead.')
  def _process_generic_batch(
      self,
      batch: List[InputT],
      model: ModelT,
      inference_args: Optional[dict[str, Any]]
  ):
    _LOGGER.warning(f"CLAUDE useing type adapter")
    # Custom batch processing
    embedding_input = self.embedding_config.type_adapter.input_fn(batch)
    prediction = self._underlying.run_inference(embedding_input, model, inference_args)
    return self.embedding_config.type_adapter.output_fn(
      batch,
      prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
    )
    
  def _process_batch(
      self,
      dict_batch: dict[str, list[Any]],
      model: ModelT,
      inference_args: Optional[dict[str, Any]]) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = collections.defaultdict(list)
    input_keys = dict_batch.keys()
    missing_columns_in_data = set(self.columns) - set(input_keys)
    if missing_columns_in_data:
      raise RuntimeError(
          f'Data does not contain the following columns '
          f': {missing_columns_in_data}.')
    for key, batch in dict_batch.items():
      if key in self.columns:
        self._validate_column_data(batch)
        prediction = self._underlying.run_inference(
            batch, model, inference_args)
        if isinstance(prediction, np.ndarray):
          prediction = prediction.tolist()
          result[key] = prediction  # type: ignore[assignment]
        else:
          result[key] = prediction  # type: ignore[assignment]
      else:
        result[key] = batch
    return result

  def run_inference(
      self,
      batch: Sequence[dict[str, list[str]]],
      model: ModelT,
      inference_args: Optional[dict[str, Any]] = None,
  ) -> list[dict[str, Union[list[float], list[str]]]]:
    """
    Runs inference on a batch of text inputs. The inputs are expected to be
    a list of dicts. Each dict should have the same keys, and the shape
    should be of the same size for a single key across the batch.
    """
    if self.embedding_config.type_adapter:
      return self._process_generic_batch(
        batch=batch,
        model=model,
        inference_args=inference_args
    )

    self._validate_batch(batch)
    dict_batch = _convert_list_of_dicts_to_dict_of_lists(list_of_dicts=batch)
    transformed_batch = self._process_batch(dict_batch, model, inference_args)
    return _convert_dict_of_lists_to_lists_of_dict(
        dict_of_lists=transformed_batch,
    )

  def get_metrics_namespace(self) -> str:
    return (
        self._underlying.get_metrics_namespace() or 'BeamML_EmbeddingHandler')

  def batch_elements_kwargs(self) -> Mapping[str, Any]:
    batch_sizes_map = {}
    if self.embedding_config.max_batch_size:
      batch_sizes_map['max_batch_size'] = self.embedding_config.max_batch_size
    if self.embedding_config.min_batch_size:
      batch_sizes_map['min_batch_size'] = self.embedding_config.min_batch_size
    return (self._underlying.batch_elements_kwargs() or batch_sizes_map)

  def __repr__(self):
    return self._underlying.__repr__()

  def validate_inference_args(self, _):
    pass


class _TextEmbeddingHandler(_EmbeddingHandler):
  """
  A ModelHandler intended to be work on list[dict[str, str]] inputs.

  The inputs to the model handler are expected to be a list of dicts.

  For example, if the original mode is used with RunInference to take a
  PCollection[E] to a PCollection[P], this ModelHandler would take a
  PCollection[dict[str, E]] to a PCollection[dict[str, P]].

  _TextEmbeddingHandler will accept an EmbeddingsManager instance, which
  contains the details of the model to be loaded and the inference_fn to be
  used. The purpose of _TextEmbeddingHandler is to generate embeddings for
  text inputs using the EmbeddingsManager instance.

  If the input is not a text column, a RuntimeError will be raised.

  This is an internal class and offers no backwards compatibility guarantees.

  Args:
    embeddings_manager: An EmbeddingsManager instance.
  """
  def _validate_column_data(self, batch):
    if not isinstance(batch[0], (str, bytes)):
      raise TypeError(
          'Embeddings can only be generated on dict[str, str].'
          f'Got dict[str, {type(batch[0])}] instead.')

  def get_metrics_namespace(self) -> str:
    return (
        self._underlying.get_metrics_namespace() or
        'BeamML_TextEmbeddingHandler')


class _ImageEmbeddingHandler(_EmbeddingHandler):
  """
  A ModelHandler intended to be work on list[dict[str, Image]] inputs.

  The inputs to the model handler are expected to be a list of dicts.

  For example, if the original mode is used with RunInference to take a
  PCollection[E] to a PCollection[P], this ModelHandler would take a
  PCollection[dict[str, E]] to a PCollection[dict[str, P]].

  _ImageEmbeddingHandler will accept an EmbeddingsManager instance, which
  contains the details of the model to be loaded and the inference_fn to be
  used. The purpose of _ImageEmbeddingHandler is to generate embeddings for
  image inputs using the EmbeddingsManager instance.

  If the input is not an Image representation column, a RuntimeError will be
  raised.

  This is an internal class and offers no backwards compatibility guarantees.

  Args:
    embeddings_manager: An EmbeddingsManager instance.
  """
  def _validate_column_data(self, batch):
    # Don't want to require framework-specific imports
    # here, so just catch columns of primatives for now.
    if isinstance(batch[0], (int, str, float, bool)):
      raise TypeError(
          'Embeddings can only be generated on dict[str, Image].'
          f'Got dict[str, {type(batch[0])}] instead.')

  def get_metrics_namespace(self) -> str:
    return (
        self._underlying.get_metrics_namespace() or
        'BeamML_ImageEmbeddingHandler')
