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

"""Tests for tagged output type hints.

This tests the implementation of BEAM-4132/BEAM-18957: type hints for tagged
outputs. Users can now specify type hints for tagged outputs using kwargs:

  @with_output_types(int, errors=str, warnings=str)
  class MyDoFn(beam.DoFn):
    ...

  beam.ParDo(MyDoFn()).with_output_types(int, errors=str)
"""

# pytype: skip-file

import unittest
from typing import Tuple

import apache_beam as beam
from apache_beam import typehints
from apache_beam.typehints import with_input_types
from apache_beam.typehints import with_output_types
from apache_beam.typehints.decorators import IOTypeHints


class IOTypeHintsTaggedOutputTest(unittest.TestCase):
  """Tests for IOTypeHints.tagged_output_types() accessor."""

  def test_empty_hints_returns_empty_dict(self):
    empty = IOTypeHints.empty()
    self.assertEqual(empty.tagged_output_types(), {})

  def test_with_tagged_types(self):
    hints = IOTypeHints.empty().with_output_types(int, errors=str, warnings=str)
    self.assertEqual(hints.tagged_output_types(), {'errors': str, 'warnings': str})

  def test_simple_output_type_with_tagged_types(self):
    """simple_output_type() should still return main type when tags present."""
    hints = IOTypeHints.empty().with_output_types(int, errors=str, warnings=str)
    self.assertEqual(hints.simple_output_type('test'), int)

  def test_without_tagged_types(self):
    """Without tagged types, tagged_output_types() returns empty dict."""
    hints = IOTypeHints.empty().with_output_types(int)
    self.assertEqual(hints.tagged_output_types(), {})
    self.assertEqual(hints.simple_output_type('test'), int)


class DecoratorStyleTaggedOutputTest(unittest.TestCase):
  """Tests for tagged output type hints using decorator style."""

  def test_decorator_type_hints(self):
    """Test that decorator correctly sets tagged output types."""

    @with_input_types(int)
    @with_output_types(int, errors=str, warnings=str)
    class MyDoFn(beam.DoFn):
      def process(self, element):
        if element < 0:
          yield beam.pvalue.TaggedOutput('errors', f'Negative: {element}')
        elif element == 0:
          yield beam.pvalue.TaggedOutput('warnings', 'Zero value')
        else:
          yield element * 2

    dofn = MyDoFn()
    hints = dofn.get_type_hints()

    self.assertEqual(hints.simple_output_type('MyDoFn'), int)
    self.assertEqual(
        hints.tagged_output_types(), {
            'errors': str, 'warnings': str
        })

  def test_decorator_pipeline_propagation(self):
    """Test that tagged types propagate through pipeline."""

    @with_input_types(int)
    @with_output_types(int, errors=str)
    class MyDoFn(beam.DoFn):
      def process(self, element):
        if element < 0:
          yield beam.pvalue.TaggedOutput('errors', f'Negative: {element}')
        else:
          yield element * 2

    with beam.Pipeline() as p:
      results = (
          p
          | beam.Create([-1, 0, 1, 2])
          | beam.ParDo(MyDoFn()).with_outputs('errors', main='main'))

      _ = results.main  # Main output (unused in this test)
      errors = results.errors

      # Verify types are propagated
      self.assertEqual(errors.element_type, str)


class MethodChainStyleTaggedOutputTest(unittest.TestCase):
  """Tests for tagged output type hints using method chain style."""

  def test_method_chain_type_hints(self):
    """Test with_output_types method chain style."""

    class SimpleDoFn(beam.DoFn):
      def process(self, element):
        if element < 0:
          yield beam.pvalue.TaggedOutput('errors', f'Negative: {element}')
        else:
          yield element * 2

    with beam.Pipeline() as p:
      results = (
          p
          | beam.Create([-1, 0, 1, 2])
          | beam.ParDo(SimpleDoFn()).with_output_types(
              int, errors=str).with_outputs('errors', main='main'))

      _ = results.main  # Main output (unused in this test)
      errors = results.errors

      self.assertEqual(errors.element_type, str)


class ComplexTypesTaggedOutputTest(unittest.TestCase):
  """Tests for tagged output type hints with complex types."""

  def test_tuple_type_hint(self):
    """Test with Tuple type hint for tagged output."""

    @with_output_types(int, failures=Tuple[str, str])
    class ProcessWithFailures(beam.DoFn):
      def process(self, element):
        try:
          if element == 'bad':
            raise ValueError("Bad element")
          yield len(element)
        except Exception as e:
          yield beam.pvalue.TaggedOutput('failures', (element, str(e)))

    with beam.Pipeline() as p:
      results = (
          p
          | beam.Create(['hello', 'bad', 'world'])
          | beam.ParDo(ProcessWithFailures()).with_outputs(
              'failures', main='main'))

      failures = results.failures

      expected_type = typehints.Tuple[str, str]
      self.assertEqual(failures.element_type, expected_type)


class BackwardsCompatibilityTest(unittest.TestCase):
  """Tests for backwards compatibility with existing code."""

  def test_decorator_without_tagged_types(self):
    """Old style decorator without tagged types should still work."""

    @with_output_types(int)
    class OldStyleDoFn(beam.DoFn):
      def process(self, element):
        yield element * 2

    dofn = OldStyleDoFn()
    hints = dofn.get_type_hints()

    self.assertEqual(hints.simple_output_type('test'), int)
    self.assertEqual(hints.tagged_output_types(), {})

  def test_method_chain_without_tagged_types(self):
    """Old style method chain without tagged types should still work."""
    with beam.Pipeline() as p:
      result = (
          p
          | beam.Create([1, 2, 3])
          | beam.Map(lambda x: x * 2).with_output_types(int))
      self.assertEqual(result.element_type, int)


class FlatMapDecoratorTaggedOutputTest(unittest.TestCase):
  """Tests for FlatMap/Map with decorated functions that have tagged outputs."""

  def test_flatmap_decorated_function_tagged_types(self):
    """Test FlatMap with @with_output_types decorated function.

    Note: For FlatMap, the output type should be Iterable[element_type] since
    the function is a generator. strip_iterable() unwraps it to element_type.
    """
    from typing import Generator

    @with_output_types(Generator[int, None, None], errors=str)
    def process_with_errors(element):
      if element < 0:
        yield beam.pvalue.TaggedOutput('errors', f'Negative: {element}')
      else:
        yield element * 2

    with beam.Pipeline() as p:
      results = (
          p
          | beam.Create([-1, 0, 1, 2])
          | beam.FlatMap(process_with_errors).with_outputs('errors', main='main'))

      _ = results.main
      errors = results.errors

      # Verify tagged output type is preserved through FlatMap
      self.assertEqual(errors.element_type, str)

  def test_map_decorated_function_tagged_types(self):
    """Test Map with @with_output_types decorated function."""

    @with_output_types(int, errors=str)
    def process_with_errors(element):
      if element < 0:
        return beam.pvalue.TaggedOutput('errors', f'Negative: {element}')
      return element * 2

    with beam.Pipeline() as p:
      results = (
          p
          | beam.Create([-1, 0, 1, 2])
          | beam.Map(process_with_errors).with_outputs('errors', main='main'))

      _ = results.main
      errors = results.errors

      # Verify tagged output type is preserved through Map
      self.assertEqual(errors.element_type, str)


class ExceptionHandlingTaggedOutputTest(unittest.TestCase):
  """Tests for exception handling tagged output type hints."""

  def test_exception_handling_type_propagation(self):
    """Test that with_exception_handling propagates type hints correctly."""

    def my_fn(x):
      if x == 0:
        raise ValueError('Zero!')
      return str(x)

    with beam.Pipeline() as p:
      good, bad = (
          p
          | beam.Create([1, 2, 0, 3])
          | beam.Map(my_fn).with_output_types(str).with_exception_handling())

      # Main output should have the explicit type hint
      self.assertEqual(good.element_type, str)

      # Dead letter output should have the correct tuple type
      # Format: Tuple[input_element, Tuple[type, str, List[str]]]
      self.assertIsNotNone(bad.element_type)
      # Check it's a Tuple type (not Any)
      self.assertTrue(
          hasattr(bad.element_type, 'tuple_types'),
          f'bad.element_type should be a Tuple, got {bad.element_type}')

  def test_exception_handling_inferred_type(self):
    """Test that with_exception_handling works without explicit type hints."""
    with beam.Pipeline() as p:
      good, bad = (
          p
          | beam.Create([1, 2, 3])
          | beam.Map(lambda x: str(x)).with_exception_handling())

      # Dead letter output should still have a proper type
      self.assertIsNotNone(bad.element_type)
      self.assertTrue(
          hasattr(bad.element_type, 'tuple_types'),
          f'bad.element_type should be a Tuple, got {bad.element_type}')


if __name__ == '__main__':
  unittest.main()
