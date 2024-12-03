import unittest
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from apache_beam.ml.rag.storage.base import (
    VectorDatabaseConfig,
    VectorDatabaseWriteTransform
)

class MockWriteTransform(beam.PTransform):
    """Mock transform that returns element."""
    
    def expand(self, pcoll):
        return pcoll | beam.Map(lambda x: x)
    

class MockDatabaseConfig(VectorDatabaseConfig):
    """Mock database config for testing."""
    def __init__(self):
        self.write_transform = MockWriteTransform()
    
    def create_write_transform(self) -> beam.PTransform:
        return self.write_transform

class VectorDatabaseBaseTest(unittest.TestCase):
    def test_write_transform_creation(self):
        """Test that write transform is created correctly."""
        config = MockDatabaseConfig()
        transform = VectorDatabaseWriteTransform(config)
        self.assertEqual(transform.database_config, config)
    
    def test_pipeline_integration(self):
        """Test writing through pipeline."""
        test_data = [
            {"id": "1", "embedding": [0.1, 0.2]},
            {"id": "2", "embedding": [0.3, 0.4]}
        ]
        
        with TestPipeline() as p:
            result = (p 
                     | beam.Create(test_data)
                     | VectorDatabaseWriteTransform(MockDatabaseConfig()))
            
            # Verify data was written
            assert_that(result, equal_to(test_data))
    
    def test_invalid_config(self):
        """Test error handling for invalid config."""
        with self.assertRaises(TypeError):
            VectorDatabaseWriteTransform(None)

if __name__ == '__main__':
    unittest.main()