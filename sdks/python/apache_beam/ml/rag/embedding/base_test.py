import unittest
from apache_beam.ml.rag.types import Chunk, ChunkContent, Embedding
from apache_beam.ml.rag.embedding.base import (
    create_rag_adapter
)

class RAGBaseEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        self.test_chunks = [
            Chunk(
                content=ChunkContent(text="This is a test sentence."),
                id="1",
                metadata={"source": "test.txt", "language": "en"}
            ),
            Chunk(
                content=ChunkContent(text="Another example."),
                id="2",
                metadata={"source": "test2.txt", "language": "en"}
            )
        ]

    def test_adapter_input_conversion(self):
        """Test the RAG type adapter converts correctly."""
        adapter = create_rag_adapter()
        
        # Test input conversion
        texts = adapter.input_fn(self.test_chunks)
        self.assertEqual(
            texts,
            ["This is a test sentence.", "Another example."]
        )
    
    def test_adapter_output_conversion(self):
        """Test the RAG type adapter converts correctly."""
        # Test output conversion
        mock_embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        # Expected outputs
        expected = [
            Embedding(
                id="1",
                dense_embedding=[0.1, 0.2, 0.3],
                metadata={'source': 'test.txt', 'language': 'en'},
                content=ChunkContent(text='This is a test sentence.' )
            ),
            Embedding(
                id="2",
                dense_embedding=[0.4, 0.5, 0.6],
                metadata={'source': 'test2.txt', 'language': 'en'},
                content=ChunkContent(text='Another example.')
            ),
        ]
        adapter = create_rag_adapter()
        
        embeddings = adapter.output_fn(self.test_chunks, mock_embeddings)
        self.assertListEqual(embeddings, expected)

if __name__ == '__main__':
    unittest.main()