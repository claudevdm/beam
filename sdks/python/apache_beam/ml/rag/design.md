# RAG with Apache Beam

Author: claudius@vdmza.com  
Last updated: Nov 8, 2024

# Overview

Apache Beam currently provides several components that can be used to build Retrieval Augmented Generation (RAG) systems, including IOs for data ingestion, MLTransform for embeddings, and Enrichment for data retrieval. However, we haven't integrated these components into a cohesive RAG solution, requiring users to write significant custom code, learn and piece together multiple API’s and manage implementation details.

# Background

A typical RAG pipeline consists of two main workflows:

## Ingestion Pipeline

1. Document Reading  
   * Read documents (text, images or other multimodal) from various sources  
   * Parse and extract content  
2. Preprocessing  
   * Split documents into chunks  
   * Extract metadata (e.g., source, chunk index)  
   * Optionally enrich with LLM-generated content or other [advanced RAG techniques](#advanced-rag-techniques)  
3. Embedding Generation  
   * Convert chunks to dense vector and (optionally) sparse vector representations  
4. Vector Storage  
   * Store embeddings along with metadata (for filtering) in vector database  
   * Store chunk text in vector storage (if possible) or document store

## Query/Enrichment Pipeline

1. Query Processing  
   * Read incoming data e.g. text or images  
   * Optional data [transformation/preprocessing](#advanced-rag-techniques)  
   * Generate data embeddings  
2. Vector Search  
   * Perform similarity search on vector database  
   * Retrieve relevant document chunks if stored separately from embeddings  
   * Optionally score and rank results  
3. LLM inference  
   * Perform LLM inference with the enriched query

# Goals

This document aims to:

1. Identify engineering and ease of use gaps in current Apache Beam components required to implement core RAG ingestion and retrieval pipelines focusing on BigQuery and Vertex AI vector databases  
2. Design components to address these gaps  
3. Package the various components required to create RAG pipelines into a RAG namespace  
4. Document their usage with working example notebooks  
5. Provide reusable templates and abstractions for common RAG operations

# Non Goals

1. We acknowledge the importance of [advanced techniques](#advanced-rag-techniques), however this design focuses only on core RAG functionality. We will explore these techniques in separate notebooks and consider incorporating them into the core design based on user feedback and interest.  
2. While this design aims to establish patterns for adding vector database support, focus will only include BigQuery and Vertex AI.

# Engineering Gaps Analysis

Currently, implementing RAG pipelines requires a mix of existing Beam components and custom code. 

Let's examine the implementation patterns for both BigQuery and Vertex AI vector search if we rely on existing components:

## BigQuery Vector Search

### Ingestion pipeline

```py
# Document reading - Supported
p | beam.io.ReadFromPubSub(...)  # including other IO transforms and document processing

# Chunking - Requires Custom Implementation
  | beam.ParDo(CustomChunkingDoFn())  

# Embedding overwrites the original text column. Requires users to copy text to a placeholder embedding field.
beam.Map(add_embedding_placeholder)

# Embedding generation - Supported
  | MLTransform().with_transform(
      VertexAITextEmbeddings(
          model_name='textembedding-gecko@latest',
          columns=['embedding_placeholder'],
      )
    )

# Vector storage - Supported
  | beam.io.WriteToBigQuery(
      table=table,
      schema=schema,
      create_disposition=CREATE_IF_NEEDED,
      write_disposition=WRITE_APPEND
    )
```

### Data/Enrichment pipeline

```py
# Define complex custom query
def vector_search_query(row: beam.Row) -> str:
    """
      Generate vector similarity search query.
    Args:
        row: Beam Row containing the embedding
    Returns:
        BigQuery SQL query that:
        - Performs vector similarity search over chunks
        - Returns chunk text, chunk_id and similarity score
    """
    return f"""
    SELECT ARRAY_AGG(
        STRUCT(
            base.chunk_text,   
            base.chunk_id,     
            distance          
        ) 
    ) as chunks
    FROM VECTOR_SEARCH(
        (SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`),
        'embedding',
        (select {row.embedding} as embedding),
        top_k => 3  # Return top 3 most relevant chunks
    )
    """


# Query reading - Supported
p | beam.io.ReadFromPubSub(...)

# Embedding overwrites the original text column. Requires users to copy text to a placeholder embedding field.
beam.Map(add_embedding_placeholder)

# Embedding generation - Supported
  | MLTransform().with_transform(
      VertexAITextEmbeddings(...)
    )

# Vector search - Requires Custom Implementation
  | beam.Map(lambda x: beam.Row(**x))
  | Enrichment(BigQueryEnrichmentHandler(
      project=project,
      query_fn=vector_search_query # No batching support
    ))
```

## Vertex AI Vector Index/Search

### Ingestion pipeline

```py
# Document reading - Supported
p | beam.io.ReadFromPubSub(...)

# Chunking - Requires Custom Implementation
  | beam.ParDo(CustomChunkingDoFn())

# Embedding overwrites the original text column. Requires users to copy text to a placeholder embedding field.
beam.Map(add_embedding_placeholder)

# Embedding generation - Supported
  | MLTransform().with_transform(
      VertexAITextEmbeddings(...)
    )

# Vector storage - Requires Custom Implementation
  | beam.ParDo(CustomVertexWriterFn(  
      project=project,
      location=location,
      index_name=index_name
    ))

# Document storage - Requires separate Document Store
| beam.io.WriteToBigQuery(
            table=table,
            schema={...}
        )
```

### Data/Enrichment pipeline

```py
# Data reading - Supported
p | beam.io.ReadFromPubSub(...)

# Embedding overwrites the original text column. Requires users to copy text to a placeholder embedding field.
beam.Map(add_embedding_placeholder)

# Embedding generation - Supported
  | MLTransform().with_transform(
      VertexAITextEmbeddings(...)
    )

# Vector search - Requires Custom Implementation
  | Enrichment(CustomVertexAiVectorSearchEnrichmentHandler(  
      endpoint_name=endpoint_name,
      num_neighbors=3
    ))

# Document retrieval - Requires Additional Query to retrieve actual chunk text
  | Enrichment(BigQueryEnrichmentHandler(...))

```

## Beam Engineering gaps

| Stage | BigQuery Vector Search | Vertex AI Vector Search | Notes |
| :---- | ----- | ----- | :---- |
| Chunking | ❌Not supported |  | Ideally emits [Chunk](#chunk) |
| Embedding | ⚠️Can deliver better user experience |  | User is required to write additional logic for retaining embeddable text Ideally emits [Embedding](#embedding) |
| Ingestion | ⚠️Can deliver better user experience and support batching | ❌Not supported | Ideally processes  [Embedding](#embedding) Potentially define a common interface for various Vector DB writers |
| Enrichment | ⚠️Not adequately supported | ❌Not supported | Ideally processes [Embedding](#embedding) |

# Cohesive RAG solution

This section outlines a cohesive RAG solution within Apache Beam, aiming to provide a user-friendly and intuitive experience for building RAG pipelines.

### Ingestion pipeline

An example of such an ingestion pipeline is:

```py
# Read Document
p | “Read document” >> beam.io.ReadFromPubSub(...)

# Set up root MLTransform
| MLTransform(...)

# Add chunking step to MLTransform
  .with_chunking(beam.ml.rag.FixedSizeChunkingHandler(columns=[‘text’], chunk_size=512, overlap=50))

# Add embedding step to MLTransform
  .with_embedding(beam.ml.rag.VertexAIEmbeddingHandler(...))

# Write to Vector DB
| beam.io.VectorDatabaseWriteTransform(
    VertexVectorIndexWriterConfig(...))
  )
```

### Data/Enrichment pipeline

An example of a data/enrichment pipeline is:

```py
# Read data to be enriched
p | beam.io.ReadFromPubSub(...)
  
# Set up root MLTransform
| “Prepare data for vector search” >> MLTransform(...)

# Add embedding to MLTransform
.with_embedding(beam.ml.rag.VertexAIEmbeddingHandler(...))

# Perform vector search
| Enrichment(beam.ml.rag.VertexAIVectorSearchEnrichmentHandler(...))

# Query LLM with extra context
| “Perform LLM Inference” >> RunInference(...)
```

## Introduce Well-Defined Inputs and Outputs of Each Step

Bundling RAG functionality in a cohesive RAG solution allows us to consider defining more specific input and output types for chunking and embedding.

### [Chunk](#chunk) {#chunk}

```py
@dataclass
class Chunk:
  id: str
  index: int
  text: str
  metadata: Dict[str, any] # e.g. document source, summary of doc
```

* Emitted by chunking transforms  
* Processed by embedding transforms

### [Embedding](#embedding) {#embedding}

```py
@dataclass
class Embedding:
    id: str,
    dense_embedding: Optional[List[Float]],
    sparse_embedding: Optional[Tuple[List[int], List[Float]]] # For Hybrid search
    metadata: Dict[str, any]
```

* Emitted by embedding transforms   
* Consumed by vector search enrichment

Using `Chunk` and `Embedding` types have the following tradeoffs

| Category | Benefits | Drawbacks |
| :---- | :---- | :---- |
| Development Experience | ✅ Improved code readability |  |
| Data Flow | ✅ Reduced information passing overhead ✅ Clear data contracts between stages | ❌ Less flexible data structures |
| Maintenance | ✅ Self-documenting interfaces | ❌ Breaking changes on refactoring |

# Chunking

Chunking is the process of dividing large text documents into smaller, manageable units called chunks, which are then stored in a vector database for efficient retrieval.

Chunking can help with

* Precision  
  * Smaller chunks lead to more precise retrieval  
* Vector Embedding Quality  
  * Most embedding models have token limits  
  * Chunking ensures the text segments fit into these limits  
* Context Window Management  
  * LLMs can also fit limited amount of tokens into their context window  
  * Smaller chunks allow fitting multiple relevant chunks in the context window in the case of a large knowledge base

Regardless of the chunking method, it is crucial to ensure that the text segment fits into both the embedding model token limit, and depending on how many chunks are retrieved that the chunks fit into the LLM context window.

## LangChain

LangChain is a popular library for working with language models that offers chunking. We leverage LangChain for its wide variety of chunking methods and ease of use. 

The Langchain library expresses chunking as a text splitter:

```py
class TextSplitter:
    """Interface for splitting text into chunks."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        length_function: Callable[[str], int] = len
    ) -> None:
        """
       Args:
            chunk_size: Maximum size of chunks to return
            chunk_overlap: Overlap in characters between chunks
            length_function: Function that measures the length of chunks
        """

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into multiple components."""

```

This interface supports a variety of chunking methods including 

* `CharacterTextSplitter`, which simply splits by number of characters  
* `RecursiveCharacterTextSplitter`, the recommended splitter which can be configured to split recursively on a list of separators  
* And many more, including an experimental `SemanticChunker`

A TextSplitter implementation generally

* Splits the document by some kind of separator  
* Merges the split document until the maximum `chunk_size` is reached

`TextSplitter` also allows specifying the length function, which is used to calculate the length of a given chunk. For example a HuggingFace tokenizer can be passed to count the number of tokens in a given chunk.

## MLTransform.with\_chunking

To allow the use of chunking in `MLTransform` pipelines we add a `MLTransformProvider` that can be extended to implement custom chunking strategies:

```py
class ChunkingTransformProvider(MLTransformProvider):
    """
    Base class for chunking transforms that can be used with 
    MLTransform.
    
    Subclasses must implement get_text_splitter_transform() to provide 
    the specific chunking implementation.
    """
    def __init__(
        self,
        document_field: str,
        metadata_fields: List[str],
        chunk_id_fn: Optional[Callable[[Type[Chunk]], str]]] = None
    ):
        self.document_column = document_column
        self.metadata_columns = metadata_columns
        self.chunk_id_fn = chunk_id_fn if chunk_id_fn is not None else

    @abc.abstractmethod
    def get_text_splitter_transform(self) -> beam.DoFn:
        """Return specific chunking DoFn implementation."""
        raise NotImplementedError

    def get_ptransform_for_processing(self, **kwargs) -> beam.PTransform:
        """Returns a PTransform that applies the chunking transform."""
        return (
            "Split text" >> self.get_text_splitter_transform()
            | "Assign chunk ID" >> beam.Map(assign_chunk_id)

# Example chunk_id_fn
def assign_chunk_id(element: Type[Chunk])
  element.id = f"{element.metadata['source_file']}_{element.index}"
  return element
```

Note the `chunk_id_fn` is used to create chunk id’s given access to the metadata and chunk index. It allows users to create deterministic chunk ids for example when updating the embedding. If not specified, we will assign a default unique id to the chunk.

To enable using LangChain text splitters:

```py
class LangChainTextSplitter(beam.DoFn):
    def __init__(
        self,
        text_splitter: langchain.text_splitter.TextSplitter,
        document_column: str,
        metadata_columns: List[str]
    ):
        self.splitter = text_splitter
        self.document_column = document_column
        self.metadata_columns = metadata_columns 

    def process(self, element):
        text_chunks = self.splitter.split_text(element[self.document_column])
        chunk_metadata = {k: element[k] for k in self.metadata_columns}

        for i, chunk_text in enumerate(chunks):
            chunk = Chunk(index=i, text=chunk_text, metadata=chunk_metadata)
            yield chunk

class LangChainChunkingProvider(ChunkingTransformProvider):
    """
    ChunkingTransformProvider implementation that uses LangChain 
    text splitters.
    
    Args:
        text_splitter: A LangChain TextSplitter instance to use for
            chunking. 
    """
    def __init__(
        self,
        document_column: str,
        metadata_columns: List[str],
        text_splitter: langchain.text_splitter.TextSplitter
    ):
        super().__init__(document_column=document_column, metadata_columns=metadata_columns)
        self.text_splitter = text_splitter
        
    def get_text_splitter_transform(self) -> beam.PTransform:
        """Returns a LangChainTextSplitter transform configured with this provider's splitter."""
        return beam.ParDo(
            LangChainTextSplitter(
                text_splitter=self.text_splitter,
                document_column=self.document_column,
                metadata_columns=self.metadata_columns))


```

## Langchain Chunking example

A pipeline using LangChain for chunking looks as follows:

```py
recursive_text_splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=5,
    length_function=len,
    separators=["\n\n", "\n", ". ", ", ", " ", ""]
)

p 
| beam.Create([{'source_file':"abc.txt", 'content':"..."}])
| MLTransform(...)
    .with_chunking(
       LangChainChunkingProvider(
         document_column="content",
         metadata_columns=["source_file"],
         text_splitter=recursive_text_splitter
       )
    )

```

# Embedding

This section recaps the existing embedding interface and proposes a RAG-embedding transform interface.

## Existing Interface

The existing embedding interface, while functional, presents some challenges for RAG pipelines:

* Overwrites the original text field and requires overhead  
* Lacks support for the `Chunk` `Embedding` dataclasses causing information passing overhead


```py

class SentenceTransformerEmbeddings(EmbeddingsManager):
  def __init__(
      self,
      model_name: str,
      columns: list[str],
      max_seq_length: Optional[int] = None,
      image_model: bool = False,
      **kwargs):
    """
    Args:
      model_name: Name of the HuggingFace model to use.
      columns: List of columns to be embedded.
      max_seq_length: Max sequence length to use for the model if applicable.
      image_model: Whether the model is generating image embeddings.
      min_batch_size: The minimum batch size to be used for inference.
      max_batch_size: The maximum batch size to be used for inference.
      large_model: Whether to share the model across processes.
    """

class VertexAITextEmbeddings(EmbeddingsManager):
  def __init__(
      self,
      model_name: str,
      columns: list[str],
      title: Optional[str] = None,
      task_type: str = DEFAULT_TASK_TYPE,
      project: Optional[str] = None,
      location: Optional[str] = None,
      credentials: Optional[Credentials] = None,
      **kwargs):
    """
    Args:
      model_name: The name of the Vertex AI Text Embedding model.
      columns: The columns containing the text to be embedded.
      task_type: The downstream task for the embeddings. Valid values are
        RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, SEMANTIC_SIMILARITY,
        CLASSIFICATION, CLUSTERING.
      title: Identifier of the text content.
      project: The default GCP project for API calls.
      location: The default location for API calls.
      credentials: Custom credentials for API calls.
        Defaults to environment credentials.
    """
```

Existing embedding transforms are appended to an MLTransform as: `MLTransform(...).with_transform(SentenceTransformerEmbeddings(...))`

Embedding transforms process elements as batches of dictionaries, and the columns argument specifies which columns should be embedded. The embedding overwrites the original text field in the output elements. For example:

```py
beam.Create({'text': 'This text will be embedded'})
| MLTransform(...).with_transform(VertexAITextEmbeddings(columns=['text'], ...)
| beam.Map(print)

Output:
{'text': [0.5,....,0.1] 
```

## RAG-Embedding Interface

This section outlines steps we can take to make the embedding interfaces more intuitive for RAG.

### Step 1: Embedding interface

Here we consider two options:

* Process `Chunks`   
* Emit `Embeddings`

| Aspect | Option A (Preferred) | Option B |
| :---- | :---- | :---- |
| Approach | Create `beam.ml.rag.EmbeddingTransformHandler` base class which is extended for individual embedding providers | Separate embedding class for each embedding provider e.g. `beam.ml.rag.RAGVertexAIEmbeddingTransform` |
| Usage | `MLTransform.with_embedding(VertexAIEmbeddingHandler(...))` | `MLTransform.with_embedding(RAGVertexAIEmbeddingTransform(...))` |

### Step 2: Bundle embedding transforms in beam.ml.rag namespace

This makes discovering RAG components easier.

### Step 3: Add MLTransform.with\_embedding or MLTransform.with\_chunk\_embedding

* Both options make the MLTransform pipeline more descriptive   
* `with_chunk_embedding` makes it clear that `Chunk` types are being processed

# Writing to Vector Databases

This section describes a common interface for writing data to vector databases while outlining the Vertex AI Vector Index writer implementation.

## Adding a common Vector DB interface

We define a common interface type for Vector DB writers that handles Embedding types:

```py
class VectorDatabaseConfig(ABC):
  @abstractmethod
  def create_write_transform(self) -> beam.PTransform:
    pass

class BigQueryVectorWriterConfig(VectorDatabaseConfig):
  def _init_(self, ...):
    # BigQuery specific configurations - managed IO params
    pass

  def create_write_transform(self) -> beam.PTransform:
    return _WriteToBigQueryVectorDatabase(self)


class VertexAIVectorWriterConfig(VectorDatabaseConfig):
  def _init_(self, ...):
    # VertexAI specific configurations
    pass

  def create_write_transform(self) -> beam.PTransform:
    return _WriteToVertexAIVectorSearch(self)

class VectorDatabaseWriteTransform(beam.PTransform):
  def _init_(self, database_config: VectorDatabaseConfig):
    self.database_config = database_config

def expand(self, pcoll):
  write_transform = self.database_config.create_write_transform()
    return pcoll | write_transform
```

## Vertex AI

One of the gaps identified is the lack of a Vertex AI Vector Index write transform.

Vertex AI stores vectors in indices that can be configured as:

* `STREAMING` \- Direct upserts through API  
* `BATCH` \- Data staged in GCS buckets (JSON/Avro/CSV)

Supports:

* Dense embeddings  
* Sparse embeddings  
* Hybrid Search (Public Review)

Document Storage considerations:

* Vertex AI vector index does not store document text  
* Requires separate document store (e.g. GCS, BigQuery)  
* Options:  
  * Provide default document stores integrated into a VertexVectorStore  
  * Let users manage document storage (requires two-step insertion)

The initial implementation will focus on supporting streaming indices.

### Vertex AI Streaming Inserts

The following snippet demonstrates how data is inserted into a streaming index:

```py
def stream_update_vector_search_index(
    project: str, location: str, index_name: str, datapoints: Sequence[dict]
) -> None:
    """Stream update an existing vector search index

    Args:
      project (str): Required. The Project ID
      location (str): Required. The region name, e.g. "us-central1"
      index_name (str): Required. The index to update. A fully-qualified index
        resource name or an index ID.  Example:
        "projects/123/locations/us-central1/indexes/my_index_id" or
        "my_index_id".
      datapoints: Sequence[dict]: Required. The datapoints to be updated. The dict
        element should be of the IndexDatapoint type.
    """
    # Initialize the Vertex AI client
    aiplatform.init(project=project, location=location)

    # Create the index instance from an existing index with stream_update
    # enabled
    my_index = aiplatform.MatchingEngineIndex(index_name=index_name)

    # Upsert the datapoints to the index
    my_index.upsert_datapoints(datapoints=datapoints)
```

Example of input `datapoints` formatted as JSON

```py
[
  {
    "id": "43",
    "embedding": [
      0.6,
      1
    ],
    "sparse_embedding": {
      "values": [
        0.1,
        0.2
      ],
      "dimensions": [
        1,
        4
      ]
    },
    "restricts": [
      {
        "namespace": "class",
        "allow": [
          "dog",
          "pet"
        ]
      },
      {
        "namespace": "category",
        "allow": [
          "canine"l
        ]
      }
    ]
  }
]

```

Note: metadata needs to be transformed into a `restricts` field.

### VertexAIVectorWriterConfig

To configure a Vertex AI Vector store writer, we need:

* GCP Project ID  
* Location  
* Index Name  
* To convert the Embedding metadata to the restricts format expected by the vertex ingestion API

```py
@dataclass
class VertexAIVectorWriterConfig:
    project: str
    location: str
    index_name: str
    metadata_to_restricts_fn: Callable[dict, list[dict]]
```

VertexAIVectorWriterConfig can be extended to support batch index inserts in the future with additional arguments, or a separate config VertexAIVectorBatchWriterConfig.

### \_WriteToVertexAIVectorSearch

Given VertexAIVectorWriterConfig, we implement a write operation using WriteToVertexAIVectorSearch.

```py
class _WriteToVertexAIVectorSearch(PTransform):
    def __init__(
      self, 
      config: Type[VertexAIVectorWriterConfig]
    )

    def expand(self, pcoll):
      return (pcoll
        | "Convert to datapoints" >> ParDo(ConvertToDatapoints(...))
        | "Write datapoints" >> ParDo(WriteDatapoints())
      )


vertex_writer_config = VertexAIVectorWriterConfig(...)

# Pipeline example
p
| VectorDatabaseWriteTransform(VectorDatabaseConfig(vertex_config=vertex_writer_config))
```

The writer implementation should provide:

* Dead Letter Queues (DLQ) for capturing and managing failed operations  
* Exponential backoff retry mechanisms for transient failures  
* Detailed error logging and monitoring

# Enrichment

In this section we discuss the options for supporting BigQuery and Vertex AI vector search during Enrichment.

## BigQuery Vector Search Enrichment

First we recap the syntax for doing vector search in BigQuery and then discuss how it can be incorporated into Enrichment.

### BigQuery VECTOR\_SEARCH

The BigQuery [`VECTOR_SEARCH`](https://cloud.google.com/bigquery/docs/reference/standard-sql/search_functions#vector_search) function is used to perform similarity searches on vector fields in BigQuery.

```
VECTOR_SEARCH(
  {TABLE base_table | base_table_query_statement}, # Table to search, or a query that prefilters the base table
  column_to_search, # Embedding column 
  TABLE query_table # Table that provides embeddings for which to find nearest neighbors
  [, query_column_to_search => query_column_to_search_value] # Embedding column name in query_table
  [, top_k => top_k_value ] # Number of nearest neighbors to return
  [, distance_type => distance_type_value ] # Distance type function to use, options are EUCLIDEAN, COSINE and DOT_PRODUCT. Default is EUCLIDEAN
  [, options => options_value ] # 
)

```

Currently, using `VECTOR_SEARCH` with BigQueryEnrichmentHandler requires a complex `query_fn`. This approach also prevents batching, which can impact performance.

We want to support BigQuery `VECTOR_SEARCH` for enrichment in a flexible way. To do this, we'll define a new data class called `BigQueryVectorSearchParameters`:

### BigQueryVectorSearchParameters

BigQueryVectorSearchParameters encapsulates the arguments required to support VECTOR\_SEARCH in enrichment:

```py
@dataclass
class BigQueryVectorSearchParameters:
    table_name: str
    embedding_column: str
    columns: str
    neighbor_count: int
    metadata_restriction_template: str
    distance_type: Optional[Type[DistanceType]]
    options: [Optional[Type[Options]] # Options mirrors additional options from VECTOR_SEARCH

vector_search_params = BigQueryVectorSearchParameters(
  table_name='sec_filings',
  embedding_column='embedding',
  columns=['chunk_text', 'document_summary'],
  neighbor_count=1,
  metadata_restriction_template="company = {company_name}"
)

vector_search_template = 
"""
   SELECT ARRAY_AGG(
        STRUCT(
            {vector_search_params.columns} # Simplified, need to prepend columns with base.column and comma separate
        ) 
    ) as chunks
    FROM VECTOR_SEARCH(
        (SELECT 
           {vector_search_params.columns} # Simplified, need to comma separate columns
         FROM `{vector_search_params.table_name}`
         WHERE {{metadata_restrictions}} # Double braces because metadata_restrictions to be populated during final query construction
         ),
        '{vector_search_params.embedding_column}',
        (select {{embedding}} as embedding), #  Double braces because embedding to be populated during final query construction
        top_k => {vector_search_params.neighbor_count},
        {format_distance_type(distance_type)} # Optional, if not provided this is empty
        {format_options(options)} # Optional, if not provided this is empty
    )
"""
```

### Enrichment Execution

The enrichment handler processes `Embedding` types to fill the remaining `{embedding}` and `{metadata_restrictions}` placeholders in the `VECTOR_SEARCH` query template:

```py
class BigQueryEnrichmentHandler:
    def __call__(
      self,
      embedding: Type[Embedding]
    ):
      self.vector_search_params.format(
        metadata_restrictions=self.metadata_restriction_template(embedding.metadata),
        embeddings=embedding.dense_embedding
      )

# Example of pipeline execution
p
| beam.Create(
     [Embedding(
         id="1", 
         dense_embedding=[0.1, 0.2], 
         metadata={'company_name': 'foo'}
      ]
  )
| Enrichment(BigQueryEnrichmentHandler(vector_search_params))
| beam.Map(print)
```

### Options for Integrating BigQuery VECTOR\_SEARCH Support in Enrichment

In this section we explore two options for supporting BigQuery vector search in enrichment.

#### Option 1: Create BigQueryVectorSearchEnrichmentHandler

```py
class BigQueryVectorSearchEnrichmentHandler(
      EnrichmentSourceHandler[Union[Embedding, List[Embedding]],Union[Embedding, List[Embedding]]]):
    def __init__(
        self,
        project: str,
        *,
        vector_search_parameters: Type[BigQueryVectorSearchParameters],
        min_batch_size: int = 1,
        max_batch_size: int = 1000,
     )
```

This handler either:

* Inserts the retrieved fields to `embedding.metadata[‘enrichment_output’]`  
* Returns tuples of `(Embedding, dict<enrichment_data>)`

#### Option 2: Modify BigQueryVectorSearchEnrichmentHandler to handle vector search

Below is the current constructor for BigQueryEnrichmentHandler:

```py
class BigQueryEnrichmentHandler(EnrichmentSourceHandler[Union[Row, List[Row]],
                                                        Union[Row, List[Row]]]):
   def __init__(
      self,
      project: str,
      *,
      table_name: str = "",
      row_restriction_template: str = "",
      fields: Optional[List[str]] = None,
      column_names: Optional[List[str]] = None,
      condition_value_fn: Optional[ConditionValueFn] = None,
      query_fn: Optional[QueryFn] = None,
      vector_search_params: Optional[Type[BigQueryVectorSearchParameters]] # New param to support vector search.
      min_batch_size: int = 1,
      max_batch_size: int = 10000,
      **kwargs,
  ):
```

To recap, existing `BigQueryEnrichmentHandler` requires:

1. `table_name` to specify the table to be queried  
2. A way to construct the `WHERE` clause which is either:  
   * `(fields OR condition_value_fn) AND row_restriction_template`   
   * `query_fn` that creates a complete BigQuery query, and does not support batching

Supporting vector search in `BigQueryEnrichmentHandler` requires either:

1. Adding a trimmed version of `BigQueryVectorSearchParameters` that just contains `VECTOR_SEARCH` specific arguments and reusing some of the existing `BigQueryEnrichmentHandler` parameters  
   1. `table_name` can be used for `VECTOR_SEACH` `base_table`  
   2. `column_names` can be reused to specify which columns to return from the `base_table`  
   3. Reuse the `row_restriction_template` along with `(fields OR condition_value_fn)` logic to build the base table prefilter statement  
2. Changing the interface of `BigQueryEnrichmentHandler` so that vector search parameters are clearly separated from regular query parameters

```py
class BigQueryEnrichmentHandler(EnrichmentSourceHandler[Union[Row, List[Row],Embedding, List[Embedding]], Union[Row, List[Row], Embedding, List[Embedding]]]):
   def __init__(
      self,
      project: str,
      *,
      query_params: Optional[Type[BigQueryParameters]] # Parameters for regular queries
      vector_search_params: Optional[Type[BigQueryVectorSearchParameters]] # New param to support vector search.
      min_batch_size: int = 1,
      max_batch_size: int = 10000,
      **kwargs,
  ):
```

#### Comparing Option 1 and Option 2:

|  | Option 1: New `BigQueryVectorSearchEnrichmentHandler`  Handler (Preferred) | Option 2: Modify Existing Handler |
| :---- | :---- | :---- |
| Code Organization | ✅ Clean separation of concerns ✅ Dedicated vector search interface | ❌ Mixed responsibilities ❌ More complex interface |
| Type Safety | ✅ Strong typing with Embedding class ✅ Clear input/output contracts | ❌ Mixed type support ❌ Complex Union types |
| Maintenance | ✅ Easier to maintain ✅ Simpler testing | ❌ More difficult to maintain ❌ Complex test scenarios |
| Code Reuse | ❌ Some code duplication ❌ Shared code needs refactoring | ✅ Maximum code reuse ✅ Single implementation |
| API Clarity | ✅ Purpose-built interface | ❌ More complex parameters |
| Future Extensions | ✅ Easy to add new features  | ❌ Risk of growing complexity |

## Vertex AI Vector Search Enrichment

First we recap the syntax for doing vector search in Vertex AI and then discuss how it can be incorporated into Enrichment.

An `aiplatform.MatchingEngineIndexEndpoint` is used to query Vertex AI Vector Index. The vector search index needs to be deployed to the `MatchingEngineIndexEndpoint`. 

Endpoints can be public, private (VPC-peering) or privacy service connect, see [setup instructions](https://cloud.google.com/vertex-ai/docs/vector-search/setup/format-structure).

Querying an endpoint is done via: `aiplatform.MatchingEngineIndexEndpoint(...).find_nearest_neighbors(...)`

Summarized `find_nearest_neighbors`:

```py
def find_nearest_neighbors(
    self,
    deployed_index_id: str,
    queries: Optional[Union[List[List[float]], List[HybridQuery]]] = None,
    num_neighbors: int = 10,
    filter: Optional[List[Namespace]] = None,
    per_crowding_attribute_neighbor_count: Optional[int] = None,
    approx_num_neighbors: Optional[int] = None,
    fraction_leaf_nodes_to_search_override: Optional[float] = None,
    return_full_datapoint: bool = False,
    numeric_filter: Optional[List[NumericNamespace]] = None,
    embedding_ids: Optional[List[str]] = None,
) -> List[List[MatchNeighbor]]:
    """Find nearest neighbors in the deployed index.
    
    Args:
        deployed_index_id: ID of the deployed index to search
        queries: Vector queries (dense or hybrid) OR
        embedding_ids: IDs to lookup embeddings before searching
        num_neighbors: Number of neighbors to return per query
        filter: Token-based filtering
        per_crowding_attribute_neighbor_count: Max neighbors per crowding tag
        approx_num_neighbors: Number of neighbors for approximate search
        fraction_leaf_nodes_to_search_override: Fraction of leaves to search (0-1)
        return_full_datapoint: Whether to return complete vector data
        numeric_filter: Numeric range filtering
    
    Returns:
        List of nearest neighbors for each query
    
    Example:
        >>> endpoint.find_nearest_neighbors(
        ...     deployed_index_id="my_index",
        ...     queries=[[1.0, 2.0, 3.0]],
        ...     num_neighbors=5,
        ...     filter=[Namespace("category", ["electronics"], [])],
        ...     return_full_datapoint=True
        ... )
    """

```

### VertexAIVectorSearchEnrichmentHandler

Here we discuss how to incorporate the `find_nearest_neighbors` call into an enrichment handler:

```py
class VertexAIVectorSearchEnrichmentHandler(EnrichmentSourceHandler):
  def __init__(
      self,
      project: str,
      location: str,
      index_endpoint_name: str,
      deployed_index_id: str,
      num_neighbors: int.
      metadata_to_filters_fn: Callable[dict, tuple]
  ):
    self.project = project
    ...
      
    def __enter__(self):
        aiplatform.init(project=self.project, location=self.location)
        self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=self.index_endpoint_name
        )
        
    def __call__(self, request: Union[Type[Chunk], Type[Chunk]):
        
        filter_to_request: Dict(tuple, List[Chunk]) = dict() 
	 
        # filter in find_neibhors applies to the entire collection of queries,
        # therefore we batch requests based on filter.
        # Theres probably be a better place to perform this kind of batching
        for chunk in request:
          filters = self.metadata_to_filters_fn(chunk.metadata)
          restrict_to_request[filters].append(chunk.id)
        
        # Do batches of queries
        for filter, chunk in restrict_to_request:
            responses.append(self.index_endpoint.find_neighbors(
                deployed_index_id=self.deployed_index_id,
                queries=[[chunk.embedding for chunk in request]], # Can also support sparse embeddings for hybrid search.
                num_neighbors=self.num_neighbors,
                filter=filter
              )
            )
        
        return responses
```

Importantly, the `find_neighbors` filter is applied to the search space for all the embeddings specified in `queries`. Therefore batching needs to happen per unique filter specification.

## Base VectorSearchEnrichmentHandler and MLTransform.with\_enrichment

We also have the option to add a base VectorSearchEnrichmentHandler for processing elements of type `Embedding`, and allow chaining to MLTransform.

This example shows how users could specify the vector database for enrichment:

```py
@dataclass
class VectorSearchConfig:
    # Exactly one needs to be set at a time
    vertex_config: Optional[TypeOf[VertexAIVectorSearchConfig]]
    bigquery_config: Optional[TypeOf[BigQueryVectorSearchConfig]]

class MLTransform:
    def with_enrichment(self, config: VectorSearchConfig):
        # Delegates to appropriate enrichment handler for given config 

# All rag operations can be chained in an MLTransform
MLTransform.with_chunking(...).with_embedding(...).with_enrichment(VectorSearchConfig(VertexAIVectorSearchConfig(...))
    
```

# Conclusion

1. Current State  
   * Existing components support parts of RAG workflow but lack integration  
   * Users must implement significant custom code  
   * No standardized patterns for common RAG operations  
2. Proposed Solution  
   * Unified RAG namespace with consistent interfaces  
   * Standard chunking transform with LangChain integration  
   * Improved embedding interfaces with type safety  
   * Vector database abstractions for BigQuery and Vertex AI  
   * Enrichment handlers optimized for vector search  
3. Benefits  
   * Reduced boilerplate code  
   * Type-safe interfaces  
   * Consistent patterns across vector stores  
   * Extensible architecture for new capabilities  
4. Implementation Plan  
   * Add chunking transform with LangChain integration  
   * Create RAG-specific embedding interfaces  
   * Implement vector database writers  
   * Develop vector search enrichment handlers  
   * Add comprehensive examples and documentation  
5. Next Steps  
   * Implement core interfaces  
   * Add test coverage  
   * Create example notebooks  
   * Document best practices  
   * Gather user feedback

# Appendix:

## [Advanced RAG techniques](#advanced-rag-techniques) {#advanced-rag-techniques}

Advanced RAG retrieval techniques can broadly be grouped into retrieval and query transformations:

Advanced retrieval strategies:

1. Reranking: Refines the initial retrieval results by using another model or method to re-rank them based on relevance.  
2. Recursive Retrieval: Performs multiple rounds of retrieval, using the results of one round to inform the next, for more in-depth exploration.  
3. Small-to-Big Retrieval: Starts with a small subset of data and gradually expands the search space, improving efficiency for large datasets.

Data transformation strategies:

1. Query Decomposition: Breaks down complex questions into smaller sub-questions that are easier to answer.  
   1. Single-step Decomposition: Divides the query into sub-questions once.  
   2. Multi-step Decomposition: Iteratively breaks down sub-questions into even smaller ones.  
2. HyDE (Hypothetical Document Embeddings): Generates a hypothetical answer to your query and uses its embeddings to find semantically similar documents.  
3. Contextual Retrieval: Adds relevant surrounding text to each chunk, providing more context and clarifying its meaning. This "contextualized chunk" now contains more information for accurate retrieval.

