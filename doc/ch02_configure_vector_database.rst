2: Configure Vector Database
=============================

Overview
--------

This chapter covers the configuration of Qdrant vector database to support hybrid search capabilities. We'll set up collections with both dense and sparse vector configurations, enabling efficient storage and retrieval of document chunks using multiple embedding strategies simultaneously.

Purpose and Architecture
------------------------

The vector database serves as the central repository for all embeddings and document chunks. By configuring it with hybrid search capabilities, we enable:

- **Dense Vector Search**: Semantic similarity matching using transformer embeddings
- **Sparse Vector Search**: Keyword-based matching using BM25 embeddings
- **Hybrid Retrieval**: Combining both search methods for optimal results
- **Scalability**: Efficient storage and retrieval of millions of vectors

Step 1: Import Required Modules
--------------------------------

Begin by importing the necessary Qdrant and LangChain components:

.. code-block:: python

    from qdrant_client.http import models as qmodels
    from langchain_qdrant import QdrantVectorStore
    from langchain_qdrant.qdrant import RetrievalMode

**Import Explanations:**

- ``qdrant_client.http.models``: Provides data models for Qdrant configuration (vectors, collections, distances)
- ``QdrantVectorStore``: LangChain wrapper for Qdrant integration
- ``RetrievalMode``: Enum specifying retrieval strategies (dense, sparse, or hybrid)

Step 2: Determine Embedding Dimension
--------------------------------------

Calculate the dimensionality of your dense embeddings:

.. code-block:: python

    embedding_dimension = len(dense_embeddings.embed_query("test"))

**Why This Matters:**

- The embedding dimension must match the vector size configured in Qdrant
- Different embedding models produce different dimensions
- ``sentence-transformers/all-mpnet-base-v2`` produces 768-dimensional vectors
- This dynamic approach ensures compatibility if you change embedding models

**Example Output:**

.. code-block:: python

    # For all-mpnet-base-v2 model
    embedding_dimension = 768

Step 3: Create Collection Initialization Function
--------------------------------------------------

Define a function to safely create Qdrant collections:

.. code-block:: python

    def ensure_collection(collection_name):
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=embedding_dimension,
                    distance=qmodels.Distance.COSINE
                ),
                sparse_vectors_config={
                    "sparse": qmodels.SparseVectorParams()
                },
            )

**Function Purpose:**

This function provides idempotent collection creation, preventing errors if the collection already exists.

Step 3a: Dense Vector Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dense vector configuration defines how semantic embeddings are stored:

.. code-block:: python

    vectors_config=qmodels.VectorParams(
        size=embedding_dimension,      # 768 for all-mpnet-base-v2
        distance=qmodels.Distance.COSINE
    )

**Configuration Details:**

- **size**: Number of dimensions in the embedding vector (768 for our model)
- **distance**: Similarity metric used for comparison

**Distance Metrics Explained:**

- **COSINE**: Measures angle between vectors (0 to 2, where 0 = identical)
  
  - Best for: Normalized embeddings, semantic similarity
  - Range: 0 (identical) to 2 (opposite)
  - Recommended for transformer embeddings

- **EUCLIDEAN**: Measures straight-line distance between points
  
  - Best for: Geometric similarity
  - Computationally more expensive
  - Suitable for non-normalized embeddings

- **DOT_PRODUCT**: Inner product of vectors
  
  - Best for: Normalized vectors with magnitude information
  - Fastest computation
  - Requires normalized embeddings

**Why COSINE for This Use Case:**

Transformer embeddings like ``all-mpnet-base-v2`` are normalized, making cosine distance ideal. It's also the most intuitive for semantic similarity (0 = identical meaning, 2 = opposite meaning).

Step 3b: Sparse Vector Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sparse vector configuration enables keyword-based retrieval:

.. code-block:: python

    sparse_vectors_config={
        "sparse": qmodels.SparseVectorParams()
    }

**Configuration Details:**

- **Named Configuration**: The "sparse" key identifies this sparse vector type
- **SparseVectorParams()**: Uses default BM25 parameters
- **Purpose**: Stores sparse embeddings for keyword matching

**Sparse vs Dense Vectors:**

.. list-table::
   :header-rows: 1

   * - Aspect
     - Dense Vectors
     - Sparse Vectors
   * - Dimensions
     - 768 (fixed)
     - Variable (thousands)
   * - Storage
     - Compact
     - Larger
   * - Search Type
     - Semantic
     - Keyword
   * - Speed
     - Fast
     - Very Fast
   * - Use Case
     - Meaning matching
     - Exact term matching

Complete Collection Setup
--------------------------

Here's the complete collection configuration code:

.. code-block:: python

    from qdrant_client.http import models as qmodels
    from langchain_qdrant import QdrantVectorStore
    from langchain_qdrant.qdrant import RetrievalMode

    # Calculate embedding dimension
    embedding_dimension = len(dense_embeddings.embed_query("test"))
    print(f"Embedding dimension: {embedding_dimension}")

    # Define collection creation function
    def ensure_collection(collection_name):
        """
        Create a Qdrant collection with hybrid search capabilities.
        
        Args:
            collection_name (str): Name of the collection to create
            
        Returns:
            None
        """
        if not client.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=embedding_dimension,
                    distance=qmodels.Distance.COSINE
                ),
                sparse_vectors_config={
                    "sparse": qmodels.SparseVectorParams()
                },
            )
            print(f"Collection {collection_name} created successfully")
        else:
            print(f"Collection {collection_name} already exists")

    # Create the child chunks collection
    ensure_collection(CHILD_COLLECTION)

Hybrid Search Capabilities
---------------------------

**What is Hybrid Search?**

Hybrid search combines dense and sparse embeddings to leverage the strengths of both:

1. **Dense Search Phase**: Finds semantically similar documents
2. **Sparse Search Phase**: Finds keyword-matching documents
3. **Fusion**: Combines results using ranking algorithms

**Example Scenario:**

Query: "How do neural networks process information?"

- **Dense Search**: Finds documents about "machine learning" and "deep learning"
- **Sparse Search**: Finds documents containing exact terms "neural networks"
- **Hybrid Result**: Returns documents that are both semantically relevant AND contain key terms

**Advantages:**

- Captures both semantic meaning and keyword relevance
- Reduces false positives from pure semantic search
- Improves recall for technical documents
- Balances precision and comprehensiveness

RetrievalMode Options
---------------------

When using QdrantVectorStore, you can specify different retrieval modes:

.. code-block:: python

    # Dense-only retrieval (semantic search)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=CHILD_COLLECTION,
        embedding=dense_embeddings,
        retrieval_mode=RetrievalMode.DENSE
    )

    # Sparse-only retrieval (keyword search)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=CHILD_COLLECTION,
        embedding=dense_embeddings,
        retrieval_mode=RetrievalMode.SPARSE
    )

    # Hybrid retrieval (recommended)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=CHILD_COLLECTION,
        embedding=dense_embeddings,
        retrieval_mode=RetrievalMode.HYBRID
    )

**Retrieval Mode Comparison:**

.. list-table::
   :header-rows: 1

   * - Mode
     - Use Case
     - Pros
     - Cons
   * - DENSE
     - Semantic understanding
     - Fast, intuitive
     - May miss keywords
   * - SPARSE
     - Exact matching
     - Precise keywords
     - Misses semantic meaning
   * - HYBRID
     - General purpose
     - Best of both worlds
     - Slightly slower

Collection Metadata and Properties
----------------------------------

After creating a collection, you can inspect its properties:

.. code-block:: python

    # Get collection information
    collection_info = client.get_collection(CHILD_COLLECTION)
    
    print(f"Collection name: {collection_info.config.params.collection_name}")
    print(f"Vector size: {collection_info.config.params.vectors.size}")
    print(f"Distance metric: {collection_info.config.params.vectors.distance}")
    print(f"Points count: {collection_info.points_count}")

**Available Information:**

- Collection configuration and parameters
- Vector dimensions and distance metrics
- Number of stored points (vectors)
- Sparse vector configuration
- Indexing status

Best Practices
--------------

**1. Collection Naming**

Use descriptive, hierarchical names:

.. code-block:: python

    # Good
    CHILD_COLLECTION = "document_child_chunks"
    PARENT_COLLECTION = "document_parent_chunks"
    
    # Avoid
    COLLECTION = "data"
    CHUNKS = "c1"

**2. Idempotent Operations**

Always check if collection exists before creating:

.. code-block:: python

    if not client.collection_exists(collection_name):
        client.create_collection(...)

**3. Dimension Consistency**

Ensure embedding dimension matches configuration:

.. code-block:: python

    # Calculate dynamically
    embedding_dimension = len(dense_embeddings.embed_query("test"))
    
    # Verify before creating collection
    assert embedding_dimension == 768, "Unexpected embedding dimension"

**4. Distance Metric Selection**

Choose appropriate distance metric for your embeddings:

.. code-block:: python

    # For normalized embeddings (transformer models)
    distance=qmodels.Distance.COSINE
    
    # For non-normalized embeddings
    distance=qmodels.Distance.EUCLIDEAN

**5. Sparse Vector Configuration**

Always include sparse vectors for hybrid search:

.. code-block:: python

    sparse_vectors_config={
        "sparse": qmodels.SparseVectorParams()
    }

Performance Considerations
--------------------------

**Memory Usage:**

- Dense vectors: ~3KB per vector (768 dimensions × 4 bytes)
- Sparse vectors: Variable, typically 1-10KB per vector
- Metadata: Additional storage for payloads

**Indexing:**

- Qdrant automatically creates indexes for efficient search
- Initial indexing may take time for large collections
- Indexes are updated incrementally as new vectors are added

**Scaling:**

- Single Qdrant instance: Millions of vectors
- Distributed Qdrant: Billions of vectors
- Partitioning: Split collections by document type or domain

Troubleshooting
---------------

**Collection Already Exists Error**

.. code-block:: python

    # Solution: Check before creating
    if not client.collection_exists(collection_name):
        client.create_collection(...)

**Dimension Mismatch Error**

.. code-block:: python

    # Problem: Embedding dimension doesn't match collection config
    # Solution: Verify dimension before creating collection
    embedding_dimension = len(dense_embeddings.embed_query("test"))
    print(f"Dimension: {embedding_dimension}")  # Should be 768

**Connection Error**

.. code-block:: python

    # Problem: Cannot connect to Qdrant
    # Solution: Ensure Qdrant is running
    # Check: client.get_collections() should return collections

**Sparse Vector Not Working**

.. code-block:: python

    # Problem: Sparse vectors not being used
    # Solution: Ensure sparse_vectors_config is included
    # Verify: Use RetrievalMode.HYBRID or RetrievalMode.SPARSE

Next Steps
----------

With the vector database configured, you can proceed to:

- Ingesting documents and creating chunks
- Generating embeddings for chunks
- Storing vectors in the configured collections
- Implementing retrieval and search functionality
- Building the complete RAG pipeline

