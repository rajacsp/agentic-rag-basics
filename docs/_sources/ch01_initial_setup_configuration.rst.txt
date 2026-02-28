1: Initial Setup and Configuration
===================================

Overview
--------

This chapter covers the foundational setup required to initialize a Retrieval-Augmented Generation (RAG) system using LangChain, Qdrant vector database, and various embedding models. The configuration establishes the core components needed for document processing, embedding generation, and semantic search capabilities.

Purpose and Architecture
------------------------

The initialization process sets up a hybrid search system that combines:

- **Dense Embeddings**: Using sentence transformers for semantic understanding
- **Sparse Embeddings**: Using BM25-based sparse embeddings for keyword matching
- **Vector Database**: Qdrant for efficient similarity search and retrieval
- **Language Model**: Ollama-based LLM for inference and generation tasks

This architecture enables powerful document retrieval and question-answering capabilities.

Step 1: Import Required Libraries
----------------------------------

The first step involves importing all necessary dependencies:

.. code-block:: python

    import os
    from pathlib import Path
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant.fastembed_sparse import FastEmbedSparse
    from qdrant_client import QdrantClient
    from langchain_ollama import ChatOllama

**Key Imports Explained:**

- ``os`` and ``pathlib.Path``: For file system operations and path management
- ``HuggingFaceEmbeddings``: Provides dense embeddings using pre-trained transformer models
- ``FastEmbedSparse``: Implements sparse embeddings using BM25 algorithm
- ``QdrantClient``: Client for interacting with Qdrant vector database
- ``ChatOllama``: Interface for local LLM inference via Ollama

Step 2: Define Directory Paths
-------------------------------

Configure the directory structure for organizing documents and data:

.. code-block:: python

    DOCS_DIR = "docs"                    # Directory containing PDF files
    MARKDOWN_DIR = "markdown"            # Directory for PDF-to-Markdown conversions
    PARENT_STORE_PATH = "parent_store"   # Directory for parent chunk JSON files
    CHILD_COLLECTION = "document_child_chunks"

**Directory Purposes:**

- ``DOCS_DIR``: Source location for original PDF documents
- ``MARKDOWN_DIR``: Stores converted markdown versions of PDFs for easier processing
- ``PARENT_STORE_PATH``: Stores JSON files containing parent chunks for hierarchical retrieval
- ``CHILD_COLLECTION``: Qdrant collection name for storing child document chunks

Step 3: Create Required Directories
------------------------------------

Ensure all necessary directories exist before processing:

.. code-block:: python

    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(MARKDOWN_DIR, exist_ok=True)
    os.makedirs(PARENT_STORE_PATH, exist_ok=True)

The ``exist_ok=True`` parameter prevents errors if directories already exist, making the code idempotent and safe for repeated execution.

Step 4: Initialize Language Model
----------------------------------

Set up the local language model using Ollama:

.. code-block:: python

    from langchain_ollama import ChatOllama
    
    llm = ChatOllama(
        model="qwen3:4b-instruct-2507-q4_K_M",
        temperature=0
    )

**Configuration Details:**

- **Model**: ``qwen3:4b-instruct-2507-q4_K_M`` - A 4-billion parameter quantized model optimized for instruction following
- **Temperature**: Set to 0 for deterministic outputs, ensuring consistent and reproducible responses

This configuration prioritizes consistency and reliability for production use cases.

Step 5: Initialize Dense Embeddings
------------------------------------

Configure dense embeddings for semantic similarity:

.. code-block:: python

    dense_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

**Model Details:**

- **Model**: ``sentence-transformers/all-mpnet-base-v2`` - A state-of-the-art sentence embedding model
- **Dimensions**: 768-dimensional embeddings
- **Use Case**: Captures semantic meaning and context of text passages
- **Performance**: Excellent balance between speed and accuracy

Dense embeddings are used for semantic search and understanding document meaning.

Step 6: Initialize Sparse Embeddings
-------------------------------------

Set up sparse embeddings for keyword-based retrieval:

.. code-block:: python

    sparse_embeddings = FastEmbedSparse(
        model_name="Qdrant/bm25"
    )

**Configuration Details:**

- **Algorithm**: BM25 (Best Matching 25) - a probabilistic ranking function
- **Use Case**: Keyword matching and traditional information retrieval
- **Advantage**: Complements dense embeddings for hybrid search

Sparse embeddings excel at finding exact keyword matches and are particularly useful for technical documents.

Step 7: Initialize Qdrant Vector Database
------------------------------------------

Connect to or create a local Qdrant instance:

.. code-block:: python

    client = QdrantClient(path="qdrant_db")

**Configuration Details:**

- **Storage**: Local file-based storage at ``qdrant_db`` directory
- **Type**: In-memory with persistent storage
- **Scalability**: Suitable for development and small-to-medium deployments

The Qdrant client provides the interface for storing and retrieving vector embeddings.

Complete Initialization Code
-----------------------------

Here's the complete initialization sequence:

.. code-block:: python

    import os
    from pathlib import Path
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant.fastembed_sparse import FastEmbedSparse
    from qdrant_client import QdrantClient
    from langchain_ollama import ChatOllama

    # Define paths
    DOCS_DIR = "docs"
    MARKDOWN_DIR = "markdown"
    PARENT_STORE_PATH = "parent_store"
    CHILD_COLLECTION = "document_child_chunks"

    # Create directories
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(MARKDOWN_DIR, exist_ok=True)
    os.makedirs(PARENT_STORE_PATH, exist_ok=True)

    # Initialize LLM
    llm = ChatOllama(
        model="qwen3:4b-instruct-2507-q4_K_M",
        temperature=0
    )

    # Initialize embeddings
    dense_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    # Initialize vector database
    client = QdrantClient(path="qdrant_db")

Key Concepts
------------

**Hybrid Search**
    Combines dense and sparse embeddings to leverage both semantic understanding and keyword matching, resulting in more accurate and relevant retrieval.

**Vector Database**
    Qdrant efficiently stores and searches high-dimensional vectors, enabling fast similarity-based document retrieval at scale.

**Embeddings**
    Numerical representations of text that capture semantic meaning, enabling computers to understand and compare text similarity.

**Local LLM**
    Running models locally via Ollama provides privacy, cost efficiency, and control over inference without relying on external APIs.

Prerequisites and Dependencies
-------------------------------

Before running this initialization code, ensure you have:

1. **Python 3.8+** installed
2. **Ollama** installed and running locally
3. **Required Python packages**:

   .. code-block:: bash

       pip install langchain langchain-huggingface langchain-qdrant qdrant-client sentence-transformers

4. **Ollama Model**: Download the required model:

   .. code-block:: bash

       ollama pull qwen3:4b-instruct-2507-q4_K_M

Troubleshooting
---------------

**Ollama Connection Error**
    Ensure Ollama is running: ``ollama serve``

**Model Not Found**
    Download the model using: ``ollama pull qwen3:4b-instruct-2507-q4_K_M``

**Memory Issues**
    The 4B model requires approximately 2-3GB RAM. For systems with less memory, consider using a smaller model.

**Embedding Model Download**
    HuggingFace embeddings are downloaded automatically on first use. Ensure internet connectivity.

Next Steps
----------

With the initial setup complete, you can proceed to:

- Document ingestion and preprocessing
- Chunk creation and embedding generation
- Vector database population
- Query processing and retrieval
- Response generation using the LLM

