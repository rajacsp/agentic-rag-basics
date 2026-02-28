Topic 5: Define Agent Tools
===========================

Overview
--------

This chapter covers the creation of retrieval tools that enable the LLM agent to access indexed documents. These tools form the bridge between the language model and the knowledge base, allowing the agent to search for relevant information and retrieve full context when needed. By defining clear, well-structured tools, we enable the agent to make intelligent decisions about what information to retrieve.

Purpose and Architecture
------------------------

Agent tools serve as the interface between the LLM and external systems. In a RAG system, tools enable:

- **Semantic Search**: Find relevant child chunks using hybrid search
- **Context Retrieval**: Access full parent chunks for comprehensive context
- **Error Handling**: Gracefully handle retrieval failures
- **Tool Binding**: Connect tools to the LLM for autonomous invocation
- **Structured Responses**: Return consistent, parseable results

The tool architecture:

1. Define retrieval functions as LangChain tools
2. Implement error handling and validation
3. Format responses for LLM consumption
4. Bind tools to the language model
5. Enable agent to invoke tools autonomously

Step 1: Import Required Libraries
----------------------------------

Begin by importing the necessary dependencies:

.. code-block:: python

    import json
    import os
    from typing import List
    from langchain_core.tools import tool

**Import Explanations:**

- ``json``: For parsing and serializing JSON data
- ``os``: For file system operations
- ``typing.List``: Type hints for function signatures
- ``langchain_core.tools.tool``: Decorator for creating LangChain tools

**Why These Imports:**

- Type hints improve code clarity and IDE support
- LangChain's tool decorator handles schema generation
- JSON support enables parent chunk retrieval
- OS operations access the parent store

Step 2: Search Child Chunks Tool
---------------------------------

Define the first tool for searching child chunks:

.. code-block:: python

    @tool
    def search_child_chunks(query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks.
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
        """
        try:
            results = child_vector_store.similarity_search(
                query,
                k=limit,
                score_threshold=0.7
            )
            if not results:
                return "NO_RELEVANT_CHUNKS"
            
            return "\n\n".join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])
        
        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"

**Tool Purpose:**

Enables the agent to search the vector database for relevant child chunks using hybrid search (dense + sparse embeddings).

**Step 2a: Tool Decorator**

.. code-block:: python

    @tool
    def search_child_chunks(query: str, limit: int) -> str:

The ``@tool`` decorator:

- Converts the function into a LangChain tool
- Automatically generates tool schema from docstring and type hints
- Makes the tool available for agent invocation
- Handles parameter validation

**Step 2b: Function Parameters**

.. code-block:: python

    def search_child_chunks(query: str, limit: int) -> str:

**Parameters:**

- **query (str)**: The search query provided by the agent
  
  - Example: "How do neural networks process information?"
  - Used for embedding generation and similarity search

- **limit (int)**: Maximum number of results to return
  
  - Typical values: 3-5 for focused results
  - Prevents overwhelming the LLM with too much context

**Return Type:**

- **str**: String representation of results
  
  - Allows LLM to parse and understand results
  - Consistent format for error handling

**Step 2c: Similarity Search**

.. code-block:: python

    results = child_vector_store.similarity_search(
        query,
        k=limit,
        score_threshold=0.7
    )

**Search Parameters:**

- **query**: Automatically embedded using dense embeddings
- **k=limit**: Return top K results
- **score_threshold=0.7**: Minimum similarity score (0-1 scale)

**Score Threshold Explanation:**

- **0.7**: Moderate threshold, balances precision and recall
- **0.8+**: High threshold, only very relevant results
- **0.5-0.7**: Lower threshold, more results but potentially less relevant
- **Cosine distance**: 0 = identical, 1 = completely different

**Why Hybrid Search:**

The vector store uses both dense and sparse embeddings:

- **Dense**: Captures semantic meaning
- **Sparse**: Captures keyword relevance
- **Combined**: Best of both worlds

**Step 2d: Result Handling**

.. code-block:: python

    if not results:
        return "NO_RELEVANT_CHUNKS"

Returns a special token when no results are found, allowing the agent to recognize and handle this case.

**Step 2e: Format Results**

.. code-block:: python

    return "\n\n".join([
        f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
        f"File Name: {doc.metadata.get('source', '')}\n"
        f"Content: {doc.page_content.strip()}"
        for doc in results
    ])

**Formatting Details:**

- **Parent ID**: Enables parent chunk retrieval
- **File Name**: Provides source attribution
- **Content**: The actual chunk text
- **Double newlines**: Separates multiple results for readability

**Example Output:**

.. code-block:: text

    Parent ID: document_parent_0
    File Name: research_paper.pdf
    Content: Neural networks are computational models inspired by biological neurons...

    Parent ID: document_parent_1
    File Name: research_paper.pdf
    Content: The architecture of neural networks consists of layers of interconnected nodes...

**Step 2f: Error Handling**

.. code-block:: python

    except Exception as e:
        return f"RETRIEVAL_ERROR: {str(e)}"

Catches any exceptions during retrieval and returns an error message, preventing agent crashes.

**Error Cases:**

- Vector store connection failures
- Invalid query format
- Database errors
- Memory issues

Step 3: Retrieve Parent Chunks Tool
------------------------------------

Define the second tool for retrieving full parent chunks:

.. code-block:: python

    @tool
    def retrieve_parent_chunks(parent_id: str) -> str:
        """Retrieve full parent chunks by their IDs.
        
        Args:
            parent_id: Parent chunk ID to retrieve
        """
        file_name = parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
        path = os.path.join(PARENT_STORE_PATH, file_name)
        
        if not os.path.exists(path):
            return "NO_PARENT_DOCUMENT"
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return (
            f"Parent ID: {parent_id}\n"
            f"File Name: {data.get('metadata', {}).get('source', 'unknown')}\n"
            f"Content: {data.get('page_content', '').strip()}"
        )

**Tool Purpose:**

Enables the agent to retrieve full parent chunks for comprehensive context, using parent_id from child chunk search results.

**Step 3a: Function Parameters**

.. code-block:: python

    def retrieve_parent_chunks(parent_id: str) -> str:

**Parameters:**

- **parent_id (str)**: The parent chunk identifier
  
  - Format: "document_parent_0"
  - Obtained from child chunk search results
  - Used to locate JSON file in parent store

**Step 3b: File Path Construction**

.. code-block:: python

    file_name = parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
    path = os.path.join(PARENT_STORE_PATH, file_name)

**Path Construction Logic:**

- Handles both formats: "parent_id" and "parent_id.json"
- Constructs full path using PARENT_STORE_PATH
- Ensures cross-platform compatibility with os.path.join

**Example:**

.. code-block:: python

    # Input: "document_parent_0"
    # Output: "parent_store/document_parent_0.json"
    
    # Input: "document_parent_0.json"
    # Output: "parent_store/document_parent_0.json"

**Step 3c: File Existence Check**

.. code-block:: python

    if not os.path.exists(path):
        return "NO_PARENT_DOCUMENT"

Validates that the parent chunk file exists before attempting to read it.

**Why This Matters:**

- Prevents FileNotFoundError exceptions
- Allows agent to handle missing parents gracefully
- Enables debugging of parent_id mismatches

**Step 3d: Load JSON Data**

.. code-block:: python

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

Reads and parses the JSON file containing the parent chunk.

**JSON Structure:**

.. code-block:: json

    {
      "page_content": "Full parent chunk text...",
      "metadata": {
        "source": "document.pdf",
        "parent_id": "document_parent_0",
        "section": "Introduction"
      }
    }

**Step 3e: Format and Return**

.. code-block:: python

    return (
        f"Parent ID: {parent_id}\n"
        f"File Name: {data.get('metadata', {}).get('source', 'unknown')}\n"
        f"Content: {data.get('page_content', '').strip()}"
    )

**Formatting Details:**

- **Parent ID**: Confirms which parent was retrieved
- **File Name**: Source attribution
- **Content**: Full parent chunk text
- **Safe access**: Uses .get() to handle missing keys

**Example Output:**

.. code-block:: text

    Parent ID: document_parent_0
    File Name: research_paper.pdf
    Content: # Introduction
    
    This section introduces the fundamental concepts of neural networks...
    [Full parent chunk content]

Step 4: Bind Tools to Language Model
-------------------------------------

Connect the tools to the LLM for autonomous invocation:

.. code-block:: python

    llm_with_tools = llm.bind_tools([
        search_child_chunks,
        retrieve_parent_chunks
    ])

**Tool Binding Details:**

- **bind_tools()**: Attaches tools to the LLM
- **Tool List**: Array of tool functions
- **Result**: LLM instance with tool capabilities

**What Binding Does:**

1. Generates tool schemas from function signatures
2. Adds tools to LLM's available functions
3. Enables LLM to decide when to use tools
4. Formats tool calls for execution

**Tool Schema Generation:**

The decorator automatically generates schemas like:

.. code-block:: json

    {
      "name": "search_child_chunks",
      "description": "Search for the top K most relevant child chunks.",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Search query string"
          },
          "limit": {
            "type": "integer",
            "description": "Maximum number of results to return"
          }
        },
        "required": ["query", "limit"]
      }
    }

Complete Tool Definition Code
------------------------------

Here's the complete tool definition code:

.. code-block:: python

    import json
    import os
    from typing import List
    from langchain_core.tools import tool

    # Configuration (from previous setup)
    PARENT_STORE_PATH = "parent_store"

    @tool
    def search_child_chunks(query: str, limit: int) -> str:
        """Search for the top K most relevant child chunks.
        
        This tool searches the vector database using hybrid search
        (dense + sparse embeddings) to find relevant document chunks.
        
        Args:
            query: Search query string (e.g., "How do neural networks work?")
            limit: Maximum number of results to return (typically 3-5)
            
        Returns:
            String containing formatted search results with parent IDs,
            file names, and chunk content. Returns "NO_RELEVANT_CHUNKS"
            if no results found, or "RETRIEVAL_ERROR" if an error occurs.
        """
        try:
            results = child_vector_store.similarity_search(
                query,
                k=limit,
                score_threshold=0.7
            )
            if not results:
                return "NO_RELEVANT_CHUNKS"
            
            return "\n\n".join([
                f"Parent ID: {doc.metadata.get('parent_id', '')}\n"
                f"File Name: {doc.metadata.get('source', '')}\n"
                f"Content: {doc.page_content.strip()}"
                for doc in results
            ])
        
        except Exception as e:
            return f"RETRIEVAL_ERROR: {str(e)}"

    @tool
    def retrieve_parent_chunks(parent_id: str) -> str:
        """Retrieve full parent chunks by their IDs.
        
        This tool retrieves the complete parent chunk from the JSON store,
        providing full context for a specific parent chunk ID.
        
        Args:
            parent_id: Parent chunk ID to retrieve (e.g., "document_parent_0")
            
        Returns:
            String containing formatted parent chunk with ID, file name,
            and full content. Returns "NO_PARENT_DOCUMENT" if not found.
        """
        file_name = parent_id if parent_id.lower().endswith(".json") else f"{parent_id}.json"
        path = os.path.join(PARENT_STORE_PATH, file_name)
        
        if not os.path.exists(path):
            return "NO_PARENT_DOCUMENT"
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return (
            f"Parent ID: {parent_id}\n"
            f"File Name: {data.get('metadata', {}).get('source', 'unknown')}\n"
            f"Content: {data.get('page_content', '').strip()}"
        )

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools([
        search_child_chunks,
        retrieve_parent_chunks
    ])

Tool Usage Workflow
-------------------

**How the Agent Uses Tools:**

1. **User Query**: "What are neural networks?"

2. **Agent Decision**: LLM decides to search for relevant information
   
   .. code-block:: python
   
       # LLM calls tool
       search_child_chunks(
           query="What are neural networks?",
           limit=3
       )

3. **Tool Execution**: Search returns relevant chunks
   
   .. code-block:: text
   
       Parent ID: doc_parent_0
       File Name: research.pdf
       Content: Neural networks are computational models...

4. **Agent Analysis**: LLM reads results and decides if more context needed

5. **Parent Retrieval**: If needed, retrieve full parent chunk
   
   .. code-block:: python
   
       retrieve_parent_chunks(parent_id="doc_parent_0")

6. **Response Generation**: LLM generates answer using all retrieved context

**Example Agent Interaction:**

.. code-block:: text

    User: "Explain how neural networks learn"
    
    Agent: I'll search for information about neural network learning.
    [Calls: search_child_chunks("neural network learning", limit=3)]
    
    Found 3 relevant chunks. Let me get more context from the parent.
    [Calls: retrieve_parent_chunks("doc_parent_1")]
    
    Based on the retrieved information, neural networks learn through
    a process called backpropagation...

Tool Design Patterns
--------------------

**1. Error Handling Pattern**

.. code-block:: python

    @tool
    def safe_tool(param: str) -> str:
        """Tool with comprehensive error handling."""
        try:
            # Main logic
            result = perform_operation(param)
            return result
        except FileNotFoundError:
            return "FILE_NOT_FOUND"
        except ValueError as e:
            return f"INVALID_INPUT: {str(e)}"
        except Exception as e:
            return f"ERROR: {str(e)}"

**2. Validation Pattern**

.. code-block:: python

    @tool
    def validated_tool(query: str, limit: int) -> str:
        """Tool with input validation."""
        # Validate inputs
        if not query or len(query.strip()) == 0:
            return "EMPTY_QUERY"
        
        if limit < 1 or limit > 100:
            return "INVALID_LIMIT"
        
        # Proceed with operation
        return perform_search(query, limit)

**3. Logging Pattern**

.. code-block:: python

    import logging
    
    logger = logging.getLogger(__name__)
    
    @tool
    def logged_tool(param: str) -> str:
        """Tool with logging."""
        logger.info(f"Tool called with param: {param}")
        try:
            result = perform_operation(param)
            logger.info(f"Tool succeeded: {result[:50]}...")
            return result
        except Exception as e:
            logger.error(f"Tool failed: {e}")
            return f"ERROR: {str(e)}"

**4. Caching Pattern**

.. code-block:: python

    from functools import lru_cache
    
    @lru_cache(maxsize=128)
    def get_parent_cached(parent_id: str) -> dict:
        """Cached parent retrieval."""
        # Load from file
        with open(f"parent_store/{parent_id}.json") as f:
            return json.load(f)
    
    @tool
    def retrieve_parent_chunks_cached(parent_id: str) -> str:
        """Retrieve with caching."""
        try:
            data = get_parent_cached(parent_id)
            return format_result(data)
        except Exception as e:
            return f"ERROR: {str(e)}"

Advanced Tool Features
----------------------

**1. Tool Metadata**

.. code-block:: python

    @tool(name="search_documents", tags=["retrieval", "search"])
    def search_child_chunks(query: str, limit: int) -> str:
        """Search for relevant document chunks."""
        # Implementation

**2. Tool with Multiple Return Types**

.. code-block:: python

    from typing import Union
    
    @tool
    def flexible_tool(param: str) -> Union[str, dict]:
        """Tool that can return different types."""
        if param == "json":
            return {"status": "success", "data": []}
        else:
            return "Success"

**3. Tool with Async Support**

.. code-block:: python

    import asyncio
    
    @tool
    async def async_tool(query: str) -> str:
        """Async tool for non-blocking operations."""
        result = await perform_async_search(query)
        return result

**4. Tool with Context**

.. code-block:: python

    @tool
    def contextual_tool(query: str, context: dict = None) -> str:
        """Tool that uses context from previous calls."""
        if context:
            # Use context from previous tool calls
            previous_results = context.get("previous_results", [])
        
        return perform_search(query)

Best Practices
--------------

**1. Clear Documentation**

.. code-block:: python

    @tool
    def well_documented_tool(query: str, limit: int) -> str:
        """Search for relevant information.
        
        This tool performs a hybrid search combining semantic and
        keyword-based matching to find the most relevant documents.
        
        Args:
            query: The search query (e.g., "How do transformers work?")
            limit: Number of results (1-10, default 5)
            
        Returns:
            Formatted string with search results or error message
            
        Raises:
            Returns error string instead of raising exceptions
        """

**2. Consistent Return Format**

.. code-block:: python

    # Good: Consistent format
    return f"Result: {data}\nSource: {source}"
    
    # Avoid: Inconsistent formats
    if success:
        return data
    else:
        return {"error": "failed"}

**3. Meaningful Error Messages**

.. code-block:: python

    # Good: Specific error messages
    return "NO_RELEVANT_CHUNKS"
    return "INVALID_PARENT_ID"
    return "DATABASE_CONNECTION_ERROR"
    
    # Avoid: Generic messages
    return "ERROR"
    return "FAILED"

**4. Type Hints**

.. code-block:: python

    # Good: Clear type hints
    @tool
    def typed_tool(query: str, limit: int) -> str:
        pass
    
    # Avoid: Missing type hints
    @tool
    def untyped_tool(query, limit):
        pass

**5. Reasonable Defaults**

.. code-block:: python

    # Good: Sensible defaults
    def search_child_chunks(query: str, limit: int = 3) -> str:
        pass
    
    # Avoid: No defaults
    def search_child_chunks(query: str, limit: int) -> str:
        pass

Performance Considerations
--------------------------

**Search Performance:**

- Average search: 100-500ms
- With score threshold: Slightly faster (fewer results)
- Hybrid search: Slightly slower than dense-only

**Parent Retrieval Performance:**

- File I/O: 10-50ms per parent
- JSON parsing: 1-5ms per parent
- Total: 20-100ms per parent

**Optimization Tips:**

.. code-block:: python

    # Cache frequently accessed parents
    parent_cache = {}
    
    @tool
    def retrieve_parent_chunks_cached(parent_id: str) -> str:
        if parent_id in parent_cache:
            return parent_cache[parent_id]
        
        result = retrieve_parent_chunks(parent_id)
        parent_cache[parent_id] = result
        return result
    
    # Limit search results
    search_child_chunks(query, limit=3)  # Faster than limit=10
    
    # Use higher score threshold
    score_threshold=0.8  # Fewer results, faster

Troubleshooting
---------------

**Tool Not Appearing in Agent**

.. code-block:: python

    # Problem: Tool not available to agent
    # Solution: Ensure tool is bound to LLM
    
    llm_with_tools = llm.bind_tools([
        search_child_chunks,
        retrieve_parent_chunks
    ])

**Tool Returns Empty Results**

.. code-block:: python

    # Problem: search_child_chunks returns "NO_RELEVANT_CHUNKS"
    # Solution: Check score threshold or query quality
    
    # Lower threshold for more results
    score_threshold=0.5
    
    # Verify query is meaningful
    query = "neural networks"  # Good
    query = "the"              # Bad

**Parent Not Found**

.. code-block:: python

    # Problem: retrieve_parent_chunks returns "NO_PARENT_DOCUMENT"
    # Solution: Verify parent_id format and file exists
    
    # Check parent store
    import os
    files = os.listdir("parent_store")
    print(f"Available parents: {files}")

**Tool Execution Timeout**

.. code-block:: python

    # Problem: Tool takes too long
    # Solution: Optimize search or add timeout
    
    # Reduce limit
    limit=3  # Faster than limit=10
    
    # Increase score threshold
    score_threshold=0.8  # Fewer results to process

Next Steps
----------

With agent tools defined, you can proceed to:

- Building the agent loop for autonomous tool invocation
- Implementing response generation with retrieved context
- Creating the complete RAG application
- Adding tool monitoring and logging
- Optimizing tool performance and caching

