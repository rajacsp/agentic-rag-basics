4: Hierarchical Document Indexing
==================================

Overview
--------

This chapter covers the hierarchical document indexing strategy using Parent/Child chunk splitting. This approach creates a two-level hierarchy where parent chunks maintain document context and structure, while child chunks enable fine-grained retrieval. This strategy significantly improves retrieval quality and context preservation in RAG systems.

Purpose and Architecture
------------------------

Hierarchical indexing addresses a fundamental challenge in RAG systems: balancing context preservation with retrieval precision. The Parent/Child strategy provides:

- **Context Preservation**: Parent chunks maintain document structure and broader context
- **Precise Retrieval**: Child chunks enable fine-grained, relevant results
- **Efficient Storage**: Reduces redundancy while maintaining relationships
- **Flexible Querying**: Retrieve child chunks, augment with parent context
- **Scalability**: Handles documents of varying sizes and structures

The indexing workflow:

1. Load Markdown documents
2. Split by headers to create parent chunks
3. Merge small parents and split large parents
4. Create child chunks from parents
5. Store children in vector database
6. Store parents in JSON files for context retrieval

Step 1: Import Required Libraries
----------------------------------

Begin by importing the necessary dependencies:

.. code-block:: python

    import os
    import glob
    import json
    from pathlib import Path
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter
    )

**Import Explanations:**

- ``os``: File system operations
- ``glob``: Pattern-based file discovery
- ``json``: JSON serialization for storing parent chunks
- ``pathlib.Path``: Object-oriented path handling
- ``MarkdownHeaderTextSplitter``: Splits Markdown by headers
- ``RecursiveCharacterTextSplitter``: Splits text recursively by delimiters

Step 2: Merge Small Parents Function
-------------------------------------

Define a function to merge parent chunks that are too small:

.. code-block:: python

    def merge_small_parents(chunks, min_size):
        """
        Merge parent chunks that are smaller than min_size.
        
        Args:
            chunks (list): List of Document objects
            min_size (int): Minimum size threshold in characters
            
        Returns:
            list: Merged chunks
        """
        if not chunks:
            return []
        
        merged, current = [], None
        
        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in current.metadata:
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"
                    else:
                        current.metadata[k] = v
            
            if len(current.page_content) >= min_size:
                merged.append(current)
                current = None
        
        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    if k in merged[-1].metadata:
                        merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                    else:
                        merged[-1].metadata[k] = v
            else:
                merged.append(current)
        
        return merged

**Function Purpose:**

Prevents fragmentation by merging chunks smaller than the minimum size threshold. This ensures parent chunks have sufficient context.

**Algorithm:**

1. Iterate through chunks sequentially
2. Accumulate content until reaching min_size
3. Merge metadata with arrow notation (``->`` indicates merged sections)
4. Append to result when threshold is reached
5. Handle remaining chunks at the end

**Example:**

.. code-block:: python

    # Input: 3 chunks of 500, 300, 600 characters (min_size=1000)
    # Process:
    #   - Chunk 1 (500): accumulate
    #   - Chunk 2 (300): merge with Chunk 1 (800 total)
    #   - Chunk 3 (600): merge with accumulated (1400 total) -> append
    # Output: 1 merged chunk of 1400 characters

**Metadata Merging:**

When chunks are merged, metadata is combined with arrow notation:

.. code-block:: python

    # Original metadata
    chunk1.metadata = {"section": "Introduction"}
    chunk2.metadata = {"section": "Background"}
    
    # After merge
    merged.metadata = {"section": "Introduction -> Background"}

Step 3: Split Large Parents Function
-------------------------------------

Define a function to split parent chunks that are too large:

.. code-block:: python

    def split_large_parents(chunks, max_size, splitter):
        """
        Split parent chunks that exceed max_size.
        
        Args:
            chunks (list): List of Document objects
            max_size (int): Maximum size threshold in characters
            splitter: RecursiveCharacterTextSplitter instance
            
        Returns:
            list: Split chunks
        """
        split_chunks = []
        
        for chunk in chunks:
            if len(chunk.page_content) <= max_size:
                split_chunks.append(chunk)
            else:
                large_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_size,
                    chunk_overlap=splitter._chunk_overlap
                )
                sub_chunks = large_splitter.split_documents([chunk])
                split_chunks.extend(sub_chunks)
        
        return split_chunks

**Function Purpose:**

Prevents excessively large parent chunks by splitting them while maintaining overlap for context continuity.

**Algorithm:**

1. Iterate through chunks
2. Keep chunks within max_size unchanged
3. For oversized chunks, create a new splitter with max_size
4. Preserve chunk_overlap from original splitter
5. Extend results with split sub-chunks

**Why Preserve Overlap:**

Overlap ensures context continuity between split chunks:

.. code-block:: python

    # Without overlap: Information at boundaries is lost
    # Chunk 1: "...end of section"
    # Chunk 2: "start of next section..."
    
    # With overlap: Context is preserved
    # Chunk 1: "...middle of section...end of section"
    # Chunk 2: "...end of section...start of next section..."

Step 4: Clean Small Chunks Function
------------------------------------

Define a function to clean up remaining small chunks:

.. code-block:: python

    def clean_small_chunks(chunks, min_size):
        """
        Remove or merge chunks smaller than min_size.
        
        Args:
            chunks (list): List of Document objects
            min_size (int): Minimum size threshold in characters
            
        Returns:
            list: Cleaned chunks
        """
        cleaned = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < min_size:
                if cleaned:
                    # Merge with previous chunk
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata:
                            cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                        else:
                            cleaned[-1].metadata[k] = v
                elif i < len(chunks) - 1:
                    # Merge with next chunk
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    for k, v in chunk.metadata.items():
                        if k in chunks[i + 1].metadata:
                            chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                        else:
                            chunks[i + 1].metadata[k] = v
                else:
                    # Last chunk, keep as is
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)
        
        return cleaned

**Function Purpose:**

Final cleanup pass to eliminate fragmentation by merging remaining small chunks with neighbors.

**Algorithm:**

1. Iterate through chunks
2. For small chunks:
   - If previous chunk exists: merge with it
   - Else if next chunk exists: merge with it
   - Else: keep as is (last chunk)
3. For normal chunks: keep unchanged

**Merging Strategy:**

- **Prefer previous**: Maintains document flow
- **Fallback to next**: Ensures no data loss
- **Keep last**: Prevents orphaned chunks

Step 5: Initialize Vector Store
--------------------------------

Set up the vector store for child chunks:

.. code-block:: python

    if client.collection_exists(CHILD_COLLECTION):
        client.delete_collection(CHILD_COLLECTION)
        ensure_collection(CHILD_COLLECTION)
    else:
        ensure_collection(CHILD_COLLECTION)
    
    child_vector_store = QdrantVectorStore(
        client=client,
        collection_name=CHILD_COLLECTION,
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID,
        sparse_vector_name="sparse"
    )

**Configuration Details:**

- **Collection Reset**: Deletes and recreates collection for fresh indexing
- **Dense Embeddings**: Uses sentence-transformers for semantic search
- **Sparse Embeddings**: Uses BM25 for keyword matching
- **Hybrid Mode**: Combines both search methods
- **Sparse Vector Name**: Identifies sparse vectors in Qdrant

Step 6: Main Indexing Function
-------------------------------

Define the main function that orchestrates the entire indexing process:

.. code-block:: python

    def index_documents():
        """
        Index all Markdown documents using hierarchical Parent/Child strategy.
        
        Process:
        1. Load Markdown files
        2. Split by headers to create parent chunks
        3. Merge small and split large parents
        4. Create child chunks from parents
        5. Store children in vector database
        6. Store parents in JSON files
        
        Returns:
            None
        """

**Step 6a: Configure Header Splitting**

.. code-block:: python

    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    parent_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

**Configuration Details:**

- **H1 (#)**: Top-level headers
- **H2 (##)**: Section headers
- **H3 (###)**: Subsection headers
- **strip_headers=False**: Keeps headers in content for context

**Why These Headers:**

- Preserves document structure
- Creates meaningful semantic boundaries
- Maintains hierarchy information in metadata

**Step 6b: Configure Child Splitter**

.. code-block:: python

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

**Configuration Details:**

- **chunk_size=500**: Child chunks are 500 characters
- **chunk_overlap=100**: 100 character overlap between chunks
- **Recursive**: Splits by multiple delimiters (newlines, spaces, characters)

**Why These Values:**

- 500 chars ≈ 100-150 tokens (good for LLM context)
- 100 char overlap ≈ 20% (maintains context continuity)
- Recursive splitting preserves semantic boundaries

**Step 6c: Set Size Thresholds**

.. code-block:: python

    min_parent_size = 2000      # Minimum parent chunk size
    max_parent_size = 4000      # Maximum parent chunk size

**Size Rationale:**

- **min_parent_size=2000**: Ensures sufficient context (400-600 tokens)
- **max_parent_size=4000**: Prevents excessively large chunks (800-1200 tokens)
- **Ratio 1:2**: Balanced range for flexibility

**Step 6d: Process Markdown Files**

.. code-block:: python

    all_parent_pairs, all_child_chunks = [], []
    md_files = sorted(glob.glob(os.path.join(MARKDOWN_DIR, "*.md")))
    
    if not md_files:
        return
    
    for doc_path_str in md_files:
        doc_path = Path(doc_path_str)
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                md_text = f.read()
        except Exception as e:
            continue

**Processing Details:**

- **Sorted files**: Ensures consistent ordering
- **Error handling**: Skips files that can't be read
- **UTF-8 encoding**: Handles special characters

**Step 6e: Create Parent Chunks**

.. code-block:: python

    parent_chunks = parent_splitter.split_text(md_text)
    merged_parents = merge_small_parents(parent_chunks, min_parent_size)
    split_parents = split_large_parents(merged_parents, max_parent_size, child_splitter)
    cleaned_parents = clean_small_chunks(split_parents, min_parent_size)

**Processing Pipeline:**

1. **split_text()**: Split by headers
2. **merge_small_parents()**: Combine small chunks
3. **split_large_parents()**: Break oversized chunks
4. **clean_small_chunks()**: Final cleanup

**Step 6f: Create Child Chunks and Store**

.. code-block:: python

    for i, p_chunk in enumerate(cleaned_parents):
        parent_id = f"{doc_path.stem}_parent_{i}"
        p_chunk.metadata.update({
            "source": doc_path.stem + ".pdf",
            "parent_id": parent_id
        })
        all_parent_pairs.append((parent_id, p_chunk))
        
        children = child_splitter.split_documents([p_chunk])
        all_child_chunks.extend(children)

**Details:**

- **parent_id**: Unique identifier for each parent chunk
- **source**: Original PDF filename
- **children**: Split from parent for fine-grained retrieval
- **Metadata propagation**: Children inherit parent metadata

**Step 6g: Store in Vector Database**

.. code-block:: python

    if not all_child_chunks:
        return
    
    try:
        child_vector_store.add_documents(all_child_chunks)
    except Exception as e:
        return

Stores all child chunks in Qdrant with both dense and sparse embeddings.

**Step 6h: Store Parent Chunks as JSON**

.. code-block:: python

    for item in os.listdir(PARENT_STORE_PATH):
        os.remove(os.path.join(PARENT_STORE_PATH, item))
    
    for parent_id, doc in all_parent_pairs:
        doc_dict = {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        filepath = os.path.join(PARENT_STORE_PATH, f"{parent_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc_dict, f, ensure_ascii=False, indent=2)

**Storage Details:**

- **Clear old files**: Removes previous parent chunks
- **JSON format**: Easy to load and parse
- **Filename**: Uses parent_id for quick lookup
- **UTF-8 encoding**: Preserves special characters

**Step 6i: Execute Indexing**

.. code-block:: python

    index_documents()

Complete Indexing Pipeline
---------------------------

Here's the complete hierarchical indexing code:

.. code-block:: python

    import os
    import glob
    import json
    from pathlib import Path
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter
    )
    from langchain_qdrant import QdrantVectorStore
    from langchain_qdrant.qdrant import RetrievalMode

    # Configuration
    MARKDOWN_DIR = "markdown"
    PARENT_STORE_PATH = "parent_store"
    CHILD_COLLECTION = "document_child_chunks"

    def merge_small_parents(chunks, min_size):
        """Merge small parent chunks."""
        if not chunks:
            return []
        
        merged, current = [], None
        
        for chunk in chunks:
            if current is None:
                current = chunk
            else:
                current.page_content += "\n\n" + chunk.page_content
                for k, v in chunk.metadata.items():
                    if k in current.metadata:
                        current.metadata[k] = f"{current.metadata[k]} -> {v}"
                    else:
                        current.metadata[k] = v
            
            if len(current.page_content) >= min_size:
                merged.append(current)
                current = None
        
        if current:
            if merged:
                merged[-1].page_content += "\n\n" + current.page_content
                for k, v in current.metadata.items():
                    if k in merged[-1].metadata:
                        merged[-1].metadata[k] = f"{merged[-1].metadata[k]} -> {v}"
                    else:
                        merged[-1].metadata[k] = v
            else:
                merged.append(current)
        
        return merged

    def split_large_parents(chunks, max_size, splitter):
        """Split large parent chunks."""
        split_chunks = []
        
        for chunk in chunks:
            if len(chunk.page_content) <= max_size:
                split_chunks.append(chunk)
            else:
                large_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_size,
                    chunk_overlap=splitter._chunk_overlap
                )
                sub_chunks = large_splitter.split_documents([chunk])
                split_chunks.extend(sub_chunks)
        
        return split_chunks

    def clean_small_chunks(chunks, min_size):
        """Clean up remaining small chunks."""
        cleaned = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk.page_content) < min_size:
                if cleaned:
                    cleaned[-1].page_content += "\n\n" + chunk.page_content
                    for k, v in chunk.metadata.items():
                        if k in cleaned[-1].metadata:
                            cleaned[-1].metadata[k] = f"{cleaned[-1].metadata[k]} -> {v}"
                        else:
                            cleaned[-1].metadata[k] = v
                elif i < len(chunks) - 1:
                    chunks[i + 1].page_content = chunk.page_content + "\n\n" + chunks[i + 1].page_content
                    for k, v in chunk.metadata.items():
                        if k in chunks[i + 1].metadata:
                            chunks[i + 1].metadata[k] = f"{v} -> {chunks[i + 1].metadata[k]}"
                        else:
                            chunks[i + 1].metadata[k] = v
                else:
                    cleaned.append(chunk)
            else:
                cleaned.append(chunk)
        
        return cleaned

    def index_documents():
        """Index all documents with hierarchical Parent/Child strategy."""
        headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        
        min_parent_size = 2000
        max_parent_size = 4000
        
        all_parent_pairs, all_child_chunks = [], []
        md_files = sorted(glob.glob(os.path.join(MARKDOWN_DIR, "*.md")))
        
        if not md_files:
            print("No Markdown files found")
            return
        
        print(f"Processing {len(md_files)} Markdown files...")
        
        for doc_path_str in md_files:
            doc_path = Path(doc_path_str)
            try:
                with open(doc_path, "r", encoding="utf-8") as f:
                    md_text = f.read()
            except Exception as e:
                print(f"Error reading {doc_path}: {e}")
                continue
            
            parent_chunks = parent_splitter.split_text(md_text)
            merged_parents = merge_small_parents(parent_chunks, min_parent_size)
            split_parents = split_large_parents(merged_parents, max_parent_size, child_splitter)
            cleaned_parents = clean_small_chunks(split_parents, min_parent_size)
            
            for i, p_chunk in enumerate(cleaned_parents):
                parent_id = f"{doc_path.stem}_parent_{i}"
                p_chunk.metadata.update({
                    "source": doc_path.stem + ".pdf",
                    "parent_id": parent_id
                })
                all_parent_pairs.append((parent_id, p_chunk))
                
                children = child_splitter.split_documents([p_chunk])
                all_child_chunks.extend(children)
        
        if not all_child_chunks:
            print("No child chunks created")
            return
        
        print(f"Storing {len(all_child_chunks)} child chunks in vector database...")
        try:
            child_vector_store.add_documents(all_child_chunks)
        except Exception as e:
            print(f"Error storing chunks: {e}")
            return
        
        print(f"Storing {len(all_parent_pairs)} parent chunks as JSON...")
        for item in os.listdir(PARENT_STORE_PATH):
            os.remove(os.path.join(PARENT_STORE_PATH, item))
        
        for parent_id, doc in all_parent_pairs:
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            filepath = os.path.join(PARENT_STORE_PATH, f"{parent_id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(doc_dict, f, ensure_ascii=False, indent=2)
        
        print(f"Indexing complete: {len(all_parent_pairs)} parents, {len(all_child_chunks)} children")

    # Execute indexing
    index_documents()

Hierarchical Retrieval Strategy
-------------------------------

**How Parent/Child Retrieval Works:**

1. **Query Processing**: Convert user query to embeddings
2. **Child Retrieval**: Find most relevant child chunks using hybrid search
3. **Parent Lookup**: Retrieve parent chunks using parent_id from child metadata
4. **Context Augmentation**: Combine child chunks with parent context
5. **Response Generation**: Use LLM with augmented context

**Example Retrieval Flow:**

.. code-block:: python

    # Query: "How do neural networks work?"
    
    # Step 1: Find relevant child chunks
    child_results = child_vector_store.similarity_search(query, k=3)
    # Returns: 3 child chunks about neural networks
    
    # Step 2: Get parent IDs from child metadata
    parent_ids = [child.metadata["parent_id"] for child in child_results]
    # Returns: ["doc1_parent_2", "doc1_parent_2", "doc2_parent_1"]
    
    # Step 3: Load parent chunks
    parents = []
    for parent_id in set(parent_ids):
        with open(f"parent_store/{parent_id}.json") as f:
            parent = json.load(f)
            parents.append(parent)
    
    # Step 4: Combine for context
    context = "\n\n".join([p["page_content"] for p in parents])
    
    # Step 5: Generate response with LLM
    response = llm.invoke(f"Context: {context}\n\nQuestion: {query}")

Data Structure and Storage
--------------------------

**Vector Database Structure:**

.. code-block:: text

    Qdrant Collection: document_child_chunks
    ├── Dense Vectors (768 dimensions)
    │   └── Semantic embeddings from sentence-transformers
    ├── Sparse Vectors (BM25)
    │   └── Keyword embeddings
    └── Metadata
        ├── source: "document.pdf"
        ├── parent_id: "document_parent_0"
        ├── section: "Introduction"
        └── ...

**Parent Store Structure:**

.. code-block:: text

    parent_store/
    ├── document_parent_0.json
    ├── document_parent_1.json
    ├── document_parent_2.json
    └── ...

**JSON File Format:**

.. code-block:: json

    {
      "page_content": "Full parent chunk text...",
      "metadata": {
        "source": "document.pdf",
        "parent_id": "document_parent_0",
        "section": "Introduction",
        "subsection": "Background"
      }
    }

Best Practices
--------------

**1. Size Configuration**

.. code-block:: python

    # For technical documents
    min_parent_size = 2000      # ~400 tokens
    max_parent_size = 4000      # ~800 tokens
    child_chunk_size = 500      # ~100 tokens
    
    # For narrative documents
    min_parent_size = 3000
    max_parent_size = 6000
    child_chunk_size = 800

**2. Header Configuration**

.. code-block:: python

    # For well-structured documents
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    
    # For documents with many levels
    headers_to_split_on = [
        ("#", "H1"), ("##", "H2"), ("###", "H3"),
        ("####", "H4"), ("#####", "H5")
    ]

**3. Overlap Configuration**

.. code-block:: python

    # For technical documents (preserve context)
    chunk_overlap = 100         # ~20% overlap
    
    # For narrative documents (less overlap needed)
    chunk_overlap = 50          # ~10% overlap

**4. Error Handling**

.. code-block:: python

    def index_documents_safe():
        """Index with comprehensive error handling."""
        try:
            # ... indexing code ...
        except Exception as e:
            print(f"Indexing failed: {e}")
            # Log error, notify user, etc.

**5. Progress Tracking**

.. code-block:: python

    from tqdm import tqdm
    
    for doc_path_str in tqdm(md_files, desc="Processing documents"):
        # ... process document ...

Performance Considerations
--------------------------

**Indexing Speed:**

- Average document: 1-5 seconds
- Large documents (100+ pages): 10-30 seconds
- Batch of 100 documents: 2-5 minutes

**Storage Requirements:**

- Dense vectors: ~3KB per child chunk
- Sparse vectors: ~5KB per child chunk
- Parent JSON: ~2KB per parent chunk
- Total: ~10KB per child chunk

**Memory Usage:**

- Processing: ~100-200MB per document
- Vector database: ~1GB per 100,000 chunks
- Parent store: ~200MB per 100,000 chunks

**Optimization Tips:**

.. code-block:: python

    # Process in batches
    batch_size = 10
    for i in range(0, len(md_files), batch_size):
        batch = md_files[i:i+batch_size]
        # Process batch
    
    # Use smaller chunks for faster retrieval
    child_chunk_size = 300  # Faster but less context
    
    # Reduce overlap for faster processing
    chunk_overlap = 50      # Less redundancy

Troubleshooting
---------------

**Empty Child Chunks**

.. code-block:: python

    # Problem: No child chunks created
    # Solution: Check parent chunk sizes
    
    if not all_child_chunks:
        print("Debug: Check parent chunk creation")
        print(f"Parent chunks: {len(cleaned_parents)}")
        for p in cleaned_parents:
            print(f"  Size: {len(p.page_content)}")

**Vector Store Errors**

.. code-block:: python

    # Problem: Error adding documents to vector store
    # Solution: Verify collection exists and is properly configured
    
    try:
        child_vector_store.add_documents(all_child_chunks)
    except Exception as e:
        print(f"Error: {e}")
        # Recreate collection
        ensure_collection(CHILD_COLLECTION)

**Parent Lookup Failures**

.. code-block:: python

    # Problem: Cannot find parent chunks
    # Solution: Verify parent_id in child metadata
    
    for child in all_child_chunks:
        if "parent_id" not in child.metadata:
            print(f"Missing parent_id in child: {child}")

**Memory Issues**

.. code-block:: python

    # Problem: Out of memory during indexing
    # Solution: Process documents in smaller batches
    
    batch_size = 5
    for i in range(0, len(md_files), batch_size):
        batch = md_files[i:i+batch_size]
        # Process batch
        # Clear memory between batches

Next Steps
----------

With hierarchical indexing complete, you can proceed to:

- Implementing retrieval functions using parent/child strategy
- Building query processing pipelines
- Integrating with LLM for response generation
- Implementing hybrid search optimization
- Building the complete RAG application

