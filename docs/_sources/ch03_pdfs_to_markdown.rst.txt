3: PDFs to Markdown
===================

Overview
--------

This chapter covers the conversion of PDF documents to Markdown format, a crucial preprocessing step in the RAG pipeline. Converting PDFs to Markdown enables easier text extraction, better preservation of document structure, and improved compatibility with downstream processing tasks.

Purpose and Workflow
--------------------

PDF documents are binary files with complex formatting and layout information. Converting them to Markdown provides:

- **Structured Text**: Preserves document hierarchy and formatting
- **Machine Readability**: Easier for text processing and chunking
- **Format Compatibility**: Works seamlessly with LangChain and other text processing tools
- **Preservation of Content**: Maintains important structural information while removing formatting noise
- **Scalability**: Batch processing of multiple PDFs

The conversion workflow:

1. Load PDF document using PyMuPDF
2. Extract text and structure using pymupdf4llm
3. Clean and normalize the extracted Markdown
4. Save to output directory with proper naming

Step 1: Import Required Libraries
----------------------------------

Begin by importing the necessary dependencies:

.. code-block:: python

    import os
    import pymupdf
    import pymupdf4llm
    from pathlib import Path
    import glob

**Import Explanations:**

- ``os``: Operating system interface for environment variables
- ``pymupdf``: PyMuPDF library for PDF manipulation (also known as fitz)
- ``pymupdf4llm``: Extension for converting PDFs to Markdown optimized for LLMs
- ``pathlib.Path``: Object-oriented path handling
- ``glob``: Pattern-based file discovery

**Why These Libraries:**

- **PyMuPDF**: Fast, efficient PDF processing with layout awareness
- **pymupdf4llm**: Specifically designed for LLM-friendly Markdown conversion
- **pathlib**: Modern, cross-platform path operations
- **glob**: Flexible file pattern matching

Step 2: Configure Environment Variables
----------------------------------------

Set up environment variables for optimal processing:

.. code-block:: python

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

**Configuration Details:**

- **TOKENIZERS_PARALLELISM**: Controls parallel tokenization
- **Value "false"**: Disables parallelism to avoid warnings and potential conflicts
- **Why**: Prevents race conditions when processing multiple PDFs sequentially

**Other Useful Environment Variables:**

.. code-block:: python

    # Suppress warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Control logging verbosity
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    
    # Set number of threads
    os.environ["OMP_NUM_THREADS"] = "4"

Step 3: Single PDF Conversion Function
---------------------------------------

Define a function to convert a single PDF to Markdown:

.. code-block:: python

    def pdf_to_markdown(pdf_path, output_dir):
        """
        Convert a single PDF file to Markdown format.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str): Directory to save the Markdown file
            
        Returns:
            None
        """
        doc = pymupdf.open(pdf_path)
        md = pymupdf4llm.to_markdown(
            doc,
            header=False,
            footer=False,
            page_separators=True,
            ignore_images=True,
            write_images=False,
            image_path=None
        )
        md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
        output_path = Path(output_dir) / Path(doc.name).stem
        Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))

**Function Parameters:**

- ``pdf_path``: Full path to the PDF file to convert
- ``output_dir``: Directory where Markdown file will be saved

**Step 3a: Open PDF Document**

.. code-block:: python

    doc = pymupdf.open(pdf_path)

Opens the PDF file and creates a document object for processing.

**Step 3b: Convert to Markdown**

.. code-block:: python

    md = pymupdf4llm.to_markdown(
        doc,
        header=False,           # Exclude page headers
        footer=False,           # Exclude page footers
        page_separators=True,   # Add separators between pages
        ignore_images=True,     # Skip image processing
        write_images=False,     # Don't save images to disk
        image_path=None         # No image output directory
    )

**Parameter Explanations:**

- **header=False**: Removes header content (page numbers, titles)
- **footer=False**: Removes footer content (page numbers, dates)
- **page_separators=True**: Adds ``---`` between pages for structure
- **ignore_images=True**: Skips image extraction (faster processing)
- **write_images=False**: Doesn't save images separately
- **image_path=None**: No directory for image output

**Why These Settings:**

- Removing headers/footers reduces noise in the text
- Page separators help identify page boundaries
- Ignoring images speeds up conversion and reduces storage
- Optimized for text-based RAG systems

**Alternative Configurations:**

.. code-block:: python

    # For documents with important images
    md = pymupdf4llm.to_markdown(
        doc,
        header=False,
        footer=False,
        page_separators=True,
        ignore_images=False,        # Include images
        write_images=True,          # Save images
        image_path="./extracted_images"
    )

    # For documents with important headers/footers
    md = pymupdf4llm.to_markdown(
        doc,
        header=True,                # Include headers
        footer=True,                # Include footers
        page_separators=True,
        ignore_images=True,
        write_images=False,
        image_path=None
    )

**Step 3c: Clean Markdown Content**

.. code-block:: python

    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')

This two-step encoding/decoding process handles problematic characters:

1. **encode('utf-8', errors='surrogatepass')**: Converts string to bytes, preserving invalid UTF-8 sequences
2. **decode('utf-8', errors='ignore')**: Converts back to string, removing invalid sequences

**Why This Matters:**

- PDFs may contain corrupted or invalid UTF-8 characters
- This approach gracefully handles encoding errors
- Prevents crashes during processing
- Ensures clean, valid Markdown output

**Step 3d: Save Markdown File**

.. code-block:: python

    output_path = Path(output_dir) / Path(doc.name).stem
    Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))

Saves the cleaned Markdown to disk:

1. **Path(output_dir) / Path(doc.name).stem**: Creates output path with original filename
2. **.with_suffix(".md")**: Ensures .md extension
3. **.write_bytes()**: Writes UTF-8 encoded content to file

**Example:**

.. code-block:: python

    # Input: docs/research_paper.pdf
    # Output: markdown/research_paper.md

Step 4: Batch PDF Conversion Function
--------------------------------------

Define a function to convert multiple PDFs:

.. code-block:: python

    def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
        """
        Convert multiple PDF files to Markdown format.
        
        Args:
            path_pattern (str): Glob pattern for PDF files (e.g., "docs/*.pdf")
            overwrite (bool): Whether to overwrite existing Markdown files
            
        Returns:
            None
        """
        output_dir = Path(MARKDOWN_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for pdf_path in map(Path, glob.glob(path_pattern)):
            md_path = (output_dir / pdf_path.stem).with_suffix(".md")
            if overwrite or not md_path.exists():
                pdf_to_markdown(pdf_path, output_dir)

**Function Parameters:**

- ``path_pattern``: Glob pattern to match PDF files
- ``overwrite``: Whether to re-convert existing Markdown files

**Step 4a: Create Output Directory**

.. code-block:: python

    output_dir = Path(MARKDOWN_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

Creates the output directory if it doesn't exist:

- **parents=True**: Creates parent directories as needed
- **exist_ok=True**: Doesn't error if directory already exists

**Step 4b: Iterate Over PDF Files**

.. code-block:: python

    for pdf_path in map(Path, glob.glob(path_pattern)):

Finds all PDFs matching the pattern and converts each one:

- **glob.glob(path_pattern)**: Finds files matching pattern
- **map(Path, ...)**: Converts strings to Path objects
- **for pdf_path in ...: **: Iterates over each PDF

**Step 4c: Check for Existing Files**

.. code-block:: python

    md_path = (output_dir / pdf_path.stem).with_suffix(".md")
    if overwrite or not md_path.exists():
        pdf_to_markdown(pdf_path, output_dir)

Implements smart conversion logic:

- **overwrite=True**: Always re-convert
- **overwrite=False**: Skip if Markdown already exists
- Saves time on repeated runs

Complete Conversion Pipeline
-----------------------------

Here's the complete PDF to Markdown conversion code:

.. code-block:: python

    import os
    import pymupdf
    import pymupdf4llm
    from pathlib import Path
    import glob

    # Configure environment
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Define paths (from previous setup)
    DOCS_DIR = "docs"
    MARKDOWN_DIR = "markdown"

    def pdf_to_markdown(pdf_path, output_dir):
        """Convert a single PDF to Markdown."""
        doc = pymupdf.open(pdf_path)
        md = pymupdf4llm.to_markdown(
            doc,
            header=False,
            footer=False,
            page_separators=True,
            ignore_images=True,
            write_images=False,
            image_path=None
        )
        md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
        output_path = Path(output_dir) / Path(doc.name).stem
        Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))
        print(f"Converted: {pdf_path} -> {output_path}.md")

    def pdfs_to_markdowns(path_pattern, overwrite: bool = False):
        """Convert multiple PDFs to Markdown."""
        output_dir = Path(MARKDOWN_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(glob.glob(path_pattern))
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in map(Path, pdf_files):
            md_path = (output_dir / pdf_path.stem).with_suffix(".md")
            if overwrite or not md_path.exists():
                pdf_to_markdown(pdf_path, output_dir)
            else:
                print(f"Skipping (already exists): {md_path}")

    # Execute conversion
    pdfs_to_markdowns(f"{DOCS_DIR}/*.pdf")

Usage Examples
--------------

**Convert All PDFs in Directory**

.. code-block:: python

    # Convert all PDFs in docs/ directory
    pdfs_to_markdowns("docs/*.pdf")

**Convert Specific PDF Pattern**

.. code-block:: python

    # Convert only research papers
    pdfs_to_markdowns("docs/research_*.pdf")

**Force Re-conversion**

.. code-block:: python

    # Re-convert all PDFs, overwriting existing Markdown
    pdfs_to_markdowns("docs/*.pdf", overwrite=True)

**Convert Single PDF**

.. code-block:: python

    # Convert one PDF
    pdf_to_markdown("docs/document.pdf", "markdown")

Output Structure
----------------

After conversion, your directory structure looks like:

.. code-block:: text

    project/
    ├── docs/
    │   ├── research_paper.pdf
    │   ├── technical_guide.pdf
    │   └── manual.pdf
    └── markdown/
        ├── research_paper.md
        ├── technical_guide.md
        └── manual.md

**Markdown File Contents:**

Each Markdown file contains:

- Extracted text from the PDF
- Preserved heading hierarchy
- Page separators (``---``)
- Lists and formatting
- Code blocks (if present)
- Tables (if present)

**Example Markdown Output:**

.. code-block:: markdown

    # Document Title

    ## Section 1

    This is the content of section 1.

    ---

    ## Section 2

    Content continues on the next page.

    - Bullet point 1
    - Bullet point 2

Best Practices
--------------

**1. Organize PDFs by Type**

.. code-block:: python

    # Separate different document types
    pdfs_to_markdowns("docs/research/*.pdf")
    pdfs_to_markdowns("docs/manuals/*.pdf")
    pdfs_to_markdowns("docs/reports/*.pdf")

**2. Handle Large Batches**

.. code-block:: python

    # Process with progress tracking
    import tqdm
    
    pdf_files = glob.glob("docs/*.pdf")
    for pdf_path in tqdm.tqdm(pdf_files):
        pdf_to_markdown(pdf_path, MARKDOWN_DIR)

**3. Error Handling**

.. code-block:: python

    def pdf_to_markdown_safe(pdf_path, output_dir):
        """Convert PDF with error handling."""
        try:
            doc = pymupdf.open(pdf_path)
            md = pymupdf4llm.to_markdown(doc, header=False, footer=False)
            md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')
            output_path = Path(output_dir) / Path(doc.name).stem
            Path(output_path).with_suffix(".md").write_bytes(md_cleaned.encode('utf-8'))
            print(f"✓ Converted: {pdf_path}")
        except Exception as e:
            print(f"✗ Error converting {pdf_path}: {e}")

**4. Verify Conversions**

.. code-block:: python

    # Check conversion results
    markdown_files = list(Path(MARKDOWN_DIR).glob("*.md"))
    print(f"Successfully converted {len(markdown_files)} files")
    
    for md_file in markdown_files:
        size = md_file.stat().st_size
        print(f"  {md_file.name}: {size:,} bytes")

**5. Clean Up Invalid Files**

.. code-block:: python

    # Remove empty or corrupted Markdown files
    for md_file in Path(MARKDOWN_DIR).glob("*.md"):
        if md_file.stat().st_size < 100:  # Less than 100 bytes
            print(f"Removing small file: {md_file}")
            md_file.unlink()

Performance Considerations
--------------------------

**Processing Speed:**

- Average PDF: 1-5 seconds per document
- Large PDFs (500+ pages): 10-30 seconds
- Batch processing: Linear with number of documents

**Memory Usage:**

- Per PDF: ~50-200MB depending on size
- Parallel processing: Multiply by number of workers
- Recommendation: Process sequentially for stability

**Optimization Tips:**

.. code-block:: python

    # Skip image processing for faster conversion
    ignore_images=True
    
    # Use page separators for better chunking
    page_separators=True
    
    # Skip headers/footers to reduce noise
    header=False
    footer=False

Troubleshooting
---------------

**PDF Opening Error**

.. code-block:: python

    # Problem: "Cannot open PDF"
    # Solution: Verify file exists and is valid
    
    import os
    if os.path.exists(pdf_path):
        print("File exists")
    else:
        print("File not found")

**Encoding Errors**

.. code-block:: python

    # Problem: Invalid UTF-8 characters
    # Solution: Already handled by cleaning function
    
    md_cleaned = md.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='ignore')

**Empty Markdown Output**

.. code-block:: python

    # Problem: Converted file is empty
    # Possible causes:
    # - PDF is image-only (scanned document)
    # - PDF is corrupted
    # - PDF uses unsupported encoding
    
    # Solution: Check file size
    if Path(md_path).stat().st_size == 0:
        print(f"Warning: Empty output for {pdf_path}")

**Memory Issues with Large PDFs**

.. code-block:: python

    # Problem: Out of memory with large PDFs
    # Solution: Process one at a time, not in parallel
    
    for pdf_path in glob.glob("docs/*.pdf"):
        pdf_to_markdown(pdf_path, MARKDOWN_DIR)
        # File is closed after each conversion

Next Steps
----------

With PDFs converted to Markdown, you can proceed to:

- Chunking Markdown documents into smaller pieces
- Generating embeddings for each chunk
- Storing chunks in the vector database
- Implementing retrieval and search functionality
- Building the complete RAG pipeline

