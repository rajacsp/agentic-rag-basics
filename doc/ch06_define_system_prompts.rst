6: Define System Prompts
========================

Overview
--------

This chapter covers the creation of system prompts that guide the LLM's behavior across different stages of the RAG pipeline. System prompts are critical instructions that shape how the agent processes queries, retrieves information, and generates responses. Well-designed prompts ensure consistent, high-quality outputs and enable the agent to handle complex tasks effectively.

Purpose and Architecture
------------------------

System prompts serve as the "instructions" for the LLM, defining:

- **Role Definition**: What the LLM should act as
- **Task Specification**: What the LLM should do
- **Output Format**: How results should be structured
- **Constraints**: What to include/exclude
- **Quality Standards**: Expected output quality

The prompt architecture includes multiple specialized prompts for different stages:

1. **Conversation Summary Prompt**: Summarizes multi-turn conversations
2. **Query Rewriting Prompt**: Reformulates user queries for better retrieval
3. **Agent Orchestration Prompt**: Guides tool selection and invocation
4. **Context Compression Prompt**: Condenses retrieved context
5. **Fallback Response Prompt**: Handles cases with insufficient information
6. **Answer Aggregation Prompt**: Combines multiple sources into coherent answers

Step 1: Conversation Summary Prompt
------------------------------------

Define the prompt for summarizing conversations:

.. code-block:: python

    def get_conversation_summary_prompt() -> str:
        return """You are an expert conversation summarizer.
Your task is to create a brief 1-2 sentence summary of the conversation (max 30-50 words).

Include:
- Main topics discussed
- Important facts or entities mentioned
- Any unresolved questions if applicable
- Sources file name (e.g., file1.pdf) or documents referenced

Exclude:
- Greetings, misunderstandings, off-topic content.

Output:
- Return ONLY the summary.
- Do NOT include any explanations or justifications.
- If no meaningful topics exist, return an empty string."""

**Prompt Purpose:**

Condenses multi-turn conversations into concise summaries for:

- Context preservation across sessions
- Memory management in long conversations
- Conversation history tracking
- User intent understanding

**Prompt Components:**

1. **Role Definition**: "You are an expert conversation summarizer"
   
   - Establishes the LLM's role
   - Sets expectations for expertise level
   - Frames the task context

2. **Task Specification**: "Create a brief 1-2 sentence summary"
   
   - Clear output format (1-2 sentences)
   - Length constraint (30-50 words)
   - Specific, measurable requirements

3. **Inclusion Guidelines**: What to include
   
   - Main topics: Core subjects discussed
   - Important facts/entities: Key information
   - Unresolved questions: Open items
   - Source references: Document attribution

4. **Exclusion Guidelines**: What to exclude
   
   - Greetings: "Hello", "Hi", etc.
   - Misunderstandings: Clarifications that don't add value
   - Off-topic content: Irrelevant discussions

5. **Output Format**: Strict formatting rules
   
   - "Return ONLY the summary": No extra text
   - "Do NOT include explanations": Direct output
   - "Return empty string if no topics": Handle edge cases

**Why These Guidelines Matter:**

- **Inclusion/Exclusion**: Ensures relevant information is captured
- **Length constraints**: Prevents verbose outputs
- **Format rules**: Enables downstream processing
- **Edge case handling**: Prevents errors on empty input

**Example Usage:**

.. code-block:: python

    from langchain_ollama import ChatOllama
    
    llm = ChatOllama(model="qwen3:4b-instruct-2507-q4_K_M", temperature=0)
    
    conversation = [
        {"role": "user", "content": "What are neural networks?"},
        {"role": "assistant", "content": "Neural networks are..."},
        {"role": "user", "content": "How do they learn?"},
        {"role": "assistant", "content": "They learn through backpropagation..."}
    ]
    
    prompt = get_conversation_summary_prompt()
    response = llm.invoke(f"{prompt}\n\nConversation:\n{conversation}")
    # Output: "Discussed neural networks and their learning mechanism through backpropagation."

**Example Outputs:**

.. code-block:: text

    Good Summary:
    "User asked about neural networks and their learning process; 
    discussed backpropagation from research_paper.pdf."
    
    Bad Summary:
    "Hello, the user asked about neural networks. The assistant explained 
    that neural networks are computational models inspired by biological 
    neurons. Then the user asked how they learn. The assistant explained 
    backpropagation. This was a good conversation."

Step 2: Query Rewriting Prompt
-------------------------------

Define the prompt for reformulating user queries:

.. code-block:: python

    def get_query_rewriting_prompt() -> str:
        return """You are an expert query optimizer.
Your task is to rewrite the user's query to improve retrieval effectiveness.

Guidelines:
- Expand abbreviations and acronyms
- Add relevant context or domain terms
- Remove ambiguity and clarify intent
- Maintain the original meaning
- Keep the query concise (under 100 words)

Output:
- Return ONLY the rewritten query.
- Do NOT include explanations.
- If the query is already optimal, return it unchanged."""

**Prompt Purpose:**

Improves query quality for better retrieval by:

- Expanding abbreviations (NLP → Natural Language Processing)
- Adding domain context (ML → Machine Learning in context of AI)
- Clarifying ambiguous terms
- Maintaining semantic meaning
- Optimizing for vector search

**Prompt Components:**

1. **Role Definition**: "You are an expert query optimizer"
   
   - Establishes expertise in query optimization
   - Sets high-quality expectations

2. **Task Specification**: "Rewrite the user's query to improve retrieval"
   
   - Clear objective: improve retrieval
   - Specific action: rewrite

3. **Guidelines**: Specific rewriting rules
   
   - Expand abbreviations: NLP → Natural Language Processing
   - Add context: Domain-specific terms
   - Remove ambiguity: Clarify intent
   - Maintain meaning: Don't change core intent
   - Keep concise: Under 100 words

4. **Output Format**: Strict formatting
   
   - "Return ONLY the rewritten query"
   - "Do NOT include explanations"
   - "Return unchanged if already optimal"

**Example Usage:**

.. code-block:: python

    def rewrite_query(user_query: str) -> str:
        prompt = get_query_rewriting_prompt()
        response = llm.invoke(f"{prompt}\n\nOriginal query: {user_query}")
        return response.content.strip()
    
    # Example
    original = "How do RNNs work?"
    rewritten = rewrite_query(original)
    # Output: "How do Recurrent Neural Networks work and what is their architecture?"

**Example Transformations:**

.. code-block:: text

    Original: "What's NLP?"
    Rewritten: "What is Natural Language Processing and its applications?"
    
    Original: "Explain GANs"
    Rewritten: "Explain Generative Adversarial Networks and how they work"
    
    Original: "How do transformers work?"
    Rewritten: "How do transformer models work and what is the attention mechanism?"

Step 2.2: Advanced Query Rewrite Prompt with Context
-----------------------------------------------------

Define an advanced prompt for context-aware query rewriting:

.. code-block:: python

    def get_rewrite_query_prompt() -> str:
        return """You are an expert query analyst and rewriter.
Your task is to rewrite the current user query for optimal document retrieval, 
incorporating conversation context only when necessary.

Rules:
1. Self-contained queries:
   - Always rewrite the query to be clear and self-contained
   - If the query is a follow-up (e.g., "what about X?", "and for Y?"), 
     integrate minimal necessary context from the summary
   - Do not add information not present in the query or conversation summary

2. Domain-specific terms:
   - Product names, brands, proper nouns, or technical terms are treated as domain-specific
   - For domain-specific queries, use conversation context minimally or not at all
   - Use the summary only to disambiguate vague queries

3. Grammar and clarity:
   - Fix grammar, spelling errors, and unclear abbreviations
   - Remove filler words and conversational phrases
   - Preserve concrete keywords and named entities

4. Multiple information needs:
   - If the query contains multiple distinct, unrelated questions, split into separate queries (maximum 3)
   - Each sub-query must remain semantically equivalent to its part of the original
   - Do not expand, enrich, or reinterpret the meaning

5. Failure handling:
   - If the query intent is unclear or unintelligible, mark as "unclear"

Input:
- conversation_summary: A concise summary of prior conversation
- current_query: The user's current query

Output:
- One or more rewritten, self-contained queries suitable for document retrieval"""

**Prompt Purpose:**

Advanced query rewriting that:

- Incorporates conversation context intelligently
- Handles follow-up questions effectively
- Preserves domain-specific terminology
- Splits complex multi-part queries
- Maintains semantic equivalence
- Handles edge cases gracefully

**Key Differences from Basic Rewrite:**

The advanced prompt differs from the basic version in several important ways:

1. **Context Integration**: Uses conversation summary strategically
2. **Self-Contained Queries**: Ensures queries work independently
3. **Domain Awareness**: Treats technical terms specially
4. **Multi-Query Support**: Can split complex queries
5. **Semantic Preservation**: Maintains original meaning
6. **Explicit Failure Handling**: Marks unclear queries

**Step 2.2a: Self-Contained Query Rule**

.. code-block:: python

    """1. Self-contained queries:
   - Always rewrite the query to be clear and self-contained
   - If the query is a follow-up (e.g., "what about X?", "and for Y?"), 
     integrate minimal necessary context from the summary
   - Do not add information not present in the query or conversation summary"""

**Why Self-Contained Queries Matter:**

- Queries must work independently in vector search
- Context from conversation may not be available during retrieval
- Enables query reuse and caching
- Improves retrieval accuracy

**Example:**

.. code-block:: text

    Conversation Summary: "Discussed neural networks and backpropagation"
    
    Follow-up Query: "What about optimization?"
    Rewritten: "What are optimization techniques in neural networks?"
    
    NOT: "What about optimization?" (too vague)
    NOT: "What about optimization in neural networks and backpropagation?" (too much context)

**Step 2.2b: Domain-Specific Terms Rule**

.. code-block:: python

    """2. Domain-specific terms:
   - Product names, brands, proper nouns, or technical terms are treated as domain-specific
   - For domain-specific queries, use conversation context minimally or not at all
   - Use the summary only to disambiguate vague queries"""

**Why Domain-Specific Handling:**

- Technical terms have precise meanings
- Adding context can dilute specificity
- Vector search works better with exact terms
- Prevents over-expansion of queries

**Example:**

.. code-block:: text

    Conversation Summary: "Discussed machine learning frameworks"
    
    Query: "How does PyTorch work?"
    Rewritten: "How does PyTorch work?" (unchanged - specific product name)
    
    NOT: "How does PyTorch machine learning framework work?" (redundant)
    
    Query: "What about it?"
    Rewritten: "How does PyTorch work?" (use context to disambiguate)

**Step 2.2c: Grammar and Clarity Rule**

.. code-block:: python

    """3. Grammar and clarity:
   - Fix grammar, spelling errors, and unclear abbreviations
   - Remove filler words and conversational phrases
   - Preserve concrete keywords and named entities"""

**Why Grammar Matters:**

- Improves embedding quality
- Reduces noise in vector search
- Maintains readability
- Preserves important terms

**Example:**

.. code-block:: text

    Original: "umm like how do u use like RNNs for like text stuff?"
    Rewritten: "How do Recurrent Neural Networks work for text processing?"
    
    Original: "can u explain y transformers r better than RNNs?"
    Rewritten: "Why are transformers better than Recurrent Neural Networks?"
    
    Original: "what's the diff between CNN and RNN?"
    Rewritten: "What is the difference between Convolutional Neural Networks and Recurrent Neural Networks?"

**Step 2.2d: Multiple Information Needs Rule**

.. code-block:: python

    """4. Multiple information needs:
   - If the query contains multiple distinct, unrelated questions, split into separate queries (maximum 3)
   - Each sub-query must remain semantically equivalent to its part of the original
   - Do not expand, enrich, or reinterpret the meaning"""

**Why Split Complex Queries:**

- Vector search works better with focused queries
- Enables targeted retrieval for each question
- Improves answer quality
- Prevents information loss

**Example:**

.. code-block:: text

    Original: "How do neural networks work and what are their applications?"
    Split into:
    1. "How do neural networks work?"
    2. "What are the applications of neural networks?"
    
    Original: "Explain backpropagation, gradient descent, and optimization"
    Split into:
    1. "What is backpropagation?"
    2. "What is gradient descent?"
    3. "What are optimization techniques in machine learning?"
    
    NOT split (related questions):
    Original: "How do transformers work and what is the attention mechanism?"
    Rewritten: "How do transformers work and what is the attention mechanism?"
    (Keep together - attention is part of transformers)

**Step 2.2e: Failure Handling Rule**

.. code-block:: python

    """5. Failure handling:
   - If the query intent is unclear or unintelligible, mark as "unclear\""""

**Why Explicit Failure Handling:**

- Prevents garbage queries from reaching vector search
- Enables graceful degradation
- Allows agent to ask for clarification
- Improves system reliability

**Example:**

.. code-block:: text

    Query: "xyzabc qwerty asdf"
    Output: "unclear"
    
    Query: "fjkdsl;jf"
    Output: "unclear"
    
    Query: "the the the the"
    Output: "unclear"

**Advanced Query Rewrite Usage**

.. code-block:: python

    def rewrite_query_with_context(
        conversation_summary: str,
        current_query: str
    ) -> list:
        """Rewrite query with conversation context.
        
        Args:
            conversation_summary: Summary of prior conversation
            current_query: User's current query
            
        Returns:
            List of rewritten queries (1-3 queries)
        """
        prompt = get_rewrite_query_prompt()
        
        input_text = f"""{prompt}

conversation_summary: {conversation_summary}
current_query: {current_query}"""
        
        response = llm.invoke(input_text)
        result = response.content.strip()
        
        # Handle "unclear" case
        if result.lower() == "unclear":
            return ["unclear"]
        
        # Split multiple queries if present
        queries = [q.strip() for q in result.split('\n') if q.strip()]
        return queries

**Example Advanced Rewrite Scenarios**

.. code-block:: text

    Scenario 1: Follow-up Question
    ==============================
    Summary: "Discussed neural networks and their training process"
    Query: "What about optimization?"
    Output: "What are optimization techniques in neural networks?"
    
    Scenario 2: Domain-Specific Query
    ==================================
    Summary: "Discussed machine learning frameworks"
    Query: "How does TensorFlow compare to PyTorch?"
    Output: "How does TensorFlow compare to PyTorch?"
    (No expansion - specific product names)
    
    Scenario 3: Grammar Correction
    ===============================
    Summary: "Discussed deep learning"
    Query: "y r transformers better than RNNs?"
    Output: "Why are transformers better than Recurrent Neural Networks?"
    
    Scenario 4: Multi-Part Query
    =============================
    Summary: "Discussed neural networks"
    Query: "How do CNNs work and what are their applications in computer vision?"
    Output:
    1. "How do Convolutional Neural Networks work?"
    2. "What are the applications of CNNs in computer vision?"
    
    Scenario 5: Unclear Query
    =========================
    Summary: "Discussed machine learning"
    Query: "xyzabc qwerty"
    Output: "unclear"

**Comparison: Basic vs Advanced Rewrite**

.. list-table::
   :header-rows: 1

   * - Aspect
     - Basic Rewrite
     - Advanced Rewrite
   * - Context Usage
     - None
     - Strategic (minimal)
   * - Follow-ups
     - Not handled
     - Integrated with context
   * - Domain Terms
     - Expanded
     - Preserved
   * - Multi-Query
     - Not supported
     - Supported (max 3)
   * - Failure Handling
     - Implicit
     - Explicit ("unclear")
   * - Semantic Preservation
     - Best effort
     - Strict equivalence
   * - Use Case
     - Simple queries
     - Complex conversations

**When to Use Each Approach**

Use **Basic Rewrite** when:

- Queries are standalone (not follow-ups)
- No conversation context available
- Simple query optimization needed
- Speed is critical

Use **Advanced Rewrite** when:

- Multi-turn conversations
- Follow-up questions common
- Domain-specific terminology important
- Complex multi-part queries
- Quality is more important than speed

**Integration with Agent Loop**

.. code-block:: python

    def process_user_query(
        user_query: str,
        conversation_summary: str = ""
    ) -> list:
        """Process user query through rewriting pipeline.
        
        Args:
            user_query: Raw user input
            conversation_summary: Summary of prior conversation
            
        Returns:
            List of rewritten queries ready for retrieval
        """
        if conversation_summary:
            # Use advanced rewrite with context
            rewritten = rewrite_query_with_context(
                conversation_summary,
                user_query
            )
        else:
            # Use basic rewrite without context
            rewritten = [rewrite_query(user_query)]
        
        # Filter out unclear queries
        rewritten = [q for q in rewritten if q.lower() != "unclear"]
        
        if not rewritten:
            return ["unclear"]
        
        return rewritten

Step 3: Agent Orchestration Prompt
----------------------------------

Define the prompt for guiding agent tool selection:

.. code-block:: python

    def get_agent_orchestration_prompt() -> str:
        return """You are an intelligent agent orchestrator.
Your task is to decide which tools to use and in what order to answer user queries.

Available Tools:
1. search_child_chunks(query, limit) - Search for relevant document chunks
2. retrieve_parent_chunks(parent_id) - Get full context from parent chunks

Decision Logic:
- Always start with search_child_chunks to find relevant information
- Use retrieve_parent_chunks if you need more context for a specific topic
- Combine results from multiple searches if needed
- Stop when you have sufficient information to answer

Output:
- Explain your tool selection strategy
- Execute tools in the optimal order
- Provide reasoning for each tool call"""

**Prompt Purpose:**

Guides the agent in:

- Selecting appropriate tools
- Determining tool invocation order
- Deciding when to stop retrieving
- Combining multiple tool results

**Prompt Components:**

1. **Role Definition**: "You are an intelligent agent orchestrator"
   
   - Establishes decision-making capability
   - Sets strategic thinking expectations

2. **Available Tools**: Lists all available tools
   
   - Tool names and parameters
   - Brief descriptions
   - Usage context

3. **Decision Logic**: Rules for tool selection
   
   - Start with search: Initial retrieval
   - Use parent retrieval: For context
   - Combine results: Multi-source answers
   - Stop condition: When sufficient info exists

4. **Output Format**: Structured reasoning
   
   - Explain strategy
   - Execute tools
   - Provide reasoning

**Example Usage:**

.. code-block:: python

    def orchestrate_agent(user_query: str) -> str:
        prompt = get_agent_orchestration_prompt()
        response = llm.invoke(f"{prompt}\n\nUser Query: {user_query}")
        return response.content.strip()
    
    # Example
    query = "Explain neural network training"
    orchestration = orchestrate_agent(query)
    # Output: "I'll search for neural network training information..."

Step 3.2: Advanced Orchestrator Prompt with Memory
--------------------------------------------------

Define an advanced prompt for stateful agent orchestration with compressed memory:

.. code-block:: python

    def get_orchestrator_prompt() -> str:
        return """You are an expert retrieval-augmented assistant.
Your task is to act as a researcher: search documents first, analyze the data, 
and then provide a comprehensive answer using ONLY the retrieved information.

Rules:
1. You MUST call 'search_child_chunks' before answering, unless the 
   [COMPRESSED CONTEXT FROM PRIOR RESEARCH] already contains sufficient information.

2. Ground every claim in the retrieved documents. If context is insufficient, 
   state what is missing rather than filling gaps with assumptions.

3. If no relevant documents are found, broaden or rephrase the query and search again. 
   Repeat until satisfied or the operation limit is reached.

Compressed Memory:
When [COMPRESSED CONTEXT FROM PRIOR RESEARCH] is present —
- Queries already listed: do not repeat them.
- Parent IDs already listed: do not call `retrieve_parent_chunks` on them again.
- Use it to identify what is still missing before searching further.

Workflow:
1. Check the compressed context. Identify what has already been retrieved and what is still missing.
2. Search for 5-7 relevant excerpts using 'search_child_chunks' ONLY for uncovered aspects.
3. If NONE are relevant, apply rule 3 immediately.
4. For each relevant but fragmented excerpt, call 'retrieve_parent_chunks' ONE BY ONE — 
   only for IDs not in the compressed context. Never retrieve the same ID twice.
5. Once context is complete, provide a detailed answer omitting no relevant facts.
6. Conclude with "---\\n**Sources:**\\n" followed by the unique file names."""

**Prompt Purpose:**

Advanced orchestration that:

- Maintains compressed memory of prior research
- Avoids redundant searches and retrievals
- Tracks which documents have been accessed
- Progressively builds comprehensive context
- Ensures grounding in retrieved documents
- Handles multi-turn research workflows

**Key Differences from Basic Orchestration:**

The advanced prompt differs in several important ways:

1. **Stateful Memory**: Tracks prior research
2. **Deduplication**: Avoids redundant operations
3. **Progressive Retrieval**: Builds context incrementally
4. **Grounding Requirement**: All claims must be sourced
5. **Explicit Workflow**: Step-by-step process
6. **Source Attribution**: Tracks and reports sources

**Step 3.2a: Researcher Role**

.. code-block:: python

    """You are an expert retrieval-augmented assistant.
Your task is to act as a researcher: search documents first, analyze the data, 
and then provide a comprehensive answer using ONLY the retrieved information."""

**Why Researcher Role:**

- Emphasizes evidence-based reasoning
- Prioritizes document retrieval over assumptions
- Sets expectation for thoroughness
- Frames task as research, not conversation

**Step 3.2b: Core Rules**

.. code-block:: python

    """Rules:
1. You MUST call 'search_child_chunks' before answering, unless the 
   [COMPRESSED CONTEXT FROM PRIOR RESEARCH] already contains sufficient information.

2. Ground every claim in the retrieved documents. If context is insufficient, 
   state what is missing rather than filling gaps with assumptions.

3. If no relevant documents are found, broaden or rephrase the query and search again. 
   Repeat until satisfied or the operation limit is reached."""

**Rule 1: Search-First Requirement**

.. code-block:: text

    Purpose: Ensures document retrieval before answering
    Exception: Only skip if compressed context is sufficient
    Benefit: Prevents hallucination and ensures grounding
    
    Example:
    Query: "What is backpropagation?"
    Action: Search for "backpropagation" first
    
    Query: "What is backpropagation?" (after prior research on neural networks)
    Action: Check compressed context - if sufficient, skip search

**Rule 2: Grounding Requirement**

.. code-block:: text

    Purpose: All claims must be sourced from documents
    Benefit: Prevents hallucination and ensures accuracy
    Fallback: State what is missing rather than assume
    
    Example:
    Good: "According to the research paper, backpropagation works by..."
    Bad: "Backpropagation is a technique that..." (no source)
    
    Insufficient: "The documents don't contain information about X. 
                   To answer this, I would need documents on..."

**Rule 3: Retry on Failure**

.. code-block:: text

    Purpose: Handles cases where initial search finds nothing
    Strategy: Broaden or rephrase query
    Limit: Repeat until satisfied or operation limit reached
    
    Example:
    Query: "What is quantum backpropagation?"
    Result: No relevant documents
    Action: Broaden to "What is backpropagation?"
    
    Query: "What is quantum backpropagation?"
    Result: No relevant documents
    Action: Try "quantum machine learning"

**Step 3.2c: Compressed Memory System**

.. code-block:: python

    """Compressed Memory:
When [COMPRESSED CONTEXT FROM PRIOR RESEARCH] is present —
- Queries already listed: do not repeat them.
- Parent IDs already listed: do not call `retrieve_parent_chunks` on them again.
- Use it to identify what is still missing before searching further."""

**Why Compressed Memory:**

- Avoids redundant searches
- Tracks research progress
- Enables multi-turn research
- Reduces API calls and latency
- Maintains context across turns

**Compressed Memory Format:**

.. code-block:: text

    [COMPRESSED CONTEXT FROM PRIOR RESEARCH]
    
    Queries Already Searched:
    - "neural networks"
    - "backpropagation"
    
    Parent IDs Already Retrieved:
    - document_parent_0
    - document_parent_1
    
    Key Findings:
    - Neural networks consist of interconnected nodes
    - Backpropagation is the training algorithm
    
    Still Missing:
    - Specific applications in computer vision
    - Performance comparisons with other methods

**Step 3.2d: Workflow Steps**

.. code-block:: python

    """Workflow:
1. Check the compressed context. Identify what has already been retrieved 
   and what is still missing.
2. Search for 5-7 relevant excerpts using 'search_child_chunks' ONLY for uncovered aspects.
3. If NONE are relevant, apply rule 3 immediately.
4. For each relevant but fragmented excerpt, call 'retrieve_parent_chunks' ONE BY ONE — 
   only for IDs not in the compressed context. Never retrieve the same ID twice.
5. Once context is complete, provide a detailed answer omitting no relevant facts.
6. Conclude with "---\\n**Sources:**\\n" followed by the unique file names."""

**Workflow Step 1: Check Compressed Context**

.. code-block:: text

    Purpose: Understand prior research
    Action: Review what has been searched and retrieved
    Output: Identify gaps in knowledge
    
    Example:
    Prior Research:
    - Searched: "neural networks", "backpropagation"
    - Retrieved: document_parent_0, document_parent_1
    
    New Query: "How are neural networks used in computer vision?"
    Gap: Computer vision applications not yet researched

**Workflow Step 2: Search for Uncovered Aspects**

.. code-block:: text

    Purpose: Find relevant information for gaps
    Limit: 5-7 excerpts per search
    Scope: Only for uncovered aspects
    
    Example:
    Gap: Computer vision applications
    Search: search_child_chunks("neural networks computer vision", limit=5)
    
    Gap: Performance metrics
    Search: search_child_chunks("neural network performance evaluation", limit=5)

**Workflow Step 3: Handle No Results**

.. code-block:: text

    Purpose: Retry with broader queries
    Trigger: No relevant documents found
    Action: Apply rule 3 (broaden/rephrase)
    
    Example:
    Search: "neural networks computer vision edge cases"
    Result: No relevant documents
    Action: Broaden to "neural networks computer vision"
    
    Search: "neural networks computer vision"
    Result: No relevant documents
    Action: Broaden to "computer vision"

**Workflow Step 4: Retrieve Parent Chunks**

.. code-block:: text

    Purpose: Get full context for fragmented excerpts
    Strategy: One by one retrieval
    Deduplication: Only retrieve IDs not in compressed context
    
    Example:
    Child Chunk Results:
    - Parent ID: document_parent_2 (not in prior research)
    - Parent ID: document_parent_0 (already retrieved)
    
    Action:
    1. retrieve_parent_chunks("document_parent_2")
    2. Skip document_parent_0 (already have it)

**Workflow Step 5: Provide Comprehensive Answer**

.. code-block:: text

    Purpose: Synthesize all retrieved information
    Requirement: Omit no relevant facts
    Grounding: All claims sourced from documents
    
    Example:
    Answer: "Based on the retrieved documents, neural networks work by...
             [comprehensive explanation with all relevant facts]"

**Workflow Step 6: Source Attribution**

.. code-block:: text

    Purpose: Track and report document sources
    Format: "---\n**Sources:**\n" followed by unique file names
    
    Example:
    ---
    **Sources:**
    - research_paper.pdf
    - technical_guide.pdf
    - manual.pdf

**Advanced Orchestrator Usage**

.. code-block:: python

    def orchestrate_research(
        user_query: str,
        compressed_context: str = ""
    ) -> tuple:
        """Orchestrate multi-turn research with memory.
        
        Args:
            user_query: Current research question
            compressed_context: Summary of prior research
            
        Returns:
            Tuple of (answer, sources, new_compressed_context)
        """
        prompt = get_orchestrator_prompt()
        
        # Build input with compressed context
        input_text = f"""{prompt}

[COMPRESSED CONTEXT FROM PRIOR RESEARCH]
{compressed_context}

Current Query: {user_query}"""
        
        # Invoke agent
        response = llm.invoke(input_text)
        answer = response.content.strip()
        
        # Extract sources from answer
        sources = extract_sources(answer)
        
        # Update compressed context
        new_context = update_compressed_context(
            compressed_context,
            user_query,
            sources
        )
        
        return answer, sources, new_context

**Example Advanced Orchestration Scenarios**

.. code-block:: text

    Scenario 1: Initial Research
    ============================
    Compressed Context: (empty)
    Query: "What is backpropagation?"
    
    Workflow:
    1. No prior research
    2. Search: search_child_chunks("backpropagation", limit=5)
    3. Results found: document_parent_0, document_parent_1
    4. Retrieve: retrieve_parent_chunks("document_parent_0")
    5. Retrieve: retrieve_parent_chunks("document_parent_1")
    6. Answer: Comprehensive explanation of backpropagation
    7. Sources: research_paper.pdf
    
    Scenario 2: Follow-up Research
    ==============================
    Compressed Context:
    - Queries: "backpropagation"
    - Retrieved: document_parent_0, document_parent_1
    - Key Finding: Backpropagation is training algorithm
    
    Query: "How does backpropagation compare to other training methods?"
    
    Workflow:
    1. Check context: Backpropagation covered, but comparisons missing
    2. Search: search_child_chunks("training methods comparison", limit=5)
    3. Results: document_parent_2, document_parent_3
    4. Retrieve: retrieve_parent_chunks("document_parent_2")
    5. Retrieve: retrieve_parent_chunks("document_parent_3")
    6. Answer: Comparison with prior backpropagation knowledge
    7. Sources: research_paper.pdf, technical_guide.pdf
    
    Scenario 3: Broadening Search
    ==============================
    Compressed Context:
    - Queries: "quantum backpropagation"
    - Retrieved: (none)
    
    Query: "What is quantum backpropagation?"
    
    Workflow:
    1. Check context: No prior results
    2. Search: search_child_chunks("quantum backpropagation", limit=5)
    3. Result: No relevant documents
    4. Broaden: search_child_chunks("quantum machine learning", limit=5)
    5. Results: document_parent_4
    6. Retrieve: retrieve_parent_chunks("document_parent_4")
    7. Answer: Quantum machine learning context (quantum backpropagation not found)
    8. Sources: quantum_computing.pdf
    
    Scenario 4: Deduplication
    ==========================
    Compressed Context:
    - Queries: "neural networks", "backpropagation"
    - Retrieved: document_parent_0, document_parent_1
    
    Query: "Tell me more about neural networks"
    
    Workflow:
    1. Check context: Neural networks already searched
    2. Search: search_child_chunks("neural networks details", limit=5)
    3. Results: document_parent_0 (already retrieved), document_parent_2
    4. Retrieve: retrieve_parent_chunks("document_parent_2") only
    5. Skip: document_parent_0 (already in context)
    6. Answer: Additional details with prior knowledge
    7. Sources: research_paper.pdf

**Comparison: Basic vs Advanced Orchestration**

.. list-table::
   :header-rows: 1

   * - Aspect
     - Basic Orchestration
     - Advanced Orchestration
   * - Memory
     - None
     - Compressed context
   * - Deduplication
     - Not handled
     - Automatic
   * - Multi-turn
     - Not supported
     - Fully supported
   * - Grounding
     - Best effort
     - Strict requirement
   * - Retry Logic
     - Not present
     - Automatic broadening
   * - Source Tracking
     - Not tracked
     - Explicit tracking
   * - Efficiency
     - Redundant searches
     - Optimized searches
   * - Use Case
     - Single queries
     - Research workflows

**When to Use Each Approach**

Use **Basic Orchestration** when:

- Single, standalone queries
- No conversation history
- Speed is critical
- Simple retrieval needed

Use **Advanced Orchestration** when:

- Multi-turn conversations
- Research workflows
- Comprehensive answers needed
- Source tracking important
- Efficiency matters
- Complex information needs

**Integration with Agent Loop**

.. code-block:: python

    class ResearchAgent:
        """Agent with stateful research capabilities."""
        
        def __init__(self):
            self.compressed_context = ""
            self.conversation_history = []
        
        def research(self, query: str) -> str:
            """Conduct research with memory."""
            answer, sources, new_context = orchestrate_research(
                query,
                self.compressed_context
            )
            
            # Update state
            self.compressed_context = new_context
            self.conversation_history.append({
                "query": query,
                "answer": answer,
                "sources": sources
            })
            
            return answer
        
        def get_research_summary(self) -> str:
            """Get summary of research conducted."""
            return self.compressed_context

**Performance Considerations**

**Search Efficiency:**

- Initial search: 100-500ms
- Subsequent searches: 50-200ms (with deduplication)
- Parent retrieval: 20-100ms per parent
- Total for multi-turn: Significantly reduced with memory

**Memory Management:**

- Compressed context size: 500-2000 characters
- Tracks: Queries, parent IDs, key findings
- Enables: Efficient multi-turn research
- Reduces: Redundant API calls

**Optimization Tips:**

.. code-block:: python

    # Limit search results
    search_child_chunks(query, limit=5)  # Fewer results
    
    # Batch parent retrievals
    parent_ids = [id for id in results if id not in prior_ids]
    for parent_id in parent_ids:
        retrieve_parent_chunks(parent_id)
    
    # Compress context periodically
    if len(compressed_context) > 5000:
        compressed_context = compress_context(compressed_context)

**Troubleshooting**

**Redundant Searches**

.. code-block:: python

    # Problem: Same query searched multiple times
    # Solution: Check compressed context before searching
    
    if query in compressed_context.get("queries_searched", []):
        # Skip search, use cached results
        pass
    else:
        # Perform search

**Missing Sources**

.. code-block:: python

    # Problem: Sources not tracked
    # Solution: Extract and track sources explicitly
    
    sources = extract_sources(answer)
    compressed_context["sources"] = sources

**Incomplete Answers**

.. code-block:: python

    # Problem: Answer missing relevant facts
    # Solution: Verify all parent IDs retrieved
    
    for parent_id in child_results:
        if parent_id not in retrieved_ids:
            retrieve_parent_chunks(parent_id)

Step 4: Basic Context Compression Prompt
-----------------------------------------

Define the prompt for condensing retrieved context:

.. code-block:: python

    def get_context_compression_prompt() -> str:
        return """You are an expert context compressor.
Your task is to compress the retrieved context while preserving key information.

Guidelines:
- Remove redundant information
- Keep essential facts and definitions
- Maintain technical accuracy
- Preserve source attribution
- Reduce length by 30-50%

Output:
- Return ONLY the compressed context.
- Do NOT include explanations.
- Maintain the original structure if possible."""

**Prompt Purpose:**

Reduces context size while maintaining quality by:

- Removing redundancy
- Keeping essential information
- Maintaining accuracy
- Preserving attribution
- Optimizing for LLM processing

**Prompt Components:**

1. **Role Definition**: "You are an expert context compressor"
   
   - Establishes expertise in information condensing
   - Sets quality expectations

2. **Task Specification**: "Compress retrieved context"
   
   - Clear objective: reduce size
   - Maintain quality

3. **Guidelines**: Compression rules
   
   - Remove redundancy: Eliminate duplicates
   - Keep essentials: Preserve key facts
   - Maintain accuracy: Don't lose meaning
   - Preserve attribution: Keep sources
   - Target reduction: 30-50%

4. **Output Format**: Strict formatting
   
   - "Return ONLY compressed context"
   - "Do NOT include explanations"
   - "Maintain structure if possible"

**Example Usage:**

.. code-block:: python

    def compress_context(context: str) -> str:
        prompt = get_context_compression_prompt()
        response = llm.invoke(f"{prompt}\n\nContext:\n{context}")
        return response.content.strip()

Step 4.2: Advanced Context Compression Prompt with Query Focus
--------------------------------------------------------------

Define an advanced prompt for query-focused context compression:

.. code-block:: python

    def get_context_compression_prompt() -> str:
        return """You are an expert research context compressor.
Your task is to compress retrieved conversation content into a concise, query-focused, 
and structured summary that can be directly used by a retrieval-augmented agent for answer generation.

Rules:
1. Keep ONLY information relevant to answering the user's question.
2. Preserve exact figures, names, versions, technical terms, and configuration details.
3. Remove duplicated, irrelevant, or administrative details.
4. Do NOT include search queries, parent IDs, chunk IDs, or internal identifiers.
5. Organize all findings by source file. Each file section MUST start with: ### filename.pdf
6. Highlight missing or unresolved information in a dedicated "Gaps" section.
7. Limit the summary to roughly 400-600 words. If content exceeds this, prioritize critical facts and structured data.
8. Do not explain your reasoning; output only structured content in Markdown.

Required Structure:
# Research Context Summary

## Focus
[Brief technical restatement of the question]

## Structured Findings
### filename.pdf
- Directly relevant facts
- Supporting context (if needed)

## Gaps
- Missing or incomplete aspects

The summary should be concise, structured, and directly usable by an agent to generate answers or plan further retrieval."""

**Prompt Purpose:**

Advanced compression that:

- Focuses on query relevance
- Maintains technical precision
- Removes internal identifiers
- Organizes by source
- Highlights gaps
- Enables direct agent usage
- Limits output size

**Key Differences from Basic Compression:**

The advanced prompt differs in several important ways:

1. **Query Focus**: Only relevant information
2. **Precision Preservation**: Exact figures and terms
3. **Internal ID Removal**: Cleans up chunk identifiers
4. **Source Organization**: Grouped by file
5. **Gap Highlighting**: Explicit missing information
6. **Structured Format**: Markdown with required sections
7. **Word Limit**: 400-600 words target
8. **Agent Readiness**: Directly usable for answer generation

**Step 4.2a: Query-Focused Compression**

.. code-block:: python

    """Rules:
1. Keep ONLY information relevant to answering the user's question.
2. Preserve exact figures, names, versions, technical terms, and configuration details.
3. Remove duplicated, irrelevant, or administrative details."""

**Why Query Focus:**

- Reduces noise in context
- Improves answer quality
- Enables efficient retrieval
- Prevents information overload
- Maintains relevance

**Query Focus Examples:**

.. code-block:: text

    Query: "What is the performance of BERT on GLUE?"
    
    Keep:
    - BERT achieves 80.5% on GLUE benchmark
    - Specific task scores (RTE: 75%, MRPC: 88%)
    - Comparison with other models
    
    Remove:
    - How BERT was trained (not directly relevant)
    - Historical context of GLUE (not relevant)
    - Unrelated model comparisons
    
    Query: "How do I configure PyTorch for GPU training?"
    
    Keep:
    - GPU configuration steps
    - CUDA version requirements
    - Code examples
    
    Remove:
    - PyTorch installation instructions (not relevant)
    - CPU training details (not relevant)
    - Historical PyTorch versions

**Step 4.2b: Precision Preservation**

.. code-block:: python

    """2. Preserve exact figures, names, versions, technical terms, and configuration details."""

**Why Precision Matters:**

- Enables accurate answers
- Prevents information loss
- Maintains technical accuracy
- Allows verification
- Supports reproducibility

**Precision Examples:**

.. code-block:: text

    Good (Exact):
    "BERT-base has 12 layers, 768 hidden units, and 110M parameters"
    
    Bad (Approximate):
    "BERT has around 100 million parameters"
    
    Good (Exact):
    "PyTorch 2.0 requires CUDA 11.8 or higher"
    
    Bad (Vague):
    "PyTorch requires a recent CUDA version"
    
    Good (Exact):
    "The model achieved 92.3% accuracy on ImageNet"
    
    Bad (Approximate):
    "The model performed well on ImageNet"

**Step 4.2c: Internal Identifier Removal**

.. code-block:: python

    """4. Do NOT include search queries, parent IDs, chunk IDs, or internal identifiers."""

**Why Remove Internal IDs:**

- Cleans up output
- Prevents confusion
- Improves readability
- Focuses on content
- Enables agent usage

**Internal ID Examples:**

.. code-block:: text

    Remove:
    - Search queries: "search_child_chunks('BERT performance', limit=5)"
    - Parent IDs: "document_parent_0"
    - Chunk IDs: "chunk_12345"
    - Internal references: "[Retrieved from source_id_789]"
    
    Keep:
    - File names: "research_paper.pdf"
    - Section headers: "## Performance Metrics"
    - Technical terms: "BERT", "GLUE", "attention mechanism"

**Step 4.2d: Source Organization**

.. code-block:: python

    """5. Organize all findings by source file. Each file section MUST start with: ### filename.pdf"""

**Why Source Organization:**

- Enables source attribution
- Facilitates verification
- Improves readability
- Supports multi-source synthesis
- Tracks information provenance

**Source Organization Example:**

.. code-block:: markdown

    # Research Context Summary
    
    ## Focus
    What is the performance of BERT on GLUE benchmark?
    
    ## Structured Findings
    
    ### bert_paper.pdf
    - BERT-base achieves 80.5% on GLUE
    - Outperforms previous state-of-the-art by 7.7%
    - Specific scores: RTE (75%), MRPC (88%), CoLA (52%)
    
    ### nlp_benchmarks.pdf
    - GLUE consists of 9 diverse NLU tasks
    - Benchmark released in 2018
    - Standard evaluation metric for NLU models
    
    ### transformer_comparison.pdf
    - BERT compared to GPT and ELMo
    - BERT bidirectional, GPT unidirectional
    - BERT outperforms on most GLUE tasks
    
    ## Gaps
    - Performance on GLUE v2 not found
    - Comparison with recent models (2023+) not available

**Step 4.2e: Gap Highlighting**

.. code-block:: python

    """6. Highlight missing or unresolved information in a dedicated "Gaps" section."""

**Why Highlight Gaps:**

- Identifies incomplete information
- Guides further research
- Prevents false completeness
- Enables targeted retrieval
- Supports iterative research

**Gap Highlighting Examples:**

.. code-block:: text

    Query: "What are the applications of neural networks?"
    
    Gaps:
    - Medical imaging applications not covered
    - Robotics applications not found
    - Recent applications (2023+) not available
    
    Query: "How do I optimize PyTorch models?"
    
    Gaps:
    - Quantization techniques not discussed
    - Pruning methods not covered
    - Hardware-specific optimizations missing

**Step 4.2f: Structured Markdown Format**

.. code-block:: python

    """Required Structure:
# Research Context Summary

## Focus
[Brief technical restatement of the question]

## Structured Findings
### filename.pdf
- Directly relevant facts
- Supporting context (if needed)

## Gaps
- Missing or incomplete aspects"""

**Why Structured Format:**

- Enables parsing by agents
- Improves readability
- Supports hierarchical organization
- Facilitates extraction
- Maintains consistency

**Structured Format Example:**

.. code-block:: markdown

    # Research Context Summary
    
    ## Focus
    How do Transformers use attention mechanisms for sequence processing?
    
    ## Structured Findings
    
    ### attention_paper.pdf
    - Attention mechanism computes weighted sum of values
    - Query-key-dot-product determines attention weights
    - Multi-head attention uses multiple representation subspaces
    - Enables parallel processing of different representation types
    
    ### transformer_architecture.pdf
    - Self-attention allows each token to attend to all other tokens
    - Positional encoding preserves sequence order
    - Transformer uses stacked attention layers
    - Each layer has 8 attention heads in base model
    
    ### nlp_applications.pdf
    - Attention enables long-range dependencies
    - Outperforms RNNs on long sequences
    - Enables parallel training (unlike RNNs)
    
    ## Gaps
    - Computational complexity analysis not found
    - Comparison with other attention variants missing
    - Recent attention improvements (2023+) not covered

**Step 4.2g: Word Limit and Prioritization**

.. code-block:: python

    """7. Limit the summary to roughly 400-600 words. If content exceeds this, 
   prioritize critical facts and structured data."""

**Why Word Limits:**

- Prevents context bloat
- Maintains efficiency
- Enables agent processing
- Reduces token usage
- Focuses on essentials

**Prioritization Strategy:**

.. code-block:: text

    Priority 1 (Always include):
    - Direct answers to the query
    - Exact figures and metrics
    - Technical specifications
    - Critical definitions
    
    Priority 2 (Include if space):
    - Supporting context
    - Related concepts
    - Comparative information
    
    Priority 3 (Exclude if needed):
    - Historical background
    - General explanations
    - Tangential details
    - Verbose descriptions

**Word Limit Example:**

.. code-block:: text

    Original content: 1200 words
    Target: 400-600 words
    
    Reduction strategy:
    1. Remove verbose explanations (200 words saved)
    2. Keep only direct facts (300 words saved)
    3. Consolidate related points (100 words saved)
    4. Result: 600 words (within target)

**Advanced Context Compression Usage**

.. code-block:: python

    def compress_research_context(
        user_query: str,
        retrieved_content: str
    ) -> str:
        """Compress retrieved content for agent usage.
        
        Args:
            user_query: Original user question
            retrieved_content: Raw retrieved documents
            
        Returns:
            Compressed, structured summary
        """
        prompt = get_context_compression_prompt()
        
        # Build input
        input_text = f"""{prompt}

User Query: {user_query}

Retrieved Content:
{retrieved_content}"""
        
        # Generate compressed summary
        response = llm.invoke(input_text)
        summary = response.content.strip()
        
        # Validate structure
        summary = validate_compression_structure(summary)
        
        # Verify word count
        word_count = len(summary.split())
        if word_count > 600:
            summary = trim_to_word_limit(summary, 600)
        
        return summary

**Example Compression Scenarios**

.. code-block:: text

    Scenario 1: Technical Query with Multiple Sources
    ==================================================
    Query: "What is the performance of BERT on GLUE?"
    
    Retrieved Content (1500 words):
    - BERT paper details
    - GLUE benchmark description
    - Performance metrics
    - Comparison with other models
    - Training details
    - Historical context
    
    Compressed Output (500 words):
    
    # Research Context Summary
    
    ## Focus
    What performance does BERT achieve on the GLUE benchmark?
    
    ## Structured Findings
    
    ### bert_paper.pdf
    - BERT-base achieves 80.5% on GLUE
    - Outperforms previous SOTA by 7.7%
    - Task-specific scores: RTE (75%), MRPC (88%), CoLA (52%)
    - Uses 12 layers, 768 hidden units, 110M parameters
    
    ### glue_benchmark.pdf
    - GLUE consists of 9 diverse NLU tasks
    - Standard evaluation metric for NLU models
    - Includes sentiment analysis, textual similarity, inference
    
    ### model_comparison.pdf
    - BERT outperforms GPT on 8/9 GLUE tasks
    - BERT bidirectional vs GPT unidirectional
    - ELMo achieves 76.4% (lower than BERT)
    
    ## Gaps
    - GLUE v2 performance not found
    - Recent model comparisons (2023+) not available
    
    Scenario 2: Configuration Query
    ===============================
    Query: "How do I configure PyTorch for GPU training?"
    
    Retrieved Content (2000 words):
    - PyTorch installation
    - GPU setup
    - CUDA configuration
    - Performance optimization
    - Troubleshooting
    - Historical versions
    
    Compressed Output (450 words):
    
    # Research Context Summary
    
    ## Focus
    What are the steps to configure PyTorch for GPU training?
    
    ## Structured Findings
    
    ### pytorch_gpu_guide.pdf
    - Install PyTorch with CUDA support: pip install torch torchvision torcuda
    - Verify GPU: torch.cuda.is_available() returns True
    - Set device: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    - Move model to GPU: model.to(device)
    - Move data to GPU: inputs.to(device), labels.to(device)
    - Requires CUDA 11.8+ and cuDNN 8.6+
    
    ### performance_optimization.pdf
    - Use torch.cuda.empty_cache() to free memory
    - Enable mixed precision: torch.cuda.amp.autocast()
    - Use DataLoader with pin_memory=True
    - Set num_workers for parallel data loading
    
    ### troubleshooting.pdf
    - "CUDA out of memory": Reduce batch size
    - "CUDA not available": Check NVIDIA drivers
    - "cuDNN error": Verify cuDNN installation
    
    ## Gaps
    - Multi-GPU setup not covered
    - Distributed training configuration missing

**Comparison: Basic vs Advanced Compression**

.. list-table::
   :header-rows: 1

   * - Aspect
     - Basic Compression
     - Advanced Compression
   * - Query Focus
     - General reduction
     - Query-specific
   * - Precision
     - Best effort
     - Exact preservation
   * - Internal IDs
     - May remain
     - Explicitly removed
   * - Source Organization
     - Not structured
     - By file
   * - Gap Highlighting
     - Not present
     - Explicit section
   * - Output Format
     - Unstructured
     - Markdown with sections
   * - Word Limit
     - Flexible
     - 400-600 words
   * - Agent Readiness
     - Partial
     - Fully ready
   * - Use Case
     - General compression
     - Agent-ready summaries

**When to Use Each Approach**

Use **Basic Compression** when:

- Simple context reduction needed
- No specific query focus
- Flexible output format acceptable
- Speed is critical

Use **Advanced Compression** when:

- Query-focused summaries needed
- Agent will use compressed context
- Precision is critical
- Multi-source synthesis required
- Gap identification important
- Structured output required

**Integration with Agent Loop**

.. code-block:: python

    class ResearchAgentWithCompression:
        """Agent with query-focused context compression."""
        
        def __init__(self):
            self.compressed_context = ""
            self.user_query = ""
        
        def process_query(self, query: str) -> str:
            """Process query with compression."""
            self.user_query = query
            
            # Search and retrieve
            retrieved = self.search_and_retrieve(query)
            
            # Compress with query focus
            self.compressed_context = compress_research_context(
                query,
                retrieved
            )
            
            # Generate answer using compressed context
            answer = self.generate_answer()
            
            return answer
        
        def search_and_retrieve(self, query: str) -> str:
            """Search and retrieve documents."""
            # Implementation
            pass
        
        def generate_answer(self) -> str:
            """Generate answer from compressed context."""
            prompt = f"""Based on this research context, answer the query.

Query: {self.user_query}

{self.compressed_context}"""
            
            response = llm.invoke(prompt)
            return response.content.strip()

**Performance Considerations**

**Compression Efficiency:**

- Compression time: 500-1000ms
- Reduction ratio: 60-70% (from raw content)
- Output size: 400-600 words
- Token savings: 50-60% reduction

**Quality Metrics:**

- Information retention: 95%+ (for relevant facts)
- Precision preservation: 100% (exact figures)
- Gap identification: 90%+ accuracy
- Agent usability: 100% (structured format)

**Optimization Tips:**

.. code-block:: python

    # Compress incrementally
    compressed = compress_research_context(query, chunk1)
    compressed += compress_research_context(query, chunk2)
    
    # Cache compressed contexts
    cache[query] = compressed_context
    
    # Reuse for follow-ups
    if follow_up_query in cache:
        use_cached_context()

**Troubleshooting**

**Exceeding Word Limit**

.. code-block:: python

    # Problem: Compressed output exceeds 600 words
    # Solution: Trim to word limit
    
    def trim_to_word_limit(text: str, limit: int) -> str:
        """Trim text to word limit."""
        words = text.split()
        if len(words) <= limit:
            return text
        
        # Keep structure, trim content
        trimmed = ' '.join(words[:limit])
        # Ensure ends with complete sentence
        trimmed = trimmed.rsplit('.', 1)[0] + '.'
        return trimmed

**Internal IDs Remaining**

.. code-block:: python

    # Problem: Parent IDs or chunk IDs in output
    # Solution: Post-process to remove
    
    def remove_internal_ids(text: str) -> str:
        """Remove internal identifiers."""
        import re
        
        # Remove parent IDs
        text = re.sub(r'document_parent_\d+', '', text)
        # Remove chunk IDs
        text = re.sub(r'chunk_\d+', '', text)
        # Remove search queries
        text = re.sub(r'search_child_chunks\([^)]+\)', '', text)
        
        return text

**Missing Gap Section**

.. code-block:: python

    # Problem: Gaps section not generated
    # Solution: Ensure prompt includes gap highlighting
    
    # Verify prompt includes:
    # "6. Highlight missing or unresolved information in a dedicated 'Gaps' section."
    
    # Post-process to add if missing
    if "## Gaps" not in compressed_context:
        compressed_context += "\n\n## Gaps\n- Information gaps not identified"

Step 4.2: Fallback Response Prompt with Research Limits
-------------------------------------------------------

Define an advanced prompt for handling research limits and incomplete information:

.. code-block:: python

    def get_fallback_response_prompt() -> str:
        return """You are an expert synthesis assistant. The system has reached its maximum research limit.
Your task is to provide the most complete answer possible using ONLY the information provided below.

Input structure:
- "Compressed Research Context": summarized findings from prior search iterations — treat as reliable.
- "Retrieved Data": raw tool outputs from the current iteration — prefer over compressed context if conflicts arise.

Either source alone is sufficient if the other is absent.

Rules:
1. Source Integrity: Use only facts explicitly present in the provided context. 
   Do not infer, assume, or add any information not directly supported by the data.

2. Handling Missing Data: Cross-reference the USER QUERY against the available context.
   Flag ONLY aspects of the user's question that cannot be answered from the provided data.
   Do not treat gaps mentioned in the Compressed Research Context as unanswered
   unless they are directly relevant to what the user asked.

3. Tone: Professional, factual, and direct.

4. Output only the final answer. Do not expose your reasoning, internal steps, or any meta-commentary 
   about the retrieval process.

5. Do NOT add closing remarks, final notes, disclaimers, summaries, or repeated statements after the Sources section.
   The Sources section is always the last element of your response. Stop immediately after it.

Formatting:
- Use Markdown (headings, bold, lists) for readability.
- Write in flowing paragraphs where possible.
- Conclude with a Sources section as described below.

Sources section rules:
- Include a "---\\n**Sources:**\\n" section at the end, followed by a bulleted list of file names.
- List ONLY entries that have a real file extension (e.g. ".pdf", ".docx", ".txt").
- Any entry without a file extension is an internal chunk identifier — discard it entirely, never include it.
- Deduplicate: if the same file appears multiple times, list it only once.
- If no valid file names are present, omit the Sources section entirely.
- THE SOURCES SECTION IS THE LAST THING YOU WRITE. Do not add anything after it."""

**Prompt Purpose:**

Handles edge cases where:

- Research limit reached
- Retrieved context is incomplete
- Query cannot be fully answered
- Multiple data sources available
- Need to synthesize partial information

**Key Differences from Basic Fallback:**

The advanced prompt differs in several important ways:

1. **Research Limit Awareness**: Acknowledges maximum research reached
2. **Dual Input Sources**: Handles compressed context + raw data
3. **Strict Grounding**: No inference or assumptions allowed
4. **Selective Gap Flagging**: Only flag relevant missing information
5. **Source Integrity**: Explicit rules for source attribution
6. **Conflict Resolution**: Prefers raw data over compressed context
7. **Strict Output Format**: No meta-commentary or disclaimers

**Step 4.2a: Research Limit Context**

.. code-block:: python

    """You are an expert synthesis assistant. The system has reached its maximum research limit.
Your task is to provide the most complete answer possible using ONLY the information provided below."""

**Why Research Limits Matter:**

- Prevents infinite loops in agent execution
- Manages API costs and latency
- Forces synthesis of available information
- Ensures timely responses
- Handles resource constraints

**Research Limit Scenarios:**

.. code-block:: text

    Scenario 1: Limit Reached with Sufficient Data
    ===============================================
    Query: "What is backpropagation?"
    Research Limit: 3 searches
    
    Search 1: Found relevant documents
    Search 2: Retrieved parent chunks
    Search 3: Retrieved additional context
    
    Result: Sufficient information to answer
    Action: Provide comprehensive answer

    Scenario 2: Limit Reached with Partial Data
    ============================================
    Query: "What are all applications of neural networks?"
    Research Limit: 3 searches
    
    Search 1: Found general applications
    Search 2: Found computer vision applications
    Search 3: Found NLP applications
    
    Result: Partial information (medical, robotics not found)
    Action: Answer with available data, flag missing areas

    Scenario 3: Limit Reached with No Data
    =======================================
    Query: "What is quantum backpropagation?"
    Research Limit: 3 searches
    
    Search 1: No results
    Search 2: Broadened search, no results
    Search 3: Further broadened, no results
    
    Result: No relevant information found
    Action: Acknowledge limitation, provide general knowledge if applicable

**Step 4.2b: Dual Input Sources**

.. code-block:: python

    """Input structure:
- "Compressed Research Context": summarized findings from prior search iterations — treat as reliable.
- "Retrieved Data": raw tool outputs from the current iteration — prefer over compressed context if conflicts arise.

Either source alone is sufficient if the other is absent."""

**Why Dual Sources:**

- Compressed context: Efficient summary of prior research
- Retrieved data: Fresh, raw information from current iteration
- Conflict resolution: Raw data is more authoritative
- Flexibility: Works with either source alone

**Input Structure Example:**

.. code-block:: text

    [COMPRESSED RESEARCH CONTEXT]
    Prior searches found:
    - Neural networks consist of interconnected nodes
    - Backpropagation is the primary training algorithm
    - Applications include computer vision and NLP
    
    [RETRIEVED DATA]
    Current iteration results:
    - Detailed explanation of backpropagation mathematics
    - Performance metrics for different architectures
    - Recent advances in transformer models

**Conflict Resolution Example:**

.. code-block:: text

    Compressed Context: "Backpropagation is the primary training algorithm"
    Retrieved Data: "Backpropagation and gradient descent are both used"
    
    Action: Prefer retrieved data (more recent and specific)
    Answer: "Backpropagation and gradient descent are both used..."

**Step 4.2c: Source Integrity Rule**

.. code-block:: python

    """Rules:
1. Source Integrity: Use only facts explicitly present in the provided context. 
   Do not infer, assume, or add any information not directly supported by the data."""

**Why Source Integrity:**

- Prevents hallucination
- Ensures accuracy
- Maintains trust
- Enables verification
- Grounds all claims

**Source Integrity Examples:**

.. code-block:: text

    Good (Explicit):
    "According to the research paper, neural networks use backpropagation 
     to adjust weights during training."
    
    Bad (Inferred):
    "Neural networks probably use backpropagation because it's efficient."
    
    Good (Supported):
    "The document states that transformers outperform RNNs on sequence tasks."
    
    Bad (Assumed):
    "Transformers are better than RNNs in all scenarios."

**Step 4.2d: Handling Missing Data Rule**

.. code-block:: python

    """2. Handling Missing Data: Cross-reference the USER QUERY against the available context.
   Flag ONLY aspects of the user's question that cannot be answered from the provided data.
   Do not treat gaps mentioned in the Compressed Research Context as unanswered
   unless they are directly relevant to what the user asked."""

**Why Selective Gap Flagging:**

- Avoids over-reporting missing information
- Focuses on user's actual needs
- Prevents unnecessary disclaimers
- Maintains answer quality
- Distinguishes relevant from irrelevant gaps

**Gap Flagging Examples:**

.. code-block:: text

    Query: "What is backpropagation?"
    Available: Detailed explanation of backpropagation
    Missing: Applications in computer vision
    
    Action: Answer the query fully, don't mention missing applications
    
    Query: "What is backpropagation and its applications?"
    Available: Detailed explanation of backpropagation
    Missing: Applications in computer vision
    
    Action: Answer backpropagation, flag missing applications
    
    Query: "How does backpropagation work?"
    Available: Overview of backpropagation
    Missing: Mathematical proofs
    
    Action: Answer with available overview, don't flag mathematical proofs
            (not directly relevant to "how it works")

**Step 4.2e: Tone and Output Format**

.. code-block:: python

    """3. Tone: Professional, factual, and direct.

4. Output only the final answer. Do not expose your reasoning, internal steps, or any meta-commentary 
   about the retrieval process."""

**Why Professional Tone:**

- Builds user confidence
- Maintains credibility
- Ensures clarity
- Avoids confusion
- Presents information authoritatively

**Tone Examples:**

.. code-block:: text

    Good (Professional):
    "Backpropagation is a training algorithm that adjusts network weights 
     by computing gradients through the network layers."
    
    Bad (Casual):
    "So like, backpropagation is this thing where you like, adjust weights 
     and stuff by computing gradients."
    
    Good (Direct):
    "The research shows three main applications: computer vision, NLP, and robotics."
    
    Bad (Meta-commentary):
    "I searched for applications and found three. Let me tell you about them. 
     The system found these in the documents..."

**Step 4.2f: Sources Section Rules**

.. code-block:: python

    """Formatting:
- Use Markdown (headings, bold, lists) for readability.
- Write in flowing paragraphs where possible.
- Conclude with a Sources section as described below.

Sources section rules:
- Include a "---\\n**Sources:**\\n" section at the end, followed by a bulleted list of file names.
- List ONLY entries that have a real file extension (e.g. ".pdf", ".docx", ".txt").
- Any entry without a file extension is an internal chunk identifier — discard it entirely, never include it.
- Deduplicate: if the same file appears multiple times, list it only once.
- If no valid file names are present, omit the Sources section entirely.
- THE SOURCES SECTION IS THE LAST THING YOU WRITE. Do not add anything after it."""

**Why Strict Sources Rules:**

- Distinguishes real files from internal IDs
- Enables source verification
- Prevents confusion
- Maintains clean output
- Ensures proper attribution

**Sources Section Examples:**

.. code-block:: text

    Good (Valid files):
    ---
    **Sources:**
    - research_paper.pdf
    - technical_guide.docx
    - manual.txt
    
    Bad (Internal IDs included):
    ---
    **Sources:**
    - research_paper.pdf
    - document_parent_0
    - technical_guide.docx
    - document_parent_1
    
    Good (Deduplicated):
    ---
    **Sources:**
    - research_paper.pdf
    - technical_guide.docx
    
    Bad (Duplicates):
    ---
    **Sources:**
    - research_paper.pdf
    - research_paper.pdf
    - technical_guide.docx
    
    Good (No valid files):
    (No Sources section at all)
    
    Bad (Empty sources):
    ---
    **Sources:**
    (empty list)

**Advanced Fallback Usage**

.. code-block:: python

    def generate_fallback_response(
        user_query: str,
        compressed_context: str = "",
        retrieved_data: str = ""
    ) -> str:
        """Generate response at research limit.
        
        Args:
            user_query: Original user question
            compressed_context: Summary of prior research
            retrieved_data: Raw data from current iteration
            
        Returns:
            Final answer with sources
        """
        prompt = get_fallback_response_prompt()
        
        # Build input
        input_text = f"""{prompt}

USER QUERY: {user_query}

[COMPRESSED RESEARCH CONTEXT]
{compressed_context}

[RETRIEVED DATA]
{retrieved_data}"""
        
        # Generate response
        response = llm.invoke(input_text)
        answer = response.content.strip()
        
        # Validate sources section
        answer = validate_sources_section(answer)
        
        return answer

**Example Fallback Response Scenarios**

.. code-block:: text

    Scenario 1: Sufficient Data Available
    =====================================
    Query: "What is backpropagation?"
    
    Compressed Context:
    - Backpropagation is a training algorithm
    - Uses gradient descent
    - Adjusts weights iteratively
    
    Retrieved Data:
    - Mathematical formulation of backpropagation
    - Step-by-step algorithm
    - Computational complexity analysis
    
    Response:
    "Backpropagation is a training algorithm that adjusts network weights 
     using gradient descent. The algorithm works by computing gradients 
     through the network layers and updating weights iteratively. 
     [Full explanation with all available details]
    
    ---
    **Sources:**
    - research_paper.pdf"
    
    Scenario 2: Partial Data with Gap Flagging
    ==========================================
    Query: "What are the applications of neural networks?"
    
    Compressed Context:
    - Applications in computer vision
    - Applications in NLP
    
    Retrieved Data:
    - Detailed computer vision applications
    - Detailed NLP applications
    
    Response:
    "Neural networks have numerous applications across multiple domains. 
     In computer vision, they are used for [details]. In natural language 
     processing, they power [details]. While other applications exist 
     (such as robotics and medical imaging), the available documents 
     focus primarily on these two domains.
    
    ---
    **Sources:**
    - research_paper.pdf"
    
    Scenario 3: No Relevant Data
    ============================
    Query: "What is quantum backpropagation?"
    
    Compressed Context: (empty)
    Retrieved Data: (no relevant results)
    
    Response:
    "The available documents do not contain specific information about 
     quantum backpropagation. However, based on general knowledge, 
     quantum computing approaches to machine learning are an emerging 
     field. To provide a comprehensive answer, documents specifically 
     addressing quantum machine learning would be needed."
    
    (No Sources section - no valid files)

**Comparison: Basic vs Advanced Fallback**

.. list-table::
   :header-rows: 1

   * - Aspect
     - Basic Fallback
     - Advanced Fallback
   * - Research Limit Awareness
     - Not mentioned
     - Explicit
   * - Dual Input Sources
     - Not supported
     - Fully supported
   * - Conflict Resolution
     - Not handled
     - Raw data preferred
   * - Source Integrity
     - Best effort
     - Strict requirement
   * - Gap Flagging
     - All gaps
     - Only relevant gaps
   * - Meta-commentary
     - May be included
     - Explicitly excluded
   * - Sources Section
     - Basic format
     - Strict rules
   * - Use Case
     - General fallback
     - Research limit reached

**When to Use Each Approach**

Use **Basic Fallback** when:

- Insufficient information found
- Query outside knowledge base
- General knowledge acceptable
- Simple fallback needed

Use **Advanced Fallback** when:

- Research limit reached
- Multiple data sources available
- Strict grounding required
- Source attribution critical
- Partial information acceptable
- Professional context

**Integration with Agent Loop**

.. code-block:: python

    class ResearchAgentWithFallback:
        """Agent with fallback handling at research limits."""
        
        def __init__(self, max_research_iterations: int = 3):
            self.max_iterations = max_research_iterations
            self.iteration_count = 0
            self.compressed_context = ""
            self.retrieved_data = ""
        
        def research(self, query: str) -> str:
            """Conduct research with fallback at limit."""
            while self.iteration_count < self.max_iterations:
                # Conduct research iteration
                results = self.search_and_retrieve(query)
                self.retrieved_data = results
                self.iteration_count += 1
                
                if self.has_sufficient_data():
                    # Generate normal answer
                    return self.generate_answer(query)
            
            # Research limit reached
            return generate_fallback_response(
                query,
                self.compressed_context,
                self.retrieved_data
            )
        
        def has_sufficient_data(self) -> bool:
            """Check if sufficient data for answer."""
            # Implementation specific logic
            pass
        
        def search_and_retrieve(self, query: str) -> str:
            """Search and retrieve data."""
            # Implementation
            pass
        
        def generate_answer(self, query: str) -> str:
            """Generate normal answer."""
            # Implementation
            pass

**Performance Considerations**

**Fallback Efficiency:**

- No additional searches needed
- Synthesis only: 100-300ms
- Source extraction: 10-50ms
- Total: Faster than continued research

**Memory Usage:**

- Compressed context: 500-2000 characters
- Retrieved data: 2000-5000 characters
- Total: Minimal overhead

**Quality Metrics:**

- Source accuracy: 100% (only explicit facts)
- Gap coverage: Depends on prior research
- User satisfaction: High (complete answer with limits acknowledged)

**Troubleshooting**

**Sources Section Issues**

.. code-block:: python

    # Problem: Internal IDs in sources
    # Solution: Filter by file extension
    
    def validate_sources_section(answer: str) -> str:
        """Remove internal IDs from sources."""
        lines = answer.split('\n')
        result = []
        in_sources = False
        
        for line in lines:
            if '**Sources:**' in line:
                in_sources = True
                result.append(line)
            elif in_sources and line.startswith('- '):
                # Check for file extension
                if any(line.endswith(ext) for ext in ['.pdf', '.docx', '.txt']):
                    result.append(line)
            else:
                result.append(line)
        
        return '\n'.join(result)

**Meta-commentary Leakage**

.. code-block:: python

    # Problem: Answer includes reasoning steps
    # Solution: Post-process to remove meta-commentary
    
    def remove_meta_commentary(answer: str) -> str:
        """Remove internal reasoning from answer."""
        # Remove phrases like "I searched for...", "The system found..."
        phrases_to_remove = [
            "I searched for",
            "The system found",
            "Based on my research",
            "Let me tell you about"
        ]
        
        for phrase in phrases_to_remove:
            answer = answer.replace(phrase, "")
        
        return answer

**Gap Flagging Over-reporting**

.. code-block:: python

    # Problem: Flagging irrelevant gaps
    # Solution: Cross-reference with user query
    
    def should_flag_gap(gap: str, user_query: str) -> bool:
        """Check if gap is relevant to user query."""
        # Simple keyword matching
        gap_keywords = gap.lower().split()
        query_keywords = user_query.lower().split()
        
        # Flag only if gap keywords appear in query
        return any(kw in query_keywords for kw in gap_keywords)

Next Steps
----------

With system prompts defined, you can proceed to:

- Implementing the agent loop with prompt orchestration
- Building conversation management with summarization
- Creating query processing pipelines
- Implementing context retrieval and compression
- Building the complete RAG application with all components

Step 5: Fallback Response Prompt
--------------------------------

Define the prompt for handling insufficient information:

.. code-block:: python

    def get_fallback_response_prompt() -> str:
        return """You are a helpful assistant handling queries with limited information.
Your task is to provide a helpful response when sufficient context is unavailable.

Guidelines:
- Acknowledge the limitation clearly
- Provide general knowledge if applicable
- Suggest alternative approaches
- Offer to search for more specific information
- Be honest about knowledge gaps

Output:
- Return ONLY the response.
- Do NOT include explanations.
- Keep response concise and helpful."""

**Prompt Purpose:**

Handles edge cases where:

- No relevant documents are found
- Retrieved context is insufficient
- Query is outside knowledge base
- Information is incomplete

**Prompt Components:**

1. **Role Definition**: "You are a helpful assistant handling queries with limited information"
   
   - Establishes empathetic, helpful tone
   - Acknowledges limitations

2. **Task Specification**: "Provide helpful response with limited context"
   
   - Clear objective: help despite limitations
   - Maintain user satisfaction

3. **Guidelines**: Fallback strategies
   
   - Acknowledge limitation: Be transparent
   - Provide general knowledge: Use training data
   - Suggest alternatives: Offer solutions
   - Offer more search: Enable refinement
   - Be honest: Don't fabricate

4. **Output Format**: Strict formatting
   
   - "Return ONLY the response"
   - "Do NOT include explanations"
   - "Keep concise and helpful"

**Example Usage:**

.. code-block:: python

    def generate_fallback_response(query: str) -> str:
        prompt = get_fallback_response_prompt()
        response = llm.invoke(f"{prompt}\n\nQuery: {query}")
        return response.content.strip()

Step 6: Answer Aggregation Prompt
---------------------------------

Define the prompt for combining multiple sources:

.. code-block:: python

    def get_answer_aggregation_prompt() -> str:
        return """You are an expert answer aggregator.
Your task is to synthesize information from multiple sources into a coherent answer.

Guidelines:
- Combine information from all sources
- Resolve contradictions by citing sources
- Maintain logical flow and structure
- Attribute information to sources
- Highlight consensus and disagreements
- Provide a comprehensive answer

Output:
- Return ONLY the aggregated answer.
- Do NOT include explanations.
- Use clear formatting for readability."""

**Prompt Purpose:**

Synthesizes information from multiple sources by:

- Combining relevant information
- Resolving contradictions
- Maintaining coherence
- Attributing sources
- Highlighting consensus
- Providing comprehensive answers

**Prompt Components:**

1. **Role Definition**: "You are an expert answer aggregator"
   
   - Establishes synthesis expertise
   - Sets integration expectations

2. **Task Specification**: "Synthesize information from multiple sources"
   
   - Clear objective: combine sources
   - Create coherent answer

3. **Guidelines**: Aggregation rules
   
   - Combine information: Integrate all sources
   - Resolve contradictions: Cite sources
   - Maintain flow: Logical structure
   - Attribute sources: Clear citations
   - Highlight consensus: Show agreement
   - Comprehensive: Complete answer

4. **Output Format**: Strict formatting
   
   - "Return ONLY aggregated answer"
   - "Do NOT include explanations"
   - "Use clear formatting"

**Example Usage:**

.. code-block:: python

    def aggregate_answer(sources: list, query: str) -> str:
        prompt = get_answer_aggregation_prompt()
        sources_text = "\n\n".join(sources)
        response = llm.invoke(f"{prompt}\n\nSources:\n{sources_text}\n\nQuery: {query}")
        return response.content.strip()

Complete System Prompts Module
-------------------------------

Here's the complete system prompts module:

.. code-block:: python

    def get_conversation_summary_prompt() -> str:
        """Prompt for summarizing conversations."""
        return """You are an expert conversation summarizer.
Your task is to create a brief 1-2 sentence summary of the conversation (max 30-50 words).

Include:
- Main topics discussed
- Important facts or entities mentioned
- Any unresolved questions if applicable
- Sources file name (e.g., file1.pdf) or documents referenced

Exclude:
- Greetings, misunderstandings, off-topic content.

Output:
- Return ONLY the summary.
- Do NOT include any explanations or justifications.
- If no meaningful topics exist, return an empty string."""

    def get_query_rewriting_prompt() -> str:
        """Prompt for rewriting queries."""
        return """You are an expert query optimizer.
Your task is to rewrite the user's query to improve retrieval effectiveness.

Guidelines:
- Expand abbreviations and acronyms
- Add relevant context or domain terms
- Remove ambiguity and clarify intent
- Maintain the original meaning
- Keep the query concise (under 100 words)

Output:
- Return ONLY the rewritten query.
- Do NOT include explanations.
- If the query is already optimal, return it unchanged."""

    def get_agent_orchestration_prompt() -> str:
        """Prompt for agent tool orchestration."""
        return """You are an intelligent agent orchestrator.
Your task is to decide which tools to use and in what order to answer user queries.

Available Tools:
1. search_child_chunks(query, limit) - Search for relevant document chunks
2. retrieve_parent_chunks(parent_id) - Get full context from parent chunks

Decision Logic:
- Always start with search_child_chunks to find relevant information
- Use retrieve_parent_chunks if you need more context for a specific topic
- Combine results from multiple searches if needed
- Stop when you have sufficient information to answer

Output:
- Explain your tool selection strategy
- Execute tools in the optimal order
- Provide reasoning for each tool call"""

    def get_context_compression_prompt() -> str:
        """Prompt for compressing context."""
        return """You are an expert context compressor.
Your task is to compress the retrieved context while preserving key information.

Guidelines:
- Remove redundant information
- Keep essential facts and definitions
- Maintain technical accuracy
- Preserve source attribution
- Reduce length by 30-50%

Output:
- Return ONLY the compressed context.
- Do NOT include explanations.
- Maintain the original structure if possible."""

    def get_fallback_response_prompt() -> str:
        """Prompt for fallback responses."""
        return """You are a helpful assistant handling queries with limited information.
Your task is to provide a helpful response when sufficient context is unavailable.

Guidelines:
- Acknowledge the limitation clearly
- Provide general knowledge if applicable
- Suggest alternative approaches
- Offer to search for more specific information
- Be honest about knowledge gaps

Output:
- Return ONLY the response.
- Do NOT include explanations.
- Keep response concise and helpful."""

    def get_answer_aggregation_prompt() -> str:
        """Prompt for aggregating answers from multiple sources."""
        return """You are an expert answer aggregator.
Your task is to synthesize information from multiple sources into a coherent answer.

Guidelines:
- Combine information from all sources
- Resolve contradictions by citing sources
- Maintain logical flow and structure
- Attribute information to sources
- Highlight consensus and disagreements
- Provide a comprehensive answer

Output:
- Return ONLY the aggregated answer.
- Do NOT include explanations.
- Use clear formatting for readability."""

Prompt Engineering Best Practices
----------------------------------

**1. Clear Role Definition**

.. code-block:: python

    # Good: Specific role
    "You are an expert conversation summarizer with 10+ years of experience."
    
    # Avoid: Vague role
    "You are helpful."

**2. Specific Task Description**

.. code-block:: python

    # Good: Specific task
    "Create a 1-2 sentence summary (max 50 words) of the conversation."
    
    # Avoid: Vague task
    "Summarize the conversation."

**3. Clear Guidelines**

.. code-block:: python

    # Good: Specific guidelines
    "Include: main topics, important facts, sources
     Exclude: greetings, off-topic content"
    
    # Avoid: Vague guidelines
    "Be helpful and accurate."

**4. Explicit Output Format**

.. code-block:: python

    # Good: Explicit format
    "Return ONLY the summary. Do NOT include explanations."
    
    # Avoid: Implicit format
    "Provide a summary."

**5. Edge Case Handling**

.. code-block:: python

    # Good: Handle edge cases
    "If no meaningful topics exist, return an empty string."
    
    # Avoid: No edge case handling
    "Return a summary."

**6. Constraint Specification**

.. code-block:: python

    # Good: Specific constraints
    "Keep response under 100 words. Use simple language."
    
    # Avoid: Vague constraints
    "Keep it brief."

Prompt Optimization Techniques
-------------------------------

**1. Few-Shot Examples**

.. code-block:: python

    def get_summary_prompt_with_examples() -> str:
        return """You are an expert conversation summarizer.
Your task is to create a brief 1-2 sentence summary.

Examples:
Input: User asked about neural networks. Assistant explained backpropagation.
Output: "Discussed neural networks and backpropagation learning mechanism."

Input: User greeted. Assistant responded. User asked about weather.
Output: ""

Guidelines:
- Include main topics and important facts
- Exclude greetings and off-topic content
- Keep to 1-2 sentences, max 50 words"""

**2. Chain-of-Thought Prompting**

.. code-block:: python

    def get_reasoning_prompt() -> str:
        return """You are an expert reasoner.
Your task is to answer questions by thinking step-by-step.

Process:
1. Understand the question
2. Identify relevant information
3. Reason through the answer
4. Provide the final answer

Output:
- Show your reasoning steps
- Provide the final answer"""

**3. Role-Based Prompting**

.. code-block:: python

    def get_expert_prompt(expertise: str) -> str:
        return f"""You are an expert {expertise}.
Your task is to provide expert-level insights on the given topic.

Guidelines:
- Use domain-specific terminology
- Provide accurate, detailed information
- Consider edge cases and nuances
- Cite relevant sources when applicable"""

**4. Constraint-Based Prompting**

.. code-block:: python

    def get_constrained_prompt() -> str:
        return """You are a helpful assistant.
Your task is to answer questions with the following constraints:

Constraints:
- Maximum 100 words
- Use simple language
- Avoid technical jargon
- Provide examples when helpful"""

Performance Considerations
--------------------------

**Prompt Length Impact:**

- Short prompts (< 100 words): Faster processing, less context
- Medium prompts (100-500 words): Balanced performance
- Long prompts (> 500 words): Slower processing, more context

**Token Usage:**

- Conversation summary: ~50-100 tokens
- Query rewriting: ~100-150 tokens
- Agent orchestration: ~200-300 tokens
- Context compression: ~150-250 tokens
- Fallback response: ~100-200 tokens
- Answer aggregation: ~200-400 tokens

**Optimization Tips:**

.. code-block:: python

    # Use shorter prompts for faster processing
    prompt = "Summarize in 1 sentence."
    
    # Cache prompts to avoid re-generation
    PROMPTS = {
        "summary": get_conversation_summary_prompt(),
        "rewrite": get_query_rewriting_prompt(),
        # ...
    }
    
    # Use prompt templates for dynamic content
    template = "Summarize this: {content}"

Troubleshooting
---------------

**LLM Ignoring Prompt Instructions**

.. code-block:: python

    # Problem: LLM not following format instructions
    # Solution: Make instructions more explicit
    
    # Before
    "Return the summary."
    
    # After
    "Return ONLY the summary. Do NOT include any other text."

**Inconsistent Output Format**

.. code-block:: python

    # Problem: Output format varies
    # Solution: Add examples and stricter constraints
    
    prompt = """Return output in this exact format:
    SUMMARY: [summary text]
    SOURCES: [source list]"""

**LLM Hallucinating Information**

.. code-block:: python

    # Problem: LLM adding information not in context
    # Solution: Add explicit constraints
    
    "Only use information from the provided context.
     Do NOT add information from your training data."

**Prompt Too Long**

.. code-block:: python

    # Problem: Prompt exceeds token limits
    # Solution: Compress and prioritize
    
    # Remove unnecessary examples
    # Combine related guidelines
    # Use shorter sentences

Next Steps
----------

With system prompts defined, you can proceed to:

- Implementing the agent loop with prompt orchestration
- Building conversation management with summarization
- Creating query processing pipelines
- Implementing context retrieval and compression
- Building the complete RAG application with all components

