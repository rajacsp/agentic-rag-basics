Topic 8: Agent Configuration
============================

Overview
--------

This chapter covers the configuration of agent execution parameters, including hard limits on tool calls and iterations, token counting for context management, and dynamic compression thresholds. Proper configuration prevents infinite loops, manages memory efficiently, and ensures predictable agent behavior.

Purpose and Architecture
------------------------

Agent configuration serves as the "control system" for the RAG agent, managing:

- **Execution Limits**: Maximum tool calls and iterations
- **Token Management**: Counting and tracking token usage
- **Context Compression**: Dynamic thresholds based on token growth
- **Resource Constraints**: Memory and latency management
- **Safety Guardrails**: Preventing runaway execution

The configuration architecture includes:

1. **Hard Limits**: Tool calls and iterations
2. **Token Thresholds**: Initial and dynamic
3. **Token Counting**: Estimating context size
4. **Compression Strategy**: Adaptive compression

Step 1: Import Required Libraries
----------------------------------

Begin by importing the necessary dependencies:

.. code-block:: python

    import tiktoken

**Import Explanations:**

- ``tiktoken``: OpenAI's token counting library for accurate token estimation

**Why This Import:**

- Provides accurate token counting for various models
- Enables precise context size estimation
- Supports multiple model encodings
- Essential for token-based compression decisions

Step 2: Define Hard Limits
---------------------------

Define maximum execution limits to prevent infinite loops:

.. code-block:: python

    MAX_TOOL_CALLS = 8       # Maximum tool calls per agent run
    MAX_ITERATIONS = 10      # Maximum agent loop iterations

**Step 2a: Maximum Tool Calls**

.. code-block:: python

    MAX_TOOL_CALLS = 8

**Purpose:**

- Limits total tool invocations per agent run
- Prevents excessive API calls
- Controls costs and latency
- Ensures timely responses

**Why 8 Calls:**

- Typical research needs 2-4 searches + 2-4 retrievals
- Provides buffer for complex queries
- Balances thoroughness with efficiency
- Prevents runaway tool usage

**Tool Call Scenarios:**

.. code-block:: text

    Scenario 1: Simple Query
    ========================
    Query: "What is backpropagation?"
    
    Tool Calls:
    1. search_child_chunks("backpropagation", limit=5)
    2. retrieve_parent_chunks("document_parent_0")
    3. retrieve_parent_chunks("document_parent_1")
    
    Total: 3 calls (within limit)
    
    Scenario 2: Complex Query
    ==========================
    Query: "Compare backpropagation with other training methods"
    
    Tool Calls:
    1. search_child_chunks("backpropagation", limit=5)
    2. search_child_chunks("training methods comparison", limit=5)
    3. retrieve_parent_chunks("document_parent_0")
    4. retrieve_parent_chunks("document_parent_1")
    5. retrieve_parent_chunks("document_parent_2")
    6. retrieve_parent_chunks("document_parent_3")
    
    Total: 6 calls (within limit)
    
    Scenario 3: Excessive Queries
    ==============================
    Query: "Tell me everything about neural networks"
    
    Tool Calls:
    1. search_child_chunks("neural networks", limit=5)
    2. search_child_chunks("neural network architecture", limit=5)
    3. search_child_chunks("neural network training", limit=5)
    4. search_child_chunks("neural network applications", limit=5)
    5. retrieve_parent_chunks("document_parent_0")
    6. retrieve_parent_chunks("document_parent_1")
    7. retrieve_parent_chunks("document_parent_2")
    8. retrieve_parent_chunks("document_parent_3")
    
    Total: 8 calls (at limit - stops here)

**Adjusting MAX_TOOL_CALLS:**

.. code-block:: text

    Conservative (4-5 calls):
    - Limited resources
    - Cost-sensitive applications
    - Real-time requirements
    
    Balanced (6-8 calls):
    - Standard RAG applications
    - Good balance of quality and efficiency
    - Recommended for most use cases
    
    Generous (10-15 calls):
    - Research-focused applications
    - Comprehensive answers needed
    - Cost not a primary concern

**Step 2b: Maximum Iterations**

.. code-block:: python

    MAX_ITERATIONS = 10

**Purpose:**

- Limits agent loop iterations
- Prevents infinite loops
- Controls execution time
- Ensures bounded behavior

**Why 10 Iterations:**

- Typical research completes in 2-5 iterations
- Provides buffer for complex research
- Prevents runaway loops
- Reasonable timeout (usually < 30 seconds)

**Iteration Scenarios:**

.. code-block:: text

    Scenario 1: Quick Answer
    ========================
    Query: "What is backpropagation?"
    
    Iterations:
    1. Analyze query
    2. Search for information
    3. Retrieve parent chunks
    4. Generate answer
    
    Total: 4 iterations (within limit)
    
    Scenario 2: Multi-Question Research
    ====================================
    Query: "What is backpropagation and how is it used?"
    
    Iterations:
    1. Analyze query (split into 2 questions)
    2. Process question 1: Search
    3. Process question 1: Retrieve
    4. Process question 1: Generate answer
    5. Process question 2: Search
    6. Process question 2: Retrieve
    7. Process question 2: Generate answer
    8. Aggregate answers
    
    Total: 8 iterations (within limit)
    
    Scenario 3: Iterative Refinement
    =================================
    Query: "Complex research question"
    
    Iterations:
    1. Analyze query
    2. Search iteration 1
    3. Retrieve iteration 1
    4. Evaluate results
    5. Search iteration 2 (refined)
    6. Retrieve iteration 2
    7. Evaluate results
    8. Search iteration 3 (further refined)
    9. Retrieve iteration 3
    10. Generate final answer
    
    Total: 10 iterations (at limit - stops here)

**Adjusting MAX_ITERATIONS:**

.. code-block:: text

    Conservative (3-5 iterations):
    - Real-time applications
    - Latency-sensitive
    - Simple queries only
    
    Balanced (8-10 iterations):
    - Standard RAG applications
    - Good balance of quality and speed
    - Recommended for most use cases
    
    Generous (15-20 iterations):
    - Research-focused applications
    - Comprehensive answers needed
    - Batch processing acceptable

Step 3: Define Token Thresholds
--------------------------------

Define token-based compression thresholds:

.. code-block:: python

    BASE_TOKEN_THRESHOLD = 2000     # Initial token threshold for compression
    TOKEN_GROWTH_FACTOR = 0.9       # Multiplier applied after each compression

**Step 3a: Base Token Threshold**

.. code-block:: python

    BASE_TOKEN_THRESHOLD = 2000

**Purpose:**

- Initial token limit before compression
- Triggers compression when exceeded
- Balances context size and quality
- Prevents context bloat

**Why 2000 Tokens:**

- Approximately 1500-2000 words
- Sufficient for most queries
- Leaves room for LLM processing
- Typical context window: 4000-8000 tokens

**Token Threshold Scenarios:**

.. code-block:: text

    Scenario 1: Small Context
    =========================
    Context tokens: 800
    Threshold: 2000
    
    Action: No compression needed
    
    Scenario 2: Medium Context
    ===========================
    Context tokens: 1800
    Threshold: 2000
    
    Action: No compression needed (just under)
    
    Scenario 3: Large Context
    ==========================
    Context tokens: 2500
    Threshold: 2000
    
    Action: Compress context
    New threshold: 2000 * 0.9 = 1800
    
    Scenario 4: Very Large Context
    ===============================
    Context tokens: 3500
    Threshold: 1800 (after compression)
    
    Action: Compress again
    New threshold: 1800 * 0.9 = 1620

**Adjusting BASE_TOKEN_THRESHOLD:**

.. code-block:: text

    Conservative (1000-1500 tokens):
    - Limited context window
    - Aggressive compression
    - Smaller models
    
    Balanced (2000-3000 tokens):
    - Standard context window
    - Moderate compression
    - Recommended for most cases
    
    Generous (4000-5000 tokens):
    - Large context window
    - Minimal compression
    - Large models (GPT-4, Claude)

**Step 3b: Token Growth Factor**

.. code-block:: python

    TOKEN_GROWTH_FACTOR = 0.9

**Purpose:**

- Multiplier for dynamic threshold adjustment
- Reduces threshold after each compression
- Prevents context re-expansion
- Enables progressive compression

**How It Works:**

1. Initial threshold: 2000 tokens
2. After compression: 2000 × 0.9 = 1800 tokens
3. After next compression: 1800 × 0.9 = 1620 tokens
4. Continues until reaching minimum

**Token Growth Factor Examples:**

.. code-block:: text

    Factor 0.9 (10% reduction):
    ===========================
    Iteration 1: 2000 tokens
    Iteration 2: 1800 tokens (2000 × 0.9)
    Iteration 3: 1620 tokens (1800 × 0.9)
    Iteration 4: 1458 tokens (1620 × 0.9)
    Iteration 5: 1312 tokens (1458 × 0.9)
    
    Factor 0.8 (20% reduction):
    ===========================
    Iteration 1: 2000 tokens
    Iteration 2: 1600 tokens (2000 × 0.8)
    Iteration 3: 1280 tokens (1600 × 0.8)
    Iteration 4: 1024 tokens (1280 × 0.8)
    
    Factor 0.95 (5% reduction):
    ===========================
    Iteration 1: 2000 tokens
    Iteration 2: 1900 tokens (2000 × 0.95)
    Iteration 3: 1805 tokens (1900 × 0.95)
    Iteration 4: 1715 tokens (1805 × 0.95)

**Adjusting TOKEN_GROWTH_FACTOR:**

.. code-block:: text

    Aggressive (0.7-0.8):
    - Rapid compression
    - Minimal context retention
    - Limited context window
    
    Balanced (0.85-0.95):
    - Moderate compression
    - Good context retention
    - Recommended for most cases
    
    Conservative (0.95-1.0):
    - Minimal compression
    - Maximum context retention
    - Large context window

Step 4: Token Counting Function
--------------------------------

Define the function for estimating context tokens:

.. code-block:: python

    def estimate_context_tokens(messages: list) -> int:
        """Estimate total tokens in message list.
        
        Args:
            messages: List of message objects with content
            
        Returns:
            Estimated token count
        """
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return sum(
            len(encoding.encode(str(msg.content))) 
            for msg in messages 
            if hasattr(msg, 'content') and msg.content
        )

**Step 4a: Model-Specific Encoding**

.. code-block:: python

    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")

**Purpose:**

- Gets appropriate tokenizer for model
- Falls back to standard encoding if unavailable
- Ensures accurate token counting

**Why This Matters:**

- Different models use different tokenizers
- GPT-4 uses cl100k_base encoding
- Fallback ensures robustness
- Prevents crashes on unavailable models

**Supported Models:**

.. code-block:: text

    GPT-4 models:
    - gpt-4
    - gpt-4-turbo
    - gpt-4-32k
    
    GPT-3.5 models:
    - gpt-3.5-turbo
    - text-davinci-003
    - text-davinci-002
    
    Fallback:
    - cl100k_base (most common)
    - p50k_base (older models)
    - r50k_base (legacy)

**Step 4b: Token Encoding**

.. code-block:: python

    return sum(
        len(encoding.encode(str(msg.content))) 
        for msg in messages 
        if hasattr(msg, 'content') and msg.content
    )

**Purpose:**

- Encodes each message to tokens
- Sums total token count
- Handles missing content gracefully

**How It Works:**

1. Iterate through messages
2. Check if message has content attribute
3. Convert content to string
4. Encode to tokens
5. Count tokens
6. Sum all counts

**Example Usage:**

.. code-block:: python

    from langchain_core.messages import HumanMessage, AIMessage
    
    messages = [
        HumanMessage(content="What is backpropagation?"),
        AIMessage(content="Backpropagation is a training algorithm..."),
        HumanMessage(content="How does it work?"),
        AIMessage(content="It works by computing gradients...")
    ]
    
    tokens = estimate_context_tokens(messages)
    # Output: ~150 tokens

**Token Counting Examples:**

.. code-block:: text

    Short message:
    "What is backpropagation?"
    Tokens: ~5
    
    Medium message:
    "Backpropagation is a training algorithm that adjusts network weights 
     using gradient descent. It computes gradients by propagating errors 
     backwards through the network."
    Tokens: ~40
    
    Long message:
    "Neural networks consist of interconnected nodes organized in layers. 
     During training, the network learns by adjusting the connections between 
     nodes through a process called backpropagation. This process calculates 
     how much each connection contributed to errors and adjusts them accordingly. 
     For example, in a network trained on image recognition, early layers might 
     learn to detect edges, while later layers learn to recognize shapes."
    Tokens: ~120

Complete Agent Configuration Module
------------------------------------

Here's the complete agent configuration code:

.. code-block:: python

    import tiktoken

    # Hard limits
    MAX_TOOL_CALLS = 8              # Maximum tool calls per agent run
    MAX_ITERATIONS = 10             # Maximum agent loop iterations

    # Token thresholds
    BASE_TOKEN_THRESHOLD = 2000     # Initial token threshold for compression
    TOKEN_GROWTH_FACTOR = 0.9       # Multiplier applied after each compression

    def estimate_context_tokens(messages: list) -> int:
        """Estimate total tokens in message list.
        
        Uses tiktoken to accurately count tokens based on model encoding.
        Falls back to cl100k_base if model-specific encoding unavailable.
        
        Args:
            messages: List of message objects with content attribute
            
        Returns:
            Estimated token count for all messages
            
        Example:
            >>> messages = [HumanMessage(content="Hello")]
            >>> tokens = estimate_context_tokens(messages)
            >>> print(tokens)
            2
        """
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
        except:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return sum(
            len(encoding.encode(str(msg.content))) 
            for msg in messages 
            if hasattr(msg, 'content') and msg.content
        )

    # Configuration validation
    def validate_configuration():
        """Validate configuration parameters."""
        assert MAX_TOOL_CALLS > 0, "MAX_TOOL_CALLS must be positive"
        assert MAX_ITERATIONS > 0, "MAX_ITERATIONS must be positive"
        assert BASE_TOKEN_THRESHOLD > 0, "BASE_TOKEN_THRESHOLD must be positive"
        assert 0 < TOKEN_GROWTH_FACTOR <= 1, "TOKEN_GROWTH_FACTOR must be between 0 and 1"

Configuration Usage Patterns
-----------------------------

**Pattern 1: Checking Tool Call Limit**

.. code-block:: python

    def should_continue_tool_calls(tool_call_count: int) -> bool:
        """Check if more tool calls are allowed."""
        return tool_call_count < MAX_TOOL_CALLS

**Pattern 2: Checking Iteration Limit**

.. code-block:: python

    def should_continue_iterations(iteration_count: int) -> bool:
        """Check if more iterations are allowed."""
        return iteration_count < MAX_ITERATIONS

**Pattern 3: Dynamic Threshold Calculation**

.. code-block:: python

    def calculate_dynamic_threshold(compression_count: int) -> int:
        """Calculate dynamic token threshold after compressions."""
        threshold = BASE_TOKEN_THRESHOLD
        for _ in range(compression_count):
            threshold = int(threshold * TOKEN_GROWTH_FACTOR)
        return threshold

**Pattern 4: Compression Decision**

.. code-block:: python

    def should_compress_context(
        messages: list,
        compression_count: int
    ) -> bool:
        """Decide if context should be compressed."""
        current_tokens = estimate_context_tokens(messages)
        threshold = calculate_dynamic_threshold(compression_count)
        return current_tokens > threshold

**Pattern 5: Complete Agent Loop**

.. code-block:: python

    def agent_loop(query: str, messages: list) -> str:
        """Main agent loop with limits and compression."""
        tool_call_count = 0
        iteration_count = 0
        compression_count = 0
        
        while should_continue_iterations(iteration_count):
            iteration_count += 1
            
            # Check if compression needed
            if should_compress_context(messages, compression_count):
                messages = compress_messages(messages)
                compression_count += 1
            
            # Execute agent step
            tool_calls = execute_agent_step(query, messages)
            
            # Check tool call limit
            if tool_call_count + len(tool_calls) > MAX_TOOL_CALLS:
                # Limit tool calls
                tool_calls = tool_calls[:MAX_TOOL_CALLS - tool_call_count]
            
            tool_call_count += len(tool_calls)
            
            # Execute tools
            for tool_call in tool_calls:
                result = execute_tool(tool_call)
                messages.append(result)
            
            # Check if done
            if is_done(messages):
                break
        
        return generate_final_answer(messages)

Configuration Monitoring
------------------------

**Pattern 1: Log Configuration**

.. code-block:: python

    def log_configuration():
        """Log current configuration."""
        print(f"MAX_TOOL_CALLS: {MAX_TOOL_CALLS}")
        print(f"MAX_ITERATIONS: {MAX_ITERATIONS}")
        print(f"BASE_TOKEN_THRESHOLD: {BASE_TOKEN_THRESHOLD}")
        print(f"TOKEN_GROWTH_FACTOR: {TOKEN_GROWTH_FACTOR}")

**Pattern 2: Monitor Execution**

.. code-block:: python

    def monitor_execution(
        tool_call_count: int,
        iteration_count: int,
        token_count: int
    ):
        """Monitor execution against limits."""
        tool_call_pct = (tool_call_count / MAX_TOOL_CALLS) * 100
        iteration_pct = (iteration_count / MAX_ITERATIONS) * 100
        
        print(f"Tool calls: {tool_call_count}/{MAX_TOOL_CALLS} ({tool_call_pct:.1f}%)")
        print(f"Iterations: {iteration_count}/{MAX_ITERATIONS} ({iteration_pct:.1f}%)")
        print(f"Tokens: {token_count}")

**Pattern 3: Alert on Limits**

.. code-block:: python

    def check_limits_alert(
        tool_call_count: int,
        iteration_count: int
    ):
        """Alert if approaching limits."""
        if tool_call_count >= MAX_TOOL_CALLS * 0.8:
            print("WARNING: Approaching tool call limit")
        
        if iteration_count >= MAX_ITERATIONS * 0.8:
            print("WARNING: Approaching iteration limit")

Best Practices
--------------

**1. Configuration Validation**

.. code-block:: python

    # Good: Validate on startup
    validate_configuration()
    
    # Avoid: No validation
    # (may fail silently later)

**2. Reasonable Defaults**

.. code-block:: python

    # Good: Balanced defaults
    MAX_TOOL_CALLS = 8
    MAX_ITERATIONS = 10
    BASE_TOKEN_THRESHOLD = 2000
    
    # Avoid: Extreme values
    MAX_TOOL_CALLS = 100  # Too high
    MAX_ITERATIONS = 1    # Too low

**3. Environment-Specific Configuration**

.. code-block:: python

    import os
    
    # Good: Environment-based configuration
    MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "8"))
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
    
    # Avoid: Hardcoded values
    MAX_TOOL_CALLS = 8

**4. Monitoring and Logging**

.. code-block:: python

    import logging
    
    logger = logging.getLogger(__name__)
    
    # Good: Log configuration and execution
    logger.info(f"Agent configured: {MAX_TOOL_CALLS} tool calls, {MAX_ITERATIONS} iterations")
    logger.info(f"Tool calls used: {tool_call_count}/{MAX_TOOL_CALLS}")
    
    # Avoid: No logging
    # (hard to debug issues)

**5. Graceful Degradation**

.. code-block:: python

    # Good: Handle limit gracefully
    if tool_call_count >= MAX_TOOL_CALLS:
        logger.warning("Tool call limit reached, generating answer with available context")
        return generate_answer(messages)
    
    # Avoid: Crash on limit
    assert tool_call_count < MAX_TOOL_CALLS

Performance Considerations
--------------------------

**Token Counting Performance:**

- Encoding initialization: 100-500ms (first call)
- Token counting: 1-5ms per message
- Total for 10 messages: 10-50ms

**Optimization Tips:**

.. code-block:: python

    # Cache encoding
    _encoding = None
    
    def get_encoding():
        global _encoding
        if _encoding is None:
            try:
                _encoding = tiktoken.encoding_for_model("gpt-4")
            except:
                _encoding = tiktoken.get_encoding("cl100k_base")
        return _encoding
    
    def estimate_context_tokens_cached(messages: list) -> int:
        """Faster token counting with cached encoding."""
        encoding = get_encoding()
        return sum(
            len(encoding.encode(str(msg.content))) 
            for msg in messages 
            if hasattr(msg, 'content') and msg.content
        )

**Memory Considerations:**

- Configuration: ~100 bytes
- Encoding object: ~1-5MB
- Token cache: ~1-10MB (if caching)

Troubleshooting
---------------

**Tool Call Limit Reached**

.. code-block:: python

    # Problem: Agent stops before finding answer
    # Solution: Increase MAX_TOOL_CALLS
    
    MAX_TOOL_CALLS = 12  # Increased from 8

**Iteration Limit Reached**

.. code-block:: python

    # Problem: Agent stops mid-research
    # Solution: Increase MAX_ITERATIONS
    
    MAX_ITERATIONS = 15  # Increased from 10

**Token Counting Errors**

.. code-block:: python

    # Problem: tiktoken not installed
    # Solution: Install tiktoken
    
    # pip install tiktoken

**Compression Too Aggressive**

.. code-block:: python

    # Problem: Context compressed too much
    # Solution: Increase BASE_TOKEN_THRESHOLD or TOKEN_GROWTH_FACTOR
    
    BASE_TOKEN_THRESHOLD = 3000  # Increased from 2000
    TOKEN_GROWTH_FACTOR = 0.95   # Increased from 0.9

**Compression Not Happening**

.. code-block:: python

    # Problem: Context never compressed
    # Solution: Lower BASE_TOKEN_THRESHOLD
    
    BASE_TOKEN_THRESHOLD = 1500  # Decreased from 2000

Next Steps
----------

With agent configuration defined, you can proceed to:

- Building the agent graph with configured limits
- Implementing node functions with limit checks
- Creating the complete agent workflow
- Integrating with LangGraph for execution
- Building the complete RAG application with monitoring

