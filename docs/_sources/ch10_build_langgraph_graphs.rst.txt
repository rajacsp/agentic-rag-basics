Topic 10: Build the LangGraph Graphs
====================================

Overview
--------

This chapter covers the assembly of the complete LangGraph workflow, combining the agent subgraph for individual question research with the main graph for conversation orchestration. The result is a sophisticated multi-agent architecture that handles conversation memory, query analysis, parallel research, and answer synthesis.

Purpose and Architecture
------------------------

The complete LangGraph system serves as the "execution engine" for the entire RAG pipeline, managing:

- **Agent Subgraph**: Processes individual questions with tool execution and context compression
- **Main Graph**: Orchestrates conversation flow, query rewriting, and answer aggregation
- **Checkpointing**: Maintains conversation state across interactions
- **Interrupts**: Enables human-in-the-loop clarification
- **Parallel Execution**: Spawns independent agents for multiple questions

The graph architecture includes:

1. **Agent Subgraph**: Question-level research and tool execution
2. **Main Graph**: Conversation-level orchestration
3. **Checkpointer**: State persistence and recovery
4. **Interrupt Points**: User interaction opportunities

Step 1: Import Required Libraries
----------------------------------

Begin by importing the necessary dependencies:

.. code-block:: python

    from langgraph.graph import START, END, StateGraph
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import InMemorySaver

**Import Explanations:**

- ``langgraph.graph.START, END``: Special nodes marking graph entry and exit
- ``langgraph.graph.StateGraph``: Graph builder for state-based workflows
- ``langgraph.prebuilt.ToolNode``: Pre-built node for tool execution
- ``langgraph.checkpoint.memory.InMemorySaver``: In-memory state checkpointing

**Why These Imports:**

- **StateGraph**: Enables state-based graph construction
- **ToolNode**: Simplifies tool execution without custom code
- **InMemorySaver**: Enables conversation persistence
- **START/END**: Marks graph boundaries

Step 2: Create Checkpointer
---------------------------

Define the state persistence mechanism:

.. code-block:: python

    checkpointer = InMemorySaver()

**Purpose:**

- Saves conversation state between interactions
- Enables recovery from interrupts
- Maintains conversation history
- Supports multi-turn conversations

**Why In-Memory:**

- Fast access
- Suitable for single-session applications
- No external dependencies
- Sufficient for most use cases

**Alternative Checkpointers:**

.. code-block:: python

    # PostgreSQL checkpointer
    from langgraph.checkpoint.postgres import PostgresSaver
    checkpointer = PostgresSaver(connection_string="postgresql://...")
    
    # SQLite checkpointer
    from langgraph.checkpoint.sqlite import SqliteSaver
    checkpointer = SqliteSaver(db_path="./checkpoints.db")

Step 3: Build Agent Subgraph
-----------------------------

Assemble the agent subgraph for individual question research:

.. code-block:: python

    agent_builder = StateGraph(AgentState)
    
    # Add nodes
    agent_builder.add_node("orchestrator", orchestrator)
    agent_builder.add_node("tools", ToolNode([search_child_chunks, retrieve_parent_chunks]))
    agent_builder.add_node("compress_context", compress_context)
    agent_builder.add_node("fallback_response", fallback_response)
    agent_builder.add_node("should_compress_context", should_compress_context)
    agent_builder.add_node("collect_answer", collect_answer)
    
    # Add edges
    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator_call,
        {
            "tools": "tools",
            "fallback_response": "fallback_response",
            "collect_answer": "collect_answer"
        }
    )
    agent_builder.add_edge("tools", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", END)
    
    # Compile agent subgraph
    agent_subgraph = agent_builder.compile()

**Step 3a: Create StateGraph**

.. code-block:: python

    agent_builder = StateGraph(AgentState)

**Purpose:**

- Creates graph builder for AgentState
- Defines state schema for agent execution
- Enables node and edge addition

**Step 3b: Add Nodes**

.. code-block:: python

    agent_builder.add_node("orchestrator", orchestrator)
    agent_builder.add_node("tools", ToolNode([search_child_chunks, retrieve_parent_chunks]))
    agent_builder.add_node("compress_context", compress_context)
    agent_builder.add_node("fallback_response", fallback_response)
    agent_builder.add_node("should_compress_context", should_compress_context)
    agent_builder.add_node("collect_answer", collect_answer)

**Purpose:**

- Registers all processing nodes
- ToolNode automatically executes tools
- Each node transforms state

**Why ToolNode:**

- Simplifies tool execution
- Handles tool result formatting
- Manages error handling
- Reduces boilerplate code

**Step 3c: Add Edges**

.. code-block:: python

    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator_call,
        {
            "tools": "tools",
            "fallback_response": "fallback_response",
            "collect_answer": "collect_answer"
        }
    )
    agent_builder.add_edge("tools", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", END)

**Purpose:**

- Defines graph flow
- START edge: Entry point
- Conditional edges: Routing logic
- Regular edges: Deterministic flow
- END edge: Exit point

**Step 3d: Compile Agent Subgraph**

.. code-block:: python

    agent_subgraph = agent_builder.compile()

**Purpose:**

- Finalizes graph structure
- Enables execution
- Validates graph connectivity
- Returns executable graph

**Agent Subgraph Structure**

.. code-block:: text

    START
      ↓
    orchestrator
      ↙    ↓    ↘
    tools  fallback  collect_answer
      ↓      ↓         ↓
    compress  collect_answer
      ↓         ↓
    orchestrator  END
      ↓
    (loop)

Step 4: Build Main Graph
------------------------

Assemble the main graph for conversation orchestration:

.. code-block:: python

    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("summarize_history", summarize_history)
    graph_builder.add_node("rewrite_query", rewrite_query)
    graph_builder.add_node("request_clarification", request_clarification)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("aggregate_answers", aggregate_answers)
    
    # Add edges
    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification", "rewrite_query")
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)
    
    # Compile main graph with checkpointer and interrupt
    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"]
    )

**Step 4a: Create StateGraph**

.. code-block:: python

    graph_builder = StateGraph(State)

**Purpose:**

- Creates graph builder for main State
- Defines state schema for conversation
- Enables node and edge addition

**Step 4b: Add Nodes**

.. code-block:: python

    graph_builder.add_node("summarize_history", summarize_history)
    graph_builder.add_node("rewrite_query", rewrite_query)
    graph_builder.add_node("request_clarification", request_clarification)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("aggregate_answers", aggregate_answers)

**Purpose:**

- Registers conversation-level nodes
- "agent" node is the compiled subgraph
- Each node processes conversation state

**Why Subgraph as Node:**

- Encapsulates agent logic
- Enables parallel execution
- Maintains separation of concerns
- Simplifies main graph

**Step 4c: Add Edges**

.. code-block:: python

    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification", "rewrite_query")
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)

**Purpose:**

- Defines conversation flow
- START edge: Entry point
- Conditional edges: Routing based on query clarity
- Multi-source edge: Aggregates from all agents
- END edge: Exit point

**Multi-Source Edge:**

.. code-block:: python

    graph_builder.add_edge(["agent"], "aggregate_answers")

This edge handles multiple agent instances (from Send commands) converging to aggregation.

**Step 4d: Compile Main Graph**

.. code-block:: python

    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"]
    )

**Purpose:**

- Finalizes graph structure
- Enables execution
- Adds checkpointing for state persistence
- Adds interrupt point for user interaction

**Checkpointer Parameter:**

- Saves state before each node
- Enables recovery from interrupts
- Supports multi-turn conversations

**Interrupt Before Parameter:**

- Pauses execution before specified node
- Allows user input
- Enables human-in-the-loop

**Main Graph Structure**

.. code-block:: text

    START
      ↓
    summarize_history
      ↓
    rewrite_query
      ↙        ↘
    request_clarification  [agent] (parallel for each question)
      ↓              ↓
    rewrite_query  aggregate_answers
      ↓              ↓
    (loop)          END

Complete Graph Assembly Code
-----------------------------

Here's the complete graph assembly code:

.. code-block:: python

    from langgraph.graph import START, END, StateGraph
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import InMemorySaver

    # Create checkpointer
    checkpointer = InMemorySaver()

    # Build agent subgraph
    agent_builder = StateGraph(AgentState)
    
    agent_builder.add_node("orchestrator", orchestrator)
    agent_builder.add_node("tools", ToolNode([search_child_chunks, retrieve_parent_chunks]))
    agent_builder.add_node("compress_context", compress_context)
    agent_builder.add_node("fallback_response", fallback_response)
    agent_builder.add_node("should_compress_context", should_compress_context)
    agent_builder.add_node("collect_answer", collect_answer)
    
    agent_builder.add_edge(START, "orchestrator")
    agent_builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator_call,
        {
            "tools": "tools",
            "fallback_response": "fallback_response",
            "collect_answer": "collect_answer"
        }
    )
    agent_builder.add_edge("tools", "should_compress_context")
    agent_builder.add_edge("compress_context", "orchestrator")
    agent_builder.add_edge("fallback_response", "collect_answer")
    agent_builder.add_edge("collect_answer", END)
    
    agent_subgraph = agent_builder.compile()

    # Build main graph
    graph_builder = StateGraph(State)
    
    graph_builder.add_node("summarize_history", summarize_history)
    graph_builder.add_node("rewrite_query", rewrite_query)
    graph_builder.add_node("request_clarification", request_clarification)
    graph_builder.add_node("agent", agent_subgraph)
    graph_builder.add_node("aggregate_answers", aggregate_answers)
    
    graph_builder.add_edge(START, "summarize_history")
    graph_builder.add_edge("summarize_history", "rewrite_query")
    graph_builder.add_conditional_edges("rewrite_query", route_after_rewrite)
    graph_builder.add_edge("request_clarification", "rewrite_query")
    graph_builder.add_edge(["agent"], "aggregate_answers")
    graph_builder.add_edge("aggregate_answers", END)
    
    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"]
    )

Graph Execution
---------------

**Basic Execution:**

.. code-block:: python

    # Single query
    result = agent_graph.invoke({
        "messages": [HumanMessage(content="What is backpropagation?")],
        "questionIsClear": False,
        "conversation_summary": "",
        "originalQuery": "",
        "rewrittenQuestions": [],
        "agent_answers": []
    })
    
    print(result["messages"][-1].content)

**Multi-Turn Conversation:**

.. code-block:: python

    # First query
    config = {"configurable": {"thread_id": "user_123"}}
    
    result1 = agent_graph.invoke({
        "messages": [HumanMessage(content="What is backpropagation?")],
        "questionIsClear": False,
        "conversation_summary": "",
        "originalQuery": "",
        "rewrittenQuestions": [],
        "agent_answers": []
    }, config=config)
    
    # Second query (uses conversation history)
    result2 = agent_graph.invoke({
        "messages": [HumanMessage(content="How is it used?")],
        "questionIsClear": False,
        "conversation_summary": "",
        "originalQuery": "",
        "rewrittenQuestions": [],
        "agent_answers": []
    }, config=config)

**With Interrupts:**

.. code-block:: python

    config = {"configurable": {"thread_id": "user_123"}}
    
    # First execution (may interrupt)
    result = agent_graph.invoke({
        "messages": [HumanMessage(content="unclear query")],
        ...
    }, config=config)
    
    # Check if interrupted
    if result.get("interrupted"):
        # Get clarification from user
        clarification = input("Please clarify: ")
        
        # Resume execution
        result = agent_graph.invoke({
            "messages": [HumanMessage(content=clarification)],
            ...
        }, config=config)

Graph Architecture Explained
----------------------------

**Agent Subgraph Flow:**

.. code-block:: text

    Purpose: Process individual questions with tool execution and context compression
    
    START
      ↓
    orchestrator (invoke LLM with tools)
      ↓
    route_after_orchestrator_call
      ↙        ↓         ↘
    tools  fallback  collect_answer
      ↓      ↓         ↓
    should_compress  collect_answer
      ↙        ↘
    compress  orchestrator
      ↓         ↓
    orchestrator (loop)
      ↓
    collect_answer
      ↓
    END

**Main Graph Flow:**

.. code-block:: text

    Purpose: Orchestrate complete conversation workflow
    
    START
      ↓
    summarize_history (extract conversation context)
      ↓
    rewrite_query (analyze clarity, reformulate)
      ↙        ↘
    request_clarification  [agent] (parallel for each question)
      ↓              ↓
    rewrite_query  aggregate_answers (synthesize results)
      ↓              ↓
    (loop)          END

**Why This Architecture:**

.. code-block:: text

    1. Separation of Concerns
       - Agent subgraph: Question-level research
       - Main graph: Conversation-level orchestration
       - Clear boundaries and responsibilities
    
    2. Parallel Execution
       - Multiple agents run simultaneously
       - Each processes one question independently
       - Results converge at aggregation
    
    3. State Management
       - Checkpointing preserves conversation state
       - Enables multi-turn interactions
       - Supports recovery from interrupts
    
    4. Human-in-the-Loop
       - Interrupt before clarification request
       - User provides input
       - Execution resumes with clarification
    
    5. Context Compression
       - Agent subgraph compresses context
       - Prevents token bloat
       - Tracks executed operations
    
    6. Graceful Degradation
       - Fallback response on limit breach
       - Always returns useful answer
       - Respects resource constraints

Best Practices
--------------

**1. Graph Validation**

.. code-block:: python

    # Good: Validate graph structure
    try:
        agent_graph = graph_builder.compile()
        print("Graph compiled successfully")
    except Exception as e:
        print(f"Graph compilation error: {e}")
    
    # Avoid: No validation
    agent_graph = graph_builder.compile()

**2. Checkpointer Configuration**

.. code-block:: python

    # Good: Use appropriate checkpointer
    from langgraph.checkpoint.memory import InMemorySaver
    checkpointer = InMemorySaver()
    
    # For production
    from langgraph.checkpoint.postgres import PostgresSaver
    checkpointer = PostgresSaver(connection_string="...")

**3. Interrupt Points**

.. code-block:: python

    # Good: Strategic interrupt points
    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"]
    )
    
    # Avoid: Too many interrupts
    interrupt_before=["node1", "node2", "node3", ...]

**4. Thread IDs**

.. code-block:: python

    # Good: Use thread IDs for multi-user
    config = {"configurable": {"thread_id": "user_123"}}
    result = agent_graph.invoke(input_state, config=config)
    
    # Avoid: No thread ID (shared state)
    result = agent_graph.invoke(input_state)

**5. Error Handling**

.. code-block:: python

    # Good: Handle execution errors
    try:
        result = agent_graph.invoke(input_state, config=config)
    except Exception as e:
        print(f"Execution error: {e}")
        # Handle error gracefully
    
    # Avoid: No error handling
    result = agent_graph.invoke(input_state, config=config)

Performance Considerations
--------------------------

**Graph Compilation Time:**

- Agent subgraph: 10-50ms
- Main graph: 10-50ms
- Total: 20-100ms

**Execution Time:**

- Simple query: 2-5 seconds
- Complex query: 5-15 seconds
- Multi-question: 10-30 seconds

**Memory Usage:**

- Graph structure: ~1-5MB
- Checkpointer: ~100KB per conversation
- Total: ~2-10MB per active conversation

**Optimization Tips:**

.. code-block:: python

    # Cache compiled graphs
    agent_graph = graph_builder.compile(checkpointer=checkpointer)
    
    # Reuse for multiple queries
    for query in queries:
        result = agent_graph.invoke(input_state, config=config)
    
    # Use thread IDs for multi-user
    config = {"configurable": {"thread_id": user_id}}

Troubleshooting
---------------

**Graph Compilation Errors**

.. code-block:: python

    # Problem: "Node not found" error
    # Solution: Ensure all referenced nodes are added
    
    # Check all nodes are added
    print(graph_builder.nodes)
    
    # Check all edges reference existing nodes
    print(graph_builder.edges)

**Routing Errors**

.. code-block:: python

    # Problem: Conditional edge routing fails
    # Solution: Verify routing function returns valid keys
    
    def route_function(state):
        # Must return one of the routing keys
        if condition:
            return "node1"
        else:
            return "node2"

**Interrupt Not Working**

.. code-block:: python

    # Problem: Execution doesn't interrupt
    # Solution: Ensure interrupt_before specifies correct node
    
    agent_graph = graph_builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["request_clarification"]  # Must exist
    )

**State Not Persisting**

.. code-block:: python

    # Problem: State lost between invocations
    # Solution: Use checkpointer and thread ID
    
    config = {"configurable": {"thread_id": "user_123"}}
    result = agent_graph.invoke(input_state, config=config)

**Parallel Execution Not Working**

.. code-block:: python

    # Problem: Agents not running in parallel
    # Solution: Use Send commands in routing
    
    def route_after_rewrite(state):
        return [
            Send("agent", {"question": q, "question_index": i})
            for i, q in enumerate(state["rewrittenQuestions"])
        ]

Next Steps
----------

With the complete LangGraph graphs built, you can proceed to:

- Creating the application interface (CLI, API, web UI)
- Implementing monitoring and logging
- Adding performance optimization
- Deploying to production
- Building the complete RAG application

