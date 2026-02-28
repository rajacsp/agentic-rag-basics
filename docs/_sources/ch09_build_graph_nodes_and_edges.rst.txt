Topic 9: Build Graph Node and Edge Functions
==============================================

Overview
--------

This chapter covers the creation of processing nodes and routing edges that form the LangGraph workflow. Nodes represent discrete processing steps (summarization, query rewriting, agent execution, answer aggregation), while edges define the flow between them. Together, they create a stateful, directed graph that orchestrates the entire RAG pipeline.

Purpose and Architecture
------------------------

Graph nodes and edges serve as the "execution engine" of the RAG system, managing:

- **Processing Nodes**: Discrete steps that transform state
- **Routing Edges**: Conditional logic that determines flow
- **State Transitions**: Updates to conversation and agent state
- **Message Management**: Adding, removing, and transforming messages
- **Workflow Orchestration**: Coordinating multi-step processes

The graph architecture includes:

1. **Summarize History Node**: Compresses conversation history
2. **Rewrite Query Node**: Analyzes and reformulates queries
3. **Request Clarification Node**: Handles unclear queries
4. **Route After Rewrite Edge**: Conditional routing logic
5. **Aggregate Answers Node**: Synthesizes multiple answers

Step 1: Import Required Libraries
----------------------------------

Begin by importing the necessary dependencies:

.. code-block:: python

    from langgraph.types import Send, Command
    from langchain_core.messages import (
        HumanMessage, AIMessage, SystemMessage, 
        RemoveMessage, ToolMessage
    )
    from typing import Literal

**Import Explanations:**

- ``langgraph.types.Send``: Sends state to another node
- ``langgraph.types.Command``: Controls graph execution
- ``langchain_core.messages``: Message types for conversation
- ``typing.Literal``: Type hints for routing decisions

**Why These Imports:**

- **Send**: Enables dynamic node routing
- **Message types**: Represent different message roles
- **RemoveMessage**: Cleans up message history
- **Literal**: Type-safe routing decisions

Step 2: Summarize History Node
-------------------------------

Define the node that compresses conversation history:

.. code-block:: python

    def summarize_history(state: State):
        """Summarize conversation history for context preservation.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with conversation summary
        """
        if len(state["messages"]) < 4:
            return {"conversation_summary": ""}
        
        relevant_msgs = [
            msg for msg in state["messages"][:-1]
            if isinstance(msg, (HumanMessage, AIMessage)) 
            and not getattr(msg, "tool_calls", None)
        ]
        
        if not relevant_msgs:
            return {"conversation_summary": ""}
        
        conversation = "Conversation history:\n"
        for msg in relevant_msgs[-6:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            conversation += f"{role}: {msg.content}\n"
        
        summary_response = llm.with_config(temperature=0.2).invoke([
            SystemMessage(content=get_conversation_summary_prompt()),
            HumanMessage(content=conversation)
        ])
        
        return {
            "conversation_summary": summary_response.content,
            "agent_answers": [{"__reset__": True}]
        }

**Step 2a: Minimum Message Check**

.. code-block:: python

    if len(state["messages"]) < 4:
        return {"conversation_summary": ""}

**Purpose:**

- Skips summarization for short conversations
- Avoids unnecessary processing
- Preserves all context for short histories

**Why 4 Messages:**

- Minimum for meaningful summary
- Typically 2 exchanges (user + assistant × 2)
- Below this, full history is already concise

**Step 2b: Filter Relevant Messages**

.. code-block:: python

    relevant_msgs = [
        msg for msg in state["messages"][:-1]
        if isinstance(msg, (HumanMessage, AIMessage)) 
        and not getattr(msg, "tool_calls", None)
    ]

**Purpose:**

- Excludes tool messages and system messages
- Removes last message (current query)
- Focuses on conversation history only

**Why This Filtering:**

- Tool messages are implementation details
- System messages are instructions, not conversation
- Last message is current query, not history
- Keeps summary focused on prior context

**Step 2c: Build Conversation String**

.. code-block:: python

    conversation = "Conversation history:\n"
    for msg in relevant_msgs[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        conversation += f"{role}: {msg.content}\n"

**Purpose:**

- Formats messages for LLM consumption
- Limits to last 6 messages (3 exchanges)
- Creates readable conversation format

**Why Last 6 Messages:**

- Provides sufficient context
- Prevents excessive token usage
- Balances completeness and efficiency
- Typical conversation window

**Step 2d: Generate Summary**

.. code-block:: python

    summary_response = llm.with_config(temperature=0.2).invoke([
        SystemMessage(content=get_conversation_summary_prompt()),
        HumanMessage(content=conversation)
    ])

**Purpose:**

- Uses LLM to create concise summary
- Low temperature (0.2) for consistency
- Applies conversation summary prompt

**Why Low Temperature:**

- Ensures consistent, deterministic summaries
- Reduces variability
- Focuses on factual content

**Step 2e: Reset Agent Answers**

.. code-block:: python

    return {
        "conversation_summary": summary_response.content,
        "agent_answers": [{"__reset__": True}]
    }

**Purpose:**

- Resets accumulated answers for new query
- Clears prior research results
- Prepares for new agent execution

**Summarize History Usage Example**

.. code-block:: python

    # Conversation with 8 messages
    state = State(
        messages=[
            HumanMessage(content="What is backpropagation?"),
            AIMessage(content="Backpropagation is..."),
            HumanMessage(content="How does it work?"),
            AIMessage(content="It works by..."),
            HumanMessage(content="What are applications?"),
            AIMessage(content="Applications include..."),
            HumanMessage(content="Tell me more"),
            AIMessage(content="More details...")
        ],
        conversation_summary="",
        agent_answers=[]
    )
    
    result = summarize_history(state)
    # Output:
    # {
    #     "conversation_summary": "Discussed backpropagation, how it works, and applications",
    #     "agent_answers": [{"__reset__": True}]
    # }

Step 3: Rewrite Query Node
---------------------------

Define the node that analyzes and reformulates queries:

.. code-block:: python

    def rewrite_query(state: State):
        """Analyze query clarity and rewrite for optimal retrieval.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated state with query analysis results
        """
        last_message = state["messages"][-1]
        conversation_summary = state.get("conversation_summary", "")
        
        context_section = (
            f"Conversation Context:\n{conversation_summary}\n" 
            if conversation_summary.strip() 
            else ""
        ) + f"User Query:\n{last_message.content}\n"
        
        llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
        response = llm_with_structure.invoke([
            SystemMessage(content=get_rewrite_query_prompt()),
            HumanMessage(content=context_section)
        ])
        
        if response.questions and response.is_clear:
            delete_all = [
                RemoveMessage(id=m.id) 
                for m in state["messages"] 
                if not isinstance(m, SystemMessage)
            ]
            return {
                "questionIsClear": True,
                "messages": delete_all,
                "originalQuery": last_message.content,
                "rewrittenQuestions": response.questions
            }
        
        clarification = (
            response.clarification_needed 
            if response.clarification_needed and len(response.clarification_needed.strip()) > 10 
            else "I need more information to understand your question."
        )
        return {
            "questionIsClear": False,
            "messages": [AIMessage(content=clarification)]
        }

**Step 3a: Extract Query and Context**

.. code-block:: python

    last_message = state["messages"][-1]
    conversation_summary = state.get("conversation_summary", "")

**Purpose:**

- Gets current user query
- Retrieves prior conversation context
- Prepares input for analysis

**Step 3b: Build Context Section**

.. code-block:: python

    context_section = (
        f"Conversation Context:\n{conversation_summary}\n" 
        if conversation_summary.strip() 
        else ""
    ) + f"User Query:\n{last_message.content}\n"

**Purpose:**

- Combines context and query
- Includes context only if available
- Formats for LLM consumption

**Step 3c: Structured Output**

.. code-block:: python

    llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke([
        SystemMessage(content=get_rewrite_query_prompt()),
        HumanMessage(content=context_section)
    ])

**Purpose:**

- Uses structured output for QueryAnalysis
- Low temperature (0.1) for consistency
- Ensures valid response format

**Why Structured Output:**

- Guarantees response format
- Enables type checking
- Prevents parsing errors
- Ensures is_clear and questions fields

**Step 3d: Handle Clear Questions**

.. code-block:: python

    if response.questions and response.is_clear:
        delete_all = [
            RemoveMessage(id=m.id) 
            for m in state["messages"] 
            if not isinstance(m, SystemMessage)
        ]
        return {
            "questionIsClear": True,
            "messages": delete_all,
            "originalQuery": last_message.content,
            "rewrittenQuestions": response.questions
        }

**Purpose:**

- Clears message history for clean agent start
- Stores original and rewritten queries
- Sets clear flag for routing

**Why Clear Messages:**

- Prevents tool messages from prior attempts
- Starts agent with clean slate
- Reduces context noise
- Improves agent focus

**Step 3e: Handle Unclear Questions**

.. code-block:: python

    clarification = (
        response.clarification_needed 
        if response.clarification_needed and len(response.clarification_needed.strip()) > 10 
        else "I need more information to understand your question."
    )
    return {
        "questionIsClear": False,
        "messages": [AIMessage(content=clarification)]
    }

**Purpose:**

- Returns clarification request to user
- Uses LLM-generated clarification if available
- Falls back to default message

**Why Length Check:**

- Ensures meaningful clarification
- Prevents empty or too-short messages
- Validates LLM output

**Rewrite Query Usage Example**

.. code-block:: python

    # Clear query
    state = State(
        messages=[HumanMessage(content="How do neural networks work?")],
        conversation_summary=""
    )
    
    result = rewrite_query(state)
    # Output:
    # {
    #     "questionIsClear": True,
    #     "messages": [],  # Cleared
    #     "originalQuery": "How do neural networks work?",
    #     "rewrittenQuestions": ["How do neural networks work?"]
    # }
    
    # Unclear query
    state = State(
        messages=[HumanMessage(content="xyzabc qwerty")],
        conversation_summary=""
    )
    
    result = rewrite_query(state)
    # Output:
    # {
    #     "questionIsClear": False,
    #     "messages": [AIMessage(content="I need more information...")]
    # }

Step 4: Request Clarification Node
-----------------------------------

Define the node that handles unclear queries:

.. code-block:: python

    def request_clarification(state: State):
        """Handle unclear query by requesting clarification.
        
        Args:
            state: Current conversation state
            
        Returns:
            Empty dict (state already updated with clarification message)
        """
        return {}

**Purpose:**

- Placeholder node for clarification workflow
- Allows user to provide clearer query
- Enables loop back to rewrite_query

**Why Separate Node:**

- Enables graph routing
- Allows user interaction
- Supports multi-turn clarification
- Maintains clean graph structure

Step 5: Route After Rewrite Edge
---------------------------------

Define the routing logic after query rewriting:

.. code-block:: python

    def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
        """Route to clarification or agent based on query clarity.
        
        Args:
            state: Current conversation state
            
        Returns:
            Routing decision or list of Send commands
        """
        if not state.get("questionIsClear", False):
            return "request_clarification"
        else:
            return [
                Send("agent", {
                    "question": query,
                    "question_index": idx,
                    "messages": []
                })
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]

**Step 5a: Clarity Check**

.. code-block:: python

    if not state.get("questionIsClear", False):
        return "request_clarification"

**Purpose:**

- Routes to clarification if query unclear
- Prevents agent execution on bad queries
- Enables user feedback loop

**Step 5b: Dynamic Agent Routing**

.. code-block:: python

    else:
        return [
            Send("agent", {
                "question": query,
                "question_index": idx,
                "messages": []
            })
            for idx, query in enumerate(state["rewrittenQuestions"])
        ]

**Purpose:**

- Creates Send command for each rewritten question
- Enables parallel agent execution
- Passes question index for tracking

**Why Send Commands:**

- Enables dynamic routing
- Supports parallel execution
- Passes state to agent nodes
- Maintains question ordering

**Route After Rewrite Usage Example**

.. code-block:: python

    # Clear query with 2 questions
    state = State(
        questionIsClear=True,
        rewrittenQuestions=[
            "How do neural networks work?",
            "What are their applications?"
        ]
    )
    
    result = route_after_rewrite(state)
    # Output: [
    #     Send("agent", {"question": "How do neural networks work?", "question_index": 0, "messages": []}),
    #     Send("agent", {"question": "What are their applications?", "question_index": 1, "messages": []})
    # ]
    
    # Unclear query
    state = State(
        questionIsClear=False
    )
    
    result = route_after_rewrite(state)
    # Output: "request_clarification"

Step 6: Aggregate Answers Node
-------------------------------

Define the node that synthesizes multiple answers:

.. code-block:: python

    def aggregate_answers(state: State):
        """Aggregate multiple agent answers into single response.
        
        Args:
            state: Current conversation state with agent answers
            
        Returns:
            Updated state with aggregated answer
        """
        if not state.get("agent_answers"):
            return {"messages": [AIMessage(content="No answers were generated.")]}
        
        sorted_answers = sorted(
            state["agent_answers"],
            key=lambda x: x["index"]
        )
        
        formatted_answers = ""
        for i, ans in enumerate(sorted_answers, start=1):
            formatted_answers += (
                f"\nAnswer {i}:\n"
                f"{ans['answer']}\n"
            )
        
        user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}
Retrieved answers:{formatted_answers}""")
        
        synthesis_response = llm.invoke([
            SystemMessage(content=get_aggregation_prompt()),
            user_message
        ])
        
        return {"messages": [AIMessage(content=synthesis_response.content)]}

**Step 6a: Check for Answers**

.. code-block:: python

    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers were generated.")]}

**Purpose:**

- Handles case where no answers produced
- Returns graceful error message
- Prevents crashes on empty results

**Step 6b: Sort Answers**

.. code-block:: python

    sorted_answers = sorted(
        state["agent_answers"],
        key=lambda x: x["index"]
    )

**Purpose:**

- Orders answers by question index
- Maintains question sequence
- Ensures consistent output order

**Step 6c: Format Answers**

.. code-block:: python

    formatted_answers = ""
    for i, ans in enumerate(sorted_answers, start=1):
        formatted_answers += (
            f"\nAnswer {i}:\n"
            f"{ans['answer']}\n"
        )

**Purpose:**

- Creates readable answer format
- Numbers answers sequentially
- Prepares for LLM aggregation

**Step 6d: Synthesize Answers**

.. code-block:: python

    user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}
Retrieved answers:{formatted_answers}""")
    
    synthesis_response = llm.invoke([
        SystemMessage(content=get_aggregation_prompt()),
        user_message
    ])

**Purpose:**

- Uses LLM to synthesize answers
- Applies aggregation prompt
- Creates coherent final response

**Aggregate Answers Usage Example**

.. code-block:: python

    # Multiple answers
    state = State(
        originalQuery="What is backpropagation and its applications?",
        agent_answers=[
            {
                "index": 0,
                "answer": "Backpropagation is a training algorithm..."
            },
            {
                "index": 1,
                "answer": "Applications include computer vision..."
            }
        ]
    )
    
    result = aggregate_answers(state)
    # Output:
    # {
    #     "messages": [AIMessage(content="Synthesized answer combining both...")]
    # }

Complete Graph Nodes and Edges Module
--------------------------------------

Here's the complete nodes and edges code:

.. code-block:: python

    from langgraph.types import Send, Command
    from langchain_core.messages import (
        HumanMessage, AIMessage, SystemMessage,
        RemoveMessage, ToolMessage
    )
    from typing import Literal

    def summarize_history(state: State):
        """Summarize conversation history."""
        if len(state["messages"]) < 4:
            return {"conversation_summary": ""}
        
        relevant_msgs = [
            msg for msg in state["messages"][:-1]
            if isinstance(msg, (HumanMessage, AIMessage))
            and not getattr(msg, "tool_calls", None)
        ]
        
        if not relevant_msgs:
            return {"conversation_summary": ""}
        
        conversation = "Conversation history:\n"
        for msg in relevant_msgs[-6:]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            conversation += f"{role}: {msg.content}\n"
        
        summary_response = llm.with_config(temperature=0.2).invoke([
            SystemMessage(content=get_conversation_summary_prompt()),
            HumanMessage(content=conversation)
        ])
        
        return {
            "conversation_summary": summary_response.content,
            "agent_answers": [{"__reset__": True}]
        }

    def rewrite_query(state: State):
        """Analyze and rewrite query."""
        last_message = state["messages"][-1]
        conversation_summary = state.get("conversation_summary", "")
        
        context_section = (
            f"Conversation Context:\n{conversation_summary}\n"
            if conversation_summary.strip()
            else ""
        ) + f"User Query:\n{last_message.content}\n"
        
        llm_with_structure = llm.with_config(temperature=0.1).with_structured_output(QueryAnalysis)
        response = llm_with_structure.invoke([
            SystemMessage(content=get_rewrite_query_prompt()),
            HumanMessage(content=context_section)
        ])
        
        if response.questions and response.is_clear:
            delete_all = [
                RemoveMessage(id=m.id)
                for m in state["messages"]
                if not isinstance(m, SystemMessage)
            ]
            return {
                "questionIsClear": True,
                "messages": delete_all,
                "originalQuery": last_message.content,
                "rewrittenQuestions": response.questions
            }
        
        clarification = (
            response.clarification_needed
            if response.clarification_needed and len(response.clarification_needed.strip()) > 10
            else "I need more information to understand your question."
        )
        return {
            "questionIsClear": False,
            "messages": [AIMessage(content=clarification)]
        }

    def request_clarification(state: State):
        """Request clarification for unclear query."""
        return {}

    def route_after_rewrite(state: State) -> Literal["request_clarification", "agent"]:
        """Route based on query clarity."""
        if not state.get("questionIsClear", False):
            return "request_clarification"
        else:
            return [
                Send("agent", {
                    "question": query,
                    "question_index": idx,
                    "messages": []
                })
                for idx, query in enumerate(state["rewrittenQuestions"])
            ]

    def aggregate_answers(state: State):
        """Aggregate multiple answers."""
        if not state.get("agent_answers"):
            return {"messages": [AIMessage(content="No answers were generated.")]}
        
        sorted_answers = sorted(
            state["agent_answers"],
            key=lambda x: x["index"]
        )
        
        formatted_answers = ""
        for i, ans in enumerate(sorted_answers, start=1):
            formatted_answers += (
                f"\nAnswer {i}:\n"
                f"{ans['answer']}\n"
            )
        
        user_message = HumanMessage(content=f"""Original user question: {state["originalQuery"]}
Retrieved answers:{formatted_answers}""")
        
        synthesis_response = llm.invoke([
            SystemMessage(content=get_aggregation_prompt()),
            user_message
        ])
        
        return {"messages": [AIMessage(content=synthesis_response.content)]}

Graph Structure
---------------

**Typical Graph Flow:**

.. code-block:: text

    User Input
        ↓
    summarize_history
        ↓
    rewrite_query
        ↓
    route_after_rewrite
        ↙           ↘
    request_clarification    agent (parallel for each question)
        ↓                           ↓
    (loop back)              aggregate_answers
                                    ↓
                            Final Response

**Graph Construction Example:**

.. code-block:: python

    from langgraph.graph import StateGraph, END

    # Create graph
    graph = StateGraph(State)

    # Add nodes
    graph.add_node("summarize_history", summarize_history)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("request_clarification", request_clarification)
    graph.add_node("agent", agent_node)  # Defined separately
    graph.add_node("aggregate_answers", aggregate_answers)

    # Add edges
    graph.add_edge("summarize_history", "rewrite_query")
    graph.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "request_clarification": "request_clarification",
            "agent": "agent"
        }
    )
    graph.add_edge("request_clarification", "rewrite_query")
    graph.add_edge("agent", "aggregate_answers")
    graph.add_edge("aggregate_answers", END)

    # Compile graph
    compiled_graph = graph.compile()

Best Practices
--------------

**1. Message Type Checking**

.. code-block:: python

    # Good: Check message type
    if isinstance(msg, HumanMessage):
        # Handle user message
        pass
    
    # Avoid: String comparison
    if msg.type == "human":
        # Fragile and error-prone
        pass

**2. Safe Dictionary Access**

.. code-block:: python

    # Good: Use get with default
    summary = state.get("conversation_summary", "")
    
    # Avoid: Direct access
    summary = state["conversation_summary"]  # May raise KeyError

**3. Structured Output**

.. code-block:: python

    # Good: Use structured output
    llm_with_structure = llm.with_structured_output(QueryAnalysis)
    response = llm_with_structure.invoke(...)
    
    # Avoid: Parse string output
    response_str = llm.invoke(...)
    # Manual parsing is error-prone

**4. Temperature Configuration**

.. code-block:: python

    # Good: Low temperature for consistency
    llm.with_config(temperature=0.1).invoke(...)
    
    # Avoid: Default temperature
    llm.invoke(...)  # May be too high for deterministic tasks

**5. Error Handling**

.. code-block:: python

    # Good: Handle empty results
    if not state.get("agent_answers"):
        return {"messages": [AIMessage(content="No answers...")]}
    
    # Avoid: Assume data exists
    answers = state["agent_answers"]  # May crash if empty

Performance Considerations
--------------------------

**Node Execution Time:**

- summarize_history: 500-1000ms (LLM call)
- rewrite_query: 300-600ms (LLM call + structured output)
- request_clarification: 1-10ms (no LLM)
- aggregate_answers: 500-1000ms (LLM call)

**Message Management:**

- RemoveMessage: 1-5ms per message
- Message filtering: 1-2ms per message
- Total for 10 messages: 10-50ms

**Optimization Tips:**

.. code-block:: python

    # Cache LLM configurations
    summarizer_llm = llm.with_config(temperature=0.2)
    rewriter_llm = llm.with_config(temperature=0.1)
    
    # Batch message operations
    delete_all = [RemoveMessage(id=m.id) for m in messages]
    
    # Limit message history
    relevant_msgs = relevant_msgs[-6:]  # Keep last 6

Troubleshooting
---------------

**Messages Not Clearing**

.. code-block:: python

    # Problem: Messages not removed
    # Solution: Ensure RemoveMessage is returned
    
    delete_all = [RemoveMessage(id=m.id) for m in state["messages"]]
    return {"messages": delete_all}

**Routing Not Working**

.. code-block:: python

    # Problem: Route not following conditional
    # Solution: Verify return type matches routing keys
    
    # Routing definition
    graph.add_conditional_edges(
        "rewrite_query",
        route_after_rewrite,
        {
            "request_clarification": "request_clarification",
            "agent": "agent"
        }
    )
    
    # Function must return one of these keys
    def route_after_rewrite(state):
        if condition:
            return "request_clarification"
        else:
            return "agent"

**Structured Output Errors**

.. code-block:: python

    # Problem: Structured output validation fails
    # Solution: Ensure response matches QueryAnalysis schema
    
    class QueryAnalysis(BaseModel):
        is_clear: bool
        questions: List[str]
        clarification_needed: str
    
    # LLM must return valid JSON matching schema

**Send Commands Not Working**

.. code-block:: python

    # Problem: Send commands not routing
    # Solution: Ensure agent node exists and accepts state
    
    # Define agent node
    graph.add_node("agent", agent_node)
    
    # Send command must match node name
    Send("agent", {"question": query, ...})

Step 7: Agent Subgraph Nodes and Edges
---------------------------------------

Define the nodes and edges for the agent subgraph that handles individual question research:

.. code-block:: python

    from langgraph.types import Command
    from langchain_core.messages import RemoveMessage
    from typing import Set

    def orchestrator(state: AgentState):
        """Orchestrate agent research with tool invocation.
        
        Args:
            state: Agent execution state
            
        Returns:
            Updated state with LLM response and tool calls
        """
        context_summary = state.get("context_summary", "").strip()
        sys_msg = SystemMessage(content=get_orchestrator_prompt())
        
        summary_injection = (
            [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
            if context_summary else []
        )
        
        if not state.get("messages"):
            human_msg = HumanMessage(content=state["question"])
            force_search = HumanMessage(
                content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP TO ANSWER THIS QUESTION."
            )
            response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])
            return {
                "messages": [human_msg, response],
                "tool_call_count": len(response.tool_calls or []),
                "iteration_count": 1
            }
        
        response = llm_with_tools.invoke([sys_msg] + summary_injection + state["messages"])
        tool_calls = response.tool_calls if hasattr(response, "tool_calls") else []
        return {
            "messages": [response],
            "tool_call_count": len(tool_calls) if tool_calls else 0,
            "iteration_count": 1
        }

**Step 7a: Orchestrator Purpose**

.. code-block:: text

    Purpose:
    - Invokes LLM with tools for research
    - Injects compressed context from prior iterations
    - Forces initial search for new questions
    - Tracks tool calls and iterations

**Step 7b: Context Injection**

.. code-block:: python

    context_summary = state.get("context_summary", "").strip()
    summary_injection = (
        [HumanMessage(content=f"[COMPRESSED CONTEXT FROM PRIOR RESEARCH]\n\n{context_summary}")]
        if context_summary else []
    )

**Purpose:**

- Provides prior research context to LLM
- Enables iterative refinement
- Prevents redundant searches
- Maintains research continuity

**Step 7c: Force Initial Search**

.. code-block:: python

    if not state.get("messages"):
        human_msg = HumanMessage(content=state["question"])
        force_search = HumanMessage(
            content="YOU MUST CALL 'search_child_chunks' AS THE FIRST STEP..."
        )
        response = llm_with_tools.invoke([sys_msg] + summary_injection + [human_msg, force_search])

**Purpose:**

- Ensures first step is always search
- Prevents LLM from skipping retrieval
- Guarantees document access
- Enforces research discipline

**Step 7d: Routing Decision Node**

.. code-block:: python

    def route_after_orchestrator_call(state: AgentState) -> Literal["tool", "fallback_response", "collect_answer"]:
        """Route based on tool calls and limits.
        
        Args:
            state: Agent execution state
            
        Returns:
            Routing decision: "tool", "fallback_response", or "collect_answer"
        """
        iteration = state.get("iteration_count", 0)
        tool_count = state.get("tool_call_count", 0)
        
        if iteration >= MAX_ITERATIONS or tool_count > MAX_TOOL_CALLS:
            return "fallback_response"
        
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None) or []
        
        if not tool_calls:
            return "collect_answer"
        
        return "tool"

**Purpose:**

- Checks execution limits
- Determines next step
- Routes to tool execution or answer collection
- Handles limit breaches

**Routing Logic:**

.. code-block:: text

    1. Check if limits exceeded
       → Yes: Route to fallback_response
       → No: Continue
    
    2. Check if LLM called tools
       → Yes: Route to tool execution
       → No: Route to collect_answer

**Step 7e: Fallback Response Node**

.. code-block:: python

    def fallback_response(state: AgentState):
        """Generate response when research limit reached.
        
        Args:
            state: Agent execution state
            
        Returns:
            Updated state with fallback response
        """
        seen = set()
        unique_contents = []
        for m in state["messages"]:
            if isinstance(m, ToolMessage) and m.content not in seen:
                unique_contents.append(m.content)
                seen.add(m.content)
        
        context_summary = state.get("context_summary", "").strip()
        
        context_parts = []
        if context_summary:
            context_parts.append(
                f"## Compressed Research Context (from prior iterations)\n\n{context_summary}"
            )
        if unique_contents:
            context_parts.append(
                "## Retrieved Data (current iteration)\n\n" +
                "\n\n".join(
                    f"--- DATA SOURCE {i} ---\n{content}"
                    for i, content in enumerate(unique_contents, 1)
                )
            )
        
        context_text = (
            "\n\n".join(context_parts)
            if context_parts
            else "No data was retrieved from the documents."
        )
        
        prompt_content = (
            f"USER QUERY: {state.get('question')}\n\n"
            f"{context_text}\n\n"
            f"INSTRUCTION:\nProvide the best possible answer using only the data above."
        )
        
        response = llm.invoke([
            SystemMessage(content=get_fallback_response_prompt()),
            HumanMessage(content=prompt_content)
        ])
        
        return {"messages": [response]}

**Purpose:**

- Generates answer when limits reached
- Uses all available context
- Deduplicates tool results
- Ensures graceful degradation

**Step 7f: Context Compression Decision**

.. code-block:: python

    def should_compress_context(state: AgentState) -> Command[Literal["compress_context", "orchestrator"]]:
        """Decide if context should be compressed.
        
        Args:
            state: Agent execution state
            
        Returns:
            Command to compress or continue
        """
        messages = state["messages"]
        
        new_ids: Set[str] = set()
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    if tc["name"] == "retrieve_parent_chunks":
                        raw = tc["args"].get("parent_id") or tc["args"].get("id") or tc["args"].get("ids") or []
                        if isinstance(raw, str):
                            new_ids.add(f"parent::{raw}")
                        else:
                            new_ids.update(f"parent::{r}" for r in raw)
                    
                    elif tc["name"] == "search_child_chunks":
                        query = tc["args"].get("query", "")
                        if query:
                            new_ids.add(f"search::{query}")
                break
        
        updated_ids = state.get("retrieval_keys", set()) | new_ids
        
        current_token_messages = estimate_context_tokens(messages)
        current_token_summary = estimate_context_tokens(
            [HumanMessage(content=state.get("context_summary", ""))]
        )
        current_tokens = current_token_messages + current_token_summary
        
        max_allowed = BASE_TOKEN_THRESHOLD + int(current_token_summary * TOKEN_GROWTH_FACTOR)
        
        goto = "compress_context" if current_tokens > max_allowed else "orchestrator"
        return Command(update={"retrieval_keys": updated_ids}, goto=goto)

**Purpose:**

- Tracks retrieved documents
- Estimates token usage
- Decides if compression needed
- Routes to compression or next iteration

**Token Calculation:**

.. code-block:: text

    current_tokens = messages_tokens + summary_tokens
    max_allowed = BASE_TOKEN_THRESHOLD + (summary_tokens × TOKEN_GROWTH_FACTOR)
    
    if current_tokens > max_allowed:
        compress_context()
    else:
        continue_orchestrator()

**Step 7g: Compress Context Node**

.. code-block:: python

    def compress_context(state: AgentState):
        """Compress research context to reduce token usage.
        
        Args:
            state: Agent execution state
            
        Returns:
            Updated state with compressed context
        """
        messages = state["messages"]
        existing_summary = state.get("context_summary", "").strip()
        
        if not messages:
            return {}
        
        conversation_text = f"USER QUESTION:\n{state.get('question')}\n\nConversation to compress:\n\n"
        if existing_summary:
            conversation_text += f"[PRIOR COMPRESSED CONTEXT]\n{existing_summary}\n\n"
        
        for msg in messages[1:]:
            if isinstance(msg, AIMessage):
                tool_calls_info = ""
                if getattr(msg, "tool_calls", None):
                    calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in msg.tool_calls)
                    tool_calls_info = f" | Tool calls: {calls}"
                conversation_text += f"[ASSISTANT{tool_calls_info}]\n{msg.content or '(tool call only)'}\n\n"
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "tool")
                conversation_text += f"[TOOL RESULT — {tool_name}]\n{msg.content}\n\n"
        
        summary_response = llm.invoke([
            SystemMessage(content=get_context_compression_prompt()),
            HumanMessage(content=conversation_text)
        ])
        new_summary = summary_response.content
        
        retrieved_ids: Set[str] = state.get("retrieval_keys", set())
        if retrieved_ids:
            parent_ids = sorted(r for r in retrieved_ids if r.startswith("parent::"))
            search_queries = sorted(
                r.replace("search::", "") for r in retrieved_ids if r.startswith("search::")
            )
            
            block = "\n\n---\n**Already executed (do NOT repeat):**\n"
            if parent_ids:
                block += "Parent chunks retrieved:\n" + "\n".join(
                    f"- {p.replace('parent::', '')}" for p in parent_ids
                ) + "\n"
            if search_queries:
                block += "Search queries already run:\n" + "\n".join(f"- {q}" for q in search_queries) + "\n"
            new_summary += block
        
        return {
            "context_summary": new_summary,
            "messages": [RemoveMessage(id=m.id) for m in messages[1:]]
        }

**Purpose:**

- Compresses research context
- Tracks executed searches and retrievals
- Prevents redundant operations
- Reduces token usage

**Compression Strategy:**

1. Build conversation text from messages
2. Include prior compressed context
3. Compress with LLM
4. Add "already executed" section
5. Clear message history

**Step 7h: Collect Answer Node**

.. code-block:: python

    def collect_answer(state: AgentState):
        """Collect final answer from agent research.
        
        Args:
            state: Agent execution state
            
        Returns:
            Updated state with final answer
        """
        last_message = state["messages"][-1]
        is_valid = (
            isinstance(last_message, AIMessage) and
            last_message.content and
            not last_message.tool_calls
        )
        answer = last_message.content if is_valid else "Unable to generate an answer."
        
        return {
            "final_answer": answer,
            "agent_answers": [{
                "index": state["question_index"],
                "question": state["question"],
                "answer": answer
            }]
        }

**Purpose:**

- Extracts final answer from LLM
- Validates answer quality
- Formats for aggregation
- Tracks question index

**Agent Subgraph Flow**

.. code-block:: text

    orchestrator
        ↓
    route_after_orchestrator_call
        ↙        ↓         ↘
    tool    collect_answer  fallback_response
        ↓         ↓              ↓
    execute   (end)           (end)
        ↓
    should_compress_context
        ↙              ↘
    compress_context  orchestrator
        ↓
    orchestrator (loop)

**Why This Architecture**

.. code-block:: text

    1. Summarization
       - Maintains conversational context without overwhelming LLM
       - Enables multi-turn research
       - Preserves prior findings
    
    2. Query Rewriting
       - Ensures search queries are precise and unambiguous
       - Uses context intelligently
       - Improves retrieval quality
    
    3. Human-in-the-Loop
       - Catches unclear queries before wasting retrieval resources
       - Enables user clarification
       - Prevents failed research
    
    4. Parallel Execution
       - Send API spawns independent agent subgraphs for each sub-question
       - Enables simultaneous research
       - Improves throughput
    
    5. Context Compression
       - Keeps agent's working memory lean across long retrieval loops
       - Prevents redundant fetches
       - Manages token usage
       - Tracks executed operations
    
    6. Fallback Response
       - Ensures graceful degradation
       - Agent always returns something useful
       - Respects execution limits
       - Synthesizes available data
    
    7. Answer Collection & Aggregation
       - Extracts clean final answers from agents
       - Aggregates into single coherent response
       - Maintains question ordering
       - Enables multi-question synthesis

**Agent Subgraph Construction Example**

.. code-block:: python

    from langgraph.graph import StateGraph, END

    # Create agent subgraph
    agent_graph = StateGraph(AgentState)

    # Add nodes
    agent_graph.add_node("orchestrator", orchestrator)
    agent_graph.add_node("tool", tool_executor)  # Defined separately
    agent_graph.add_node("compress_context", compress_context)
    agent_graph.add_node("collect_answer", collect_answer)
    agent_graph.add_node("fallback_response", fallback_response)

    # Add edges
    agent_graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator_call,
        {
            "tool": "tool",
            "collect_answer": "collect_answer",
            "fallback_response": "fallback_response"
        }
    )
    agent_graph.add_edge("tool", "should_compress_context")
    agent_graph.add_conditional_edges(
        "should_compress_context",
        lambda s: "compress_context" if s.get("should_compress") else "orchestrator",
        {
            "compress_context": "compress_context",
            "orchestrator": "orchestrator"
        }
    )
    agent_graph.add_edge("compress_context", "orchestrator")
    agent_graph.add_edge("collect_answer", END)
    agent_graph.add_edge("fallback_response", END)

    # Compile agent subgraph
    agent_subgraph = agent_graph.compile()

**Integration with Main Graph**

.. code-block:: python

    # In main graph
    graph.add_node("agent", agent_subgraph.invoke)
    
    # Route to agent with Send
    return [
        Send("agent", {
            "question": query,
            "question_index": idx,
            "messages": []
        })
        for idx, query in enumerate(state["rewrittenQuestions"])
    ]

Next Steps
----------

With graph nodes and edges defined, you can proceed to:

- Building the agent node for tool execution
- Implementing the complete graph compilation
- Creating the graph execution loop
- Adding monitoring and logging
- Building the complete RAG application

