Topic 7: Define State and Data Models
=====================================

Overview
--------

This chapter covers the creation of state structures and data models that track conversation context, agent execution, and query analysis throughout the RAG pipeline. Well-designed state models enable efficient information flow, support multi-turn conversations, and facilitate debugging and monitoring of agent behavior.

Purpose and Architecture
------------------------

State and data models serve as the "memory" of the RAG system, tracking:

- **Conversation State**: Messages, summaries, and context
- **Agent Execution State**: Tool calls, iterations, and progress
- **Query Analysis**: Question clarity and reformulation
- **Retrieval State**: Tracking retrieved documents and keys
- **Answer Aggregation**: Collecting and managing answers

The state architecture includes:

1. **State**: Main conversation and agent state
2. **AgentState**: Detailed agent execution tracking
3. **QueryAnalysis**: Query clarity and reformulation results

Step 1: Import Required Libraries
----------------------------------

Begin by importing the necessary dependencies:

.. code-block:: python

    from langgraph.graph import MessagesState
    from pydantic import BaseModel, Field
    from typing import List, Annotated, Set
    import operator

**Import Explanations:**

- ``langgraph.graph.MessagesState``: Base state class for message-based workflows
- ``pydantic.BaseModel``: Data validation and serialization
- ``pydantic.Field``: Field metadata and descriptions
- ``typing.List, Annotated, Set``: Type hints for complex types
- ``operator``: Operators for state reduction (add, union)

**Why These Imports:**

- **LangGraph**: Enables stateful graph-based agent workflows
- **Pydantic**: Provides data validation and schema generation
- **Type hints**: Enable IDE support and runtime validation
- **Operators**: Support state accumulation and merging

Step 2: Define State Reducer Functions
---------------------------------------

Define functions for managing state updates:

.. code-block:: python

    def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
        """Accumulate new items or reset if reset flag present.
        
        Args:
            existing: Current list of items
            new: New items to add
            
        Returns:
            Updated list (reset if __reset__ flag present, else accumulated)
        """
        if new and any(item.get('__reset__') for item in new):
            return []
        return existing + new

    def set_union(a: Set[str], b: Set[str]) -> Set[str]:
        """Union two sets.
        
        Args:
            a: First set
            b: Second set
            
        Returns:
            Union of both sets
        """
        return a | b

**Step 2a: Accumulate or Reset Function**

.. code-block:: python

    def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:

**Purpose:**

- Manages agent answer accumulation
- Supports resetting answers when needed
- Enables multi-turn answer collection

**How It Works:**

1. Check if new items contain ``__reset__`` flag
2. If reset flag present: return empty list (discard existing)
3. Otherwise: append new items to existing list

**Example Usage:**

.. code-block:: text

    Scenario 1: Normal Accumulation
    ===============================
    existing = [{"answer": "First answer"}]
    new = [{"answer": "Second answer"}]
    
    Result: [{"answer": "First answer"}, {"answer": "Second answer"}]
    
    Scenario 2: Reset
    =================
    existing = [{"answer": "First answer"}]
    new = [{"__reset__": True}, {"answer": "New answer"}]
    
    Result: [] (reset triggered, existing discarded)

**Why Accumulate or Reset:**

- Supports multi-turn conversations
- Enables answer collection across iterations
- Allows resetting when starting new research
- Prevents answer duplication

**Step 2b: Set Union Function**

.. code-block:: python

    def set_union(a: Set[str], b: Set[str]) -> Set[str]:

**Purpose:**

- Merges retrieval keys from multiple iterations
- Tracks unique documents accessed
- Prevents duplicate tracking

**How It Works:**

1. Takes two sets as input
2. Returns union (all unique elements)

**Example Usage:**

.. code-block:: text

    Scenario 1: Merging Keys
    ========================
    a = {"doc1.pdf", "doc2.pdf"}
    b = {"doc2.pdf", "doc3.pdf"}
    
    Result: {"doc1.pdf", "doc2.pdf", "doc3.pdf"}
    
    Scenario 2: Empty Set
    =====================
    a = {"doc1.pdf"}
    b = set()
    
    Result: {"doc1.pdf"}

**Why Set Union:**

- Tracks all accessed documents
- Prevents duplicate entries
- Enables retrieval deduplication
- Supports multi-iteration research

Step 3: Define Main State Class
--------------------------------

Define the main conversation and agent state:

.. code-block:: python

    class State(MessagesState):
        """Main state for conversation and agent execution.
        
        Extends MessagesState with additional fields for RAG pipeline.
        """
        questionIsClear: bool = False
        conversation_summary: str = ""
        originalQuery: str = ""
        rewrittenQuestions: List[str] = []
        agent_answers: Annotated[List[dict], accumulate_or_reset] = []

**State Purpose:**

Tracks:

- Conversation messages (from MessagesState)
- Question clarity
- Conversation summaries
- Original and rewritten queries
- Accumulated agent answers

**Step 3a: MessagesState Base**

.. code-block:: python

    class State(MessagesState):

**What MessagesState Provides:**

- ``messages``: List of conversation messages
- Message history management
- Built-in message handling

**Why Extend MessagesState:**

- Provides message management out-of-the-box
- Integrates with LangGraph workflows
- Enables message-based state transitions

**Step 3b: Question Clarity Field**

.. code-block:: python

    questionIsClear: bool = False

**Purpose:**

- Tracks if user's question is clear
- Determines if clarification needed
- Guides query rewriting

**Example Values:**

.. code-block:: text

    True: "What is backpropagation?" (clear)
    False: "xyzabc qwerty" (unclear)
    False: "what about it?" (ambiguous follow-up)

**Step 3c: Conversation Summary Field**

.. code-block:: python

    conversation_summary: str = ""

**Purpose:**

- Stores compressed conversation history
- Enables context preservation
- Supports multi-turn conversations

**Example Value:**

.. code-block:: text

    "Discussed neural networks and their training process using backpropagation. 
     User asked about optimization techniques."

**Step 3d: Query Fields**

.. code-block:: python

    originalQuery: str = ""
    rewrittenQuestions: List[str] = []

**Purpose:**

- ``originalQuery``: User's original input
- ``rewrittenQuestions``: Reformulated queries for retrieval

**Example Values:**

.. code-block:: text

    originalQuery: "How do RNNs work?"
    rewrittenQuestions: [
        "How do Recurrent Neural Networks work?",
        "What is the architecture of RNNs?"
    ]

**Step 3e: Agent Answers Field**

.. code-block:: python

    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

**Purpose:**

- Accumulates answers from agent iterations
- Supports multi-turn answer collection
- Can be reset when needed

**Example Value:**

.. code-block:: python

    [
        {
            "question": "How do RNNs work?",
            "answer": "RNNs process sequences...",
            "sources": ["paper.pdf"]
        },
        {
            "question": "What are applications?",
            "answer": "RNNs are used in...",
            "sources": ["guide.pdf"]
        }
    ]

**State Usage Example**

.. code-block:: python

    # Initialize state
    state = State(
        messages=[],
        questionIsClear=False,
        conversation_summary="",
        originalQuery="How do neural networks work?",
        rewrittenQuestions=[],
        agent_answers=[]
    )
    
    # Update state
    state.questionIsClear = True
    state.rewrittenQuestions = [
        "How do neural networks work?",
        "What is the architecture of neural networks?"
    ]
    
    # Accumulate answers
    state.agent_answers.append({
        "question": "How do neural networks work?",
        "answer": "Neural networks consist of interconnected nodes...",
        "sources": ["research_paper.pdf"]
    })

Step 4: Define Agent State Class
---------------------------------

Define detailed agent execution state:

.. code-block:: python

    class AgentState(MessagesState):
        """State for detailed agent execution tracking.
        
        Tracks tool calls, iterations, and retrieval progress.
        """
        tool_call_count: Annotated[int, operator.add] = 0
        iteration_count: Annotated[int, operator.add] = 0
        question: str = ""
        question_index: int = 0
        context_summary: str = ""
        retrieval_keys: Annotated[Set[str], set_union] = set()
        final_answer: str = ""
        agent_answers: List[dict] = []

**AgentState Purpose:**

Tracks:

- Tool invocation count
- Iteration progress
- Current question being processed
- Retrieved document keys
- Final answer generation
- Detailed answer collection

**Step 4a: Tool Call Tracking**

.. code-block:: python

    tool_call_count: Annotated[int, operator.add] = 0

**Purpose:**

- Counts tool invocations
- Monitors agent activity
- Detects excessive tool usage

**How It Works:**

- Uses ``operator.add`` to accumulate counts
- Increments on each tool call
- Enables monitoring and limits

**Example Usage:**

.. code-block:: text

    Initial: tool_call_count = 0
    After search: tool_call_count = 1
    After retrieve: tool_call_count = 2
    After another search: tool_call_count = 3

**Step 4b: Iteration Tracking**

.. code-block:: python

    iteration_count: Annotated[int, operator.add] = 0

**Purpose:**

- Counts agent iterations
- Monitors research progress
- Enforces iteration limits

**Example Usage:**

.. code-block:: text

    Initial: iteration_count = 0
    After iteration 1: iteration_count = 1
    After iteration 2: iteration_count = 2
    After iteration 3: iteration_count = 3

**Step 4c: Question Processing**

.. code-block:: python

    question: str = ""
    question_index: int = 0

**Purpose:**

- ``question``: Current question being processed
- ``question_index``: Index in multi-question list

**Example Usage:**

.. code-block:: text

    Scenario: Processing 3 questions
    
    Iteration 1:
    - question = "How do neural networks work?"
    - question_index = 0
    
    Iteration 2:
    - question = "What are their applications?"
    - question_index = 1
    
    Iteration 3:
    - question = "How are they trained?"
    - question_index = 2

**Step 4d: Context and Retrieval Tracking**

.. code-block:: python

    context_summary: str = ""
    retrieval_keys: Annotated[Set[str], set_union] = set()

**Purpose:**

- ``context_summary``: Compressed research context
- ``retrieval_keys``: Set of unique retrieved documents

**Example Usage:**

.. code-block:: text

    context_summary: "Found information about neural networks, 
                      backpropagation, and training methods."
    
    retrieval_keys: {"research_paper.pdf", "technical_guide.pdf", "manual.pdf"}

**Step 4e: Answer Fields**

.. code-block:: python

    final_answer: str = ""
    agent_answers: List[dict] = []

**Purpose:**

- ``final_answer``: Final synthesized answer
- ``agent_answers``: Detailed answers for each question

**Example Usage:**

.. code-block:: python

    final_answer: "Neural networks are computational models that learn 
                   through backpropagation. They have applications in 
                   computer vision, NLP, and robotics."
    
    agent_answers: [
        {
            "question": "How do neural networks work?",
            "answer": "Neural networks consist of...",
            "sources": ["paper.pdf"]
        }
    ]

**AgentState Usage Example**

.. code-block:: python

    # Initialize agent state
    agent_state = AgentState(
        messages=[],
        tool_call_count=0,
        iteration_count=0,
        question="How do neural networks work?",
        question_index=0,
        context_summary="",
        retrieval_keys=set(),
        final_answer="",
        agent_answers=[]
    )
    
    # Update during execution
    agent_state.tool_call_count += 1  # After search
    agent_state.iteration_count += 1  # After iteration
    agent_state.retrieval_keys.add("research_paper.pdf")  # After retrieval
    
    # Set final answer
    agent_state.final_answer = "Neural networks work by..."

Step 5: Define Query Analysis Model
------------------------------------

Define the data model for query analysis results:

.. code-block:: python

    class QueryAnalysis(BaseModel):
        """Result of query clarity and reformulation analysis.
        
        Determines if question is clear and provides reformulated questions.
        """
        is_clear: bool = Field(
            description="Indicates if the user's question is clear and answerable."
        )
        questions: List[str] = Field(
            description="List of rewritten, self-contained questions."
        )
        clarification_needed: str = Field(
            description="Explanation if the question is unclear."
        )

**QueryAnalysis Purpose:**

Captures:

- Question clarity assessment
- Reformulated questions
- Clarification requirements

**Step 5a: Clarity Assessment**

.. code-block:: python

    is_clear: bool = Field(
        description="Indicates if the user's question is clear and answerable."
    )

**Purpose:**

- Determines if question is understandable
- Guides downstream processing
- Triggers clarification if needed

**Example Values:**

.. code-block:: text

    True: "What is backpropagation?" (clear)
    False: "xyzabc qwerty" (unclear)
    False: "what about it?" (ambiguous)

**Step 5b: Reformulated Questions**

.. code-block:: python

    questions: List[str] = Field(
        description="List of rewritten, self-contained questions."
    )

**Purpose:**

- Provides optimized questions for retrieval
- Handles multi-part queries
- Improves search effectiveness

**Example Values:**

.. code-block:: python

    # Single question
    ["How do neural networks work?"]
    
    # Multi-part question split
    [
        "How do neural networks work?",
        "What are the applications of neural networks?"
    ]

**Step 5c: Clarification Needed**

.. code-block:: python

    clarification_needed: str = Field(
        description="Explanation if the question is unclear."
    )

**Purpose:**

- Explains why question is unclear
- Guides user clarification
- Enables helpful error messages

**Example Values:**

.. code-block:: text

    "": (if is_clear is True)
    
    "The question contains unclear abbreviations and lacks context. 
     Could you rephrase using full terms?"
    
    "The question is ambiguous. Are you asking about the architecture, 
     training process, or applications?"

**QueryAnalysis Usage Example**

.. code-block:: python

    # Clear question
    analysis = QueryAnalysis(
        is_clear=True,
        questions=["How do neural networks work?"],
        clarification_needed=""
    )
    
    # Unclear question
    analysis = QueryAnalysis(
        is_clear=False,
        questions=[],
        clarification_needed="The question is unclear. Could you provide more context?"
    )
    
    # Multi-part question
    analysis = QueryAnalysis(
        is_clear=True,
        questions=[
            "How do neural networks work?",
            "What are their applications?"
        ],
        clarification_needed=""
    )

Complete State and Data Models Module
--------------------------------------

Here's the complete state and data models code:

.. code-block:: python

    from langgraph.graph import MessagesState
    from pydantic import BaseModel, Field
    from typing import List, Annotated, Set
    import operator

    # State reducer functions
    def accumulate_or_reset(existing: List[dict], new: List[dict]) -> List[dict]:
        """Accumulate new items or reset if reset flag present."""
        if new and any(item.get('__reset__') for item in new):
            return []
        return existing + new

    def set_union(a: Set[str], b: Set[str]) -> Set[str]:
        """Union two sets."""
        return a | b

    # Main state class
    class State(MessagesState):
        """Main state for conversation and agent execution."""
        questionIsClear: bool = False
        conversation_summary: str = ""
        originalQuery: str = ""
        rewrittenQuestions: List[str] = []
        agent_answers: Annotated[List[dict], accumulate_or_reset] = []

    # Agent execution state class
    class AgentState(MessagesState):
        """State for detailed agent execution tracking."""
        tool_call_count: Annotated[int, operator.add] = 0
        iteration_count: Annotated[int, operator.add] = 0
        question: str = ""
        question_index: int = 0
        context_summary: str = ""
        retrieval_keys: Annotated[Set[str], set_union] = set()
        final_answer: str = ""
        agent_answers: List[dict] = []

    # Query analysis model
    class QueryAnalysis(BaseModel):
        """Result of query clarity and reformulation analysis."""
        is_clear: bool = Field(
            description="Indicates if the user's question is clear and answerable."
        )
        questions: List[str] = Field(
            description="List of rewritten, self-contained questions."
        )
        clarification_needed: str = Field(
            description="Explanation if the question is unclear."
        )

State Management Patterns
-------------------------

**Pattern 1: State Initialization**

.. code-block:: python

    def initialize_state(user_query: str) -> State:
        """Initialize state for new conversation."""
        return State(
            messages=[],
            questionIsClear=False,
            conversation_summary="",
            originalQuery=user_query,
            rewrittenQuestions=[],
            agent_answers=[]
        )

**Pattern 2: State Updates**

.. code-block:: python

    def update_state(state: State, updates: dict) -> State:
        """Update state with new values."""
        for key, value in updates.items():
            setattr(state, key, value)
        return state

**Pattern 3: Answer Accumulation**

.. code-block:: python

    def add_answer(state: State, answer: dict) -> State:
        """Add answer to state."""
        state.agent_answers.append(answer)
        return state

**Pattern 4: State Reset**

.. code-block:: python

    def reset_answers(state: State) -> State:
        """Reset accumulated answers."""
        state.agent_answers = [{"__reset__": True}]
        return state

State Transitions
-----------------

**Typical State Flow:**

.. code-block:: text

    1. Initialize State
       ↓
    2. Analyze Query
       - Set questionIsClear
       - Set rewrittenQuestions
       ↓
    3. Agent Execution
       - Increment iteration_count
       - Increment tool_call_count
       - Update retrieval_keys
       - Update context_summary
       ↓
    4. Answer Generation
       - Set final_answer
       - Accumulate agent_answers
       ↓
    5. Conversation Summary
       - Update conversation_summary
       ↓
    6. Return Final State

**Example State Transition:**

.. code-block:: python

    # Step 1: Initialize
    state = State(
        messages=[],
        questionIsClear=False,
        originalQuery="How do neural networks work?",
        rewrittenQuestions=[],
        agent_answers=[]
    )
    
    # Step 2: Query Analysis
    state.questionIsClear = True
    state.rewrittenQuestions = ["How do neural networks work?"]
    
    # Step 3: Agent Execution
    agent_state = AgentState(
        messages=state.messages,
        tool_call_count=0,
        iteration_count=0,
        question=state.rewrittenQuestions[0],
        question_index=0,
        context_summary="",
        retrieval_keys=set(),
        final_answer="",
        agent_answers=[]
    )
    
    # During execution
    agent_state.tool_call_count += 1  # search
    agent_state.tool_call_count += 1  # retrieve
    agent_state.iteration_count += 1
    agent_state.retrieval_keys.add("paper.pdf")
    agent_state.context_summary = "Found information about neural networks"
    
    # Step 4: Answer Generation
    agent_state.final_answer = "Neural networks are computational models..."
    agent_state.agent_answers.append({
        "question": "How do neural networks work?",
        "answer": agent_state.final_answer,
        "sources": list(agent_state.retrieval_keys)
    })
    
    # Step 5: Update main state
    state.agent_answers = agent_state.agent_answers
    state.conversation_summary = "Discussed neural networks and their architecture"

Best Practices
--------------

**1. Type Hints**

.. code-block:: python

    # Good: Clear type hints
    class State(MessagesState):
        question: str = ""
        answers: List[dict] = []
    
    # Avoid: Missing type hints
    class State(MessagesState):
        question = ""
        answers = []

**2. Field Descriptions**

.. code-block:: python

    # Good: Descriptive fields
    class QueryAnalysis(BaseModel):
        is_clear: bool = Field(
            description="Indicates if the user's question is clear and answerable."
        )
    
    # Avoid: No descriptions
    class QueryAnalysis(BaseModel):
        is_clear: bool

**3. Default Values**

.. code-block:: python

    # Good: Sensible defaults
    class State(MessagesState):
        questionIsClear: bool = False
        agent_answers: List[dict] = []
    
    # Avoid: No defaults
    class State(MessagesState):
        questionIsClear: bool
        agent_answers: List[dict]

**4. State Reducers**

.. code-block:: python

    # Good: Custom reducers for complex logic
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []
    
    # Avoid: No reducer logic
    agent_answers: List[dict] = []

**5. Immutability**

.. code-block:: python

    # Good: Create new state
    new_state = State(**state.dict(), question_is_clear=True)
    
    # Avoid: Mutate existing state
    state.question_is_clear = True

Performance Considerations
--------------------------

**Memory Usage:**

- State object: ~1-5KB
- Messages list: ~100 bytes per message
- Agent answers: ~500 bytes per answer
- Total: Scales with conversation length

**State Serialization:**

- JSON serialization: 1-5ms
- Pydantic validation: 1-2ms
- Total: Minimal overhead

**Optimization Tips:**

.. code-block:: python

    # Limit message history
    state.messages = state.messages[-10:]  # Keep last 10 messages
    
    # Compress old answers
    if len(state.agent_answers) > 5:
        state.agent_answers = compress_answers(state.agent_answers)
    
    # Clear retrieval keys periodically
    if len(state.retrieval_keys) > 100:
        state.retrieval_keys = set()

Troubleshooting
---------------

**State Not Updating**

.. code-block:: python

    # Problem: State changes not reflected
    # Solution: Ensure returning updated state
    
    def update_state(state: State) -> State:
        state.question_is_clear = True
        return state  # Must return state

**Accumulation Not Working**

.. code-block:: python

    # Problem: Answers not accumulating
    # Solution: Verify reducer function
    
    # Check that accumulate_or_reset is used
    agent_answers: Annotated[List[dict], accumulate_or_reset] = []

**Type Validation Errors**

.. code-block:: python

    # Problem: Pydantic validation errors
    # Solution: Ensure correct types
    
    # Good
    state.agent_answers = [{"answer": "text"}]
    
    # Bad
    state.agent_answers = "answer text"  # Wrong type

Next Steps
----------

With state and data models defined, you can proceed to:

- Building the agent graph with state transitions
- Implementing node functions for each stage
- Creating the complete agent workflow
- Integrating with LangGraph for execution
- Building the complete RAG application

