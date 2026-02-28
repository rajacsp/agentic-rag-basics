# Agentic RAG Basics

A comprehensive guide to building production-ready Retrieval-Augmented Generation (RAG) systems with agentic workflows, hierarchical document indexing, and multi-agent orchestration.

## Overview

This documentation covers the complete pipeline for building an Agentic RAG system, from initial setup through deployment. Learn how to integrate vector databases, implement intelligent document retrieval, design agent tools, and orchestrate complex multi-agent workflows using LangGraph.

## Key Features

- **Hierarchical Document Indexing**: Organize documents with parent-child relationships for intelligent retrieval
- **Vector Database Integration**: Configure and optimize vector stores (Pinecone, Weaviate, Chroma)
- **Intelligent Agent Orchestration**: Build multi-agent systems with conversation memory and context management
- **Tool Integration**: Define custom tools for agents with proper validation and error handling
- **LangGraph Workflows**: Assemble complex state machines with conditional routing and parallel processing
- **Context Compression**: Implement token-aware memory management to handle long conversations
- **Human-in-the-Loop**: Add clarification requests and user feedback mechanisms
- **Chat Interface**: Deploy with Gradio for interactive conversations with persistence

## Documentation Structure

The documentation is organized into 11 chapters:

1. **Initial Setup and Configuration** - Environment setup, dependencies, and project structure
2. **Configure Vector Database** - Vector store selection, configuration, and optimization
3. **PDFs to Markdown** - Document preprocessing and conversion strategies
4. **Hierarchical Document Indexing** - Parent-child chunk relationships and retrieval strategies
5. **Define Agent Tools** - Tool design, validation, and integration patterns
6. **Define System Prompts** - Prompt engineering for orchestrators, retrievers, and synthesizers
7. **Define State and Data Models** - TypedDict schemas and state management
8. **Agent Configuration** - Token budgets, iteration limits, and compression thresholds
9. **Build Graph Nodes and Edges** - Node functions and conditional routing logic
10. **Build LangGraph Graphs** - Assembling agent subgraphs and main orchestration graphs
11. **Create Chat Interface** - Gradio deployment with conversation persistence

## Quick Start

### Prerequisites

- Python 3.10+
- pip or conda
- API keys for LLM provider (OpenAI, Anthropic, etc.)
- Vector database account (optional, can use in-memory)

### Installation

```bash
# Clone the repository
git clone https://github.com/rajacsp/agentic-rag-basics.git
cd agentic-rag-basics

# Install dependencies
pip install -r requirements.txt
```

### Building Documentation

```bash
cd doc
make html
```

The built documentation will be available in `docs/` directory.

## Project Structure

```
agentic-rag-basics/
├── doc/                          # Documentation source files
│   ├── ch01_initial_setup_configuration.rst
│   ├── ch02_configure_vector_database.rst
│   ├── ch03_pdfs_to_markdown.rst
│   ├── ch04_hierarchical_document_indexing.rst
│   ├── ch05_define_agent_tools.rst
│   ├── ch06_define_system_prompts.rst
│   ├── ch07_define_state_and_data_models.rst
│   ├── ch08_agent_configuration.rst
│   ├── ch09_build_graph_nodes_and_edges.rst
│   ├── ch10_build_langgraph_graphs.rst
│   ├── ch11_create_chat_interface.rst
│   ├── conf.py                   # Sphinx configuration
│   ├── index.rst                 # Documentation index
│   └── Makefile                  # Build documentation
├── docs/                         # Generated HTML documentation
├── README.md                     # This file
└── LICENSE                       # License information
```

## Core Concepts

### Agentic RAG

An agentic RAG system combines:
- **Retrieval**: Fetch relevant documents from a vector database
- **Augmentation**: Enhance prompts with retrieved context
- **Generation**: Use LLMs to synthesize answers
- **Agency**: Enable agents to make decisions, call tools, and iterate

### Multi-Agent Architecture

The system uses two levels of agents:
- **Orchestrator Agent**: Routes queries, manages conversation context, and decides when to clarify
- **Retrieval Agents**: Execute in parallel to search and retrieve relevant documents

### Hierarchical Indexing

Documents are indexed with parent-child relationships:
- **Parent chunks**: Full document sections for context
- **Child chunks**: Smaller, searchable segments for retrieval
- **Hybrid retrieval**: Fetch child chunks, return parent context

## Key Technologies

- **LangChain**: LLM framework and tool integration
- **LangGraph**: State machine and workflow orchestration
- **Vector Databases**: Pinecone, Weaviate, Chroma, or FAISS
- **LLMs**: OpenAI GPT-4, Anthropic Claude, or open-source models
- **Gradio**: Web interface for chat interactions
- **Sphinx**: Documentation generation

## Performance Considerations

- Token budgets prevent runaway costs
- Context compression manages memory for long conversations
- Hierarchical indexing balances retrieval speed and context quality
- Parallel agent execution reduces latency
- Checkpointing enables conversation persistence

## Troubleshooting

### Common Issues

**Vector database connection fails**
- Verify API keys and credentials
- Check network connectivity
- Ensure database is running (for local deployments)

**Agent loops indefinitely**
- Check MAX_ITERATIONS and MAX_TOOL_CALLS limits
- Review tool definitions for infinite loops
- Verify conditional routing logic

**Memory issues with long conversations**
- Enable context compression
- Reduce chunk sizes
- Implement conversation summarization

**Poor retrieval quality**
- Adjust chunk sizes and overlap
- Review embedding model selection
- Tune similarity thresholds

## Contributing

Contributions are welcome. Please ensure:
- Documentation is updated for new features
- Code examples are tested
- Performance implications are documented

## License

See LICENSE file for details.

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Vector Database Guides](https://www.pinecone.io/learn/)
- [RAG Best Practices](https://docs.anthropic.com/en/docs/build-a-rag-chatbot-with-claude)

## Support

For questions or issues:
1. Check the documentation chapters
2. Review code examples in the doc/code directory
3. Open an issue on GitHub

---

**Last Updated**: February 2026
**Version**: 1.0
