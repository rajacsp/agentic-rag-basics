Agentic RAG Basics
==================

Welcome to the Agentic RAG Basics documentation. This comprehensive guide covers the setup, configuration, and implementation of a complete Retrieval-Augmented Generation (RAG) system with agentic capabilities, conversation memory, and human-in-the-loop support using LangChain, Qdrant, LangGraph, and local language models.

Contents
--------

.. toctree::
   :maxdepth: 2

   preface
   ch01_initial_setup_configuration
   ch02_configure_vector_database
   ch03_pdfs_to_markdown
   ch04_hierarchical_document_indexing
   ch05_define_agent_tools
   ch06_define_system_prompts
   ch07_define_state_and_data_models
   ch08_agent_configuration
   ch09_build_graph_nodes_and_edges
   ch10_build_langgraph_graphs
   ch11_create_chat_interface

.. This will display as:
   Preface
   1: Initial Setup and Configuration
   2: Configure Vector Database
   3: PDFs to Markdown
   4: Hierarchical Document Indexing
   5: Define Agent Tools
   6: Define System Prompts
   7: Define State and Data Models
   8: Agent Configuration
   9: Build Graph Node and Edge Functions
   10: Build the LangGraph Graphs
   11: Create Chat Interface

Key Features
------------

- **Hybrid Search**: Combines dense and sparse embeddings for optimal retrieval
- **Hierarchical Indexing**: Parent/child chunk strategy for context preservation
- **Multi-Agent Architecture**: Parallel processing of multiple questions
- **Conversation Memory**: Multi-turn context preservation with compression
- **Human-in-the-Loop**: Intelligent clarification requests for unclear queries
- **Graceful Degradation**: Fallback responses when research limits are reached
- **Token Management**: Dynamic context compression based on token usage
- **State Persistence**: Checkpointing for conversation recovery

System Architecture
-------------------

**Document Processing**: Convert PDFs to Markdown and create hierarchical chunks

**Vector Storage**: Store embeddings in Qdrant with hybrid search capabilities

**Agent System**: LangGraph-based multi-agent architecture with tool execution

**Chat Interface**: Gradio web UI with conversation persistence and human-in-the-loop support

**Complete Pipeline**: End-to-end document research and question answering
