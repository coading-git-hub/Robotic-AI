# Context-Aware RAG Agent with FastAPI

This specification defines the implementation of a FastAPI backend with OpenAI Agent SDK that answers queries using retrieved book content and optional user-selected text. The system prioritizes selected text in the agent context, strictly grounds responses in book content, and handles queries safely with fallback mechanisms.

## Overview

The system provides:
1. FastAPI endpoints for query processing with optional selected text
2. Context-aware agent reasoning that prioritizes user-selected text
3. Strict grounding in book content to prevent hallucinations
4. Fallback handling for insufficient context
5. Stateless API design for cloud compatibility

## Key Components

- Input validation for query and selected text parameters
- Context retrieval from Qdrant with prioritization logic
- Agent context preparation emphasizing selected text priority
- OpenAI Agent SDK integration for response generation
- Response validation to ensure grounding in provided context
- FastAPI endpoints for query processing and health checks

## Files in this Specification

- `spec.md`: Feature requirements and user scenarios
- `plan.md`: Implementation architecture and design
- `research.md`: Technical research and decision documentation
- `data-model.md`: Data structure definitions
- `contracts/api-contract.md`: API interface specifications
- `quickstart.md`: Setup and usage instructions
- `checklists/`: Implementation checklists

## Technical Decisions

- **Model**: OpenAI gpt-4-turbo for agent responses
- **Architecture**: Stateless FastAPI service with context prioritization
- **Context Handling**: Selected text prioritized over retrieved content
- **Grounding**: Strict validation to prevent hallucinations
- **Configuration**: Environment variables via python-dotenv