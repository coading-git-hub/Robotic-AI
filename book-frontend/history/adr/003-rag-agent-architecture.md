# ADR-003: RAG Agent Architecture with FastAPI

**Date**: 2025-12-13
**Status**: Accepted
**Authors**: Claude Code

## Context

For the Context-Aware RAG Agent with FastAPI, we need to make key architectural decisions about:
1. How to integrate the OpenAI Agent SDK with FastAPI for RAG responses
2. How to prioritize user-selected text in the agent context
3. How to ensure responses are strictly grounded in book content
4. How to handle fallback scenarios when context is insufficient
5. How to maintain stateless operation while providing context-aware responses

The system needs to accept user queries with optional selected text, retrieve relevant content from Qdrant, combine contexts with proper prioritization, generate grounded responses, and handle queries safely without hallucinations.

## Decision

### Agent Integration: OpenAI Chat Completions API

We will use OpenAI's Chat Completions API instead of the newer Assistants API for the following reasons:
- **Simplicity**: More straightforward integration with explicit context control
- **Transparency**: Clear visibility into the entire context provided to the model
- **Control**: Ability to precisely structure the system message with context prioritization
- **Compatibility**: Works well with the existing retrieval system

### Context Prioritization: Explicit System Message Structure

We will structure the agent context with explicit demarcation and prioritization:
- **Selected Text**: Marked as "USER SELECTED TEXT (HIGHEST PRIORITY)" with highest relevance score
- **Retrieved Content**: Marked with source information and similarity scores
- **System Message**: Explicitly instructs the agent to prioritize selected text and ground responses in provided context

### Grounding Validation: Multi-Layer Approach

We will implement grounding validation through multiple layers:
- **Pre-response**: Explicit instructions in system message to use only provided context
- **Post-response**: Validation of response content against provided context
- **Fallback Handling**: Clear responses when context is insufficient
- **Source Attribution**: Include sources in responses for transparency

### API Architecture: Stateless FastAPI Service

We will implement a completely stateless FastAPI service with:
- No session memory between requests
- All context explicitly passed in each request
- Request-response pattern only
- Proper input validation and error handling

## Rationale

### OpenAI Chat Completions API
- **Simplicity**: Easier to implement and debug than Assistants API
- **Control**: Direct control over context structure and prioritization
- **Transparency**: Clear understanding of what context is provided to the model
- **Performance**: Lower latency than Assistants API for simple queries

### Explicit Context Prioritization
- **Clarity**: Clear demarcation helps agent understand priority hierarchy
- **Flexibility**: Easy to adjust priority levels or add new context types
- **Maintainability**: Simple to understand and modify context structure
- **Reliability**: Consistent behavior across different query types

### Multi-Layer Grounding Validation
- **Effectiveness**: Multiple layers provide robust hallucination prevention
- **Transparency**: Clear visibility into grounding quality
- **Safety**: Prevents spreading of incorrect information
- **User Experience**: Clear fallbacks when context insufficient

### Stateless Architecture
- **Scalability**: No session state to manage across instances
- **Reliability**: No state corruption or synchronization issues
- **Simplicity**: Easier to reason about and debug
- **Cloud-Native**: Better fit for containerized deployments

## Alternatives Considered

### Agent Integration
1. **OpenAI Assistants API** - More complex, less transparent context control
2. **Custom LLM Integration** - More development work, less reliable
3. **Function Calling Approach** - Good alternative but more complex implementation

### Context Prioritization
1. **Embedding-based prioritization** - Less transparent to the agent
2. **Separate processing streams** - More complex implementation
3. **No explicit prioritization** - Would not meet requirements

### Grounding Approaches
1. **Post-hoc validation only** - Less effective than prevention
2. **No validation** - Would not prevent hallucinations
3. **External fact-checking** - More complex and slower

## Consequences

### Positive
- OpenAI Chat Completions provide reliable, controllable responses
- Explicit context structure ensures proper prioritization
- Multi-layer validation prevents hallucinations effectively
- Stateless design enables scalable deployment
- Clear fallback handling maintains user trust

### Negative
- Chat Completions API requires full context in each request
- Explicit structure adds some complexity to context preparation
- Validation layers add computational overhead
- Stateless design means no conversation history

## Implementation

The implementation will be in `rag_agent_api.py` with the following characteristics:
- FastAPI endpoints for query processing and health checks
- OpenAI Chat Completions API integration with structured context
- Explicit context prioritization with selected text emphasis
- Multi-layer validation to prevent hallucinations
- Stateless operation with comprehensive error handling
- Environment variable configuration for all service credentials