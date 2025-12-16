# ADR-004: Chatbot Frontend Integration Architecture

**Date**: 2025-12-13
**Status**: Accepted
**Authors**: Claude Code

## Context

For the RAG Chatbot Frontend Integration, we need to make key architectural decisions about:
1. How to integrate the chatbot UI into the Docusaurus book frontend
2. How to implement the floating icon and click-to-open interface
3. How to capture and pass selected text as context
4. How to communicate with the Agent SDK backend
5. How to structure the components for maintainability and performance

The system needs to seamlessly integrate into the existing book frontend without affecting performance or user experience, while providing a robust chat interface that connects to the backend agent service.

## Decision

### UI Architecture: Embedded Widget with Click-to-Open Interface

We will implement a floating action button with slide-in chat panel approach:
- **Floating Icon**: Circular 60px button positioned at bottom-right corner
- **Click-to-Open**: Slide-in panel that appears when icon is clicked
- **Widget Design**: Self-contained component that doesn't interfere with main content
- **Accessibility**: Full WCAG 2.1 AA compliance with keyboard navigation

### Component Architecture: Modular React Components

We will structure the implementation as modular React components:
- `FloatingChatIcon`: Displays persistent icon with accessibility features
- `ChatWidget`: Main interface with message history and input controls
- `TextSelectionHandler`: Captures selected text using window.getSelection()
- `APIClient`: Handles communication with Agent SDK backend
- `ChatbotIntegration`: Main component that combines all parts

### Text Selection: JavaScript-Based Capture

We will use the window.getSelection() API with event listeners:
- **Capture Method**: Monitor mouseup and keyup events to detect text selection
- **Processing**: Extract selected text and validate length/format
- **Context Passing**: Include selected text in query requests to backend
- **User Feedback**: Visual indication when text is captured

### Backend Communication: Fetch API with Error Handling

We will use the fetch API for communication with the Agent SDK:
- **Protocol**: HTTPS requests to Agent SDK endpoints
- **Endpoints**: /api/agent/query for queries, /api/health for status
- **Error Handling**: Comprehensive error handling with fallbacks
- **Performance**: Timeout and retry mechanisms for reliability

### Deployment: GitHub Pages with External Backend

We will deploy the frontend on GitHub Pages connecting to external Agent SDK:
- **Frontend**: Static site hosted on GitHub Pages
- **Backend**: Agent SDK service hosted separately
- **Connection**: Secure API communication over HTTPS
- **Configuration**: Environment variable-based settings

## Rationale

### Embedded Widget with Click-to-Open
- **Non-intrusive**: Minimal impact on main content and page layout
- **Familiar**: Common UI pattern that users understand
- **Accessible**: Can be implemented with proper accessibility features
- **Performant**: Only loads when needed, minimal initial page impact

### Modular Component Architecture
- **Maintainability**: Clear separation of concerns
- **Reusability**: Components can be used independently
- **Testability**: Each component can be tested separately
- **Scalability**: Easy to add new features or modify components

### JavaScript Text Selection
- **Native**: Uses browser's built-in selection API
- **Reliable**: Works across different content types
- **Flexible**: Can be enhanced with additional logic
- **Efficient**: No external dependencies required

### Fetch API Communication
- **Modern**: Standard browser API with good support
- **Flexible**: Promise-based with async/await
- **Secure**: Built-in CORS and security features
- **Debuggable**: Good error reporting and debugging tools

### GitHub Pages Deployment
- **Cost-effective**: Free hosting for static content
- **Reliable**: High availability and performance
- **Scalable**: Handles traffic automatically
- **Simple**: Easy to deploy and maintain

## Alternatives Considered

### UI Approaches
1. **Always-visible panel** - Would take up too much screen real estate
2. **Modal dialog** - Could interfere more with page navigation
3. **Inline integration** - Would require more complex layout management

### Text Selection Methods
1. **Range API directly** - More complex to implement than getSelection()
2. **Custom selection tool** - Would require more development effort
3. **No text selection** - Would not meet requirements for selected-text mode

### Communication Methods
1. **WebSocket connection** - More complex than needed for this use case
2. **GraphQL API** - Overkill for simple query/response pattern
3. **Server-sent events** - Not needed for request/response pattern

## Consequences

### Positive
- Embedded widget provides non-intrusive user experience
- Modular components enable maintainable and testable code
- JavaScript text selection works across all content types
- Fetch API provides reliable communication with proper error handling
- GitHub Pages deployment is cost-effective and scalable

### Negative
- Additional JavaScript increases bundle size
- Floating widget may slightly impact performance
- External backend dependency requires network connectivity
- Text selection may not work with all content types (e.g., images)

## Implementation

The implementation will be in `chatbot-frontend-components.js` with the following characteristics:
- React components for all UI elements
- Styled-components for responsive styling
- window.getSelection() for text capture
- fetch API for Agent SDK communication
- Environment variable configuration for backend URL
- Comprehensive error handling and fallbacks