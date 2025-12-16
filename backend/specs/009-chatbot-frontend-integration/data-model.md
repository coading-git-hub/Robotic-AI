# Data Model: RAG Chatbot Frontend Integration

**Feature**: 009-chatbot-frontend-integration
**Created**: 2025-12-13
**Status**: Draft

## Entity: ChatMessage

### Description
Represents a single message in the chat conversation between user and agent.

### Fields
- `id` (string): Unique identifier for the message
- `content` (string): The text content of the message
- `sender` (string): The sender of the message ('user' or 'agent')
- `timestamp` (datetime): When the message was created/sent
- `type` (string): Type of message ('query', 'response', 'system')
- `sources` (list[dict], optional): Sources referenced in agent response (nullable)
- `confidence` (float, optional): Confidence score for agent response (nullable)

### Validation Rules
- `id` must be unique within the chat session
- `content` must not be empty
- `sender` must be one of: 'user', 'agent'
- `timestamp` must be current or past time
- `type` must be one of: 'query', 'response', 'system'
- `confidence` must be between 0.0 and 1.0 if provided

### Relationships
- One ChatMessage belongs to one ChatSession (many-to-one)

## Entity: ChatSession

### Description
Represents a single chat session containing multiple messages and context information.

### Fields
- `sessionId` (string): Unique identifier for the chat session
- `messages` (list[ChatMessage]): List of messages in the conversation
- `selectedText` (string, optional): Text selected by user in the book (nullable)
- `createdAt` (datetime): When the session was created
- `lastActive` (datetime): When the session was last used
- `isActive` (bool): Whether the session is currently active

### Validation Rules
- `sessionId` must be unique
- `messages` list must not exceed configured maximum
- `selectedText` length must be within configured limits if provided
- `createdAt` must be before or equal to `lastActive`

### Relationships
- One ChatSession contains many ChatMessage (one-to-many)
- One ChatSession has one active ChatWidget state (one-to-one)

## Entity: APIRequest

### Description
Represents a request sent from the frontend to the FastAPI agent backend.

### Fields
- `requestId` (string): Unique identifier for the API request
- `query` (string): The user's query to the agent
- `selectedText` (string, optional): Text selected by user in the book (nullable)
- `timestamp` (datetime): When the request was sent
- `backendUrl` (string): URL of the FastAPI backend
- `userPreferences` (dict, optional): User preferences for the request (nullable)

### Validation Rules
- `requestId` must be unique
- `query` must not be empty
- `selectedText` length must be within configured limits if provided
- `backendUrl` must be a valid URL
- `timestamp` must be current or past time

### Relationships
- One APIRequest maps to one APIResponse (one-to-one, optional)

## Entity: APIResponse

### Description
Represents a response received from the FastAPI agent backend.

### Fields
- `responseId` (string): Unique identifier for the API response
- `requestRef` (string): Reference to the original APIRequest
- `response` (string): The agent's response to the query
- `sources` (list[dict]): List of sources used to generate the answer
- `confidence` (float): Confidence score for the response (0.0-1.0)
- `groundedInContext` (bool): Whether the response is grounded in provided context
- `processingTimeMs` (int): Time taken to process the request in milliseconds
- `timestamp` (datetime): When the response was received
- `fallbackUsed` (bool): Whether a fallback response was used

### Validation Rules
- `responseId` must be unique
- `requestRef` must reference an existing APIRequest
- `response` must not be empty
- `confidence` must be between 0.0 and 1.0
- `processingTimeMs` must be non-negative

### Relationships
- One APIResponse belongs to one APIRequest (one-to-one)
- One APIResponse generates many ChatMessage (one-to-many, when response is split)

## Entity: ChatWidgetState

### Description
Represents the current state of the chat widget UI component.

### Fields
- `widgetId` (string): Unique identifier for the widget instance
- `isOpen` (bool): Whether the chat interface is currently open
- `isMinimized` (bool): Whether the chat interface is minimized
- `position` (dict): Position coordinates (x, y) of the widget
- `size` (dict): Size dimensions (width, height) of the widget
- `currentInput` (string): Current text in the input field
- `isLoading` (bool): Whether the widget is currently loading
- `errorState` (string, optional): Current error state if any (nullable)
- `sessionRef` (string): Reference to the current ChatSession

### Validation Rules
- `widgetId` must be unique per page
- `isOpen` and `isMinimized` cannot both be true
- `position` coordinates must be non-negative
- `size` dimensions must be positive
- `sessionRef` must reference an existing ChatSession

### Relationships
- One ChatWidgetState belongs to one ChatSession (one-to-one)
- One ChatWidgetState maps to one active UI instance (one-to-one)

## Frontend Component State Models

### FloatingChatIcon State
```
{
  "isVisible": "bool - Whether the icon is currently visible",
  "position": "string - Position of the icon (bottom-right, bottom-left, etc.)",
  "isHovered": "bool - Whether the icon is currently being hovered",
  "animationState": "string - Current animation state (idle, pulse, etc.)"
}
```

### ChatWidget State
```
{
  "isOpen": "bool - Whether the chat interface is open",
  "messages": "array[ChatMessage] - Current message history",
  "inputValue": "string - Current value in the input field",
  "isLoading": "bool - Whether a response is being loaded",
  "selectedText": "string - Currently captured selected text",
  "error": "string - Current error message if any"
}
```

### TextSelectionHandler State
```
{
  "selectedText": "string - Currently selected text from the page",
  "selectionRange": "object - Range object for the current selection",
  "isTextSelected": "bool - Whether text is currently selected",
  "lastSelectionTime": "datetime - When text was last selected"
}
```