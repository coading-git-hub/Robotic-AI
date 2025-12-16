# RAG Chatbot Frontend Integration

This package provides a React-based chatbot frontend integration for Docusaurus documentation sites. It enables users to ask questions about book content using a RAG (Retrieval-Augmented Generation) system with optional text selection context.

## Features

- Floating chatbot icon that appears on all book pages
- Text selection capture to provide context to the chatbot
- Responsive chat interface with message history
- Integration with FastAPI agent backend
- Accessibility features and keyboard navigation
- Error handling and connection status indicators
- Mobile-responsive design

## Installation

1. Install the required dependencies:

```bash
npm install react react-dom axios styled-components
```

2. Install Docusaurus dependencies if integrating with a Docusaurus site:

```bash
npm install @docusaurus/core
```

## Usage

### Standalone React Component

```jsx
import React from 'react';
import ChatbotContainer from './src/components/ChatbotContainer';

function App() {
  return (
    <div className="App">
      {/* Your existing content */}
      <ChatbotContainer />
    </div>
  );
}

export default App;
```

### Docusaurus Integration

1. Add the plugin to your `docusaurus.config.js`:

```js
module.exports = {
  // ... other config
  plugins: [
    // ... other plugins
    [
      './src/plugins/docusaurus-plugin-chatbot',
      {
        backendUrl: process.env.BACKEND_API_URL || 'http://localhost:8000',
        enableByDefault: true,
      },
    ],
  ],
};
```

## Environment Variables

Create a `.env` file with the following variables:

```env
# Backend API Configuration
REACT_APP_BACKEND_API_URL=http://localhost:8000

# Chat Widget Configuration
CHAT_WIDGET_POSITION=bottom-right
CHAT_WIDGET_SIZE=60px

# Default Messages and Fallbacks
REACT_APP_DEFAULT_FALLBACK_MESSAGE=I cannot answer based on the provided context

# Text Selection Limits
MAX_SELECTED_TEXT_LENGTH=2000

# API Timeout Configuration
API_TIMEOUT=30000
```

## Components

### FloatingChatIcon
- Fixed position chat icon that appears on all pages
- Opens chat interface when clicked
- Accessible with keyboard navigation

### ChatWidget
- Main chat interface with message history
- Input area with text selection display
- Loading states and error handling
- Source attribution for responses

### TextSelectionHandler
- Captures selected text from the page
- Integrates selected text with chat context
- Handles different content types

### APIClient
- Communicates with the backend agent API
- Handles requests and responses
- Implements retry logic and error handling

## API Endpoints Used

- `POST /api/agent/query` - Send user queries to the backend
- `GET /api/health` - Check backend health status
- `POST /api/agent/validate` - Validate agent responses (for development)

## Development

To run the development server:

```bash
npm start
```

To build for production:

```bash
npm run build
```

## Deployment

### GitHub Pages

1. Build the project: `npm run build`
2. Configure GitHub Pages in your repository settings
3. Point to the `build/` directory

### Docusaurus Deployment

1. Add the plugin to your Docusaurus configuration
2. Build with `npm run build`
3. Deploy according to your Docusaurus hosting setup

## Error Handling

The chatbot includes comprehensive error handling:

- Backend connection status indicator
- Graceful handling of API errors
- Input validation and sanitization
- Connection retry logic

## Accessibility

The chatbot follows WCAG 2.1 AA standards:

- Proper ARIA labels and roles
- Keyboard navigation support
- Sufficient color contrast
- Focus indicators

## Security

- Input sanitization to prevent XSS
- Environment variable configuration
- Secure API communication

## License

MIT