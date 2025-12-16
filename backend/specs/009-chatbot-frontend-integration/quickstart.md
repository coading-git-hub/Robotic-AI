# Quickstart Guide: RAG Chatbot Frontend Integration

**Feature**: 009-chatbot-frontend-integration
**Created**: 2025-12-13

## Overview

This guide provides quick setup instructions for integrating the RAG chatbot into the Docusaurus book frontend. The system adds a floating chatbot icon that opens a chat interface, captures selected text for context, communicates with the Agent SDK backend, and displays responses. The integrated system will be deployed on GitHub Pages.

## Prerequisites

- Node.js >= 16.14.0
- Docusaurus project (v2.x) with existing book content
- Running Agent SDK backend (from spec 008)
- Git for version control
- GitHub account for Pages deployment

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies
```bash
npm install
# Or if using yarn
yarn install
```

### 3. Install Chatbot Component Dependencies
```bash
npm install react react-dom axios styled-components
# Add any additional dependencies needed for the chatbot components
```

### 4. Configure Backend Connection
Update your Docusaurus configuration (`docusaurus.config.js`) to include the backend API URL:

```javascript
module.exports = {
  // ... existing config
  themeConfig: {
    // ... existing theme config
    chatbot: {
      backendUrl: process.env.BACKEND_API_URL || 'https://your-agent-sdk-backend.com',
      widgetPosition: 'bottom-right', // Options: bottom-left, bottom-right
      widgetSize: '60px', // Size of the floating icon
      maxSelectedTextLength: 2000 // Maximum length of selected text
    }
  }
};
```

### 5. Prepare Environment Variables
Create a `.env` file with the following content:

```env
BACKEND_API_URL=https://your-agent-sdk-backend.com
CHAT_WIDGET_POSITION=bottom-right
CHAT_WIDGET_SIZE=60px
DEFAULT_FALLBACK_MESSAGE=I cannot answer based on the provided context
MAX_SELECTED_TEXT_LENGTH=2000
REQUEST_TIMEOUT_MS=30000
```

## Integration Steps

### 1. Add Chatbot Components
Create the following components in your Docusaurus project:

- `src/components/ChatWidget/` - Main chat interface component
- `src/components/FloatingChatIcon.js` - Floating icon component
- `src/components/TextSelectionHandler.js` - Text selection capture component
- `src/components/APIClient.js` - Backend communication service

### 2. Integrate into Layout
Add the floating chat icon to your main layout by modifying `src/pages/Layout.js` or by using Docusaurus swizzling:

```bash
npm run swizzle @docusaurus/Theme-classic Layout -- --eject
```

Then add the chat components to your layout file.

### 3. Build and Test Locally
```bash
npm run build
npm run serve
```

## Usage

### 1. Floating Chat Icon
- The chatbot icon will appear on all book pages
- Click the icon to open the chat interface
- The icon remains visible as users navigate through the book

### 2. Text Selection
- Select text from any part of the book content
- The selected text will be captured and can be used as context
- When submitting a query, selected text is automatically included

### 3. Chat Interaction
- Type your question in the chat input
- If you have selected text, it will be included as context
- Submit the query to get a response from the RAG agent
- Responses will be displayed in the chat interface with source attribution

## Configuration Options

- `BACKEND_API_URL`: URL of the FastAPI agent backend
- `CHAT_WIDGET_POSITION`: Position of floating widget (bottom-left, bottom-right)
- `CHAT_WIDGET_SIZE`: Size of floating widget in pixels
- `DEFAULT_FALLBACK_MESSAGE`: Message when context is insufficient
- `MAX_SELECTED_TEXT_LENGTH`: Maximum length of selected text to send
- `REQUEST_TIMEOUT_MS`: Timeout for API requests in milliseconds

## Deployment to GitHub Pages

### 1. Set up GitHub Actions
Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18

      - name: Install dependencies
        run: npm install

      - name: Build website
        run: npm run build
        env:
          BACKEND_API_URL: ${{ secrets.BACKEND_API_URL }}

      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./build
```

### 2. Configure GitHub Pages
- Go to your repository Settings > Pages
- Select Source: GitHub Actions
- The site will be deployed automatically on pushes to main

## Troubleshooting

### Common Issues

1. **Chat icon not appearing**: Check that components are properly integrated into the layout
2. **Backend connection errors**: Verify BACKEND_API_URL is correctly configured
3. **Text selection not working**: Check browser compatibility and event listeners
4. **Slow response times**: Verify FastAPI backend performance and network connectivity

### Testing Commands

Check frontend functionality:
```bash
npm run start
```

Verify backend connectivity:
```bash
curl -X GET https://your-backend-url/api/health
```

## Next Steps

- Monitor chatbot usage and response quality
- Gather user feedback on the chatbot experience
- Optimize performance based on usage patterns
- Add additional features based on user needs