# RAG Chatbot Frontend Integration

This specification defines the implementation of RAG chatbot frontend integration into the Docusaurus book frontend. The system integrates a floating chatbot icon that opens a chat interface, captures selected text for context, communicates with the Agent SDK backend, and displays responses. The integrated system will be deployed on GitHub Pages.

## Overview

The system provides:
1. Floating chatbot icon visible on all book pages
2. Click-to-open chat interface with embedded widget
3. Text selection capture from book content
4. Communication with Agent SDK backend
5. Display of agent responses with source attribution
6. GitHub Pages deployment with backend connection

## Key Components

- FloatingChatIcon: Displays persistent chat icon on all pages
- ChatWidget: Main chat interface component
- TextSelectionHandler: Captures selected text from book content
- APIClient: Handles communication with Agent SDK backend
- Integration components for Docusaurus layout

## Files in this Specification

- `spec.md`: Feature requirements and user scenarios
- `plan.md`: Implementation architecture and design
- `research.md`: Technical research and decision documentation
- `data-model.md`: Data structure definitions
- `contracts/api-contract.md`: API interface specifications
- `quickstart.md`: Setup and usage instructions
- `checklists/`: Implementation checklists

## Technical Decisions

- **UI**: Embedded widget with click-to-open interface
- **Text Selection**: JavaScript-based capture using window.getSelection()
- **Backend**: Agent SDK API communication via fetch API
- **Deployment**: GitHub Pages for frontend, separate backend hosting
- **Architecture**: React components integrated into Docusaurus