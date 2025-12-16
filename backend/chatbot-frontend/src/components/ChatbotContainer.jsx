import React, { useState, useEffect } from 'react';
import FloatingChatIcon from './FloatingChatIcon';
import ChatWidget from './ChatWidget';
import TextSelectionHandler from './TextSelectionHandler';
import APIClient from './APIClient';
import { error as logError, info } from '../utils/logger';
import { handleError } from '../utils/errorHandler';
import config from '../utils/config';

// Main chatbot container component
const ChatbotContainer = () => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const [connectionStatus, setConnectionStatus] = useState('checking'); // 'checking', 'connected', 'disconnected'

  // Initialize API client with backend URL from environment or default
  const apiClient = new APIClient(
    process.env.REACT_APP_BACKEND_API_URL ||
    process.env.BACKEND_API_URL ||
    config.backendUrl ||
    'http://localhost:8000'
  );

  // Check backend connection on component mount
  useEffect(() => {
    const checkConnection = async () => {
      try {
        setConnectionStatus('checking');
        const healthResult = await apiClient.checkHealth();
        if (healthResult.success) {
          setConnectionStatus('connected');
          info('Successfully connected to backend API');
        } else {
          setConnectionStatus('disconnected');
          logError('Failed to connect to backend API', healthResult.error);
        }
      } catch (error) {
        setConnectionStatus('disconnected');
        logError('Error checking backend connection', error);
      }
    };

    checkConnection();
  }, []);

  // Handle toggling the chat open/closed
  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
  };

  // Handle closing the chat
  const closeChat = () => {
    setIsChatOpen(false);
  };

  // Handle text selection
  const handleTextSelected = (text) => {
    setSelectedText(text);
  };

  // Handle sending a message
  const handleSendMessage = async (query, contextText = null) => {
    // Add user message to the chat
    const userMessage = {
      id: Date.now(),
      text: query,
      sender: 'user'
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Send query to backend with selected text context
      const result = await apiClient.sendQuery(query, contextText || selectedText);

      if (result.success) {
        // Add agent response to the chat
        const agentMessage = {
          id: Date.now() + 1,
          text: result.data.answer,
          sender: 'agent',
          sources: result.data.sources,
          confidence: result.data.confidence,
          grounded_in_context: result.data.grounded_in_context
        };

        setMessages(prev => [...prev, agentMessage]);
      } else {
        // Handle error response using error handler
        const handledError = handleError(result.error, 'api-sendQuery');
        const errorMessage = {
          id: Date.now() + 1,
          text: handledError.message || 'Sorry, I encountered an error processing your request. Please try again.',
          sender: 'agent'
        };

        setMessages(prev => [...prev, errorMessage]);
        logError('API Error:', result.error);
      }
    } catch (error) {
      // Handle unexpected errors
      const handledError = handleError(error, 'unexpected-sendMessage');
      const errorMessage = {
        id: Date.now() + 1,
        text: handledError.message || 'Sorry, I encountered an unexpected error. Please try again.',
        sender: 'agent'
      };

      setMessages(prev => [...prev, errorMessage]);
      logError('Unexpected error in handleSendMessage:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Initialize with a welcome message
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          id: 1,
          text: 'Hello! I\'m your book assistant. You can ask me questions about the book content, or select text on the page to provide additional context.',
          sender: 'agent'
        }
      ]);
    }
  }, [messages.length]);

  // Add connection status indicator to messages if disconnected
  useEffect(() => {
    if (connectionStatus === 'disconnected' && messages.length > 0) {
      // Check if we already have a connection error message
      const hasConnectionError = messages.some(msg => msg.id === 'connection-error');

      if (!hasConnectionError) {
        const connectionErrorMessage = {
          id: 'connection-error',
          text: '⚠️ Note: The backend service is currently unavailable. Responses may be delayed or unavailable.',
          sender: 'system',
          isSystemMessage: true
        };

        setMessages(prev => [...prev, connectionErrorMessage]);
      }
    }
  }, [connectionStatus, messages]);

  return (
    <>
      <TextSelectionHandler onTextSelected={handleTextSelected} />
      <FloatingChatIcon onToggleChat={toggleChat} />
      <ChatWidget
        isOpen={isChatOpen}
        onClose={closeChat}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
        messages={messages}
        selectedText={selectedText}
      />
    </>
  );
};

export default ChatbotContainer;