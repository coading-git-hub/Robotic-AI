/*
 * RAG Chatbot Frontend Components
 *
 * This file contains the React components for the RAG chatbot frontend integration.
 * These components include:
 * - FloatingChatIcon: A floating icon that opens the chat widget
 * - ChatWidget: The main chat interface
 * - TextSelectionHandler: Captures selected text from the page
 * - APIClient: Handles communication with the Agent SDK backend
 */

import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';

// Styled components for the chat interface
const FloatingIcon = styled.div`
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 1000;
  transition: all 0.3s ease;

  &:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
  }

  &:active {
    transform: scale(0.95);
  }
`;

const ChatContainer = styled.div`
  position: fixed;
  bottom: 100px;
  right: 20px;
  width: 380px;
  height: 500px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
  display: flex;
  flex-direction: column;
  z-index: 1000;
  overflow: hidden;
  transition: all 0.3s ease;
`;

const ChatHeader = styled.div`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const ChatMessages = styled.div`
  flex: 1;
  padding: 16px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background-color: #f9f9f9;
`;

const MessageBubble = styled.div`
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 18px;
  line-height: 1.4;

  ${({ isUser }) => isUser
    ? `
      align-self: flex-end;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border-bottom-right-radius: 4px;
    `
    : `
      align-self: flex-start;
      background: white;
      color: #333;
      border: 1px solid #e0e0e0;
      border-bottom-left-radius: 4px;
    `
  }
`;

const ChatInput = styled.div`
  display: flex;
  padding: 16px;
  background: white;
  border-top: 1px solid #e0e0e0;
`;

const Input = styled.input`
  flex: 1;
  padding: 12px 16px;
  border: 1px solid #e0e0e0;
  border-radius: 24px;
  outline: none;
  font-size: 14px;

  &:focus {
    border-color: #667eea;
  }
`;

const SendButton = styled.button`
  margin-left: 8px;
  padding: 12px 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 24px;
  cursor: pointer;
  font-weight: 600;

  &:hover {
    opacity: 0.9;
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  color: white;
  font-size: 18px;
  cursor: pointer;
  padding: 0;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    opacity: 0.8;
  }
`;

// Floating Chat Icon Component
export const FloatingChatIcon = ({ onToggleChat }) => {
  return (
    <FloatingIcon onClick={onToggleChat} aria-label="Open chatbot">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H16.5C16.2348 17 15.9804 16.8946 15.7929 16.7071C15.6054 16.5196 15.5 16.2652 15.5 16V8C15.5 7.73478 15.6054 7.48043 15.7929 7.29289C15.9804 7.10536 16.2348 7 16.5 7H19C19.5304 7 20.0391 7.21071 20.4142 7.58579C20.7893 7.96086 21 8.46957 21 9V15ZM13.5 9C13.5 8.46957 13.2893 7.96086 12.9142 7.58579C12.5391 7.21071 12.0304 7 11.5 7H3C2.46957 7 1.96086 7.21071 1.58579 7.58579C1.21071 7.96086 1 8.46957 1 9V17C1 17.5304 1.21071 18.0391 1.58579 18.4142C1.96086 18.7893 2.46957 19 3 19H11.5C12.0304 19 12.5391 18.7893 12.9142 18.4142C13.2893 18.0391 13.5 17.5304 13.5 17V9Z" fill="currentColor"/>
      </svg>
    </FloatingIcon>
  );
};

// Chat Widget Component
export const ChatWidget = ({ isOpen, onClose, backendUrl }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef(null);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle sending a message
  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      content: inputValue,
      sender: 'user',
      timestamp: new Date().toISOString(),
      type: 'query'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Send query to Agent SDK backend
      const response = await fetch(`${backendUrl}/api/agent/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: inputValue,
          selected_text: selectedText,
          user_preferences: {
            context_priority: 'selected_text_first',
            response_length: 'medium'
          }
        })
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }

      const data = await response.json();

      const agentMessage = {
        id: Date.now() + 1,
        content: data.answer,
        sender: 'agent',
        timestamp: new Date().toISOString(),
        type: 'response',
        sources: data.sources,
        confidence: data.confidence
      };

      setMessages(prev => [...prev, agentMessage]);
      setSelectedText(''); // Clear selected text after sending
    } catch (error) {
      console.error('Error sending message:', error);

      const errorMessage = {
        id: Date.now() + 1,
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        sender: 'agent',
        timestamp: new Date().toISOString(),
        type: 'response'
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Handle Enter key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!isOpen) return null;

  return (
    <ChatContainer>
      <ChatHeader>
        <h3>Book Assistant</h3>
        <CloseButton onClick={onClose} aria-label="Close chat">
          ×
        </CloseButton>
      </ChatHeader>

      <ChatMessages>
        {messages.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#999', marginTop: '20px' }}>
            Ask me anything about the book content!
          </div>
        ) : (
          messages.map((message) => (
            <MessageBubble key={message.id} isUser={message.sender === 'user'}>
              {message.content}
              {message.sources && message.sources.length > 0 && (
                <div style={{ marginTop: '8px', fontSize: '0.8em', opacity: 0.8 }}>
                  Sources: {message.sources.slice(0, 2).map((source, idx) =>
                    <span key={idx} title={source.title}>
                      {source.title.substring(0, 20)}{source.title.length > 20 ? '...' : ''}
                      {idx < Math.min(2, message.sources.length) - 1 ? ', ' : ''}
                    </span>
                  )}
                </div>
              )}
            </MessageBubble>
          ))
        )}
        {isLoading && (
          <MessageBubble isUser={false}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: '#ccc',
                marginRight: '4px',
                animation: 'pulse 1.5s infinite'
              }}></div>
              <div style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: '#ccc',
                marginRight: '4px',
                animation: 'pulse 1.5s infinite 0.3s'
              }}></div>
              <div style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: '#ccc',
                animation: 'pulse 1.5s infinite 0.6s'
              }}></div>
            </div>
          </MessageBubble>
        )}
        <div ref={messagesEndRef} />
      </ChatMessages>

      <ChatInput>
        <Input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about the book content..."
          disabled={isLoading}
        />
        <SendButton onClick={handleSendMessage} disabled={!inputValue.trim() || isLoading}>
          Send
        </SendButton>
      </ChatInput>

      <style jsx>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </ChatContainer>
  );
};

// Text Selection Handler Component
export const TextSelectionHandler = ({ onTextSelected }) => {
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();

      // Only trigger if there's actual selected text
      if (selectedText && selectedText.length > 0) {
        // Limit the length of selected text
        const limitedText = selectedText.length > 2000
          ? selectedText.substring(0, 2000) + '...'
          : selectedText;

        onTextSelected(limitedText);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('keyup', handleSelection); // For keyboard selection

    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('keyup', handleSelection);
    };
  }, [onTextSelected]);

  return null; // This component doesn't render anything
};

// API Client Service Component
export const APIClient = ({ backendUrl }) => {
  const sendQuery = async (query, selectedText = '') => {
    try {
      const response = await fetch(`${backendUrl}/api/agent/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          query,
          selected_text: selectedText,
          user_preferences: {
            context_priority: 'selected_text_first',
            response_length: 'medium'
          }
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API Client error:', error);
      throw error;
    }
  };

  const checkHealth = async () => {
    try {
      const response = await fetch(`${backendUrl}/api/health`);
      if (!response.ok) {
        throw new Error(`Health check failed with status: ${response.status}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  };

  return {
    sendQuery,
    checkHealth
  };
};

// Main Integration Component
export const ChatbotIntegration = ({ backendUrl = 'https://your-agent-sdk-backend.com' }) => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [capturedText, setCapturedText] = useState('');

  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
  };

  const handleTextSelected = (text) => {
    setCapturedText(text);
  };

  return (
    <>
      <TextSelectionHandler onTextSelected={handleTextSelected} />
      <FloatingChatIcon onToggleChat={toggleChat} />
      <ChatWidget
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
        backendUrl={backendUrl}
        selectedText={capturedText}
      />
    </>
  );
};

export default ChatbotIntegration;