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

  @media (max-width: 480px) {
    width: 50px;
    height: 50px;
    bottom: 15px;
    right: 15px;
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

  @media (max-width: 480px) {
    width: calc(100% - 40px);
    height: calc(100vh - 120px);
    bottom: 80px;
    right: 20px;
    left: 20px;
  }

  @media (max-width: 768px) {
    width: 340px;
    height: 450px;
    bottom: 80px;
    right: 15px;
  }
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

  @media (max-width: 480px) {
    max-width: 90%;
    padding: 10px 12px;
    font-size: 14px;
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

  @media (max-width: 480px) {
    padding: 10px 12px;
    font-size: 14px;
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

  @media (max-width: 480px) {
    padding: 10px 16px;
    font-size: 14px;
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
export const ChatWidget = ({ isOpen, onClose, backendUrl, selectedText: propSelectedText = '' }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentSelectedText, setCurrentSelectedText] = useState('');
  const [backendStatus, setBackendStatus] = useState('checking'); // 'checking', 'connected', 'disconnected'
  const [pulseOpacity, setPulseOpacity] = useState(1);
  const messagesEndRef = useRef(null);

  // Update currentSelectedText when propSelectedText changes
  useEffect(() => {
    setCurrentSelectedText(propSelectedText);
  }, [propSelectedText]);

  // Check backend status on component mount and periodically
  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const response = await fetch(`${backendUrl}/api/health`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' }
        });

        if (response.ok) {
          setBackendStatus('connected');
        } else {
          setBackendStatus('disconnected');
        }
      } catch (error) {
        setBackendStatus('disconnected');
      }
    };

    checkBackendStatus();

    // Set up periodic health check every 30 seconds
    const healthCheckInterval = setInterval(checkBackendStatus, 30000);

    return () => clearInterval(healthCheckInterval);
  }, [backendUrl]);

  // Handle pulse animation for checking status
  useEffect(() => {
    if (backendStatus !== 'checking') {
      setPulseOpacity(1);
      return;
    }

    let animationFrame;
    let start;

    const animate = (timestamp) => {
      if (!start) start = timestamp;
      const progress = timestamp - start;
      const cycleDuration = 1500; // 1.5 seconds
      const cycleProgress = (progress % cycleDuration) / cycleDuration;

      // Create a pulsing effect: go from 1 to 0.4 and back to 1
      const newOpacity = 1 - 0.6 * Math.abs(Math.sin(Math.PI * cycleProgress));
      setPulseOpacity(newOpacity);

      animationFrame = requestAnimationFrame(animate);
    };

    animationFrame = requestAnimationFrame(animate);

    return () => {
      if (animationFrame) {
        cancelAnimationFrame(animationFrame);
      }
    };
  }, [backendStatus]);

  // Scroll to bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle sending a message
  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    // Create the message content with selected text context if available
    const messageContent = inputValue;
    const userMessage = {
      id: Date.now(),
      content: messageContent,
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
          selected_text: currentSelectedText, // Use the current selected text
          user_preferences: {
            context_priority: 'selected_text_first',
            response_length: 'medium'
          }
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(`Backend error: ${response.status} - ${errorData.detail || response.statusText}`);
      }

      const data = await response.json();

      const agentMessage = {
        id: Date.now() + 1,
        content: data.answer || 'No response received from the agent.',
        sender: 'agent',
        timestamp: new Date().toISOString(),
        type: 'response',
        sources: data.sources || [],
        confidence: data.confidence
      };

      setMessages(prev => [...prev, agentMessage]);
      setCurrentSelectedText(''); // Clear selected text after sending
    } catch (error) {
      console.error('Error sending message:', error);

      let errorMessageContent = 'Sorry, I encountered an error processing your request. ';

      // Check if it's a network error or backend not running
      if (error.message.includes('fetch failed') || error.message.includes('Failed to fetch')) {
        errorMessageContent += 'The backend service may not be running. Please ensure the RAG agent API is accessible at: ' + backendUrl;
      } else {
        errorMessageContent += error.message;
      }

      const errorMessage = {
        id: Date.now() + 1,
        content: errorMessageContent,
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

  // Determine status indicator based on backend status
  const getStatusIndicator = () => {
    switch (backendStatus) {
      case 'connected':
        return (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '12px'
          }}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: '#4CAF50',
              boxShadow: '0 0 4px #4CAF50'
            }}></div>
            <span>Online</span>
          </div>
        );
      case 'disconnected':
        return (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '12px'
          }}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: '#F44336',
              boxShadow: '0 0 4px #F44336'
            }}></div>
            <span>Offline</span>
          </div>
        );
      case 'checking':
      default:
        return (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '12px'
          }}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: '#FFC107',
              animation: 'pulse 1.5s infinite'
            }}></div>
            <span>Checking</span>
          </div>
        );
    }
  };

  return (
    <ChatContainer role="dialog" aria-modal="true" aria-label="Book Assistant Chat">
      <ChatHeader>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <h3 style={{ margin: 0 }} aria-label="Chatbot title">Book Assistant</h3>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '10px',
            color: '#e0e0e0'
          }} aria-label={`Backend status: ${backendStatus}`}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: backendStatus === 'checking' ? '#FFC107' :
                             backendStatus === 'connected' ? '#4CAF50' : '#F44336',
              opacity: backendStatus === 'checking' ? pulseOpacity : 1,
              boxShadow: backendStatus === 'connected' ? '0 0 4px #4CAF50' :
                         backendStatus === 'disconnected' ? '0 0 4px #F44336' : 'none'
            }} aria-hidden="true"></div>
            <span>
              {backendStatus === 'connected' ? 'Online' :
               backendStatus === 'disconnected' ? 'Offline' : 'Checking'}
            </span>
          </div>
        </div>
        <CloseButton onClick={onClose} aria-label="Close chat">
          ×
        </CloseButton>
      </ChatHeader>

      <ChatMessages role="log" aria-live="polite" aria-label="Chat messages">
        {messages.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#999', marginTop: '20px' }}>
            Ask me anything about the book content!
          </div>
        ) : (
          messages.map((message) => (
            <MessageBubble
              key={message.id}
              isUser={message.sender === 'user'}
              role="listitem"
              aria-label={`${message.sender}: ${message.content}`}
            >
              {message.content}
              {message.sources && message.sources.length > 0 && (
                <div style={{ marginTop: '8px', fontSize: '0.8em', opacity: 0.8 }} aria-label="Sources">
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
          <MessageBubble isUser={false} aria-label="Loading response">
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <div style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: '#ccc',
                marginRight: '4px',
                animation: 'pulse 1.5s infinite'
              }} aria-hidden="true"></div>
              <div style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: '#ccc',
                marginRight: '4px',
                animation: 'pulse 1.5s infinite 0.3s'
              }} aria-hidden="true"></div>
              <div style={{
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                backgroundColor: '#ccc',
                animation: 'pulse 1.5s infinite 0.6s'
              }} aria-hidden="true"></div>
            </div>
          </MessageBubble>
        )}
        <div ref={messagesEndRef} aria-hidden="true" />
      </ChatMessages>

      {/* Show selected text indicator */}
      {currentSelectedText && (
        <div
          style={{
            padding: '8px 16px',
            backgroundColor: '#e8f4fd',
            border: '1px solid #2196f3',
            borderRadius: '4px',
            margin: '8px 16px',
            fontSize: '12px',
            color: '#1565c0'
          }}
          aria-label={`Selected text: ${currentSelectedText.substring(0, 100)}${currentSelectedText.length > 100 ? '...' : ''}`}
        >
          <strong>Selected text:</strong> "{currentSelectedText.substring(0, 100)}{currentSelectedText.length > 100 ? '...' : ''}"
          <button
            onClick={() => setCurrentSelectedText('')}
            style={{
              float: 'right',
              background: 'none',
              border: 'none',
              color: '#1565c0',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: 'bold'
            }}
            aria-label="Dismiss selected text"
          >
            ×
          </button>
          <div style={{ marginTop: '4px' }}>
            <button
              onClick={() => setInputValue(`Explain this: ${currentSelectedText.substring(0, 200)}${currentSelectedText.length > 200 ? '...' : ''}`)}
              style={{
                marginTop: '4px',
                padding: '4px 8px',
                backgroundColor: '#2196f3',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '11px'
              }}
              aria-label="Ask about selected text"
            >
              Ask about this
            </button>
          </div>
        </div>
      )}

      <ChatInput role="form" aria-label="Message input">
        <Input
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask about the book content..."
          disabled={isLoading}
          aria-label="Type your message"
        />
        <SendButton
          onClick={handleSendMessage}
          disabled={!inputValue.trim() || isLoading}
          aria-label="Send message"
        >
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
export const TextSelectionHandler = ({ onTextSelected, onOpenChat }) => {
  useEffect(() => {
    const handleSelection = () => {
      // Small delay to ensure selection is complete
      setTimeout(() => {
        const selectedText = window.getSelection().toString().trim();

        // Only trigger if there's actual selected text
        if (selectedText && selectedText.length > 0) {
          // Limit the length of selected text
          const limitedText = selectedText.length > 2000
            ? selectedText.substring(0, 2000) + '...'
            : selectedText;

          onTextSelected(limitedText);

          // Automatically open the chat if it's closed when text is selected
          if (onOpenChat) {
            onOpenChat();
          }
        }
      }, 10); // Small delay to ensure selection is complete
    };

    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('keyup', handleSelection); // For keyboard selection
    document.addEventListener('touchend', handleSelection); // For mobile touch selection

    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('keyup', handleSelection);
      document.removeEventListener('touchend', handleSelection);
    };
  }, [onTextSelected, onOpenChat]);

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
export const ChatbotIntegration = ({ backendUrl = 'http://localhost:8000' }) => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [capturedText, setCapturedText] = useState('');

  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
  };

  const openChat = () => {
    setIsChatOpen(true);
  };

  const closeChat = () => {
    setIsChatOpen(false);
  };

  const handleTextSelected = (text) => {
    setCapturedText(text);
  };

  return (
    <>
      <TextSelectionHandler onTextSelected={handleTextSelected} onOpenChat={openChat} />
      <FloatingChatIcon onToggleChat={toggleChat} />
      <ChatWidget
        isOpen={isChatOpen}
        onClose={closeChat}
        backendUrl={backendUrl}
        selectedText={capturedText}
      />
    </>
  );
};

export default ChatbotIntegration;