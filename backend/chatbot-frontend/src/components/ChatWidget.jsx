import React, { useState, useRef, useEffect } from 'react';
import styled from 'styled-components';

// Styled components for the chat widget
const ChatContainer = styled.div`
  position: fixed;
  bottom: 90px;
  right: 20px;
  width: 380px;
  height: 550px;
  background-color: white;
  border-radius: 16px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
  display: flex;
  flex-direction: column;
  z-index: 1000;
  overflow: hidden;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  border: 1px solid #e5e7eb;

  @media (max-width: 768px) {
    width: 320px;
    height: 500px;
    bottom: 85px;
    right: 15px;
  }
`;

const ChatHeader = styled.div`
  background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
  color: white;
  padding: 18px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
`;

const ChatTitle = styled.h3`
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const CloseButton = styled.button`
  background: rgba(255, 255, 255, 0.2);
  border: none;
  color: white;
  font-size: 20px;
  cursor: pointer;
  padding: 4px;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
  transition: background-color 0.2s;

  &:hover {
    background: rgba(255, 255, 255, 0.3);
  }

  &:focus {
    outline: 2px solid rgba(255, 255, 255, 0.5);
    outline-offset: 2px;
  }
`;

const ChatMessages = styled.div`
  flex: 1;
  padding: 16px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 14px;
  background: linear-gradient(180deg, #fafafa 0%, #ffffff 100%);
  -webkit-overflow-scrolling: touch;

  &::-webkit-scrollbar {
    width: 6px;
  }

  &::-webkit-scrollbar-track {
    background: #f1f5f9;
    border-radius: 3px;
  }

  &::-webkit-scrollbar-thumb {
    background: #c7d2fe;
    border-radius: 3px;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: #a5b4fc;
  }
`;

const Message = styled.div`
  max-width: 85%;
  padding: 12px 16px;
  border-radius: 18px;
  font-size: 14px;
  line-height: 1.5;
  word-wrap: break-word;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  animation: fadeIn 0.3s ease-out;

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  ${(props) => props.sender === 'user' && `
    align-self: flex-end;
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
    color: white;
    border-bottom-right-radius: 4px;
  `}

  ${(props) => props.sender === 'agent' && `
    align-self: flex-start;
    background-color: white;
    color: #1f2937;
    border: 1px solid #e5e7eb;
    border-bottom-left-radius: 4px;
  `}

  ${(props) => props.sender === 'system' && `
    align-self: center;
    background-color: #fef3c7;
    color: #92400e;
    border: 1px solid #fbbf24;
    border-radius: 12px;
    font-size: 12px;
    font-style: italic;
    text-align: center;
    max-width: 95%;
  `}
`;

const SourceAttribution = styled.div`
  margin-top: 8px;
  padding-top: 6px;
  border-top: 1px solid #e5e7eb;
  font-size: 11px;
  color: #6b7280;

  a {
    color: #4F46E5;
    text-decoration: none;

    &:hover {
      text-decoration: underline;
    }
  }
`;

const ChatInputArea = styled.div`
  padding: 16px;
  border-top: 1px solid #e5e7eb;
  background-color: white;
  display: flex;
  flex-direction: column;
  gap: 12px;
`;

const InputContainer = styled.div`
  display: flex;
  gap: 8px;
  margin-bottom: 4px;
`;

const SelectedTextDisplay = styled.div`
  background-color: #eff6ff;
  border: 1px solid #3b82f6;
  border-radius: 10px;
  padding: 10px 12px;
  margin-bottom: 8px;
  font-size: 12px;
  color: #1d4ed8;
  max-height: 80px;
  overflow-y: auto;
  position: relative;

  &::before {
    content: 'Context: ';
    font-weight: 600;
    color: #1e40af;
  }

  &:hover {
    background-color: #dbeafe;
  }
`;

const ChatInput = styled.textarea`
  flex: 1;
  padding: 12px 14px;
  border: 1px solid #d1d5db;
  border-radius: 12px;
  resize: none;
  font-size: 14px;
  font-family: inherit;
  min-height: 70px;
  max-height: 150px;
  transition: border-color 0.2s, box-shadow 0.2s;

  &:focus {
    outline: none;
    border-color: #6366F1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
  }

  &:disabled {
    background-color: #f9fafb;
    cursor: not-allowed;
  }
`;

const SendButton = styled.button`
  background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 12px 20px;
  cursor: pointer;
  font-weight: 500;
  font-size: 14px;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  align-self: flex-end;
  min-width: 80px;

  &:hover:not(:disabled) {
    background: linear-gradient(135deg, #4338CA 0%, #5A67D8 100%);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
  }

  &:active:not(:disabled) {
    transform: translateY(0);
  }

  &:disabled {
    background: #d1d5db;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }
`;

const LoadingIndicator = styled.div`
  align-self: flex-start;
  background-color: white;
  color: #1f2937;
  padding: 12px 16px;
  border-radius: 18px;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 8px;
  border: 1px solid #e5e7eb;

  div {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #4F46E5;
    animation: bounce 1.5s infinite ease-in-out;

    &:nth-child(2) {
      animation-delay: 0.2s;
    }

    &:nth-child(3) {
      animation-delay: 0.4s;
    }
  }

  @keyframes bounce {
    0%, 80%, 100% { transform: scale(0); }
    40% { transform: scale(1.0); }
  }
`;

// ChatWidget component
const ChatWidget = ({ isOpen, onClose, onSendMessage, isLoading, messages, selectedText }) => {
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSend = () => {
    if (inputValue.trim() && !isLoading) {
      onSendMessage(inputValue, selectedText);
      setInputValue('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Escape') {
      onClose();
    }
  };

  if (!isOpen) {
    return null;
  }

  return (
    <ChatContainer onKeyDown={handleKeyDown}>
      <ChatHeader>
        <ChatTitle>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path
              d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H16.58L19.5 21.95C19.61 22.15 19.6 22.39 19.48 22.59C19.37 22.79 19.16 22.9 18.94 22.89C18.72 22.88 18.52 22.75 18.4 22.55L15 17H5C4.46957 17 3.96086 16.7893 3.58579 16.4142C3.21071 16.0391 3 15.5304 3 15V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H19C19.5304 3 20.0391 3.21071 20.4142 3.58579C20.7893 3.96086 21 4.46957 21 5V15Z"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          Book Assistant
        </ChatTitle>
        <CloseButton onClick={onClose} aria-label="Close chat" title="Close chat (Esc)">
          ×
        </CloseButton>
      </ChatHeader>

      <ChatMessages>
        {messages.map((message, index) => (
          <Message key={`${message.id || index}-${message.sender}`} sender={message.sender}>
            {message.text}
            {message.sources && message.sources.length > 0 && (
              <SourceAttribution>
                Sources: {message.sources.map((source, idx) => (
                  <span key={idx}>
                    {idx > 0 && ', '}
                    <a href={source.url} target="_blank" rel="noopener noreferrer">
                      {source.title || 'Source'}
                    </a>
                  </span>
                ))}
              </SourceAttribution>
            )}
          </Message>
        ))}
        {isLoading && (
          <LoadingIndicator>
            <div></div>
            <div></div>
            <div></div>
          </LoadingIndicator>
        )}
        <div ref={messagesEndRef} />
      </ChatMessages>

      <ChatInputArea>
        {selectedText && (
          <SelectedTextDisplay title="Selected text context">
            {selectedText.length > 120 ? selectedText.substring(0, 120) + '...' : selectedText}
          </SelectedTextDisplay>
        )}

        <InputContainer>
          <ChatInput
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about the book content..."
            disabled={isLoading}
            aria-label="Type your question"
          />
        </InputContainer>

        <SendButton onClick={handleSend} disabled={!inputValue.trim() || isLoading}>
          {isLoading ? (
            <span>Sending...</span>
          ) : (
            <span>
              Send <span aria-hidden="true">→</span>
            </span>
          )}
        </SendButton>
      </ChatInputArea>
    </ChatContainer>
  );
};

export default ChatWidget;