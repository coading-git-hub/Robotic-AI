import React, { useState } from 'react';
import styled from 'styled-components';

// Styled component for the floating chat icon
const FloatingIcon = styled.div`
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 4px 20px rgba(79, 70, 229, 0.4);
  z-index: 1000;
  font-size: 24px;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  border: none;
  outline: none;
  user-select: none;
  -webkit-tap-highlight-color: transparent;

  &:hover {
    transform: scale(1.1) rotate(10deg);
    box-shadow: 0 6px 25px rgba(79, 70, 229, 0.6);
    background: linear-gradient(135deg, #4338CA 0%, #6D28D9 100%);
  }

  &:active {
    transform: scale(0.95) rotate(0deg);
  }

  &:focus {
    outline: 2px solid #C7D2FE;
    outline-offset: 2px;
  }

  @media (max-width: 768px) {
    bottom: 15px;
    right: 15px;
    width: 50px;
    height: 50px;
  }
`;

// SVG icon for the chat bubble
const ChatIcon = () => (
  <svg
    width="30"
    height="30"
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    focusable="false"
    aria-hidden="true"
  >
    <path
      d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H16.58L19.5 21.95C19.61 22.15 19.6 22.39 19.48 22.59C19.37 22.79 19.16 22.9 18.94 22.89C18.72 22.88 18.52 22.75 18.4 22.55L15 17H5C4.46957 17 3.96086 16.7893 3.58579 16.4142C3.21071 16.0391 3 15.5304 3 15V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H19C19.5304 3 20.0391 3.21071 20.4142 3.58579C20.7893 3.96086 21 4.46957 21 5V15Z"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

// FloatingChatIcon component
const FloatingChatIcon = ({ onToggleChat }) => {
  const [isVisible, setIsVisible] = useState(true);

  const handleClick = () => {
    onToggleChat();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onToggleChat();
    }
  };

  return (
    <FloatingIcon
      onClick={handleClick}
      aria-label="Open chatbot assistant"
      role="button"
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      <ChatIcon />
    </FloatingIcon>
  );
};

export default FloatingChatIcon;