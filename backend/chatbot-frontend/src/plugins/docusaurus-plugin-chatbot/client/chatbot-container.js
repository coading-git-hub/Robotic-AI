// Client-side chatbot container for Docusaurus integration

import React, { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import ChatbotContainer from '../../../components/ChatbotContainer';

// Function to initialize the chatbot
function initializeChatbot() {
  // Create a container element for the chatbot if it doesn't exist
  let container = document.getElementById('chatbot-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'chatbot-container';
    document.body.appendChild(container);
  }

  // Create a root and render the chatbot
  const root = createRoot(container);
  root.render(<ChatbotContainer />);

  // Clean up function to unmount the chatbot when needed
  return () => {
    root.unmount();
  };
}

// Initialize the chatbot when the DOM is ready
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    // Document is still loading, wait for DOMContentLoaded
    document.addEventListener('DOMContentLoaded', initializeChatbot);
  } else {
    // Document is already loaded, initialize immediately
    initializeChatbot();
  }
}

// Export the initialization function in case it's needed elsewhere
export { initializeChatbot };

// Export a React component that can be used directly in Docusaurus pages
export default function DocusaurusChatbot() {
  useEffect(() => {
    // This component is just a wrapper, the main initialization happens above
    // This is to ensure the chatbot is always available on all pages
  }, []);

  return null; // The chatbot renders outside of the normal React tree in the #chatbot-container
}