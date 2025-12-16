import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatbotIntegration from '../components/Chatbot/Chatbot';

// Get backend URL - using a safe approach for Docusaurus
// Since process.env might not be available in the browser, we use a try-catch approach
let BACKEND_URL = 'http://localhost:8002';

try {
  // In Docusaurus, environment variables are replaced at build time or we need to use a different approach
  // For now, default to localhost, but allow override through window object if needed
  BACKEND_URL =
    (typeof window !== 'undefined' && window.CHATBOT_BACKEND_URL) ||
    'http://localhost:8002';
} catch (e) {
  // If there's any error accessing environment variables, default to localhost
  BACKEND_URL = 'http://localhost:8002';
}

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props} />
      <ChatbotIntegration backendUrl={BACKEND_URL} />
    </>
  );
}