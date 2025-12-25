import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatbotIntegration from '../components/Chatbot/Chatbot';
import { BetterAuthProvider } from '../components/auth/BetterAuthProvider';

// Get backend URL - using a safe approach for Docusaurus
// Since we're using a unified API, both chatbot and auth use the same backend
let BACKEND_URL = 'https://kiran-ahmed-phisical-ai.hf.space/api/health';

try {
  // In Docusaurus, environment variables are replaced at build time or we need to use a different approach
  // For now, default to localhost, but allow override through window object if needed
  BACKEND_URL =
    (typeof window !== 'undefined' && window.CHATBOT_BACKEND_URL) ||
    process.env.REACT_APP_BACKEND_URL ||
    process.env.BACKEND_URL ||
    'https://kiran-ahmed-phisical-ai.hf.space/api/health';
} catch (e) {
  // If there's any error accessing environment variables, default to localhost
  BACKEND_URL = 'https://kiran-ahmed-phisical-ai.hf.space/api/health';
}

export default function Layout(props) {
  return (
    <BetterAuthProvider backendUrl={BACKEND_URL}>
      <OriginalLayout {...props} />
      <ChatbotIntegration backendUrl={BACKEND_URL} />
    </BetterAuthProvider>
  );
}