// Error handling utilities for the chatbot frontend

import { error as logError } from './logger';

// Custom error classes
export class ChatbotError extends Error {
  constructor(message, code = 'CHATBOT_ERROR', originalError = null) {
    super(message);
    this.name = 'ChatbotError';
    this.code = code;
    this.originalError = originalError;
    this.timestamp = new Date();
  }
}

export class APIError extends ChatbotError {
  constructor(message, status = null, response = null) {
    super(message, 'API_ERROR');
    this.name = 'APIError';
    this.status = status;
    this.response = response;
  }
}

export class ValidationError extends ChatbotError {
  constructor(message, field = null, value = null) {
    super(message, 'VALIDATION_ERROR');
    this.name = 'ValidationError';
    this.field = field;
    this.value = value;
  }
}

export class NetworkError extends ChatbotError {
  constructor(message, url = null, originalError = null) {
    super(message, 'NETWORK_ERROR');
    this.name = 'NetworkError';
    this.url = url;
    this.originalError = originalError;
  }
}

// Error handler function
export const handleError = (error, context = '') => {
  // Log the error
  logError(`Error in ${context || 'unknown context'}`, {
    message: error.message,
    name: error.name,
    stack: error.stack,
    code: error.code,
    timestamp: new Date().toISOString()
  });

  // Return a user-friendly error based on error type
  if (error instanceof APIError) {
    if (error.status === 404) {
      return new ChatbotError('The service is temporarily unavailable. Please try again later.', 'SERVICE_UNAVAILABLE');
    } else if (error.status >= 500) {
      return new ChatbotError('The server encountered an error. Please try again later.', 'SERVER_ERROR');
    } else if (error.status === 429) {
      return new ChatbotError('Too many requests. Please wait before trying again.', 'RATE_LIMITED');
    }
  } else if (error instanceof NetworkError) {
    return new ChatbotError('Unable to connect to the service. Please check your internet connection.', 'CONNECTION_ERROR');
  } else if (error instanceof ValidationError) {
    return new ChatbotError(`Invalid input: ${error.message}`, 'VALIDATION_ERROR');
  }

  // Default error for unknown errors
  return new ChatbotError('An unexpected error occurred. Please try again.', 'UNKNOWN_ERROR');
};

// Error boundary component for React
export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });

    // Log the error
    logError('Error caught by boundary', {
      error: error.toString(),
      errorInfo: errorInfo.toString()
    });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: '20px', textAlign: 'center', color: '#dc2626' }}>
          <h3>Something went wrong.</h3>
          <p>We're sorry, but an error occurred in the chat interface.</p>
          <button
            onClick={() => this.setState({ hasError: false, error: null, errorInfo: null })}
            style={{
              padding: '8px 16px',
              backgroundColor: '#4F46E5',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// Validation utility functions
export const validateQuery = (query) => {
  if (!query || typeof query !== 'string') {
    throw new ValidationError('Query is required and must be a string', 'query', query);
  }

  if (query.trim().length === 0) {
    throw new ValidationError('Query cannot be empty', 'query', query);
  }

  if (query.length > 2000) { // Max length from spec
    throw new ValidationError('Query is too long', 'query', query);
  }

  return true;
};

export const validateSelectedText = (selectedText) => {
  if (selectedText && typeof selectedText !== 'string') {
    throw new ValidationError('Selected text must be a string', 'selectedText', selectedText);
  }

  if (selectedText && selectedText.length > 2000) { // Max length from spec
    throw new ValidationError('Selected text is too long', 'selectedText', selectedText);
  }

  return true;
};

export const validateBackendUrl = (url) => {
  if (!url || typeof url !== 'string') {
    throw new ValidationError('Backend URL is required and must be a string', 'backendUrl', url);
  }

  try {
    new URL(url);
  } catch (e) {
    throw new ValidationError('Backend URL is not valid', 'backendUrl', url);
  }

  return true;
};

// HTTP error handling
export const handleHttpError = (error, url) => {
  if (error.response) {
    // Server responded with error status
    const { status, data } = error.response;
    let message = `HTTP Error ${status}`;

    if (data && typeof data === 'object') {
      message = data.message || data.detail || message;
    } else if (typeof data === 'string') {
      message = data;
    }

    return new APIError(message, status, data);
  } else if (error.request) {
    // Request was made but no response received
    return new NetworkError('No response received from server', url, error);
  } else {
    // Something else happened
    return new NetworkError(error.message, url, error);
  }
};