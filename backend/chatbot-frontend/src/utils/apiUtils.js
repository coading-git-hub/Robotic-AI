// API communication utilities for the chatbot frontend

import axios from 'axios';
import { handleHttpError, ChatbotError } from './errorHandler';
import { validateQueryInput, validateSelectedTextInput, validateBackendUrl } from './validation';
import { debug, info, warn, error } from './logger';
import config from './config';

// Create an axios instance with default configuration
const createApiClient = (backendUrl) => {
  return axios.create({
    baseURL: backendUrl,
    timeout: config.apiTimeout || 30000, // 30 seconds default
    headers: {
      'Content-Type': 'application/json',
    },
  });
};

// Function to send a query to the backend agent API
export const sendQueryToBackend = async (query, selectedText = null, backendUrl = null) => {
  debug('Sending query to backend', { query: query.substring(0, 50) + '...', selectedText: !!selectedText });

  try {
    // Validate inputs
    validateQueryInput(query);
    if (selectedText) {
      validateSelectedTextInput(selectedText);
    }

    // Use provided backend URL or default
    const url = backendUrl || config.backendUrl || process.env.REACT_APP_BACKEND_API_URL || 'http://localhost:8000';
    validateBackendUrl(url);

    // Create API client
    const apiClient = createApiClient(url);

    // Prepare the request payload
    const payload = {
      query: query,
      selected_text: selectedText
    };

    // Send the request
    const response = await apiClient.post('/api/agent/query', payload);

    info('Query response received', { status: response.status });

    return {
      success: true,
      data: response.data
    };
  } catch (error) {
    const handledError = handleHttpError(error, backendUrl);
    error('Query request failed', { error: handledError.message, query: query.substring(0, 50) + '...' });

    return {
      success: false,
      error: handledError
    };
  }
};

// Function to check backend health
export const checkBackendHealth = async (backendUrl = null) => {
  debug('Checking backend health');

  try {
    // Use provided backend URL or default
    const url = backendUrl || config.backendUrl || process.env.REACT_APP_BACKEND_API_URL || 'http://localhost:8000';
    validateBackendUrl(url);

    // Create API client
    const apiClient = createApiClient(url);

    // Send health check request
    const response = await apiClient.get('/api/health');

    info('Health check successful', { status: response.status });

    return {
      success: true,
      data: response.data
    };
  } catch (error) {
    const handledError = handleHttpError(error, backendUrl);
    warn('Health check failed', { error: handledError.message });

    return {
      success: false,
      error: handledError
    };
  }
};

// Function to validate agent response (for development/testing)
export const validateAgentResponse = async (query, context, response, backendUrl = null) => {
  debug('Validating agent response', { query: query.substring(0, 30) + '...' });

  try {
    // Validate inputs
    validateQueryInput(query);
    if (typeof context !== 'string' || context.trim().length === 0) {
      throw new ChatbotError('Context is required for validation', 'VALIDATION_ERROR');
    }
    if (typeof response !== 'string' || response.trim().length === 0) {
      throw new ChatbotError('Response is required for validation', 'VALIDATION_ERROR');
    }

    // Use provided backend URL or default
    const url = backendUrl || config.backendUrl || process.env.REACT_APP_BACKEND_API_URL || 'http://localhost:8000';
    validateBackendUrl(url);

    // Create API client
    const apiClient = createApiClient(url);

    // Prepare the validation payload
    const payload = {
      query: query,
      context: context,
      response: response
    };

    // Send validation request
    const validationResponse = await apiClient.post('/api/agent/validate', payload);

    info('Response validation completed', { status: validationResponse.status });

    return {
      success: true,
      data: validationResponse.data
    };
  } catch (error) {
    const handledError = handleHttpError(error, backendUrl);
    error('Response validation failed', { error: handledError.message });

    return {
      success: false,
      error: handledError
    };
  }
};

// Function to implement retry logic for API calls
export const callWithRetry = async (apiFunction, maxRetries = 3, delay = 1000) => {
  let lastError;

  for (let i = 0; i < maxRetries; i++) {
    try {
      const result = await apiFunction();
      if (result.success) {
        return result;
      }
      lastError = result.error;
    } catch (error) {
      lastError = error;
    }

    // If not the last attempt, wait before retrying
    if (i < maxRetries - 1) {
      info(`API call failed, retrying in ${delay}ms`, { attempt: i + 1, maxRetries });
      await new Promise(resolve => setTimeout(resolve, delay));
      // Exponential backoff: double the delay for the next retry
      delay *= 2;
    }
  }

  error('API call failed after all retries', { maxRetries, lastError: lastError.message });
  return {
    success: false,
    error: lastError
  };
};

// Function to get API status and capabilities
export const getApiCapabilities = async (backendUrl = null) => {
  debug('Getting API capabilities');

  try {
    // Use provided backend URL or default
    const url = backendUrl || config.backendUrl || process.env.REACT_APP_BACKEND_API_URL || 'http://localhost:8000';
    validateBackendUrl(url);

    // Create API client
    const apiClient = createApiClient(url);

    // For now, just check if the API is reachable by checking health
    const response = await apiClient.get('/api/health');

    return {
      success: true,
      capabilities: {
        agentQuery: true,
        healthCheck: true,
        responseValidation: true
      },
      version: response.data?.version || 'unknown',
      status: response.data?.status || 'unknown'
    };
  } catch (error) {
    const handledError = handleHttpError(error, backendUrl);
    warn('Could not get API capabilities', { error: handledError.message });

    return {
      success: false,
      capabilities: {},
      error: handledError
    };
  }
};

// Function to format API responses for the UI
export const formatApiResponse = (apiResponse) => {
  if (!apiResponse || !apiResponse.success) {
    return {
      answer: apiResponse?.error?.message || 'Sorry, I could not process your request.',
      sources: [],
      confidence: 0,
      groundedInContext: false
    };
  }

  return {
    answer: apiResponse.data?.answer || 'No answer provided',
    sources: apiResponse.data?.sources || [],
    confidence: apiResponse.data?.confidence || 0,
    groundedInContext: apiResponse.data?.grounded_in_context || false
  };
};