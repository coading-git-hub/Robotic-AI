// Input validation utilities for the chatbot frontend

import { ValidationError } from './errorHandler';

// Validate query input
export const validateQueryInput = (query) => {
  if (!query) {
    throw new ValidationError('Query is required', 'query', query);
  }

  if (typeof query !== 'string') {
    throw new ValidationError('Query must be a string', 'query', typeof query);
  }

  if (query.trim().length === 0) {
    throw new ValidationError('Query cannot be empty or whitespace only', 'query', query);
  }

  if (query.length > 2000) {
    throw new ValidationError('Query exceeds maximum length of 2000 characters', 'query', query);
  }

  // Check for potentially harmful content (basic XSS prevention)
  if (/<script|<iframe|<object|<embed|on\w+\s*=|javascript:/i.test(query)) {
    throw new ValidationError('Query contains potentially harmful content', 'query', query);
  }

  return true;
};

// Validate selected text
export const validateSelectedTextInput = (selectedText) => {
  if (selectedText === null || selectedText === undefined) {
    return true; // Selected text is optional
  }

  if (typeof selectedText !== 'string') {
    throw new ValidationError('Selected text must be a string', 'selectedText', typeof selectedText);
  }

  if (selectedText.length > 2000) {
    throw new ValidationError('Selected text exceeds maximum length of 2000 characters', 'selectedText', selectedText);
  }

  // Check for potentially harmful content
  if (/<script|<iframe|<object|<embed|on\w+\s*=|javascript:/i.test(selectedText)) {
    throw new ValidationError('Selected text contains potentially harmful content', 'selectedText', selectedText);
  }

  return true;
};

// Validate message object
export const validateMessage = (message) => {
  if (!message) {
    throw new ValidationError('Message is required', 'message', message);
  }

  if (typeof message !== 'object') {
    throw new ValidationError('Message must be an object', 'message', typeof message);
  }

  if (!message.text || typeof message.text !== 'string') {
    throw new ValidationError('Message text is required and must be a string', 'message.text', message.text);
  }

  if (!message.sender || !['user', 'agent'].includes(message.sender)) {
    throw new ValidationError('Message sender must be either "user" or "agent"', 'message.sender', message.sender);
  }

  if (message.text.length > 5000) {
    throw new ValidationError('Message text exceeds maximum length of 5000 characters', 'message.text', message.text);
  }

  return true;
};

// Validate chat session
export const validateChatSession = (session) => {
  if (!session) {
    throw new ValidationError('Chat session is required', 'session', session);
  }

  if (typeof session !== 'object') {
    throw new ValidationError('Chat session must be an object', 'session', typeof session);
  }

  if (!session.sessionId) {
    throw new ValidationError('Session ID is required', 'session.sessionId', session.sessionId);
  }

  if (!Array.isArray(session.messages)) {
    throw new ValidationError('Session messages must be an array', 'session.messages', typeof session.messages);
  }

  return true;
};

// Validate API response
export const validateApiResponse = (response) => {
  if (!response) {
    throw new ValidationError('API response is required', 'response', response);
  }

  if (typeof response !== 'object') {
    throw new ValidationError('API response must be an object', 'response', typeof response);
  }

  if (typeof response.answer !== 'string') {
    throw new ValidationError('API response must contain an answer string', 'response.answer', typeof response.answer);
  }

  return true;
};

// Validate backend URL
export const validateBackendUrl = (url) => {
  if (!url) {
    throw new ValidationError('Backend URL is required', 'backendUrl', url);
  }

  if (typeof url !== 'string') {
    throw new ValidationError('Backend URL must be a string', 'backendUrl', typeof url);
  }

  try {
    new URL(url);
  } catch (e) {
    throw new ValidationError('Backend URL is not valid', 'backendUrl', url);
  }

  return true;
};

// Sanitize text input (remove potentially harmful content)
export const sanitizeTextInput = (text) => {
  if (!text || typeof text !== 'string') {
    return text;
  }

  // Remove potentially harmful HTML tags and attributes
  return text
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/<iframe\b[^<]*(?:(?!<\/iframe>)<[^<]*)*<\/iframe>/gi, '')
    .replace(/<object\b[^<]*(?:(?!<\/object>)<[^<]*)*<\/object>/gi, '')
    .replace(/<embed\b[^<]*(?:(?!<\/embed>)<[^<]*)*<\/embed>/gi, '')
    .replace(/<img[^>]*on\w+\s*=("[^"]*"|'[^']*')/gi, '')
    .replace(/<[^>]*on\w+\s*=("[^"]*"|'[^']*')[^>]*>/gi, '')
    .trim();
};

// Validate and sanitize user input
export const validateAndSanitizeInput = (query, selectedText = null) => {
  // Validate query
  validateQueryInput(query);

  // Validate selected text if provided
  if (selectedText !== null && selectedText !== undefined) {
    validateSelectedTextInput(selectedText);
  }

  // Sanitize inputs
  const sanitizedQuery = sanitizeTextInput(query);
  const sanitizedSelectedText = selectedText ? sanitizeTextInput(selectedText) : null;

  return {
    query: sanitizedQuery,
    selectedText: sanitizedSelectedText
  };
};