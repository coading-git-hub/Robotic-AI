// Logging utility for the chatbot frontend

import config from './config';

// Log levels
const LOG_LEVELS = {
  DEBUG: 0,
  INFO: 1,
  WARN: 2,
  ERROR: 3
};

// Get current log level
const getCurrentLogLevel = () => {
  const level = config.getLogLevel ? config.getLogLevel() : config.logLevel || 'info';
  return LOG_LEVELS[level.toUpperCase()] || LOG_LEVELS.INFO;
};

// Check if logging is enabled for the given level
const isLoggingEnabled = (level) => {
  if (!config.enableLogging && !config.isLoggingEnabled?.()) {
    return false;
  }
  return LOG_LEVELS[level] >= getCurrentLogLevel();
};

// Format log message
const formatMessage = (level, message, meta = {}) => {
  const timestamp = new Date().toISOString();
  const formattedMessage = `[${timestamp}] ${level}: ${message}`;

  if (Object.keys(meta).length > 0) {
    return `${formattedMessage} - Meta: ${JSON.stringify(meta)}`;
  }

  return formattedMessage;
};

// Log function
const log = (level, message, meta = {}) => {
  if (!isLoggingEnabled(level)) {
    return;
  }

  const formattedMessage = formatMessage(level, message, meta);

  switch (level) {
    case 'DEBUG':
      console.debug(formattedMessage);
      break;
    case 'INFO':
      console.info(formattedMessage);
      break;
    case 'WARN':
      console.warn(formattedMessage);
      break;
    case 'ERROR':
      console.error(formattedMessage);
      break;
    default:
      console.log(formattedMessage);
  }
};

// Export log functions
export const debug = (message, meta = {}) => log('DEBUG', message, meta);
export const info = (message, meta = {}) => log('INFO', message, meta);
export const warn = (message, meta = {}) => log('WARN', message, meta);
export const error = (message, meta = {}) => log('ERROR', message, meta);

// Create a logger instance
const logger = {
  debug,
  info,
  warn,
  error,
  isLoggingEnabled: (level) => isLoggingEnabled(level),
  formatMessage
};

export default logger;