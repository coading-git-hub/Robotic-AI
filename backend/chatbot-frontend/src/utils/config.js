// Configuration utility for the chatbot frontend

// Default configuration values
const defaultConfig = {
  backendUrl: 'http://localhost:8000',
  chatWidgetPosition: 'bottom-right',
  chatWidgetSize: '60px',
  defaultFallbackMessage: 'I cannot answer based on the provided context',
  maxSelectedTextLength: 2000,
  apiTimeout: 30000, // 30 seconds
  maxMessageHistory: 50,
  enableLogging: true,
  logLevel: 'info' // 'debug', 'info', 'warn', 'error'
};

// Function to load configuration from environment variables
const loadConfig = () => {
  return {
    backendUrl: process.env.REACT_APP_BACKEND_API_URL ||
                process.env.BACKEND_API_URL ||
                defaultConfig.backendUrl,
    chatWidgetPosition: process.env.CHAT_WIDGET_POSITION || defaultConfig.chatWidgetPosition,
    chatWidgetSize: process.env.CHAT_WIDGET_SIZE || defaultConfig.chatWidgetSize,
    defaultFallbackMessage: process.env.REACT_APP_DEFAULT_FALLBACK_MESSAGE ||
                           process.env.DEFAULT_FALLBACK_MESSAGE ||
                           defaultConfig.defaultFallbackMessage,
    maxSelectedTextLength: parseInt(process.env.MAX_SELECTED_TEXT_LENGTH) ||
                          defaultConfig.maxSelectedTextLength,
    apiTimeout: parseInt(process.env.API_TIMEOUT) || defaultConfig.apiTimeout,
    maxMessageHistory: defaultConfig.maxMessageHistory,
    enableLogging: defaultConfig.enableLogging,
    logLevel: process.env.LOG_LEVEL || defaultConfig.logLevel
  };
};

// Export the loaded configuration
const config = loadConfig();

export default config;

// Export utility functions
export const getBackendUrl = () => config.backendUrl;
export const getChatWidgetPosition = () => config.chatWidgetPosition;
export const getChatWidgetSize = () => config.chatWidgetSize;
export const getDefaultFallbackMessage = () => config.defaultFallbackMessage;
export const getMaxSelectedTextLength = () => config.maxSelectedTextLength;
export const getApiTimeout = () => config.apiTimeout;
export const getMaxMessageHistory = () => config.maxMessageHistory;
export const isLoggingEnabled = () => config.enableLogging;
export const getLogLevel = () => config.logLevel;