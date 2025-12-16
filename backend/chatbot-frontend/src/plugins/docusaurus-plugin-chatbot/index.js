// Docusaurus plugin for chatbot integration

const path = require('path');

module.exports = function (context, options) {
  const { siteConfig } = context;
  const config = {
    ...options,
    backendUrl: options.backendUrl || process.env.BACKEND_API_URL || 'http://localhost:8000',
    enableByDefault: options.enableByDefault !== false, // default to true
  };

  return {
    name: 'docusaurus-plugin-chatbot',

    getClientModules() {
      return [path.resolve(__dirname, './client/chatbot-container')];
    },

    configureWebpack(config, isServer, utils) {
      return {
        resolve: {
          alias: {
            '@chatbot': path.resolve(__dirname, '../../components'),
          },
        },
      };
    },

    injectHtmlTags() {
      return {
        postBodyTags: [
          // This will be rendered after the app mounts
          {
            tagName: 'div',
            attributes: {
              id: 'chatbot-container',
            },
          },
        ],
      };
    },
  };
};