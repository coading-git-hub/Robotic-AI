// Docusaurus preset that includes the chatbot plugin

module.exports = function (context, options = {}) {
  const { siteConfig } = context;

  return {
    themes: [
      // Include any themes that might be needed
      ...(siteConfig.themes || []),
    ],
    plugins: [
      // Include the chatbot plugin
      [require.resolve('../plugins/docusaurus-plugin-chatbot'), options.chatbot || {}],

      // Include other plugins as needed
      ...(siteConfig.plugins || []),
    ],
  };
};