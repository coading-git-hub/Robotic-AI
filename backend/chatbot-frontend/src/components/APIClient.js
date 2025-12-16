import axios from 'axios';

// APIClient component
class APIClient {
  constructor(backendUrl) {
    this.backendUrl = backendUrl || process.env.REACT_APP_BACKEND_API_URL || 'http://localhost:8000';
    this.defaultTimeout = 30000; // 30 seconds
  }

  // Method to send a query to the backend agent API
  async sendQuery(query, selectedText = null) {
    try {
      const response = await axios.post(
        `${this.backendUrl}/api/agent/query`,
        {
          query: query,
          selected_text: selectedText
        },
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: this.defaultTimeout
        }
      );

      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      console.error('API Error:', error);

      // Return a structured error response
      return {
        success: false,
        error: error.response ? error.response.data : error.message,
        status: error.response ? error.response.status : null
      };
    }
  }

  // Method to check backend health
  async checkHealth() {
    try {
      const response = await axios.get(
        `${this.backendUrl}/api/health`,
        {
          timeout: 5000 // 5 seconds for health check
        }
      );

      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      console.error('Health Check Error:', error);

      return {
        success: false,
        error: error.message
      };
    }
  }

  // Method to validate agent response (for development/testing)
  async validateResponse(query, context, response) {
    try {
      const validationResponse = await axios.post(
        `${this.backendUrl}/api/agent/validate`,
        {
          query: query,
          context: context,
          response: response
        },
        {
          headers: {
            'Content-Type': 'application/json',
          },
          timeout: this.defaultTimeout
        }
      );

      return {
        success: true,
        data: validationResponse.data
      };
    } catch (error) {
      console.error('Validation Error:', error);

      return {
        success: false,
        error: error.response ? error.response.data : error.message,
        status: error.response ? error.response.status : null
      };
    }
  }
}

export default APIClient;