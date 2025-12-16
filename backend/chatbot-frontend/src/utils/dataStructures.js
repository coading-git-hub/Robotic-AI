// Data structures for the chatbot frontend

// ChatMessage class
export class ChatMessage {
  constructor(id, text, sender, timestamp = new Date(), type = 'text') {
    this.id = id;
    this.text = text;
    this.sender = sender; // 'user' or 'agent'
    this.timestamp = timestamp;
    this.type = type; // 'text', 'command', 'system', etc.
    this.sources = [];
    this.confidence = null;
    this.groundedInContext = null;
  }

  // Validate the message
  isValid() {
    return (
      this.id !== undefined &&
      typeof this.text === 'string' &&
      this.text.trim().length > 0 &&
      ['user', 'agent'].includes(this.sender) &&
      this.timestamp instanceof Date
    );
  }

  // Convert to plain object for serialization
  toObject() {
    return {
      id: this.id,
      text: this.text,
      sender: this.sender,
      timestamp: this.timestamp,
      type: this.type,
      sources: this.sources,
      confidence: this.confidence,
      groundedInContext: this.groundedInContext
    };
  }

  // Create from plain object
  static fromObject(obj) {
    const message = new ChatMessage(obj.id, obj.text, obj.sender, obj.timestamp, obj.type);
    message.sources = obj.sources || [];
    message.confidence = obj.confidence;
    message.groundedInContext = obj.groundedInContext;
    return message;
  }
}

// ChatSession class
export class ChatSession {
  constructor(sessionId, createdAt = new Date()) {
    this.sessionId = sessionId;
    this.messages = [];
    this.selectedText = null;
    this.createdAt = createdAt;
    this.updatedAt = createdAt;
    this.isActive = true;
  }

  // Add a message to the session
  addMessage(message) {
    if (!(message instanceof ChatMessage) && !message.isValid?.()) {
      throw new Error('Invalid message object');
    }

    this.messages.push(message);
    this.updatedAt = new Date();
  }

  // Get messages by sender
  getMessagesBySender(sender) {
    return this.messages.filter(message => message.sender === sender);
  }

  // Get the last message
  getLastMessage() {
    return this.messages.length > 0 ? this.messages[this.messages.length - 1] : null;
  }

  // Get message count
  getMessageCount() {
    return this.messages.length;
  }

  // Clear the session
  clear() {
    this.messages = [];
    this.selectedText = null;
    this.updatedAt = new Date();
  }

  // Update selected text
  updateSelectedText(text) {
    this.selectedText = text;
    this.updatedAt = new Date();
  }

  // Validate the session
  isValid() {
    return (
      this.sessionId !== undefined &&
      Array.isArray(this.messages) &&
      this.createdAt instanceof Date
    );
  }

  // Convert to plain object for serialization
  toObject() {
    return {
      sessionId: this.sessionId,
      messages: this.messages.map(msg => msg.toObject ? msg.toObject() : msg),
      selectedText: this.selectedText,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt,
      isActive: this.isActive
    };
  }

  // Create from plain object
  static fromObject(obj) {
    const session = new ChatSession(obj.sessionId, obj.createdAt);
    session.messages = obj.messages?.map(msg => ChatMessage.fromObject(msg)) || [];
    session.selectedText = obj.selectedText;
    session.updatedAt = obj.updatedAt;
    session.isActive = obj.isActive;
    return session;
  }
}

// API Request structure
export class APIRequest {
  constructor(query, selectedText = null, backendUrl = null) {
    this.query = query;
    this.selectedText = selectedText;
    this.timestamp = new Date();
    this.backendUrl = backendUrl;
  }

  isValid() {
    return (
      typeof this.query === 'string' &&
      this.query.trim().length > 0
    );
  }

  toPayload() {
    return {
      query: this.query,
      selected_text: this.selectedText
    };
  }
}

// API Response structure
export class APIResponse {
  constructor(response, sources = [], confidence = null, timestamp = new Date()) {
    this.response = response;
    this.sources = sources;
    this.confidence = confidence;
    this.timestamp = timestamp;
    this.groundedInContext = null;
  }

  isValid() {
    return typeof this.response === 'string';
  }

  static fromApiResponse(apiResponse) {
    const response = new APIResponse(
      apiResponse.answer,
      apiResponse.sources || [],
      apiResponse.confidence,
    );
    response.groundedInContext = apiResponse.grounded_in_context;
    return response;
  }
}