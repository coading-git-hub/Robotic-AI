// Text selection utilities for the chatbot frontend

import { validateSelectedTextInput } from './validation';

// Get currently selected text from the page
export const getSelectedText = () => {
  const selection = window.getSelection ? window.getSelection() : null;

  if (!selection) {
    return '';
  }

  // Get the selected text string
  const selectedText = selection.toString().trim();

  // Validate the selected text
  try {
    validateSelectedTextInput(selectedText);
  } catch (error) {
    // If validation fails, return empty string
    return '';
  }

  return selectedText;
};

// Get selected text with additional context (surrounding text)
export const getSelectedTextWithContext = (contextLength = 50) => {
  const selection = window.getSelection ? window.getSelection() : null;

  if (!selection || selection.rangeCount === 0) {
    return { text: '', context: '' };
  }

  const range = selection.getRangeAt(0);
  const selectedText = selection.toString().trim();

  // Get surrounding context
  const startRange = document.createRange();
  const endRange = document.createRange();

  // Set start position for context
  startRange.setStart(range.startContainer, 0);
  startRange.setEnd(range.startContainer, range.startOffset);
  const beforeText = startRange.toString().substring(Math.max(0, startRange.toString().length - contextLength));

  // Set end position for context
  endRange.setStart(range.endContainer, range.endOffset);
  endRange.setEnd(range.endContainer, range.endContainer.length);
  const afterText = endRange.toString().substring(0, contextLength);

  return {
    text: selectedText,
    context: `${beforeText}${selectedText}${afterText}`
  };
};

// Check if text is currently selected
export const isTextSelected = () => {
  const selection = window.getSelection ? window.getSelection() : null;
  return selection && selection.toString().trim().length > 0;
};

// Clear current text selection
export const clearTextSelection = () => {
  if (window.getSelection) {
    window.getSelection().removeAllRanges();
  } else if (document.selection) {
    document.selection.empty();
  }
};

// Get element containing the selection
export const getSelectionContainer = () => {
  const selection = window.getSelection ? window.getSelection() : null;

  if (!selection || selection.rangeCount === 0) {
    return null;
  }

  const range = selection.getRangeAt(0);
  return range.commonAncestorContainer;
};

// Get word count of selected text
export const getSelectedWordCount = () => {
  const selectedText = getSelectedText();
  if (!selectedText) return 0;
  return selectedText.trim().split(/\s+/).filter(word => word.length > 0).length;
};

// Get character count of selected text
export const getSelectedCharacterCount = () => {
  return getSelectedText().length;
};

// Highlight selected text with a temporary highlight
export const highlightSelectedText = (color = 'yellow') => {
  const selection = window.getSelection ? window.getSelection() : null;

  if (!selection || selection.rangeCount === 0) {
    return;
  }

  const range = selection.getRangeAt(0);
  const span = document.createElement('span');
  span.style.backgroundColor = color;
  span.style.padding = '0 1px';

  // Surround the selected content with the span
  range.surroundContents(span);
};

// Remove temporary highlights
export const removeHighlights = (container = document.body) => {
  const highlights = container.querySelectorAll('span[style*="background-color"]');
  highlights.forEach(highlight => {
    // Remove the highlight but preserve the content
    const content = document.createTextNode(highlight.textContent);
    highlight.parentNode.replaceChild(content, highlight);
  });
};