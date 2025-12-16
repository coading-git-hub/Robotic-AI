import React, { useState, useEffect } from 'react';

// TextSelectionHandler component
const TextSelectionHandler = ({ onTextSelected }) => {
  const [selectedText, setSelectedText] = useState('');

  useEffect(() => {
    const handleSelection = () => {
      const selected = window.getSelection().toString().trim();

      // Only update if there's actually selected text and it's not too long
      if (selected && selected.length <= 2000) { // Max length from spec
        setSelectedText(selected);
        onTextSelected(selected);
      } else if (!selected) {
        // If no text is selected, clear the selected text
        if (selectedText) {
          setSelectedText('');
          onTextSelected('');
        }
      }
    };

    // Add event listeners for text selection
    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('keyup', handleSelection);

    // Also listen for touch events on mobile devices
    document.addEventListener('touchend', handleSelection);

    // Cleanup event listeners on component unmount
    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('keyup', handleSelection);
      document.removeEventListener('touchend', handleSelection);
    };
  }, [onTextSelected, selectedText]);

  // Function to programmatically get selected text
  const getSelectedText = () => {
    return window.getSelection ? window.getSelection().toString().trim() : '';
  };

  // Function to clear selected text
  const clearSelectedText = () => {
    setSelectedText('');
    onTextSelected('');
  };

  return null; // This component doesn't render anything visible
};

export default TextSelectionHandler;