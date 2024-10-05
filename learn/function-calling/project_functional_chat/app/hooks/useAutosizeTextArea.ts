import { useEffect } from 'react';

// Updates the height of a <textarea> when the value changes.
export const useAutosizeTextArea = (textAreaRef: HTMLTextAreaElement | null, value: string) => {
  useEffect(() => {
    if (textAreaRef) {
      // We need to reset the height momentarily to get the correct scrollHeight for the textarea
      textAreaRef.style.height = '0px';
      const { scrollHeight } = textAreaRef;
      // We then set the height directly, outside of the render loop
      // Trying to set this with state or a ref will produce an incorrect value.
      const { style } = textAreaRef;
      style.height = `${scrollHeight}px`;
    }
  }, [textAreaRef, value]);
};
