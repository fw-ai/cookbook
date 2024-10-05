'use client';

import * as React from 'react';

export interface useCopyToClipboardProps {
  timeout?: number;
}

export function useCopyToClipboard({ timeout = 2000 }: useCopyToClipboardProps) {
  const [isCopied, setIsCopied] = React.useState<boolean>(false);

  const copyToClipboard = (value: string) => {
    if (typeof window === 'undefined' || !navigator.clipboard?.writeText) {
      return;
    }

    if (!value) {
      return;
    }

    navigator.clipboard
      .writeText(value)
      .then(() => {
        setIsCopied(true);

        setTimeout(() => {
          setIsCopied(false);
        }, timeout);
        return null;
      })
      .catch(() => {
        // eslint-disable-next-line no-console
        console.error('Failed to copy to clipboard.');
      });
  };

  return { isCopied, copyToClipboard };
}
