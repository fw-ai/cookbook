'use client';

import React, {
  FC,
  memo,
  useState,
} from 'react';

import {
  CheckIcon,
  CopyIcon,
  EyeClosedIcon,
  EyeOpenIcon,
} from '@radix-ui/react-icons';

import { Button } from '../ui/button';
import { useCopyToClipboard } from './use-copy-to-clipboard';

import SyntaxHighlighter from 'react-syntax-highlighter';
import { atelierDuneLight } from 'react-syntax-highlighter/dist/esm/styles/hljs';

interface Props {
  language: string;
  value: string;
  copyValue?: string;
  apiKeyToHide?: string;
  header?: boolean;
  showLineNumbers?: boolean;
}

type languageMap = Record<string, string | undefined>;

export const programmingLanguages: languageMap = {
  javascript: '.js',
  python: '.py',
  java: '.java',
  c: '.c',
  cpp: '.cpp',
  'c++': '.cpp',
  'c#': '.cs',
  ruby: '.rb',
  php: '.php',
  swift: '.swift',
  'objective-c': '.m',
  kotlin: '.kt',
  typescript: '.ts',
  go: '.go',
  perl: '.pl',
  rust: '.rs',
  scala: '.scala',
  haskell: '.hs',
  lua: '.lua',
  shell: '.sh',
  sql: '.sql',
  html: '.html',
  css: '.css',
  // add more file extensions here, make sure the key is same as language prop in CodeBlock.tsx component
};

export const generateRandomString = (length: number, lowercase = false) => {
  // excluding similar looking characters like Z, 2, I, 1, O, 0
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXY3456789';
  let result = '';
  // eslint-disable-next-line no-plusplus
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return lowercase ? result.toLowerCase() : result;
};

const CodeBlock: FC<Props> = memo(({ language, value, copyValue, apiKeyToHide, header = true, showLineNumbers = true }) => {
  const { isCopied, copyToClipboard } = useCopyToClipboard({ timeout: 2000 });
  const [showAPIKey, setShowAPIKey] = useState(false);
  const onCopy = () => {
    if (isCopied) return;
    copyToClipboard(copyValue || value);
  };

  const showHideAPIKey = () => {
    setShowAPIKey(!showAPIKey);
  };

  return (
    <div className="codeblock relative w-full h-full bg-zinc-100 font-sans rounded-lg">
      {header ? (
        <div className="flex w-full items-center justify-between bg-zinc-800 px-4 py-1 pr-4 text-zinc-100">
          <span className="text-xs lowercase">{language}</span>
          <div className="flex items-center space-x-1 ">
            {apiKeyToHide ? (
              <Button variant="ghost" size="icon" onClick={showHideAPIKey}>
                {showAPIKey ? <EyeClosedIcon className="h-4 w-4" /> : <EyeOpenIcon className="h-4 w-4" />}
                <span className="sr-only">Copy code</span>
              </Button>
            ) : null}
            <Button variant="ghost" size="icon" onClick={onCopy}>
              {isCopied ? <CheckIcon className="h-4 w-4" /> : <CopyIcon className="h-4 w-4" />}
              <span className="sr-only">Copy code</span>
            </Button>
          </div>
        </div>
      ) : null}
      <SyntaxHighlighter
        language={language}
        style={atelierDuneLight}
        PreTag="div"
        showLineNumbers={showLineNumbers}
        wrapLongLines
        customStyle={{
          width: '100%',
          background: 'transparent',
          padding: '1rem 1rem',
          borderRadius: '0.5rem',
          overflow: 'hidden',
        }}
        lineNumberContainerStyle={{
          display: 'none',
        }}
        codeTagProps={{
          style: {
            fontSize: '0.8rem',
            fontFamily: 'var(--font-mono)',
          },
        }}
      >
        {apiKeyToHide && !showAPIKey ? value.replace(apiKeyToHide, '*'.repeat(apiKeyToHide.length)) : value}
      </SyntaxHighlighter>
    </div>
  );
});
CodeBlock.displayName = 'CodeBlock';

export { CodeBlock };
