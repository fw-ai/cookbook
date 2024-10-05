'use client';
import { useState } from 'react';

import { LightAsync as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/cjs/styles/prism';
import bash from 'react-syntax-highlighter/dist/esm/languages/hljs/bash';
import python from 'react-syntax-highlighter/dist/esm/languages/hljs/python';
import typescript from 'react-syntax-highlighter/dist/esm/languages/hljs/typescript';

import { CopyIcon } from '@radix-ui/react-icons';

SyntaxHighlighter.registerLanguage('typescript', typescript);
SyntaxHighlighter.registerLanguage('python', python);
SyntaxHighlighter.registerLanguage('bash', bash);

interface CodeBlockProps {
  language: 'python' | 'typescript' | 'bash';
  value: string;
}

const CodeBlock = ({ language, value }: CodeBlockProps) => {
  const [copyTip, setCopyTip] = useState('Copy code');
  const handleCopy = async (text: string) => {
    navigator.clipboard.writeText(text);
    setCopyTip('Copied');
    await new Promise((resolve) => setTimeout(resolve, 500));
    setCopyTip(`Copy code`);
  };
  return (
    <div style={{ position: 'relative', fontFamily: '"Fira Code",monospace' }}>
      {/* <SyntaxHighlighter code={value} language={language} /> */}
      <SyntaxHighlighter
        style={oneDark}
        showLineNumbers={false}
        language={language}
        wrapLines={true}
        PreTag="div"
        className="mockup-code scrollbar-thin scrollbar-track-base-content/5 scrollbar-thumb-base-content/40 scrollbar-track-rounded-md scrollbar-thumb-rounded"
        useInlineStyles={true}
      >
        {String(value).replace(/\n$/u, '')}
      </SyntaxHighlighter>

      <button
        onClick={async () => await handleCopy(value)}
        className="tooltip tooltip-left absolute z-40 mr-2 mt-5"
        data-tip={copyTip}
        style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          backgroundColor: 'rgba(255, 255, 255, 0.6)',
          borderRadius: '4px',
        }}
      >
        <CopyIcon />
      </button>
    </div>
  );
};

export default CodeBlock;
