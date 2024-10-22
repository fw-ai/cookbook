import { FC, memo } from 'react';

import ReactMarkdown, { Options } from 'react-markdown';
import remarkGfm from 'remark-gfm';

import { CodeBlock } from './CodeBlock';

const MemoizedReactMarkdown: FC<Options> = memo(
  ReactMarkdown,
  (prevProps, nextProps) => prevProps.children === nextProps.children && prevProps.className === nextProps.className,
);

export default function Markdown({ content }: { content: string }) {
  return (
    <MemoizedReactMarkdown
      className="prose break-words dark:prose-invert prose-p:leading-relaxed prose-pre:p-0"
      remarkPlugins={[remarkGfm]}
      components={{
        p({ children }: { children: React.ReactNode }) {
          return <p className="mb-2 last:mb-0 text-wrap">{children}</p>;
        },
        code({ inline, className, children, ...props }) {
          if (children && Array.isArray(children) && children.length > 0) {
            if (children[0] === '▍') {
              return <span className="mt-1 animate-pulse cursor-default">▍</span>;
            }
            children[0] = (children[0] as string).replace('`▍`', '▍');
          }
          if (inline) {
            return (
              <code className={className} {...props}>
                {children}
              </code>
            );
          }
          const match = /language-(?<lang>\w+)/u.exec(className || '');

          return (
            <CodeBlock
              key={Math.random()}
              showLineNumbers={false}
              header={false}
              language={match?.[1] || ''}
              // eslint-disable-next-line require-unicode-regexp
              value={String(children).replace(/\n$/, '')}
              {...props}
            />
          );
        },
      }}
    >
      {content}
    </MemoizedReactMarkdown>
  );
}
