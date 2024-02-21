import { ChatMessage as ChatMessageInterface } from '~/components/common/types';
import { cn } from '~/lib/utils';

import { CheckIcon, CopyIcon } from '@radix-ui/react-icons';

import { Button } from '../ui/button';
import { v4 as uuidv4 } from 'uuid';
import ChatAvatar from './chat-avatar';
import Markdown from './markdown';
import { useCopyToClipboard } from './use-copy-to-clipboard';
import Toggle from './toggle';
import { stringifyObject } from '../common/utils';

export default function ChatMessage(chatMessage: ChatMessageInterface) {
  const { isCopied, copyToClipboard } = useCopyToClipboard({ timeout: 2000 });
  const functionCall = chatMessage.metadata?.functionCall;
  var functionCallContent;
  if (functionCall) {
    type FunctionCall = {
      name: string;
      arguments: { [key: string]: any };
    };
    const parsed: FunctionCall = {
      name: functionCall.name,
      arguments: JSON.parse(functionCall.arguments)
    };
    functionCallContent = '```javascript\n' + stringifyObject(parsed) + '\n```';
  }
  return (
    <div
      className={cn(
        'flex items-start gap-4 pr-5 pt-5',
        // eslint-disable-next-line @typescript-eslint/no-unnecessary-boolean-literal-compare
        chatMessage.metadata?.used === false ? 'opacity-40' : undefined,
      )}
    >
      <ChatAvatar role={chatMessage.role} />
      <div className="group flex flex-1 justify-between gap-2 ">
        <div className="flex-1">
          {chatMessage.role === 'assistant' && chatMessage.metadata?.loading === true ? (
            <div className="flex justify-start items-start mt-1">
              <div className="flex space-x-1 h-6 ">
                <span className="typing-dots animate-loader"></span>
                <span className="typing-dots animate-loader animation-delay-200"></span>
                <span className="typing-dots animate-loader animation-delay-400"></span>
              </div>
            </div>
          ) : null}
          {functionCallContent && (
            <Toggle showText="show function calls" hideText="hide function calls">
              <Markdown
                key={uuidv4()}
                content={functionCallContent}
              />
            </Toggle>
          )}
          <Markdown
            key={chatMessage.id}
            content={chatMessage.content}
          />
          {chatMessage?.metadata?.averageTokenTime ? (
            <div className="flex justify-start md:justify-end items-center mt-1">
              <span className="text-xs text-zinc-400">
                {chatMessage.metadata.firstTokenTime?.toFixed(0) || '-- '}ms initial latency{' | '}
                {chatMessage.metadata.averageTokenTime.toFixed(2)} tokens/s
              </span>
            </div>
          ) : null}
        </div>
        <Button
          onClick={() => copyToClipboard(chatMessage.content as string)}
          size="icon"
          variant="ghost"
          className="max-sm:hidden h-8 w-8 opacity-0 group-hover:opacity-100"
        >
          {isCopied ? <CheckIcon className="h-4 w-4" /> : <CopyIcon className="h-4 w-4" />}
        </Button>
      </div>
    </div>
  );
}
