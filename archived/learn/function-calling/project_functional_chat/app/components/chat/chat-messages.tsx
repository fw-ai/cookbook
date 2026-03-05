import { ChatMessage as ChatMessageInterface } from '../common/types';
import ChatActions from './chat-actions';
import ChatMessage from './chat-message';

export default function ChatMessages(props: {
  messages: ChatMessageInterface[];
  isLoading: boolean;
  reload?: () => void;
  stop?: () => void;
  children?: React.ReactNode;
}) {
  const messageLength = props.messages.length;
  const lastMessage = props.messages[messageLength - 1];

  const isLastMessageFromAssistant = messageLength > 0 && lastMessage?.role !== 'user';
  const showReload = props.reload && !props.isLoading && isLastMessageFromAssistant;
  const showStop = props.stop && props.isLoading;

  return (
    <div className="w-full pb-0 mt-4 overflow-y-auto flex-1">
      <div className="flex flex-col gap-5 divide-y pb-4">
        {props.messages.filter(m => m.metadata?.hide !== true).map((m) => (
          <ChatMessage key={m.id} {...m} />
        ))}
        {props.children}
      </div>
      <div className="flex justify-end pb-4">
        <ChatActions reload={props.reload} stop={props.stop} showReload={showReload} showStop={showStop} />
      </div>
    </div>
  );
}
