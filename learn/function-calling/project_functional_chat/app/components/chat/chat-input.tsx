import {
  useRef,
  useState,
} from 'react';

import { useAutosizeTextArea } from '~/hooks/useAutosizeTextArea';

import { PaperPlaneIcon } from '@radix-ui/react-icons';

import { Button } from '../ui/button';
import { Textarea } from '../ui/textarea';

export default function ChatInput(props: {
  onSubmit: (text: string, base64Files?: string[]) => void;
  multiModal?: boolean;
  isLoading?: boolean;
  onFileUpload?: (file: File) => void;
  onFileError?: (error: string) => void;
}) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [value, setValue] = useState('');
  const textAreaRef = useRef<HTMLTextAreaElement>(null);

  useAutosizeTextArea(textAreaRef.current, value);

  const handleChange = (evt: React.ChangeEvent<HTMLTextAreaElement>) => {
    const val = evt.target?.value;

    setValue(val);
  };

  const onSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const message = e.currentTarget.elements.namedItem('message') as HTMLInputElement;
    if (imageUrl) {
      props.onSubmit(message.value, [imageUrl]);
      setImageUrl(null);
      setValue('');
      return;
    }
    props.onSubmit(message.value);
    setTimeout(() => {
      setValue('');
    }, 10);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // ignore if the user is using IME
    if (event.nativeEvent.isComposing || event.keyCode === 229) {
      return;
    }
    // Prevents adding a new line

    if (
      ((event.which || event.keyCode) === 13 || event.key === 'Enter') &&
      !event.shiftKey &&
      !event.altKey &&
      !event.ctrlKey &&
      !event.metaKey
    ) {
      event.currentTarget.form?.requestSubmit();
    } else if (event.key === 'Enter') {
      // Adds a new line
      setValue(`${value}`);
    } else {
      setValue(event.currentTarget.value);
    }
  };

  return (
    <form
      onSubmit={onSubmit}
      className="rounded-xl bg-zinc-200 p-2 w-full overflow-hidden space-y-4 max-sm:w-[96%] mx-auto"
    >
      <div className="flex w-full items-start justify-between sm:gap-4 ">
        <Textarea
          autoFocus
          name="message"
          onKeyDown={handleKeyDown}
          placeholder="Type a message"
          ref={textAreaRef}
          rows={1}
          className="flex-1 border-none text-zinc-500 min-h-[38px] overflow-hidden resize-none text-base"
          value={value}
          onChange={handleChange}
        />
        <Button type="submit" disabled={props.isLoading}>
          <PaperPlaneIcon className="h-5 w-5 text-zinc-400" />
        </Button>
      </div>
    </form>
  );
}
