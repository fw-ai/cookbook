import {
  PauseIcon,
  ReloadIcon,
} from '@radix-ui/react-icons';

import { Button } from '../ui/button';
import { ChatHandler } from './chat.interface';

export default function ChatActions(
  props: Pick<ChatHandler, "stop" | "reload"> & {
    showReload?: boolean;
    showStop?: boolean;
  },
) {
  return (
    <div className="space-x-4">
      {props.showStop && (
        <Button variant="outline" size="sm" onClick={props.stop}>
          <PauseIcon className="mr-2 h-4 w-4" />
          Stop generating
        </Button>
      )}
      {props.showReload && (
        <Button variant="outline" size="sm" onClick={props.reload}>
          <ReloadIcon className="mr-2 h-4 w-4" />
          Regenerate
        </Button>
      )}
    </div>
  );
}
