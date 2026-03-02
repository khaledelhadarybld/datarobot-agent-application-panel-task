import { type PropsWithChildren, useEffect, useRef } from 'react';
import { Skeleton } from '@/components/ui/skeleton';
import { ChatMessagesMemo } from '@/components/ChatMessage';
import { ChatError } from '@/components/ChatError';
import {
  isErrorStateEvent,
  isMessageStateEvent,
  isStepStateEvent,
  type ChatStateEvent,
  isThinkingEvent,
} from '@/types/events';
import { StepEvent } from '@/components/StepEvent';
import { ThinkingEvent } from '@/components/ThinkingEvent.tsx';

export type ChatMessageProps = {
  isLoading: boolean;
  chatId: string;
  messages?: ChatStateEvent[];
} & PropsWithChildren;

export function ChatMessages({ children, messages, isLoading }: ChatMessageProps) {
  return (
    <div className="flex flex-col gap-2">
      {isLoading && messages?.length === 0 ? (
        <div className="space-y-4">
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-20 w-full" />
        </div>
      ) : (
        children ||
        (messages &&
          messages.map(m => {
            if (isErrorStateEvent(m)) {
              return <ChatError key={m.value.id} {...m.value} />;
            }
            if (isMessageStateEvent(m)) {
              return <ChatMessagesMemo key={m.value.id} {...m.value} />;
            }
            if (isStepStateEvent(m)) {
              return <StepEvent key={m.value.id} {...m.value} />;
            }
            if (isThinkingEvent(m)) {
              return <ThinkingEvent key={m.type} />;
            }
          }))
      )}
    </div>
  );
}
