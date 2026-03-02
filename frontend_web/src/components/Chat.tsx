import { type PropsWithChildren, useEffect, useRef } from 'react';
import { ChatMessages } from '@/components/ChatMessages';
import { ChatTextInput } from '@/components/ChatTextInput';
import { ChatProgress } from '@/components/ChatProgress';
import { useChatContext } from '@/hooks/use-chat-context';
import { ScrollArea } from '@/components/ui/scroll-area';
import { MessageResponse } from '@/api/chat/types.ts';
import { ChatStateEvent } from '@/types/events.ts';

export type ChatProps = {
  initialMessages?: MessageResponse[];
} & PropsWithChildren;

const THRESHOLD = 100;

export function useChatScroll({ chatId, events }: { chatId: string; events: ChatStateEvent[] }) {
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const shouldAutoscrollRef = useRef<boolean>(true);

  const onChatScroll = () => {
    if (!scrollContainerRef.current || !events.length) {
      return;
    }
    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    shouldAutoscrollRef.current = distanceFromBottom <= THRESHOLD;
  };

  useEffect(() => {
    if (scrollContainerRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [chatId]);

  useEffect(() => {
    if (scrollContainerRef.current && shouldAutoscrollRef.current) {
      scrollContainerRef.current.scrollTop = scrollContainerRef.current.scrollHeight;
    }
  }, [events]);

  return { scrollContainerRef, onChatScroll };
}

export function Chat({ initialMessages, children }: ChatProps) {
  const {
    chatId,
    sendMessage,
    userInput,
    setUserInput,
    combinedEvents,
    progress,
    deleteProgress,
    isLoadingHistory,
    setInitialMessages,
    isAgentRunning,
  } = useChatContext();
  useEffect(() => {
    if (initialMessages) {
      setInitialMessages(initialMessages);
    }
  }, []);

  const { scrollContainerRef, onChatScroll } = useChatScroll({ chatId, events: combinedEvents });

  return (
    <div className="flex flex-col h-full gap-4 p-2 w-full min-w-0">
      {children || (
        <>
          <ScrollArea
            className="scroll mb-5 w-full flex-1 min-h-0"
            scrollViewportRef={scrollContainerRef}
            onWheel={onChatScroll}
          >
            <div className="w-full justify-self-center">
              <ChatMessages
                isLoading={isLoadingHistory}
                messages={combinedEvents}
                chatId={chatId}
              />
              <ChatProgress progress={progress || {}} deleteProgress={deleteProgress} />
            </div>
          </ScrollArea>

          <ChatTextInput
            userInput={userInput}
            setUserInput={setUserInput}
            onSubmit={sendMessage}
            runningAgent={isAgentRunning}
          />
        </>
      )}
    </div>
  );
}
