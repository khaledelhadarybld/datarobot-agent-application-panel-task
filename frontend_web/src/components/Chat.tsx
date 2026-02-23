import { type PropsWithChildren, useEffect } from 'react';
import { ChatMessages } from '@/components/ChatMessages';
import { ChatTextInput } from '@/components/ChatTextInput';
import { ChatProgress } from '@/components/ChatProgress';
import { useChatContext } from '@/hooks/use-chat-context';
import { MessageResponse } from '@/api/chat/types.ts';

export type ChatProps = {
  initialMessages?: MessageResponse[];
} & PropsWithChildren;

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

  return (
    <div className="flex flex-col h-full gap-4 p-4 overflow-hidden w-full">
      {children || (
        <>
          <div className="flex flex-col grow gap-2 min-h-0 overflow-hidden">
            <ChatMessages isLoading={isLoadingHistory} messages={combinedEvents} chatId={chatId} />
            <ChatProgress progress={progress || {}} deleteProgress={deleteProgress} />
          </div>
          <div className="shrink-0">
            <ChatTextInput
              userInput={userInput}
              setUserInput={setUserInput}
              onSubmit={sendMessage}
              runningAgent={isAgentRunning}
            />
          </div>
        </>
      )}
    </div>
  );
}
