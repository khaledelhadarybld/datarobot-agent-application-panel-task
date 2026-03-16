import React from 'react';
import { useParams } from 'react-router-dom';
import { ChatPageContent } from '@/components/page/ChatPage.tsx';
import { useMainLayout } from '@/hooks/use-main-layout.ts';

export const ChatPage: React.FC = () => {
  const { chatId } = useParams<{ chatId: string }>();
  const { hasChat, isNewChat, isLoadingChats, addChatHandler } = useMainLayout();

  if (!chatId) {
    return null;
  }

  return (
    <ChatPageContent
      chatId={chatId}
      hasChat={hasChat}
      isNewChat={isNewChat}
      isLoadingChats={isLoadingChats}
      addChatHandler={addChatHandler}
    />
  );
};
