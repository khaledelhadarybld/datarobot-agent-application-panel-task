import React, { useEffect, useState } from 'react';
import { ChatPage as ChatPageImplementation } from '@/components/page/ChatPage.tsx';
import { useLocation } from 'react-router-dom';

export const ChatPage: React.FC = () => {
  const [chatId, setChatId] = useState<string>(() => window.location.hash?.substring(1));
  const { hash } = useLocation();
  useEffect(() => {
    if (hash) {
      setChatId(hash.substring(1));
    }
  }, [hash]);

  const setChatIdHandler = (id: string) => {
    setChatId(id);
    window.location.hash = id;
  };

  return <ChatPageImplementation chatId={chatId} setChatId={setChatIdHandler} />;
};
