import { useLayoutEffect } from 'react';
import { Outlet, useNavigate, useParams, useMatch } from 'react-router-dom';
import { ChatSidebar } from '@/components/ChatSidebar.tsx';
import { useChatList } from '@/hooks/use-chat-list.ts';
import { PATHS } from '@/constants/path.ts';

export function MainLayout() {
  const { chatId = '' } = useParams<{ chatId?: string }>();
  const navigate = useNavigate();

  const setChatIdHandler = (id: string) => {
    navigate(`/chat/${id}`);
  };

  const isChatEmptyPage = useMatch(PATHS.CHAT_EMPTY);
  const isChatSelectedPage = useMatch(PATHS.CHAT);
  const isChat = isChatEmptyPage || isChatSelectedPage;

  const {
    hasChat,
    isNewChat,
    chats,
    isLoadingChats,
    addChatHandler,
    deleteChatHandler,
    isDeletingChat,
  } = useChatList({
    chatId,
    setChatId: setChatIdHandler,
    showStartChat: !chatId,
  });

  useLayoutEffect(() => {
    if (isLoadingChats || !chats || chats?.find(c => c.id === chatId)) {
      return;
    }
    if (!isChat) {
      return;
    }
    if (!chats.length) {
      addChatHandler();
    } else {
      setChatIdHandler(chats[0].id);
    }
  }, [chats, isLoadingChats, isChat, chatId]);

  return (
    <div className="flex flex-row w-full h-svh">
      <ChatSidebar
        isLoading={isLoadingChats}
        chatId={chatId}
        chats={chats}
        onChatCreate={addChatHandler}
        onChatSelect={setChatIdHandler}
        onChatDelete={deleteChatHandler}
        isDeletingChat={isDeletingChat}
      />
      <Outlet
        context={{
          hasChat,
          isNewChat,
          isLoadingChats,
          addChatHandler,
        }}
      />
    </div>
  );
}
