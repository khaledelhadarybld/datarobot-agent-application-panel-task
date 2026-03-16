import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { v4 as uuid } from 'uuid';
import type { ChatListItem } from '@/api/chat/types';
import { useDeleteChat, useFetchChats } from '@/api/chat';
import { useAddChat, useHasChat } from '@/hooks/use-chats-state.ts';
import { queryClient } from '@/lib/query-client.ts';
import { chatsKeys } from '@/api/chat/keys.ts';

export type UseChatListParams = {
  chatId: string;
  setChatId: (id: string) => void;
  /**
   * Set to true if "No chats selected" state should be shown
   */
  showStartChat?: boolean;
};

export function useChatList({ chatId, setChatId, showStartChat = false }: UseChatListParams) {
  const [newChat, setNewChat] = useState<ChatListItem | null>(null);
  const newChatRef = useRef<ChatListItem | null>(null);
  const hasChat = useHasChat(chatId);

  const addChatToState = useAddChat();
  const { mutateAsync: deleteChatMutation, isPending: isDeletingChat } = useDeleteChat();
  const { data: chats, isLoading: isLoadingChats, refetch } = useFetchChats();

  useEffect(() => {
    if (chats?.some(chat => chat.id === newChat?.id)) {
      setNewChat(null);
    }
  }, [chats]);

  useEffect(() => {
    newChatRef.current = newChat;
  });

  useLayoutEffect(() => {
    if (!hasChat && chatId && !isLoadingChats) {
      addChatToState(chatId);
    }
    if (!hasChat && !chatId && !isLoadingChats && !showStartChat) {
      addChatHandler();
    }
  }, [hasChat, chatId, isLoadingChats]);

  const chatsWithNew = useMemo(() => {
    if (chats?.some(chat => chat.id === newChat?.id)) {
      return chats;
    }
    return newChat ? [newChat, ...(chats || [])] : chats;
  }, [chats, newChat]);

  const refetchChats = (): void => {
    queryClient.invalidateQueries({ queryKey: chatsKeys.list });
  };

  /**
   * Returns new chat id
   */
  const createChat = (name: string): string => {
    const newChatID = uuid();
    setNewChat({
      id: newChatID,
      name: name,
      userId: '',
      createdAt: new Date(),
      updatedAt: null,
    });
    addChatToState(newChatID);
    return newChatID;
  };

  const deleteChat = (chatId: string) => {
    return deleteChatMutation({ chatId }).then(() => refetch());
  };

  function addChatHandler() {
    const newChatID = createChat('New');
    setChatId(newChatID);
  }

  function deleteChatHandler(id: string, callbackFn: () => void) {
    deleteChat(id)
      .then(() => {
        refetchChats();
      })
      .catch(error => console.error(error))
      .finally(callbackFn);
  }

  return {
    isNewChat: newChat?.id === chatId,
    hasChat,
    chatId,
    setChatId,
    chats: chatsWithNew,
    newChat,
    setNewChat,
    isLoadingChats,
    refetchChats,
    deleteChat,
    isDeletingChat,
    addChatHandler,
    deleteChatHandler,
  };
}
