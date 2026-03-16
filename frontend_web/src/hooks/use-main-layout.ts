import { useOutletContext } from 'react-router-dom';

export interface MainLayoutContext {
  hasChat: boolean;
  isNewChat: boolean;
  isLoadingChats: boolean;
  addChatHandler: () => void;
}

export function useMainLayout() {
  return useOutletContext<MainLayoutContext>();
}
