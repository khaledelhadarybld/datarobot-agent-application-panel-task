import { StartNewChat } from '@/components/StartNewChat.tsx';
import { useMainLayout } from '@/hooks/use-main-layout.ts';

export function EmptyStatePage() {
  const { addChatHandler } = useMainLayout();
  return <StartNewChat createChat={addChatHandler} />;
}
