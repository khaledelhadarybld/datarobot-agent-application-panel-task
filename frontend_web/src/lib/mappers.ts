import { v4 as uuid } from 'uuid';
import type {
  Message,
  ReasoningMessageChunkEvent,
  ReasoningMessageContentEvent,
  ReasoningMessageEndEvent,
  ReasoningMessageStartEvent,
  TextMessageChunkEvent,
  TextMessageContentEvent,
  TextMessageEndEvent,
  TextMessageStartEvent,
  ToolCallEndEvent,
} from '@ag-ui/core';
import { EventType } from '@ag-ui/core';
import type { ChatStateEvent, ChatStateEventByType } from '@/types/events';
import { MessageResponse } from '@/api/chat/types.ts';
import type { ToolInvocationUIPart } from '@/types/message.ts';

type AgUiTextEvent =
  | TextMessageStartEvent
  | TextMessageContentEvent
  | TextMessageEndEvent
  | TextMessageChunkEvent;

type AgUiReasoningEvent =
  | ReasoningMessageStartEvent
  | ReasoningMessageContentEvent
  | ReasoningMessageChunkEvent
  | ReasoningMessageEndEvent;

type AgUiToolEvent = ToolCallEndEvent;

export function createTextMessageFromAgUiEvent(
  event: AgUiTextEvent,
  textMessageBuffer?: string
): MessageResponse {
  const baseMessage: MessageResponse = {
    id: event.messageId || '',
    content: {
      format: 2,
      parts: [],
      content: '',
    },
    role: 'assistant',
    createdAt: event.timestamp ? new Date(event.timestamp) : new Date(),
    threadId: '',
    resourceId: '',
  };

  // Map role, converting 'developer' to 'system' for compatibility
  const mapRole = (
    role?: 'user' | 'assistant' | 'system' | 'developer'
  ): 'user' | 'assistant' | 'system' => {
    if (!role || role === 'developer') return 'system';
    return role;
  };

  switch (event.type) {
    case EventType.TEXT_MESSAGE_START:
      return {
        ...baseMessage,
        role: mapRole(event.role),
        id: event.messageId,
      };

    case EventType.TEXT_MESSAGE_CONTENT:
      return {
        ...baseMessage,
        id: event.messageId,
        content: {
          format: 2,
          parts: [
            {
              type: 'text',
              text: textMessageBuffer + event.delta,
            },
          ],
          content: textMessageBuffer + event.delta,
        },
      };

    case EventType.TEXT_MESSAGE_END:
      return {
        ...baseMessage,
        id: event.messageId,
      };

    case EventType.TEXT_MESSAGE_CHUNK:
      return {
        ...baseMessage,
        id: event.messageId || '',
        role: mapRole(event.role),
        content: {
          format: 2,
          parts: event.delta
            ? [
                {
                  type: 'text',
                  text: event.delta,
                },
              ]
            : [],
          content: event.delta || '',
        },
      };

    default:
      return baseMessage;
  }
}

export function createToolMessageFromAgUiEvent(
  event: AgUiToolEvent,
  toolCallName: string,
  toolCallArgs: Record<string, any>
): MessageResponse {
  const baseMessage: MessageResponse = {
    id: uuid(),
    content: {
      format: 2,
      parts: [
        {
          type: 'tool-invocation',
          toolInvocation: {
            state: 'call',
            args: toolCallArgs || event.rawEvent?.args || {},
            toolCallId: event.toolCallId,
            toolName: toolCallName,
          } as any,
        },
      ],
    },
    role: 'assistant',
    createdAt: event.timestamp ? new Date(event.timestamp) : new Date(),
    threadId: '',
    resourceId: uuid(),
  };

  return baseMessage;
}

export function createTextMessageFromUserInput({
  message,
  chatId,
  messageId,
}: {
  message: string;
  chatId: string;
  messageId: string;
}): MessageResponse {
  const baseMessage: MessageResponse = {
    id: messageId,
    content: {
      format: 2,
      parts: [
        {
          type: 'text',
          text: message,
        },
      ],
      content: message,
    },
    role: 'user',
    createdAt: new Date(),
    threadId: chatId,
    resourceId: uuid(),
  };

  return baseMessage;
}

export function createCustomMessageWidget({
  toolCallName,
  toolCallArgs,
  threadId,
}: {
  toolCallName: string;
  toolCallArgs: Record<string, any>;
  threadId: string;
}): MessageResponse {
  const toolInvocation = {
    state: 'call',
    toolCallId: `call_${uuid()}`,
    toolName: toolCallName,
    args: toolCallArgs,
  };
  return {
    id: uuid(),
    content: {
      format: 2,
      parts: [
        {
          type: 'tool-invocation',
          toolInvocation,
        },
      ],
    },
    role: 'assistant',
    createdAt: new Date(),
    threadId: threadId,
    resourceId: uuid(),
  };
}

export function messageToStateEvent(message: MessageResponse): ChatStateEventByType<'message'> {
  return {
    type: 'message',
    value: message,
  };
}

function reasoningPart(reasoningText: string): {
  type: 'reasoning';
  reasoning: string;
  details: Array<{ type: 'text'; text: string }>;
} {
  return {
    type: 'reasoning',
    reasoning: reasoningText,
    details: reasoningText ? [{ type: 'text', text: reasoningText }] : [],
  };
}

// ---------------------------------------------------------------------------
// Internal helpers for messageResponseToAgUiMessage
// ---------------------------------------------------------------------------

/** Pull the plain-text body out of a MessageContent, preferring the
 *  top-level `content` field and falling back to the first text part. */
function extractText(content: MessageResponse['content']): string {
  if (content.content) return content.content;
  const firstTextPart = content.parts.find(p => p.type === 'text');
  return firstTextPart?.type === 'text' ? firstTextPart.text : '';
}

/** Return all tool-invocation UI parts found inside the content parts. */
function collectToolInvocations(parts: MessageResponse['content']['parts']): ToolInvocationUIPart[] {
  return parts.filter((p): p is ToolInvocationUIPart => p.type === 'tool-invocation');
}

/** Serialise a tool invocation's arguments into the string format AG-UI expects. */
function serialiseArgs(args: unknown): string {
  return typeof args === 'string' ? args : JSON.stringify(args ?? {});
}

// ---------------------------------------------------------------------------
// Role-specific converters  (each returns Message | null)
// ---------------------------------------------------------------------------

type RoleConverter = (id: string, content: MessageResponse['content']) => Message | null;

const roleConverters: Record<string, RoleConverter> = {
  user: (id, content) => ({ id, role: 'user', content: extractText(content) }),
  system: (id, content) => ({ id, role: 'system', content: extractText(content) }),
  reasoning: (id, content) => ({ id, role: 'reasoning', content: content.content ?? '' }),

  assistant(id, content) {
    const toolParts = collectToolInvocations(content.parts);

    if (toolParts.length > 0) {
      return {
        id,
        role: 'assistant',
        toolCalls: toolParts.map(({ toolInvocation }) => ({
          id: toolInvocation.toolCallId ?? id,
          type: 'function' as const,
          function: {
            name: toolInvocation.toolName,
            arguments: serialiseArgs(toolInvocation.args),
          },
        })),
      };
    }

    return { id, role: 'assistant', content: extractText(content) };
  },
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Transform an internal {@link MessageResponse} into the AG-UI {@link Message}
 * format so it can be included in the conversation context sent to the agent.
 *
 * @param message - The app-level message to convert.
 * @returns The AG-UI message, or `null` when the role is unrecognised.
 */
export function messageResponseToAgUiMessage(message: MessageResponse): Message | null {
  const converter = roleConverters[message.role];
  return converter ? converter(message.id, message.content) : null;
}

/**
 * Assemble the full, ordered conversation for the agent by merging persisted
 * history with in-session events and appending the latest user turn.
 *
 * Duplicates (messages already present in history that also appear in the
 * session) are removed so each message is sent exactly once.
 *
 * @param history  - Previously persisted messages (may be empty for new chats).
 * @param sessionEvents - Events accumulated in the current browser session.
 * @param userTurn - The new user message being sent right now.
 * @returns A deduped, chronologically ordered array of AG-UI Messages.
 */
export function buildConversationMessages(
  history: MessageResponse[],
  sessionEvents: ChatStateEvent[],
  userTurn: { id: string; content: string },
): Message[] {
  console.log("buildConversationMessages", history, sessionEvents, userTurn);
  const fromHistory = history
    .map(messageResponseToAgUiMessage)
    .filter((m): m is Message => m !== null);

  const fromSession = sessionEvents
    .filter((e): e is ChatStateEventByType<'message'> => e.type === 'message')
    .map(e => messageResponseToAgUiMessage(e.value))
    .filter((m): m is Message => m !== null);

  const knownIds = new Set(fromHistory.map(m => m.id));
  const uniqueSessionMessages = fromSession.filter(m => !knownIds.has(m.id));

  return [
    ...fromHistory,
    ...uniqueSessionMessages,
    { id: userTurn.id, role: 'user', content: userTurn.content },
  ];
}

export function createReasoningMessage(
  event: AgUiReasoningEvent,
  reasoningMessageBuffer = ''
): MessageResponse {
  const reasoningMessage: MessageResponse = {
    id: event.messageId ?? uuid(),
    role: 'reasoning',
    content: {
      format: 2,
      parts: [],
      content: '',
    },
    createdAt: new Date(),
    threadId: '',
    resourceId: uuid(),
  };

  switch (event.type) {
    case EventType.REASONING_MESSAGE_START:
      return { ...reasoningMessage, id: event.messageId ?? reasoningMessage.id };
    case EventType.REASONING_MESSAGE_CONTENT: {
      const fullContent = reasoningMessageBuffer + event.delta;
      return {
        ...reasoningMessage,
        id: event.messageId ?? reasoningMessage.id,
        content: {
          format: 2,
          parts: [reasoningPart(fullContent)],
          content: fullContent,
        },
      };
    }
    case EventType.REASONING_MESSAGE_END:
      return { ...reasoningMessage, id: event.messageId ?? reasoningMessage.id };

    case EventType.REASONING_MESSAGE_CHUNK: {
      return {
        ...reasoningMessage,
        id: event.messageId ?? '',
        content: {
          format: 2,
          parts: event.delta ? [reasoningPart(event.delta)] : [],
          content: event.delta,
        },
      };
    }
    default:
      return reasoningMessage;
  }
}
