import { describe, it, expect } from 'vitest';
import { unwrapMarkdownCodeBlocks } from '../src/lib/markdown';

describe('unwrapMarkdownCodeBlocks', () => {
  it('returns input unchanged when no markdown blocks present', () => {
    const input = 'Hello world';
    expect(unwrapMarkdownCodeBlocks(input)).toBe('Hello world');
  });

  it('unwraps simple text inside markdown block', () => {
    const input = '```markdown\ntest\n```';
    expect(unwrapMarkdownCodeBlocks(input)).toBe('\n\ntest\n\n\n');
  });

  it('preserves nested code blocks inside markdown block', () => {
    const input = '```markdown\ntest\n```tsx\nexport const A = () => <div></div>\n```\n```';
    const expected = '\n\ntest\n```tsx\nexport const A = () => <div></div>\n```\n\n\n';
    expect(unwrapMarkdownCodeBlocks(input)).toBe(expected);
  });

  it('handles unclosed markdown block', () => {
    const input = '```markdown\ntest\n```tsx\nexport const A = () => <div></div>\n```';
    const expected = '\n\ntest\n```tsx\nexport const A = () => <div></div>\n```\n\n';
    expect(unwrapMarkdownCodeBlocks(input)).toBe(expected);
  });

  it('handles multiple markdown blocks', () => {
    const input = '```markdown\nfirst\n```\nsome text\n```markdown\nsecond\n```';
    const expected = '\n\nfirst\n\n\nsome text\n\n\nsecond\n\n\n';
    expect(unwrapMarkdownCodeBlocks(input)).toBe(expected);
  });

  it('handles empty content inside markdown block', () => {
    const input = '```markdown\n```';
    expect(unwrapMarkdownCodeBlocks(input)).toBe('\n\n\n\n');
  });

  it('handles deeply nested code blocks', () => {
    const input = '```markdown\nouter\n```tsx\ninner tsx\n```\nafter inner\n```';
    const expected = '\n\nouter\n```tsx\ninner tsx\n```\nafter inner\n\n\n';
    expect(unwrapMarkdownCodeBlocks(input)).toBe(expected);
  });

  it('preserves content before and after markdown blocks', () => {
    const input = 'before\n```markdown\nmiddle\n```\nafter';
    const expected = 'before\n\n\nmiddle\n\n\nafter';
    expect(unwrapMarkdownCodeBlocks(input)).toBe(expected);
  });

  it('handles markdown block with multiple nested code blocks', () => {
    const input = '```markdown\ntext\n```tsx\ncode1\n```\nmore text\n```javascript\ncode2\n```\nfinal\n```';
    const expected = '\n\ntext\n```tsx\ncode1\n```\nmore text\n```javascript\ncode2\n```\nfinal\n\n\n';
    expect(unwrapMarkdownCodeBlocks(input)).toBe(expected);
  });

  it('handles bare (no-language) inner code fences without premature close', () => {
    const input = '```markdown\ntext\n```\necho hello\n```\nmore text\n```';
    const expected = '\n\ntext\n```\necho hello\n```\nmore text\n\n\n';
    expect(unwrapMarkdownCodeBlocks(input)).toBe(expected);
  });

  it('handles bare inner fences alongside language-tagged fences', () => {
    const input =
      '```markdown\nstart\n```\nbare code\n```\nmiddle\n```tsx\ntagged code\n```\nend\n```';
    const expected =
      '\n\nstart\n```\nbare code\n```\nmiddle\n```tsx\ntagged code\n```\nend\n\n\n';
    expect(unwrapMarkdownCodeBlocks(input)).toBe(expected);
  });

  it('handles multiple consecutive bare inner fence pairs', () => {
    const input = '```markdown\na\n```\nblock1\n```\nb\n```\nblock2\n```\nc\n```';
    const expected = '\n\na\n```\nblock1\n```\nb\n```\nblock2\n```\nc\n\n\n';
    expect(unwrapMarkdownCodeBlocks(input)).toBe(expected);
  });
});
