/**
 * A code editor, built from a textarea and a coloured layer behind it.
 *
 * The technique is old and it is the right one here: a `<pre>` holding
 * highlighted tokens sits underneath a `<textarea>` whose own text is
 * transparent but whose caret and selection are not. The browser handles
 * every editing behaviour — caret movement, selection, undo, IME, accessibility
 * — because it really is a textarea, and the colour is painted underneath.
 *
 * The alternative was an editor library. It was not taken: NEURAX generates
 * exactly one language, the four themes are HSL tokens that every library
 * would have to be fought into agreeing with, and the studio should not carry
 * two hundred grammars to read a `model.py`. See `pythonHighlight.ts`.
 *
 * The two things that make or break this approach:
 *
 *  - **The layers must measure identically.** Same font, size, line height,
 *    padding, wrapping and tab size, or the caret drifts from the text under
 *    it. Every one of those is set on both, from one place.
 *  - **The highlighter must not lose a character.** Guaranteed by
 *    `tokenizePython`, which is tested on exactly that property.
 */
import { useCallback, useEffect, useMemo, useRef } from 'react';

import { TOKEN_CLASS, tokenizePython } from '@/utils/pythonHighlight.ts';
import { cn } from '@/lib/utils.ts';

interface CodeEditorProps {
  value: string;
  /** Absent for a file that is shown but not edited. */
  onChange?: (next: string) => void;
  readOnly?: boolean;
  /** Colour only Python. Anything else is shown as plain monospaced text
   *  rather than lit by a tokenizer that does not know its rules. */
  language?: 'python' | 'plain';
  className?: string;
  ariaLabel?: string;
}

/** One place, so the two layers cannot drift apart. */
const SHARED_TEXT = 'font-mono text-[12px] leading-[1.6] tracking-normal';
const SHARED_BOX = 'p-3 whitespace-pre break-normal';

export function CodeEditor({
  value,
  onChange,
  readOnly = false,
  language = 'python',
  className,
  ariaLabel = 'Model code',
}: CodeEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const highlightRef = useRef<HTMLPreElement>(null);
  const gutterRef = useRef<HTMLDivElement>(null);

  const tokens = useMemo(
    () => (language === 'python' ? tokenizePython(value) : [{ kind: 'plain' as const, text: value }]),
    [language, value],
  );

  // Counted from the text, not from the tokens: a file's last line has no
  // trailing newline, and `split` is the only count that agrees with what the
  // textarea will actually render.
  const lineCount = useMemo(() => value.split('\n').length, [value]);

  /** Keep the painted layer and the gutter under the text being edited. */
  const syncScroll = useCallback(() => {
    const source = textareaRef.current;
    if (!source) return;
    if (highlightRef.current) {
      highlightRef.current.scrollTop = source.scrollTop;
      highlightRef.current.scrollLeft = source.scrollLeft;
    }
    if (gutterRef.current) gutterRef.current.scrollTop = source.scrollTop;
  }, []);

  // A file switched underneath the editor arrives scrolled to wherever the
  // last one was. Reset both layers together rather than leaving the colour
  // and the text at different offsets.
  useEffect(() => {
    syncScroll();
  }, [value, syncScroll]);

  /**
   * Tab inserts spaces instead of leaving the editor.
   *
   * In Python this is not a convenience — a file you cannot indent is a file
   * you cannot edit. Shift+Tab dedents. Tab still moves focus when the editor
   * is read-only, so keyboard navigation is never trapped in a pane there is
   * no reason to be typing in.
   */
  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key !== 'Tab' || readOnly || !onChange) return;
      event.preventDefault();

      const field = event.currentTarget;
      const { selectionStart, selectionEnd } = field;
      const lineStart = value.lastIndexOf('\n', selectionStart - 1) + 1;

      if (event.shiftKey) {
        const indent = value.slice(lineStart, lineStart + 4);
        const remove = indent.match(/^ {1,4}/)?.[0].length ?? 0;
        if (remove === 0) return;
        onChange(value.slice(0, lineStart) + value.slice(lineStart + remove));
        queueMicrotask(() => {
          field.selectionStart = Math.max(lineStart, selectionStart - remove);
          field.selectionEnd = Math.max(lineStart, selectionEnd - remove);
        });
        return;
      }

      onChange(value.slice(0, selectionStart) + '    ' + value.slice(selectionEnd));
      queueMicrotask(() => {
        field.selectionStart = selectionStart + 4;
        field.selectionEnd = selectionStart + 4;
      });
    },
    [onChange, readOnly, value],
  );

  return (
    <div className={cn('relative flex min-h-0 overflow-hidden bg-card', className)}>
      {/* Line numbers. Their own column so the text never reflows around
          them, and not selectable, so copying the file copies the file. */}
      <div
        ref={gutterRef}
        aria-hidden
        className={cn(
          SHARED_TEXT,
          'shrink-0 select-none overflow-hidden border-r border-border bg-muted/30 py-3 pl-3 pr-2 text-right text-muted-foreground/50',
        )}
      >
        {Array.from({ length: lineCount }, (_, i) => (
          <div key={i}>{i + 1}</div>
        ))}
      </div>

      <div className="relative min-w-0 flex-1">
        <pre
          ref={highlightRef}
          aria-hidden
          className={cn(SHARED_TEXT, SHARED_BOX, 'pointer-events-none absolute inset-0 m-0 overflow-hidden')}
        >
          {tokens.map((token, i) => (
            <span key={i} className={TOKEN_CLASS[token.kind]}>
              {token.text}
            </span>
          ))}
          {/* One trailing newline so the last line can scroll clear of the
              bottom edge, exactly as the textarea's own does. */}
          {'\n'}
        </pre>

        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange?.(e.target.value)}
          onScroll={syncScroll}
          onKeyDown={handleKeyDown}
          readOnly={readOnly || !onChange}
          spellCheck={false}
          autoComplete="off"
          autoCorrect="off"
          autoCapitalize="off"
          aria-label={ariaLabel}
          className={cn(
            SHARED_TEXT,
            SHARED_BOX,
            // Transparent text over the painted layer; the caret and the
            // selection stay visible and are the browser's own.
            'absolute inset-0 h-full w-full resize-none border-0 bg-transparent text-transparent caret-foreground outline-none',
            'selection:bg-primary/25 focus-visible:ring-1 focus-visible:ring-ring',
            'overflow-auto scrollbar-thin',
          )}
        />
      </div>
    </div>
  );
}
