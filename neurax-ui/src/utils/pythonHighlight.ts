/**
 * Syntax highlighting for the one language NEURAX generates.
 *
 * Written here rather than pulled in, for three reasons that are specific to
 * this application rather than general principle:
 *
 *  - There is exactly one language to colour. A highlighting library ships
 *    grammars for two hundred, and the studio would carry all of them to read
 *    a `model.py`.
 *  - The four NEURAX themes are defined as HSL tokens in `index.css`, and
 *    every library brings its own colour scheme that then has to be fought
 *    into agreement with them. Emitting class names that resolve to the
 *    theme's own `--chart-*` palette means the editor is themed by
 *    construction, in all four, including the two nobody remembers to check.
 *  - A tokenizer for Python's *lexical* structure is small and testable. It
 *    is not a parser and does not pretend to be: it knows strings, comments,
 *    numbers, keywords and decorators, which is the whole of what colour is
 *    for when you are reading a model definition.
 *
 * The one rule that matters for correctness: a token that opens a string or a
 * comment consumes to its end before anything else is considered. Highlighting
 * a `#` inside a string as a comment, or a keyword inside one as a keyword, is
 * the classic failure and it makes the file read as broken when it is fine.
 */

export type TokenKind =
  | 'plain'
  | 'keyword'
  | 'builtin'
  | 'string'
  | 'number'
  | 'comment'
  | 'decorator'
  | 'definition';

export interface Token {
  kind: TokenKind;
  text: string;
}

/** Reserved words. `self` is not one, and is deliberately absent. */
const KEYWORDS = new Set([
  'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del',
  'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in',
  'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while',
  'with', 'yield',
]);

/** Names that carry meaning in generated model code and read better lit. */
const BUILTINS = new Set([
  'True', 'False', 'None', 'self', 'super', 'int', 'float', 'str', 'bool', 'list',
  'dict', 'tuple', 'set', 'len', 'range', 'print', 'isinstance', 'enumerate', 'zip',
  'sum', 'min', 'max', 'abs', 'round', 'sorted', 'type', 'getattr', 'setattr',
]);

const IDENTIFIER_START = /[A-Za-z_]/;
const IDENTIFIER_PART = /[A-Za-z0-9_]/;
const DIGIT = /[0-9]/;

/**
 * Split Python source into coloured runs.
 *
 * Every character of the input appears in exactly one token, in order, so
 * joining the token texts reproduces the source byte for byte. That property
 * is what lets the highlighted layer sit under a textarea and stay aligned
 * with it — a highlighter that drops or reorders a character misaligns the
 * caret from that point on, and the editor becomes unusable rather than
 * merely ugly.
 */
export function tokenizePython(source: string): Token[] {
  const tokens: Token[] = [];
  let plain = '';

  const flush = () => {
    if (plain) {
      tokens.push({ kind: 'plain', text: plain });
      plain = '';
    }
  };
  const push = (kind: TokenKind, text: string) => {
    flush();
    tokens.push({ kind, text });
  };

  let i = 0;
  while (i < source.length) {
    const char = source[i];

    // ── Comments: to the end of the line, whatever is in them ──────────
    if (char === '#') {
      const end = source.indexOf('\n', i);
      const stop = end === -1 ? source.length : end;
      push('comment', source.slice(i, stop));
      i = stop;
      continue;
    }

    // ── Strings, including triple-quoted and prefixed (f, r, b, rb…) ───
    const stringStart = matchStringStart(source, i);
    if (stringStart) {
      const end = findStringEnd(source, stringStart.bodyAt, stringStart.quote);
      push('string', source.slice(i, end));
      i = end;
      continue;
    }

    // ── Decorators: `@` immediately followed by a name ─────────────────
    if (char === '@' && IDENTIFIER_START.test(source[i + 1] ?? '')) {
      let j = i + 1;
      while (j < source.length && (IDENTIFIER_PART.test(source[j]) || source[j] === '.')) j++;
      push('decorator', source.slice(i, j));
      i = j;
      continue;
    }

    // ── Numbers, including 0x, 1e-4, 1_000 and a leading dot ───────────
    if (DIGIT.test(char) || (char === '.' && DIGIT.test(source[i + 1] ?? ''))) {
      let j = i;
      while (j < source.length && /[0-9a-fA-FxXoObB._+-]/.test(source[j])) {
        // `+`/`-` belong to the number only as an exponent's sign; otherwise
        // they are operators and `1-2` would swallow the whole expression.
        if ((source[j] === '+' || source[j] === '-') && !/[eE]/.test(source[j - 1] ?? '')) break;
        j++;
      }
      push('number', source.slice(i, j));
      i = j;
      continue;
    }

    // ── Identifiers ───────────────────────────────────────────────────
    if (IDENTIFIER_START.test(char)) {
      let j = i;
      while (j < source.length && IDENTIFIER_PART.test(source[j])) j++;
      const word = source.slice(i, j);

      if (KEYWORDS.has(word)) {
        push('keyword', word);
      } else if (BUILTINS.has(word)) {
        push('builtin', word);
      } else if (followsDefiningKeyword(tokens)) {
        // The name right after `class` or `def` — the thing a reader is
        // scanning for when they open a model file.
        push('definition', word);
      } else {
        plain += word;
      }
      i = j;
      continue;
    }

    plain += char;
    i++;
  }

  flush();
  return tokens;
}

/** Is the previous meaningful token `class` or `def`? */
function followsDefiningKeyword(tokens: Token[]): boolean {
  for (let k = tokens.length - 1; k >= 0; k--) {
    const token = tokens[k];
    // Only whitespace may sit between the keyword and the name.
    if (token.kind === 'plain' && /^\s+$/.test(token.text)) continue;
    return token.kind === 'keyword' && (token.text === 'class' || token.text === 'def');
  }
  return false;
}

/** A string beginning at `i`, with any prefix, or `null`. */
function matchStringStart(source: string, i: number): { quote: string; bodyAt: number } | null {
  let j = i;
  // At most two prefix letters: `f`, `r`, `b`, `u`, and pairs like `rb`.
  while (j < i + 2 && /[fFrRbBuU]/.test(source[j] ?? '')) j++;
  const char = source[j];
  if (char !== '"' && char !== "'") return null;

  const triple = source.slice(j, j + 3);
  if (triple === '"""' || triple === "'''") {
    return { quote: triple, bodyAt: j + 3 };
  }
  return { quote: char, bodyAt: j + 1 };
}

/** Where a string that opened with `quote` ends — past the closing quote. */
function findStringEnd(source: string, from: number, quote: string): number {
  let i = from;
  while (i < source.length) {
    if (source[i] === '\\') {
      i += 2;
      continue;
    }
    if (source.startsWith(quote, i)) return i + quote.length;
    // A single-quoted string cannot cross a newline. Ending it at the line
    // break rather than running to the end of the file is what keeps one
    // unbalanced quote from colouring the whole rest of the document.
    if (quote.length === 1 && source[i] === '\n') return i;
    i++;
  }
  return source.length;
}

/**
 * The theme class for each kind.
 *
 * Colours come from the same `--chart-*` palette the analysis charts use, so
 * the editor is themed by whatever `index.css` says rather than by a second
 * palette that has to be kept in agreement with the first.
 */
export const TOKEN_CLASS: Record<TokenKind, string> = {
  plain: 'text-foreground',
  keyword: 'text-[hsl(var(--chart-5))]',
  builtin: 'text-[hsl(var(--chart-2))]',
  string: 'text-[hsl(var(--chart-3))]',
  number: 'text-[hsl(var(--chart-6))]',
  comment: 'text-muted-foreground/70 italic',
  decorator: 'text-[hsl(var(--chart-1))]',
  definition: 'text-[hsl(var(--chart-7))] font-semibold',
};
