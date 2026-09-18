// WordPiece tokenizer for BERT-family uncased models (basic tokenize:
// lowercase, strip accents, split CJK + punctuation; then greedy longest-match
// wordpiece against vocab.txt). Deterministic, no dependencies.

const MAX_WORD_CHARS = 100;

function isWhitespace(cp: number): boolean {
  return cp === 0x20 || cp === 0x09 || cp === 0x0a || cp === 0x0d || (cp >= 0x2000 && cp <= 0x200a) || cp === 0x2028 || cp === 0x2029 || cp === 0x00a0 || cp === 0x3000 || cp === 0xfeff;
}

function isControl(cp: number): boolean {
  return (cp < 0x20 && !isWhitespace(cp)) || (cp >= 0x7f && cp <= 0x9f);
}

function isCJK(cp: number): boolean {
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df) ||
    (cp >= 0x2a700 && cp <= 0x2b73f) ||
    (cp >= 0x2b740 && cp <= 0x2b81f) ||
    (cp >= 0x2b820 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x2f800 && cp <= 0x2fa1f)
  );
}

function isPunctuation(ch: string, cp: number): boolean {
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) return true;
  return /\p{P}/u.test(ch);
}

export class WordPieceTokenizer {
  private vocab = new Map<string, number>();
  clsId = 101;
  sepId = 102;
  unkId = 100;

  static async load(vocabUrl: string): Promise<WordPieceTokenizer> {
    const t = new WordPieceTokenizer();
    const text = await (await fetch(vocabUrl)).text();
    const lines = text.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const token = lines[i].replace(/\r$/, '');
      if (token.length > 0) t.vocab.set(token, i);
    }
    t.clsId = t.vocab.get('[CLS]') ?? 101;
    t.sepId = t.vocab.get('[SEP]') ?? 102;
    t.unkId = t.vocab.get('[UNK]') ?? 100;
    return t;
  }

  private basicTokens(text: string): string[] {
    // lowercase + strip accents (uncased model convention)
    const lowered = text.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
    const out: string[] = [];
    let current = '';
    const flush = () => {
      if (current) {
        out.push(current);
        current = '';
      }
    };
    for (const ch of lowered) {
      const cp = ch.codePointAt(0)!;
      if (isControl(cp)) continue;
      if (isWhitespace(cp)) {
        flush();
        continue;
      }
      if (isCJK(cp) || isPunctuation(ch, cp)) {
        flush();
        out.push(ch);
        continue;
      }
      current += ch;
    }
    flush();
    return out;
  }

  private wordPiece(word: string): string[] {
    if ([...word].length > MAX_WORD_CHARS) return ['[UNK]'];
    const pieces: string[] = [];
    let start = 0;
    const chars = [...word];
    while (start < chars.length) {
      let end = chars.length;
      let found: string | null = null;
      while (start < end) {
        const piece = (start > 0 ? '##' : '') + chars.slice(start, end).join('');
        if (this.vocab.has(piece)) {
          found = piece;
          break;
        }
        end--;
      }
      if (found === null) return ['[UNK]'];
      pieces.push(found);
      start = end;
    }
    return pieces;
  }

  /** Returns token strings and ids, including [CLS]/[SEP], capped at maxTokens total. */
  encode(text: string, maxTokens = 128): { tokens: string[]; ids: number[] } {
    const pieces: string[] = [];
    for (const word of this.basicTokens(text)) {
      for (const p of this.wordPiece(word)) pieces.push(p);
      if (pieces.length >= maxTokens - 2) break;
    }
    const capped = pieces.slice(0, maxTokens - 2);
    const tokens = ['[CLS]', ...capped, '[SEP]'];
    const ids = tokens.map((t) => this.vocab.get(t) ?? this.unkId);
    return { tokens, ids };
  }
}
