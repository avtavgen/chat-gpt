from tokenizers import ByteLevelBPETokenizer
import os
import re

CORPUS   = "data/dataset2.txt"
SAVE_DIR = "data/tokenizer"
VOCAB_SIZE = 16000
MIN_FREQ   = 2

POEM_SEP   = "<|poem|>"
STANZA_SEP = "<|stanza|>"

def normalize_poem(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    text  = re.sub(r'\n{3,}', '\n\n', '\n'.join(lines))
    return text.strip()

def format_poem(poem: str) -> str:
    """Split poem into stanzas (double newline) and rejoin with STANZA_SEP."""
    stanzas = re.split(r'\n{2,}', normalize_poem(poem))
    return f"\n{STANZA_SEP}\n".join(s.strip() for s in stanzas if s.strip())

def build_corpus(raw_poems: list[str]) -> str:
    return f"\n{POEM_SEP}\n".join(format_poem(p) for p in raw_poems)

os.makedirs(SAVE_DIR, exist_ok=True)

tokenizer = ByteLevelBPETokenizer(add_prefix_space=True)
tokenizer.train(
    files=[CORPUS],
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQ,
    special_tokens=[POEM_SEP, STANZA_SEP],
)

tokenizer.save_model(SAVE_DIR)
print(f"Tokenizer saved to {SAVE_DIR}/")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

def audit_tokenizer(tokenizer, poems: list[str]) -> None:
    lengths = [len(tokenizer.encode(p).ids) for p in poems]
    print(f"\nTokens per poem — min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)/len(lengths):.0f}")

    for sep in (POEM_SEP, STANZA_SEP):
        ids = tokenizer.encode(sep).ids
        status = "OK" if len(ids) == 1 else "SPLIT — fix this!"
        print(f"  {sep:12s} → token ID {ids}  [{status}]")

    print("\nCyrillic fragmentation check:")
    sample_words = ["любовь", "влюбленного", "кулуарах", "облака", "зарница", "подпевалой"]
    for w in sample_words:
        enc = tokenizer.encode(w)
        print(f"  {w:20s} → {len(enc.ids)} tokens: {enc.tokens}")

poems = [
    """Я впадаю в любовь,
Как в море впадает река,
И от этой любви,
Знаю я, никуда мне не деться!

Я впадаю в любовь,
Как впадающий в детство старик,
И совсем не боюсь,
Что она для кого-то потеха.""",

    """Я живу на земле
В ожидании чуда:
Белокрылой грозы
Над притихшей рекой.

Пронеслась над землей
Золотая зарница
И закрыла глаза
За далекой чертой.""",
]

corpus_text = build_corpus(poems)
print(f"\nFormatted corpus preview:\n{corpus_text}\n")

audit_tokenizer(tokenizer, poems)

# Round-trip check
sample = poems[0]
enc = tokenizer.encode(format_poem(sample))
decoded = tokenizer.decode(enc.ids)
print(f"\nRound-trip OK: {decoded == format_poem(sample)}")
print(f"Decoded:\n{decoded}")