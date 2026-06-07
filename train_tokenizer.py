from tokenizers import ByteLevelBPETokenizer
import os

CORPUS = "data/dataset.txt"
SAVE_DIR = "data/tokenizer"
VOCAB_SIZE = 8000
MIN_FREQ = 2

os.makedirs(SAVE_DIR, exist_ok=True)

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=[CORPUS],
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQ,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
)

tokenizer.save_model(SAVE_DIR)
print(f"Tokenizer saved to {SAVE_DIR}/")
print(f"Vocab size: {tokenizer.get_vocab_size()}")

# Quick sanity check
sample = "Белеет парус одинокий в тумане моря голубом"
enc = tokenizer.encode(sample)
print(f"\nSample: {sample}")
print(f"Tokens: {enc.tokens}")
print(f"IDs:    {enc.ids}")
print(f"Decoded: {tokenizer.decode(enc.ids)}")
