import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--input",      type=str, default="data/dataset2.txt")
parser.add_argument("--output",     type=str, default="data/dataset_augmented.txt")
parser.add_argument("--sep",        type=str, default="\n\n",
                    help="String that separates poems in the file (default: blank line)")
parser.add_argument("--no_reverse", action="store_true",
                    help="Skip line-reversal augmentation")
parser.add_argument("--seed",       type=int, default=42)
args = parser.parse_args()

random.seed(args.seed)

with open(args.input, "r", encoding="utf-8") as f:
    raw = f.read()

# Split on separator, strip whitespace, drop empty chunks
poems = [p.strip() for p in raw.split(args.sep) if p.strip()]
print(f"Loaded {len(poems)} poems from {args.input}")

seen   = set()
unique = []
for p in poems:
    key = p.lower().replace(" ", "")
    if key not in seen:
        seen.add(key)
        unique.append(p)

print(f"After deduplication: {len(unique)} poems ({len(poems) - len(unique)} duplicates removed)")
poems = unique

augmented = list(poems)   # original order

# Shuffled copy — different context boundaries
shuffled = list(poems)
random.shuffle(shuffled)
augmented += shuffled
print(f"Added shuffled copy: {len(shuffled)} poems")

# Line-reversed copy — each poem's lines in reverse order
# e.g. "Line1\nLine2\nLine3" → "Line3\nLine2\nLine1"
# This teaches the model rhyme endings without needing new data.
if not args.no_reverse:
    reversed_poems = [
        "\n".join(p.split("\n")[::-1])
        for p in poems
    ]
    augmented += reversed_poems
    print(f"Added line-reversed copy: {len(reversed_poems)} poems")

random.shuffle(augmented)   # mix all copies together before writing

output_text = ("\n\n").join(augmented)
with open(args.output, "w", encoding="utf-8") as f:
    f.write(output_text)

original_chars   = len(raw)
augmented_chars  = len(output_text)
print(f"\nDone.")
print(f"  Original : {original_chars:,} chars  ({len(poems)} poems)")
print(f"  Augmented: {augmented_chars:,} chars  ({len(augmented)} poems)")
print(f"  Multiplier: {augmented_chars / original_chars:.1f}×")
print(f"  Written to: {args.output}")
print(f"\nNext step: set DATA_FILE = '{args.output}' in train.py")