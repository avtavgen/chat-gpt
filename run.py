import math
import os

import torch
import torch.nn.functional as F

from generator import generate
from data_loader import CharDataset
from gpt import GPT
from tokenizer import TiktokenWrapper

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 50
n_embd = 360
n_head = 6
n_layer = 6
n_kv_head = 2
min_lr = learning_rate / 10
warmup_iters = 250
dropout = 0.1
weight_decay = 0.1

CHECKPOINT_BEST = "data/brodiaga_best.pt"
CHECKPOINT_LAST = "data/brodiaga_last.pt"
DATA_FILE = "data/dataset2.txt"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

torch.manual_seed(1337)

tok = TiktokenWrapper()
vocab_size = tok.vocab_size
print(f"Vocab size: {vocab_size}")

with open(DATA_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

ids = tok.encode_ordinary(text)
data = torch.tensor(ids, dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

loader = torch.utils.data.DataLoader(
    CharDataset(data, block_size),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

model = GPT(
    vocab_size=vocab_size,
    context_length=block_size,
    model_dim=n_embd,
    num_blocks=n_layer,
    num_heads=n_head,
    num_kv_heads=n_kv_head,
    dropout=dropout,
)
m = model.to(device)

decay_params = [p for n, p in m.named_parameters() if p.dim() >= 2]
no_decay_params = [p for n, p in m.named_parameters() if p.dim() <  2]
optimizer = torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ],
    lr=learning_rate,
)

def get_lr(step: int) -> float:
    # 1. Linear warmup
    if step < warmup_iters:
        return learning_rate * step / max(1, warmup_iters)
    # 2. Cosine decay from learning_rate → min_lr
    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (learning_rate - min_lr)


def set_lr(step: int):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr

start_iter = 0
best_val_loss = float("inf")

if os.path.exists(CHECKPOINT_BEST):
    print(f"Resuming from {CHECKPOINT_BEST}")
    ckpt = torch.load(CHECKPOINT_BEST, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    start_iter = ckpt["step"] + 1
    best_val_loss = ckpt["val_loss"]
    print(f" Resumed at step {start_iter}, best val loss {best_val_loss:.4f}")

m.train()


def get_batch(source: torch.Tensor):
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def get_loss(x, y):
    logits = m(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    return logits, loss


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split, source in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(source)
            _, loss = get_loss(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out


def save_checkpoint(path: str, step: int, val_loss: float):
    torch.save(
        {
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "vocab_size": vocab_size,
            "tokenizer_dir": "data/tokenizer",
        },
        path,
    )

for i, (x, y) in enumerate(loader):
    if i < start_iter:
        continue
    if i >= max_iters:
        break

    x, y = x.to(device), y.to(device)
    lr = set_lr(i)

    if i % eval_interval == 0 or i == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {i:5d} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f} | "
            f"lr {lr:.2e}" + (" ← best" if losses["val"] < best_val_loss else "")
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            save_checkpoint(CHECKPOINT_BEST, i, best_val_loss)

        save_checkpoint(CHECKPOINT_LAST, i, losses["val"].item())

    logits, loss = get_loss(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    optimizer.step()

print("\nLoading best checkpoint for generation...")
ckpt = torch.load(CHECKPOINT_BEST, map_location=device)
m.load_state_dict(ckpt["model_state_dict"])
m.eval()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = generate(m, context, block_size, max_new_tokens=500)[0].tolist()
print("\n--- Generated ---")
print(tok.decode(generated_ids))
