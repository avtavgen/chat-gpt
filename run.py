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
n_embd = 386
n_head = 6
n_layer = 6
dropout = 0.1

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

torch.manual_seed(1337)

tok = TiktokenWrapper()
vocab_size = tok.vocab_size
print(f"Vocab size: {vocab_size}")

with open('data/dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

ids = tok.encode_ordinary(text)
data = torch.tensor(ids, dtype=torch.long)
torch.save(data, "data/train.pt")
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

loader = torch.utils.data.DataLoader(
    CharDataset(data, block_size),
    batch_size=32,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

model = GPT(
    vocab_size=vocab_size,
    context_length=block_size,
    model_dim=n_embd,
    num_blocks=n_layer,
    num_heads=n_head,
    dropout=dropout,
)
m = model.to(device)
m.train()

total_params = sum(p.numel() for p in m.parameters())
print(f"Parameters: {total_params/1e6:.2f}M")

torch.set_float32_matmul_precision("high")

if device.type == "mps":
    torch.backends.mps.allow_tf32 = True


def get_batch(source: torch.Tensor):
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def get_loss(x, y):
    logits = m(x)  # (B, T, C)
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

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

for i, (x, y) in enumerate(loader):
    if i >= max_iters:
        break

    x, y = x.to(device), y.to(device)

    if i % eval_interval == 0 or i == max_iters - 1:
        losses = estimate_loss()
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"step {i:5d} | "
            f"train {losses['train']:.4f} | "
            f"val {losses['val']:.4f} | "
            f"lr {lr_now:.2e}"
        )

    logits, loss = get_loss(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

torch.save(
    {
        "model_state_dict": m.state_dict(),
        "vocab_size":       vocab_size,
        "tokenizer_dir":    "data/tokenizer",
    },
    "data/brodiaga.pt",
)

m.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_ids = generate(m, context, block_size, max_new_tokens=500)[0].tolist()
print(tok.decode(generated_ids))
