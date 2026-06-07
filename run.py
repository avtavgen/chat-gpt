import torch
import torch.nn.functional as F

from generator import generate
from gpt import GPT

batch_size = 64
block_size = 128
max_iters = 3000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 312
n_head = 4
n_layer = 4

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

torch.manual_seed(1337)

with open('data/data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    raw_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(raw_data) - block_size, (batch_size,))
    a = torch.stack([raw_data[i:i+block_size] for i in ix])
    b = torch.stack([raw_data[i+1:i+1+block_size] for i in ix])
    return a.to(device), b.to(device)


def get_loss(r, t):
    lgts = model(r)
    B, T, C = lgts.shape
    lss = F.cross_entropy(lgts.view(B * T, C), t.view(B * T))
    return lgts, lss


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        lsses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            r, t = get_batch(split)
            lgits, lss = get_loss(r, t)
            lsses[k] = lss.item()
        out[split] = lsses.mean()
    model.train()
    return out


model = GPT(
    vocab_size=vocab_size,
    context_length=block_size,
    model_dim=n_embd,
    num_blocks=n_layer,
    num_heads=n_head
)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for i in range(max_iters):
    if i % eval_interval == 0 or i == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    x, y = get_batch('train')
    logits, loss = get_loss(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(generate(m, context, block_size, max_new_tokens=500)[0].tolist()))
