import torch
import torch.nn.functional as F

from generator import generate
from data_loader import CharDataset
from gpt import GPT

batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 50
n_embd = 384
n_head = 6
n_layer = 6

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

torch.manual_seed(1337)

with open('data/data_new.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
torch.save(data, "data/train.pt")
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

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
    num_heads=n_head
)
m = model.to(device)
m.train()

torch.set_float32_matmul_precision("high")

if device.type == "mps":
    torch.backends.mps.allow_tf32 = True


def get_loss(r, t):
    lgts = m(r)
    B, T, C = lgts.shape
    lss = F.cross_entropy(lgts.reshape(-1, lgts.size(-1)), t.view(B * T))
    return lgts, lss


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        source = train_data if split == 'train' else val_data
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            ix = torch.randint(len(source) - block_size, (batch_size,))
            x = torch.stack([source[i:i+block_size] for i in ix]).to(device)
            y = torch.stack([source[i+1:i+block_size+1] for i in ix]).to(device)
            _, loss = get_loss(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
data = torch.load("data/train.pt")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

for i, (x, y) in enumerate(loader):

    if i >= max_iters:
        break

    x = x.to(device)
    y = y.to(device)

    if i % eval_interval == 0 or i == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    logits, loss = get_loss(x, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

torch.save({
    "model_state_dict": m.state_dict(),
    "vocab_size": vocab_size,
    "chars": chars,
}, "data/brodiaga.pt")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(generate(m, context, block_size, max_new_tokens=500)[0].tolist()))
