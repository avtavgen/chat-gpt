import torch

@torch.no_grad()
def generate(model, idx, block_size, max_new_tokens=200, temperature=0.8, top_k=40):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_token), dim=1)

    return idx
