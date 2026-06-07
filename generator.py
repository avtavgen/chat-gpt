import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(model, context, block_size, max_new_tokens=500, temperature=0.85, top_p=0.92)  -> torch.Tensor:
    model.eval()

    for _ in range(max_new_tokens):
        # Crop context to the last block_size tokens
        ctx = context[:, -block_size:]
        logits = model(ctx)[:, -1, :]  # (1, vocab_size)

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)  # (1, vocab_size)

        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumulative = sorted_probs.cumsum(dim=-1)

            # Remove tokens whose cumulative prob already exceeds top_p.
            # Shift right by one so the token that *crosses* the threshold
            # is kept (otherwise the distribution can sum to 0 on rare tokens).
            remove_mask = (cumulative - sorted_probs) > top_p
            sorted_probs[remove_mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

            next_token_sorted = torch.multinomial(sorted_probs, num_samples=1)  # (1, 1)
            next_token = sorted_idx.gather(-1, next_token_sorted)  # (1, 1)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        context = torch.cat([context, next_token], dim=1)

    return context
