import argparse
import torch
import torch.nn.functional as F

from tokenizer import TiktokenWrapper
from gpt import GPT


parser = argparse.ArgumentParser(description="Sample from a trained Russian brodiaga GPT")

parser.add_argument("--checkpoint",   type=str,   default="data/brodiaga_best.pt")
parser.add_argument("--prompt",       type=str,   default="",
                    help="Seed text (default: empty)")
parser.add_argument("--tokens",       type=int,   default=500,
                    help="Tokens to generate (default: 500)")
parser.add_argument("--temperature",  type=float, default=0.85,
                    help="Sampling temperature (default: 0.85)")
parser.add_argument("--top_p",        type=float, default=0.92,
                    help="Nucleus cutoff, 1.0=off (default: 0.92)")
parser.add_argument("--rep_penalty",  type=float, default=1.2,
                    help="Repetition penalty ≥1.0, 1.0=off (default: 1.2)")
parser.add_argument("--n",            type=int,   default=1,
                    help="Independent samples to generate (default: 1)")
parser.add_argument("--beam",         type=int,   default=0,
                    help="Beam width for beam search, 0=nucleus sampling (default: 0)")

args = parser.parse_args()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

print(f"Loading: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location=device)
vocab_size = ckpt["vocab_size"]
model_config = ckpt.get(
    "model_config",
    {
        "model_dim": 360,
        "num_blocks": 6,
        "num_heads": 6,
        "num_kv_heads": 2,
        "block_size": 256,
    },
)
block_size = model_config["block_size"]

print(f"  step={ckpt.get('step', '?')}  val_loss={ckpt.get('val_loss', '?'):.4f}")
print(f"  config: {model_config}")

tok = TiktokenWrapper(ckpt.get("tokenizer_dir", "data/tokenizer"))

model = GPT(
    vocab_size     = vocab_size,
    context_length = block_size,
    model_dim      = model_config["model_dim"],
    num_blocks     = model_config["num_blocks"],
    num_heads      = model_config["num_heads"],
    num_kv_heads   = model_config["num_kv_heads"],
    dropout        = 0.0,   # always 0 at inference
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()
print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M\n")

prompt_ids = tok.encode(args.prompt) if args.prompt else []
if args.prompt:
    print(f"Prompt ({len(prompt_ids)} tokens): {args.prompt!r}\n")
else:
    print("No prompt — generating from scratch\n")


def apply_rep_penalty(logits: torch.Tensor, seen_ids: list[int], penalty: float) -> torch.Tensor:
    """
    Divide logits of already-seen tokens by `penalty`.
    penalty=1.0 → no effect. penalty=1.3 → 30% logit reduction per repeat.
    Applied before softmax so it interacts correctly with temperature.
    """
    if penalty == 1.0 or not seen_ids:
        return logits
    unique = torch.tensor(list(set(seen_ids)), dtype=torch.long, device=logits.device)
    logits[0, unique] /= penalty
    return logits


def apply_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filter — zero out tokens outside the top-p probability mass."""
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative  = sorted_probs.cumsum(dim=-1)
    remove_mask = (cumulative - sorted_probs) > top_p
    sorted_probs[remove_mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    # Scatter back to original vocab order
    out = torch.zeros_like(probs)
    out.scatter_(-1, sorted_idx, sorted_probs)
    return out


@torch.no_grad()
def nucleus_sample(seed_ids: list[int]) -> str:
    ctx = (
        torch.tensor(seed_ids, dtype=torch.long, device=device).unsqueeze(0)
        if seed_ids
        else torch.zeros((1, 1), dtype=torch.long, device=device)
    )

    for _ in range(args.tokens):
        logits = model(ctx[:, -block_size:])[:, -1, :]

        # 1. Repetition penalty
        logits = apply_rep_penalty(logits, ctx[0].tolist(), args.rep_penalty)

        # 2. Temperature
        logits = logits / args.temperature
        probs  = F.softmax(logits, dim=-1)

        # 3. Top-p nucleus filter
        if args.top_p < 1.0:
            probs = apply_top_p(probs, args.top_p)

        next_tok = torch.multinomial(probs, 1)
        ctx = torch.cat([ctx, next_tok], dim=1)

    generated = ctx[0, len(seed_ids) if seed_ids else 1:].tolist()
    return tok.decode(generated)


@torch.no_grad()
def beam_search(seed_ids: list[int], beam_width: int) -> str:
    init = (
        torch.tensor(seed_ids, dtype=torch.long, device=device)
        if seed_ids
        else torch.zeros(1, dtype=torch.long, device=device)
    )
    # beams: list of (log_prob, token_id_list)
    beams = [(0.0, init.tolist())]

    for _ in range(args.tokens):
        candidates = []
        for log_prob, seq in beams:
            ctx    = torch.tensor(seq[-block_size:], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(ctx)[:, -1, :]
            logprobs = F.log_softmax(logits, dim=-1)[0]  # (vocab,)

            # Take top beam_width continuations for this beam
            topk_lp, topk_ids = logprobs.topk(beam_width)
            for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                candidates.append((log_prob + lp, seq + [tid]))

        # Keep only the best beam_width candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]

    best_seq = beams[0][1]
    generated = best_seq[len(seed_ids) if seed_ids else 1:]
    return tok.decode(generated)


use_beam = args.beam > 0

if use_beam and args.n > 1:
    print("Note: beam search is deterministic — --n has no effect, generating once.\n")
    args.n = 1

for i in range(args.n):
    if args.n > 1:
        print(f"{'─' * 40} Sample {i + 1} {'─' * 40}")

    if args.prompt:
        print(args.prompt, end="")

    if use_beam:
        output = beam_search(prompt_ids, args.beam)
    else:
        output = nucleus_sample(prompt_ids)

    print(output)
    if args.n > 1:
        print()