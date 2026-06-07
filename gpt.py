from transformer_block import TransformerBlock
import torch
import torch.nn as nn


class GPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int, num_kv_heads: int = None, dropout: float = 0.1):
        super().__init__()

        if num_kv_heads is None:
            num_kv_heads = max(1, num_heads // 2)

        self.word_embeddings = nn.Embedding(vocab_size, model_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    model_dim, num_heads, num_kv_heads, context_length, dropout
                )
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

        # Weight tying
        self.vocab_projection.weight = self.word_embeddings.weight

        self._init_weights()

        print(
            f"GPT | vocab={vocab_size} | dim={model_dim} | blocks={num_blocks} | "
            f"heads={num_heads} | kv_heads={num_kv_heads} | rope=True | "
            f"params={sum(p.numel() for p in self.parameters()) / 1e6:.2f}M"
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        x = self.emb_dropout(self.word_embeddings(context))
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return self.vocab_projection(x)
