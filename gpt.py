from transformer_block import TransformerBlock
import torch
import torch.nn as nn


class GPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, model_dim)
        self.position_embeddings = nn.Embedding(context_length, model_dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(model_dim, num_heads, context_length, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.final_norm = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

        self.vocab_projection.weight = self.word_embeddings.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-2 style weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, context):
        B, T = context.shape

        tok_emb = self.word_embeddings(context)
        pos_emb = self.position_embeddings(
            torch.arange(T, device=context.device)
        )

        x = self.emb_dropout(tok_emb + pos_emb)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        return self.vocab_projection(x)
