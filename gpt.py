from transformer_block import TransformerBlock
import torch
import torch.nn as nn


class GPT(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, model_dim)
        self.position_embeddings = nn.Embedding(context_length, model_dim)
        self.transformer_blocks = nn.Sequential()
        for i in range(num_blocks):
            self.transformer_blocks.append(TransformerBlock(model_dim, num_heads))
        self.final_norm = nn.LayerNorm(model_dim)
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, context):
        B, T = context.shape

        tok_emb = self.word_embeddings(context)
        pos = torch.arange(T, device=context.device)
        pos_emb = self.position_embeddings(pos)

        x = tok_emb + pos_emb
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.vocab_projection(x)

        return logits
