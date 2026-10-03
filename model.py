"""
PHASE 3: The Transformer, built from scratch in raw PyTorch
==============================================================
This is a GPT-style decoder-only transformer. Every LLM you've used
(GPT-4, Claude, LLaMA) is fundamentally this architecture, scaled up
massively. Nothing here is simplified conceptually — only the SIZE
is small, tuned for a 4GB VRAM GPU.

THE CORE IDEA IN ONE SENTENCE:
Take a sequence of tokens, turn each into a vector (embedding), let each
token "look at" (attend to) earlier tokens to gather context, then predict
the next token. Stack this attention+processing step multiple times (layers)
to build up increasingly abstract understanding.

CONFIG SIZED FOR YOUR GPU (4GB VRAM):
    n_embd=256, n_head=4, n_layer=4, block_size=128
    -> roughly 10-15M parameters, comfortably fits in 4GB with room
       for activations, gradients, and optimizer states.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 65        # set this to your tokenizer's vocab_size
    block_size: int = 128       # max context length (how many tokens the model sees at once)
    n_layer: int = 4            # number of transformer blocks stacked
    n_head: int = 4             # number of attention heads per block
    n_embd: int = 256           # embedding dimension (the "width" of the model)
    dropout: float = 0.1        # regularization: randomly zero some activations during training


class CausalSelfAttention(nn.Module):
    """
    THE HEART OF THE TRANSFORMER.

    For every token, this computes: "given everything I've seen so far
    (not the future — that's what 'causal' means), which earlier tokens
    should I pay attention to, and how much?"

    Mechanically: each token produces a Query (what am I looking for?),
    a Key (what do I contain?), and a Value (what do I actually offer?).
    We compare every token's Query against every other token's Key to get
    attention scores, mask out future positions (causal), softmax to turn
    scores into probabilities, then take a weighted sum of Values.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # One linear layer produces Q, K, V all at once (efficiency trick)
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection after attention is computed
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower-triangular matrix of 1s. Token i can only
        # attend to tokens 0..i, never i+1 onward (can't see the future).
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time (sequence length), Channels (n_embd)

        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape into (B, n_head, T, head_dim) so each head attends independently.
        # Multiple heads let the model learn different KINDS of relationships
        # simultaneously (e.g. one head tracks syntax, another tracks topic).
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: QK^T / sqrt(head_dim)
        # The scaling prevents the softmax from becoming too "peaky" as
        # head_dim grows, which would make gradients vanish.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # Apply the causal mask: set future positions to -inf so softmax
        # gives them ~0 probability.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = att @ v  # weighted sum of values -> (B, n_head, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back
        out = self.resid_dropout(self.out_proj(out))
        return out


class MLP(nn.Module):
    """
    Feed-forward network applied to each token independently (after
    attention has mixed information ACROSS tokens, this mixes information
    WITHIN each token's representation). Standard design: expand to 4x
    width, apply nonlinearity, project back down.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))


class Block(nn.Module):
    """
    One transformer block: attention (mix info across tokens) followed by
    MLP (process each token's info). LayerNorm + residual ("skip") connections
    around each sub-layer are what make deep stacks of these actually
    trainable — without residuals, gradients vanish after a few layers.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-norm design (norm BEFORE the sub-layer, not after) — this is
        # what modern GPTs use, it trains more stably than the original
        # "Attention Is All You Need" post-norm design.
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between the input embedding and output
        # projection. Cuts parameter count and is standard practice (GPT-2 does this).
        self.token_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized: {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"sequence length {T} exceeds block_size {self.config.block_size}"

        pos = torch.arange(0, T, device=idx.device)
        tok_emb = self.token_emb(idx)   # (B, T, n_embd) — "what token is this"
        pos_emb = self.pos_emb(pos)     # (T, n_embd)    — "where in the sequence is this"
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size) — raw scores per possible next token

        loss = None
        if targets is not None:
            # Cross-entropy between predicted next-token distribution and
            # the actual next token, averaged over every position in the batch.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation: predict one token, append it, repeat."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # only need the last position's prediction

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


if __name__ == "__main__":
    # Sanity check: build the model and run one forward pass
    config = GPTConfig(vocab_size=65, block_size=128, n_layer=4, n_head=4, n_embd=256)
    model = GPT(config)

    dummy_input = torch.randint(0, 65, (2, 32))   # batch of 2, sequence length 32
    dummy_targets = torch.randint(0, 65, (2, 32))
    logits, loss = model(dummy_input, dummy_targets)
    print(f"Logits shape: {logits.shape}")   # (2, 32, 65)
    print(f"Loss: {loss.item():.4f}")
