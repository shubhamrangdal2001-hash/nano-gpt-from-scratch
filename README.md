# Building an LLM From Scratch — Complete Roadmap
**Tuned for: NVIDIA GTX (4GB VRAM), 8GB system RAM**

This is an end-to-end, hands-on path to building and training a real
(small) GPT-style language model, understanding every component. The
code in this folder is a complete working implementation — not pseudocode.

---

## How the pieces fit together

```
tokenizer_char.py  →  model.py  →  train.py  →  generate.py
(text → numbers)     (architecture)  (learning)   (sampling)
```

| File | Phase | What it does |
|---|---|---|
| `tokenizer_char.py` | 2a | Simplest tokenizer: 1 character = 1 token |
| `tokenizer_bpe.py` | 2b | Real subword tokenizer (what GPT actually uses) |
| `model.py` | 3 | The transformer architecture itself |
| `train.py` | 4 | Training loop, tuned for 4GB VRAM |
| `generate.py` | 6 | Generate text from your trained model |

---

## Phase 0 — Environment setup (do this first)

```bash
# 1. Check your GPU and CUDA version
nvidia-smi
# Look at "CUDA Version" top-right of the output

# 2. Install PyTorch with CUDA support (adjust cu121 to match your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Verify PyTorch sees your GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True   NVIDIA GeForce GTX ...
```

If `torch.cuda.is_available()` returns `False`, your PyTorch build doesn't
match your CUDA driver — reinstall with the correct `cu1xx` tag from
[pytorch.org/get-started](https://pytorch.org/get-started/locally/).

---

## Phase 1 — Foundations (concepts to understand before coding)

You don't need to master these mathematically, but you should understand
*what problem each one solves*, because they show up directly in `model.py`:

- **Vectors/embeddings**: every token becomes a list of numbers (a point in
  high-dimensional space) so the network can do math on it
- **Matrix multiplication**: how embeddings get transformed layer to layer
- **Softmax**: converts raw scores into a probability distribution that
  sums to 1 — used both in attention and in the final next-token prediction
- **Gradient descent & backpropagation**: how the model learns — nudge
  every weight slightly in the direction that reduces the loss
- **Cross-entropy loss**: measures how wrong the model's predicted
  probability distribution was compared to the actual next token

If any of these are shaky, spend 2-3 days on 3Blue1Brown's neural network
series (visual, intuitive) before moving on.

---

## Phase 2 — Tokenization

Run the sanity checks:
```bash
python tokenizer_char.py
python tokenizer_bpe.py
```

**Why two tokenizers?** `train.py` uses the character-level one because
it's simpler to debug and needs zero setup. Once your model trains
successfully end-to-end, try swapping in `tokenizer_bpe.py` — you'll see
sequences get shorter (more meaning per token) and, with enough training
data, generation quality improve.

**The core insight**: an LLM never sees text. It only ever sees integers.
Tokenization is the entire bridge between human language and the math
that follows.

---

## Phase 3 — The Transformer architecture (`model.py`)

Run the sanity check:
```bash
python model.py
# Should print: Model initialized: ~3.2M parameters (small due to weight tying — see below)
#               Logits shape: torch.Size([2, 32, 65])
#               Loss: ~4.2 (this is roughly ln(65), i.e. random guessing — expected before training)
```

**Walk through the file top to bottom, in this order of importance:**

1. **`CausalSelfAttention`** — the actual innovation of the transformer
   architecture. Every token computes a Query, Key, and Value vector.
   Attention score between token A and token B = how well A's Query
   matches B's Key. These scores become weights for a weighted average
   of Values — that weighted average is what gets passed forward. The
   causal mask ensures token A can never attend to tokens that come
   after it (otherwise the model could "cheat" by looking at the answer).

2. **`MLP`** — after attention mixes information *between* tokens, the
   MLP processes each token's representation *independently*, expanding
   to 4x width and back down. This is where a lot of the model's
   "knowledge" ends up stored, per recent interpretability research.

3. **`Block`** — one attention + one MLP, wrapped in residual connections
   (`x = x + sublayer(x)`) and LayerNorm. The residual connections are
   what make it possible to stack many of these blocks without gradients
   vanishing — this is arguably as important as attention itself.

4. **`GPT`** — stacks `n_layer` blocks, adds token + positional embeddings
   at the start (position matters because attention alone has no sense of
   order), and a final linear layer that projects back to vocabulary size
   to produce next-token probabilities.

**Try this exercise**: change `n_layer` from 4 to 2, rerun `python model.py`,
watch the parameter count drop. Then try `n_embd` from 256 to 512, watch it
roughly quadruple (parameter count in linear layers scales with the square
of embedding dimension) — this is a concrete feel for *why* scaling LLMs
gets expensive so fast.

---

## Phase 4 — Training (`train.py`)

```bash
python train.py
```

What happens: downloads TinyShakespeare (~1MB of Shakespeare text) if not
present, builds the char tokenizer, then trains for 3000 steps. On a 4GB
GTX GPU this should take somewhere from 15-45 minutes depending on your
exact GPU model.

**Watch the loss numbers.** Starting loss should be around 4.2 (≈ln(65),
random guessing over 65 possible characters). By step 3000 you're looking
for something in the 1.4-1.7 range. If `val loss` stops improving while
`train loss` keeps dropping, that's overfitting — the model is memorizing
rather than generalizing (expected at this tiny scale/dataset, and fine
for learning purposes).

**Key techniques in this script, and why each is necessary at 4GB VRAM:**

- **Mixed precision (`torch.autocast` + `GradScaler`)**: stores
  activations in 16-bit floats instead of 32-bit during the forward pass,
  roughly halving memory use. `GradScaler` compensates for the reduced
  numerical range of fp16 by scaling gradients up before backward() and
  down before the optimizer step, preventing tiny gradients from
  underflowing to zero.

- **Gradient accumulation**: with `batch_size=16` and `grad_accum_steps=4`,
  you get the learning stability of an effective batch size of 64, without
  ever needing 64 sequences in VRAM simultaneously — you process 16 at a
  time, accumulate gradients, and only update weights every 4th micro-step.

- **Gradient clipping**: caps the total gradient norm at 1.0 before each
  optimizer step, preventing rare large-gradient steps from destabilizing
  training (a common failure mode in transformer training).

- **Checkpointing every `eval_interval` steps**: on a laptop, training can
  get interrupted (thermal throttling, sleep, closing the lid). Frequent
  checkpoints mean you never lose more than a few minutes of progress.

**If you hit "CUDA out of memory":** reduce, in this order —
`batch_size` (16→8→4) → `block_size` (128→64) → `n_embd`/`n_layer` in
train.py's config.

---

## Phase 5 — Scaling concepts (study, don't necessarily run)

You won't train a billion-parameter model on this laptop, but you should
understand *why* real LLMs need massive compute, for interview conversations
and general fluency:

- **Chinchilla scaling laws**: compute-optimal training roughly balances
  model size and dataset size — a model trained on too little data for its
  size is wasting parameters (this is exactly what a tiny model on
  TinyShakespeare demonstrates in miniature)
- **Distributed training (DDP/FSDP)**: splitting a model or batch across
  multiple GPUs when one isn't enough — conceptually similar to your
  gradient accumulation trick, just across machines instead of across steps
- **Mixed precision at scale**: bf16 is generally preferred over fp16 for
  large-scale training because it has a wider dynamic range (this is why
  `GradScaler` exists for fp16 — bf16 usually doesn't need it)

If you want a taste of "bigger," swap this same code onto Google Colab's
free T4 (15GB VRAM) and try `n_embd=384, n_layer=6, batch_size=32` with
the TinyStories dataset instead of TinyShakespeare — noticeably more
coherent output.

---

## Phase 6 — Generation (`generate.py`)

```bash
python generate.py
```

Loads your checkpoint and generates 300 characters starting from the
prompt `"ROMEO:"`. Try adjusting `temperature` (0.3 for conservative,
1.2 for wild) and `top_k` (10 for narrow, None for unrestricted) to feel
their effect directly.

**What `generate()` in `model.py` is actually doing**: at each step, run
the full model forward, take only the prediction at the *last* position,
convert to probabilities, sample one token, append it to the sequence,
repeat. This is exactly how ChatGPT/Claude produce text token-by-token —
same core loop, vastly larger model.

---

## Suggested timeline

| Week | Focus |
|---|---|
| 1 | Phase 0-1: environment + math refresh |
| 1-2 | Phase 2: both tokenizers, understand the difference |
| 2-3 | Phase 3: read `model.py` line by line, do the parameter-count exercise |
| 3 | Phase 4: get a successful training run, watch loss curves |
| 4 | Phase 5-6: generation, sampling experiments, optionally try Colab for a bigger run |

## Where this connects to your other work

This is a strong complement to MechSage — it demonstrates you understand
what's happening *inside* the LLMs you're orchestrating with LangGraph/RAG
in your other projects, which is exactly the kind of depth that
differentiates an "Agentic AI Engineer" from someone who only calls APIs.
Consider adding a short writeup + your loss curves to your GitHub as a
portfolio piece.
