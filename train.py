"""
PHASE 4: Training loop, tuned for a 4GB VRAM GPU / 8GB system RAM
=====================================================================
Run this on YOUR LAPTOP with your NVIDIA GTX GPU. Before running:
  1. pip install torch --index-url https://download.pytorch.org/whl/cu121
     (use the CUDA version matching your driver — see setup notes at bottom)
  2. Download the dataset (instructions below, or this script does it for you)
  3. Close other apps — your RAM is usually near full at idle, so free it up

VRAM-SAVING TECHNIQUES USED HERE (all necessary at 4GB):
  - Mixed precision (fp16) training: activations/gradients stored in 16-bit
    instead of 32-bit, roughly halving memory use, with minimal accuracy loss
  - Small batch size + gradient accumulation: simulate a larger batch size
    without needing the memory for it, by accumulating gradients over
    several small forward/backward passes before updating weights
  - Small model (~10-15M params) via model.py's default config
"""

import os
import time
import urllib.request
import torch
from model import GPT, GPTConfig
from tokenizer_char import CharTokenizer

# ---------------------------------------------------------------------------
# 1. CONFIG — tuned for 4GB VRAM. If you hit "CUDA out of memory", reduce
#    batch_size first, then block_size, then n_embd/n_layer.
# ---------------------------------------------------------------------------
DATA_PATH = "input.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

batch_size = 8           # reduced batch size for CPU training
block_size = 128         # context length in tokens
grad_accum_steps = 4     # effective batch size = batch_size * grad_accum_steps = 64
max_iters = 500         # reduced steps for quicker CPU training
eval_interval = 250      # how often to check validation loss
eval_iters = 50          # how many batches to average for eval loss
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoint.pt"

torch.manual_seed(1337)

# ---------------------------------------------------------------------------
# 2. DATA — download TinyShakespeare if not already present
# ---------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    print(f"Downloading dataset to {DATA_PATH}...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)

with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# 90/10 train/val split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Dataset: {len(text):,} characters, vocab size: {tokenizer.vocab_size}")
print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")


def get_batch(split):
    """Sample a random batch of (input, target) sequences.
    target is input shifted by one position — that's the entire supervision
    signal for language modeling: predict the next token at every position."""
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    """Average loss over several batches, for both splits, without
    gradient tracking (saves memory and gives a stabler loss estimate
    than a single noisy batch)."""
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# 3. MODEL, OPTIMIZER, MIXED PRECISION SCALER
# ---------------------------------------------------------------------------
config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_layer=4,
    n_head=4,
    n_embd=256,
    dropout=0.1,
)
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

# GradScaler prevents fp16 underflow during backprop — small gradient values
# get scaled up before backward(), then scaled back down before the optimizer
# step. This is what makes fp16 training numerically stable.
scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

if device == "cpu":
    print("\nWARNING: No CUDA GPU detected — training will be very slow on CPU.")
    print("Check your PyTorch install has CUDA support (see setup notes at bottom).\n")

# ---------------------------------------------------------------------------
# 4. TRAINING LOOP
# ---------------------------------------------------------------------------
print(f"\nTraining on device: {device}")
print(f"Effective batch size: {batch_size * grad_accum_steps} "
      f"({batch_size} x {grad_accum_steps} grad accum steps)\n")

t0 = time.time()
for it in range(max_iters):

    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss(model)
        elapsed = time.time() - t0
        print(f"step {it:5d} | train loss {losses['train']:.4f} | "
              f"val loss {losses['val']:.4f} | {elapsed:.1f}s elapsed")

        # Save a checkpoint — critical on a laptop, since training can be
        # interrupted (thermal throttling, sleep, etc.)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "iter": it,
            "vocab": tokenizer.stoi,
        }, checkpoint_path)

    optimizer.zero_grad(set_to_none=True)

    # Gradient accumulation: run several small forward/backward passes,
    # summing gradients, before one optimizer step. This simulates a
    # bigger batch size than would otherwise fit in 4GB VRAM.
    for micro_step in range(grad_accum_steps):
        xb, yb = get_batch("train")
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device=="cuda")):
            _, loss = model(xb, yb)
            loss = loss / grad_accum_steps  # normalize so gradients don't compound

        scaler.scale(loss).backward()

    # Gradient clipping prevents rare exploding-gradient steps from
    # destabilizing training — clips the total gradient norm to 1.0
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()

print(f"\nTraining complete. Final checkpoint saved to {checkpoint_path}")

"""
SETUP NOTES
============
1. Check your NVIDIA driver / CUDA version:
     nvidia-smi
   Look at the "CUDA Version" in the top right — install a PyTorch build
   that matches or is older than that version.

2. Install PyTorch with CUDA (example for CUDA 12.1):
     pip install torch --index-url https://download.pytorch.org/whl/cu121

3. Verify GPU is visible to PyTorch:
     python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

4. If you get "CUDA out of memory":
   - Reduce batch_size (try 8, then 4)
   - Reduce block_size (try 64)
   - Close other GPU-using apps (browser with hardware acceleration, etc.)

5. RAM is your other bottleneck at 8GB total. Close Chrome/heavy apps
   before training. Monitor via Task Manager while training runs.
"""
