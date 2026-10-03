"""
PHASE 6: Inference / text generation from your trained model
================================================================
Loads the checkpoint saved by train.py and generates new text.

SAMPLING STRATEGIES EXPLAINED:
- temperature: divides logits before softmax. Low (e.g. 0.3) = confident,
  repetitive, "safe" text. High (e.g. 1.2) = more random, creative, sometimes
  incoherent text. 1.0 = unmodified model distribution.
- top_k: only sample from the k most likely next tokens, discard the rest.
  Prevents the model from ever picking a wildly unlikely token.
"""

import torch
from model import GPT, GPTConfig
from tokenizer_char import CharTokenizer

checkpoint_path = "checkpoint.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config = checkpoint["config"]
stoi = checkpoint["vocab"]
itos = {i: ch for ch, i in stoi.items()}

model = GPT(config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Loaded checkpoint from step {checkpoint['iter']}")

# Rebuild a tokenizer-compatible encode/decode using the saved vocab
def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return ''.join(itos[i] for i in ids)


# Start generation from a prompt (or empty context)
prompt = "Explain what a large language model is"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

with torch.no_grad():
    output = model.generate(context, max_new_tokens=300, temperature=0.8, top_k=40)

print("\n--- Generated text ---\n")
print(decode(output[0].tolist()))
