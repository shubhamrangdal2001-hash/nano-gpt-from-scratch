"""
PHASE 2b: Byte Pair Encoding (BPE) tokenizer, built from scratch
==================================================================
This is what GPT-2, GPT-3, GPT-4, and LLaMA actually use (in more optimized
form). The algorithm is simple: start with individual bytes/characters as
tokens, then repeatedly merge the MOST FREQUENT adjacent pair into a new
token. Do this N times, and you get a vocabulary of N+256 tokens that
captures common subwords ("ing", "tion", "the ") as single tokens.

WHY IT'S BETTER THAN CHAR-LEVEL:
"the" might become 1 token instead of 3, "hello" might become 1-2 tokens
instead of 5. This means more meaning packed into the same context length.

ALGORITHM (this is the entire idea):
1. Start: represent text as a list of byte values (0-255)
2. Count all adjacent pairs of tokens, find the most frequent pair
3. Replace every occurrence of that pair with a new token id
4. Repeat steps 2-3 for `num_merges` iterations
5. Store the merge rules — you need them to encode/decode later
"""

class BPETokenizer:
    def __init__(self):
        self.merges = {}   # (id1, id2) -> new_id, in the order they were learned
        self.vocab = {}    # id -> bytes, built after training

    def train(self, text: str, vocab_size: int = 500, verbose: bool = False):
        assert vocab_size >= 256, "vocab_size must be at least 256 (raw bytes)"
        num_merges = vocab_size - 256

        # Step 1: convert text to raw UTF-8 bytes, then to a list of ints (0-255)
        tokens = list(text.encode("utf-8"))

        merges = {}
        for i in range(num_merges):
            pair_counts = self._get_pair_counts(tokens)
            if not pair_counts:
                break  # no more pairs to merge (text too short)

            # Find the single most frequent adjacent pair
            top_pair = max(pair_counts, key=pair_counts.get)

            # New token id: 256 + i (since 0-255 are reserved for raw bytes)
            new_id = 256 + i
            tokens = self._merge(tokens, top_pair, new_id)
            merges[top_pair] = new_id

            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {new_id} "
                      f"(count={pair_counts[top_pair]})")

        self.merges = merges

        # Build final vocab: map every token id to the bytes it represents
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab
        self.vocab_size = len(vocab)

    @staticmethod
    def _get_pair_counts(tokens: list[int]) -> dict:
        """Count how often each adjacent pair of tokens appears."""
        counts = {}
        for pair in zip(tokens, tokens[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def _merge(tokens: list[int], pair: tuple, new_id: int) -> list[int]:
        """Replace every occurrence of `pair` in `tokens` with `new_id`."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def encode(self, text: str) -> list[int]:
        tokens = list(text.encode("utf-8"))
        # Apply merges in the SAME ORDER they were learned during training
        while len(tokens) >= 2:
            pair_counts = self._get_pair_counts(tokens)
            # Pick the pair that was learned earliest in training (lowest new_id)
            candidate = min(pair_counts, key=lambda p: self.merges.get(p, float("inf")))
            if candidate not in self.merges:
                break  # no more applicable merges
            tokens = self._merge(tokens, candidate, self.merges[candidate])
        return tokens

    def decode(self, ids: list[int]) -> str:
        byte_string = b"".join(self.vocab[idx] for idx in ids)
        return byte_string.decode("utf-8", errors="replace")


if __name__ == "__main__":
    sample = "the quick brown fox jumps over the lazy dog. the fox runs fast." * 20
    tok = BPETokenizer()
    tok.train(sample, vocab_size=280, verbose=True)

    encoded = tok.encode("the fox jumps")
    print(f"\n'the fox jumps' -> {encoded} ({len(encoded)} tokens)")
    print(f"decoded back -> '{tok.decode(encoded)}'")
    print(f"\nFinal vocab size: {tok.vocab_size}")
