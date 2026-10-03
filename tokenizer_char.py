"""
PHASE 2a: Character-level tokenizer (from scratch)
====================================================
This is the simplest possible tokenizer. Every unique character in your
text becomes one token. No libraries, no subword logic — pure Python.

WHY START HERE:
Character-level tokenization lets you focus on the transformer architecture
first, without worrying about subword merging logic. Once your model trains
end-to-end with this, you can swap in the BPE tokenizer (tokenizer_bpe.py)
without changing anything else in your pipeline.

LIMITATION (why real LLMs don't use this):
A character-level vocab is small (~65-100 tokens for English text), but it
means every word takes many tokens to represent ("hello" = 5 tokens instead
of 1). This makes sequences longer and wastes context length. BPE fixes this.
"""

class CharTokenizer:
    def __init__(self, text: str):
        # Get every unique character in the training text, sorted for
        # reproducibility (so the same text always produces the same vocab).
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        # Two lookup tables: character -> integer id, and back again.
        # This is literally all a tokenizer is: a fixed mapping between
        # symbols and integers, because neural nets only work with numbers.
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """String -> list of integers"""
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        """List of integers -> string"""
        return ''.join(self.itos[i] for i in ids)


if __name__ == "__main__":
    # Quick sanity test
    sample = "hello world"
    tok = CharTokenizer(sample)
    print(f"Vocab size: {tok.vocab_size}")
    print(f"Vocab: {tok.stoi}")

    encoded = tok.encode("hello")
    print(f"'hello' -> {encoded}")
    print(f"{encoded} -> '{tok.decode(encoded)}'")
