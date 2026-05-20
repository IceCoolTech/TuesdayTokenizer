
# --------------------------------------------------------------------
# Helper functions used in the Tokenizer

def get_freqs(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairsxs
    Allows to update an existing dictionary of counts (optional)
    """
    counts = {} if not None else counts

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts

def merge(ids, bigram, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        if ids[i] == bigram[0] and i < len(ids) - 1 and ids[i+1] == bigram[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1

    return newids

# --------------------------------------------------------------------
# BPE Tokenizer foundation

class Tokenizer:
    """Tokenizer base class"""

    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.vocab = self._build_vocab()

    def train(self, text, vocab_size, verbose=False):
        # tokenizer can train vocabulary of size vocab_size
        raise NotImplementedError
    
    def encode(self, text):
        # tokenizer can encode string into list of integers
        raise NotImplementedError
    
    def decode(self, ids):
        # tokenizer can decode a list of int into str
        raise NotImplementedError
    
    def _build_vocab(self):
        """DOES NOT SUPPORT SPECIAL TOKENS"""
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab
    
