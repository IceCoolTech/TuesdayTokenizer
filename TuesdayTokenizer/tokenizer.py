import regex as re
from .foundation import Tokenizer, merge, get_freqs

"""
This is the byte-level Byte Pair Encoding tokenizer.

The tokenizer does not currently support the following:
- Special tokens
- Easily swappable split_patterns 

SEE FOUNDATION FOR TOKENIZER SKELETON.
"""

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class TuesdayTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)

    def train(self, text, vocab_size, verbose=False):
        # verify valid vocab_size and set number of merges to be performed
        assert vocab_size >- 256
        num_merges = vocab_size - 256

        # find and replace text using the split pattern 
        text_chunks = re.findall(self.compiled_pattern, text)
        
        # encode each text chunk with its utf-8 encoding
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]
        
        merges = {}
        # create a dictionary where key is the byte (0...255) and value (bytes([idx])) is the byte representation of itself
        # example: {0: b'\x00', 1: b'\x01', ... 65: b'A', 97: b'a', ... 255: b'\xff'}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            freqs = {}

            # for each chunk id in chunk ids, create a lookup table (dict) of pair frequencies
            for chunk_ids in ids:
                get_freqs(chunk_ids, freqs)

            # find the bigram (pair) with the highest frequency
            bigram = max(freqs, key=freqs.get)

            # set the new id for the merged pair
            idx = 256 + i

            # perform the merge for the corresponding occurrences in the list of chunk ids
            ids = [merge(chunk_ids, bigram, idx) for chunk_ids in ids]
            merges[bigram] = idx 
            vocab[idx] = vocab[bigram[0]] + vocab[bigram[1]]

            if verbose:
                print(f"Merge {i+1}/{num_merges}: {bigram} -> {idx} ({vocab[idx]}) had {freqs[bigram]} occurrences")
    
        self.merges = merges
        self.vocab = vocab

    def _encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            freqs = get_freqs(ids)
            bigram = min(freqs, key=lambda pair: self.merges.get(pair, float("inf")))
            if bigram not in self.merges:
                break
            idx = self.merges[bigram]
            ids = merge(ids, bigram, idx)
        return ids
        
    def encode_ordinary(self, text):
        """
        NOTE: Ordinary encode ignores special tokens
        """

        # process text into chunks conforming to the pattern(s) defined in split_pattern
        text_chunks = re.findall(self.compiled_pattern, text)

        ids = []
        # independently encode each chunk and join their results 
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") 
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids 
    
    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            else:
                raise ValueError(f"Invalid token id: {idx}")
            
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text        
