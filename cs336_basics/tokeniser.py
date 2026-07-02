import os
from typing import BinaryIO, Iterable, Iterator
import regex as re
import concurrent.futures
from collections import Counter
from tqdm import tqdm
import json
import pickle
import itertools
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def read_chunk(input_path: str | os.PathLike, start:int, end:int, special_tokens:list[str]):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end-start).decode("utf-8", errors="ignore")
        pieces = re.split("|".join([re.escape(p) for p in special_tokens]), chunk)
        pretokens = [pt.group() for piece in pieces for pt in re.finditer(PAT, piece)]
    return Counter(pretokens)

def get_pretoken_counts(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple, int]:
    num_processes = 64
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        concurrents = [executor.submit(read_chunk, input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

        print("PRETOKENISATION:")
        with tqdm(total=num_processes) as pbar:
            pretoken_counts = concurrents[0].result()
            pbar.update(1)
            for future in concurrent.futures.as_completed(concurrents[1:]):
                tocombine = future.result()
                pretoken_counts += tocombine
                pbar.update(1)
    return {tuple(pt.encode()): v for pt, v in pretoken_counts.items()}

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # init vocab
    vocab = {i: bytes([i]) for i in range(256)}
    for spec in special_tokens:
        vocab[len(vocab)] = spec.encode()

    pretoken_counts = get_pretoken_counts(input_path, special_tokens) 

    # do vocab creation process
    merges = []
    pair_freq = {}
    pair_to_pt = {}
    for pretoken, freq in pretoken_counts.items():
        pairs = [(pretoken[i], pretoken[i+1]) for i in range(len(pretoken) - 1)]
        for p in pairs:
            pair_freq[p] = pair_freq.setdefault(p, 0) + freq
            pair_to_pt[p] = pair_to_pt.setdefault(p, set())
            pair_to_pt[p].add(pretoken)

    print("MERGING:")
    with tqdm(total=vocab_size) as pbar:
        while len(vocab) < vocab_size:
            # if we do not have any more 
            if len(pair_freq) == 0:
                break

            most_freq_pair, _ = max(pair_freq.items(), key=lambda k: (k[1], tuple(vocab[i] for i in k[0])))

            # merge!
            merge_index = len(vocab)
            merge_bytes = vocab[most_freq_pair[0]] + vocab[most_freq_pair[1]]
            vocab[len(vocab)] = merge_bytes
            merges.append((vocab[most_freq_pair[0]], vocab[most_freq_pair[1]]))
            # update pretokens
            to_update = pair_to_pt[most_freq_pair].copy()
            for oldpt in to_update:
                old_freq = pretoken_counts.pop(oldpt)
                
                newpt = merge_new_token(oldpt, most_freq_pair, merge_index)

                new_pair_freq = Counter([(newpt[i], newpt[i+1]) for i in range(len(newpt) - 1)])
                old_pair_freq = Counter([(oldpt[i], oldpt[i+1]) for i in range(len(oldpt) - 1)])

                for pair, count in old_pair_freq.items():
                    pair_freq[pair] -= old_freq * count
                    if pair_freq[pair] <= 0:
                        pair_freq.pop(pair)

                for pair, count in new_pair_freq.items():
                    pair_freq[pair] = pair_freq.setdefault(pair, 0) + old_freq * count

                pretoken_counts[newpt] = pretoken_counts.setdefault(newpt, 0) + old_freq

                for pair in old_pair_freq.keys():
                    pair_to_pt[pair].remove(oldpt)

                for pair in new_pair_freq.keys():
                    pair_to_pt[pair] = pair_to_pt.setdefault(pair, set())
                    pair_to_pt[pair].add(newpt)
            pbar.update(1)

    return vocab, merges

def merge_new_token(oldpt: list[int], old_pair: tuple[bytes, bytes], newtoken: int) -> tuple[int]:
    newpt = []
    i = 0
    while i < len(oldpt):
        if i < len(oldpt) - 1 and (oldpt[i], oldpt[i+1]) == old_pair:
            newpt.append(newtoken)
            i += 2
        else:
            newpt.append(oldpt[i])
            i += 1
    newpt = tuple(newpt)
    return newpt

class Tokeniser:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.opposite_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        if special_tokens:
            special_tokens_sorted = sorted(special_tokens, key=lambda x: -len(x))
            self.special_tokens = special_tokens_sorted
            for st in special_tokens_sorted:
                if st.encode("utf-8", "replace") not in self.vocab.values():
                    self.vocab[len(self.vocab)] = st.encode()
            self.spectok_map = {st: tok for tok, st in self.vocab.items() if st.decode("utf-8", "replace") in special_tokens_sorted}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab, merges = files_to_vocab_merges(vocab_filepath, merges_filepath)
        return cls(vocab, merges, special_tokens)

    def encode_without_special(self, text:str) -> list[int]:
        if len(text) == 0:
            return []
        if self.special_tokens and text in self.special_tokens:
            return [self.spectok_map[text.encode("utf-8")]]

        tokens = []
        pretokens = re.findall(PAT, text)
        for pt in pretokens:
            pt_utf8 = pt.encode("utf-8")
            if pt_utf8 in self.opposite_vocab.keys():
                tokens.append(self.opposite_vocab[pt_utf8])
            else:
                for char in pt:
                    for i in tuple(char.encode("utf-8")):
                        newtok = self.opposite_vocab[i.to_bytes()]
                        tokens.append(newtok)

        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)]
            lowest_pair = pairs[0]
            lowest_pair_rep = (self.vocab[lowest_pair[0]], self.vocab[lowest_pair[1]])
            lowest_pair_idx = float("inf")
            for p in pairs:
                merg_rep = (self.vocab[p[0]], self.vocab[p[1]])
                idx = self.merges.index(merg_rep) if merg_rep in self.merges else float("inf")
                if idx < lowest_pair_idx or (idx == lowest_pair_idx and merg_rep < lowest_pair_rep):
                    lowest_pair = p
                    lowest_pair_rep = merg_rep
                    lowest_pair_idx = idx
            if lowest_pair_idx == float("inf"):
                break
            lowest_pair_bytes = b"".join(lowest_pair_rep)
            lowest_pair_vocab_idx = -1
            for k, v in self.vocab.items():
                if v == lowest_pair_bytes:
                    lowest_pair_vocab_idx = k
                    break
            tokens = merge_new_token(tokens, lowest_pair, lowest_pair_vocab_idx)
        return list(tokens)

    def encode(self, text:str) -> list[int]:
        if not self.special_tokens:
            return self.encode_without_special(text)
        pieces = re.split("(" + "|".join([re.escape(p) for p in self.special_tokens]) + ")", text)
        processed_pieces = [self.encode_without_special(p) for p in pieces]
        return list(itertools.chain(*processed_pieces))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        out = b"".join(self.vocab[i] for i in ids)
        return out.decode("utf-8", errors="replace")

def bpe_to_file(vocab: dict[int, bytes], vocab_filepath: str, merges: list[tuple[bytes, bytes]], merges_filepath: str,):
    with open(vocab_filepath, "wb") as f:
        pickle.dump(vocab, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(merges_filepath, "wb") as f:
        pickle.dump(merges, f, protocol=pickle.HIGHEST_PROTOCOL)

def files_to_vocab_merges(vocab_filepath: str, merges_filepath: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(vocab_filepath, "rb") as f:
        vocab = pickle.load(f)
    with open(merges_filepath, "rb") as f:
        merges = pickle.load(f)
    return vocab, merges
