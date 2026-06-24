import os
from typing import BinaryIO, Iterable, Iterator
import regex as re
import concurrent.futures
from collections import Counter
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
    print(f"Reading from {start} to {end}")
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end-start).decode("utf-8", errors="ignore")
        pieces = re.split("|".join([re.escape(p) for p in special_tokens]), chunk)
        pretokens = [pt.group() for piece in pieces for pt in re.finditer(PAT, piece)]
    print(f"Done from {start} to {end}")
    return Counter(pretokens)

def get_pretoken_counts(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[int, ...], int]:
    num_processes = 8
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        concurrents = [executor.submit(read_chunk, input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]

    pretoken_counts = concurrents[0].result()
    for future in concurrents[1:]:
        tocombine = future.result()
        pretoken_counts += tocombine
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

    return vocab, merges

def merge_new_token(oldpt: tuple[bytes], old_pair: tuple[bytes, bytes], newtoken: int) -> tuple[bytes]:
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
        pass
    def from_files(self, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        pass
    def encode(self, text:str) -> list[int]:
        pass
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pass
    def decode(self, ids: list[int]) -> str:
        pass

if __name__ == "__main__":
    #train_bpe("../data/TinyStoriesV2-GPT4-valid.txt", 1000, ["<|endoftext|>"])
    v, m = train_bpe("corpus.en", 1000, ["<|endoftext|>"])
    print(v)
    print(m)
