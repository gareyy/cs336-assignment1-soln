import os
from typing import Generator, Iterable
import tokeniser
import tqdm
import numpy as np
import mmap

BUFMAX = 8192

def file_to_tokens_iter(tok: tokeniser.Tokeniser, input_p: os.PathLike | str) -> Iterable[int]:
    with open(input_p, "r", encoding="utf-8", errors="ignore") as f_in:
        for t in tok.encode_iterable(f_in):
            yield t

def estimate_file_pretoken_count(tok: tokeniser.Tokeniser, input_p: os.PathLike | str) -> int:
    total_pretokens = 0
    with open(input_p, "r", encoding="utf-8", errors="ignore") as f_in:
        for l in tqdm.tqdm(f_in):
            total_pretokens += tok.estimate_pretoken_count(l)
    return total_pretokens

def file_to_token_npz(tok: tokeniser.Tokeniser, input_p: os.PathLike | str, output_p: os.PathLike | str):
    total_est_pretokens = estimate_file_pretoken_count(tok, input_p)
    towrite = np.array([], dtype=np.uint16)
    total_tokens = 0
    with tqdm.tqdm(total=total_est_pretokens) as pbar:
        buf = np.array([], dtype=np.uint16)
        for token in file_to_tokens_iter(tok, INPUT):
            buf = np.append(buf, [token])
            if len(buf) >= BUFMAX:
                towrite = np.append(towrite, buf)
                buf = np.array([], dtype=np.uint16)
            pbar.update(1)
            total_tokens += 1
    if len(buf) > 0:
        towrite = np.append(towrite, buf)
    with open(output_p, "wb") as f:
        np.save(f, towrite)

if __name__ == "__main__":
    tok = tokeniser.Tokeniser.from_files("tinystories_voc", "tinystories_mer", ["<|endoftext|>"])
    INPUT = "/home/wumeno/Documents/cs336-assignment1-soln/data/TinyStoriesV2-GPT4-train.txt"
    ENCODED = "tinystories_encoded"
    file_to_token_npz(tok, INPUT, ENCODED)
