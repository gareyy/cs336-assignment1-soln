import os
from typing import Generator, Iterable
import tokeniser
import tqdm
import numpy as np
import mmap
import concurrent.futures as futures

BUFMAX = 8192
NUM_PROCESSES = 512

def estimate_file_pretoken_count(tok: tokeniser.Tokeniser, input_p: os.PathLike | str) -> int:
    total_pretokens = 0
    with open(input_p, "r", encoding="utf-8", errors="ignore") as f_in:
        for l in tqdm.tqdm(f_in):
            total_pretokens += tok.estimate_pretoken_count(l)
    return total_pretokens

def file_to_tokens_iter(tok: tokeniser.Tokeniser, input_p: os.PathLike | str) -> Iterable[int]:
    with open(input_p, "r", encoding="utf-8", errors="ignore") as f_in:
        for t in tok.encode_iterable(f_in):
            yield t

def process_chunk(tok: tokeniser.Tokeniser, input_path : os.PathLike | str, start:int, end:int, index:int) -> tuple[np.ndarray, int]:
    with open(input_path, "r") as f:
        f.seek(start)
        chunk = f.read(end-start)
        #chunk = f.read(end-start).decode("utf-8", errors="ignore")
    lines = chunk.splitlines(keepends=True)
    towrite = np.array([], dtype=np.uint16)
    buf = np.array([], dtype=np.uint16)
    for line in lines:
        tokens = tok.encode(line)
        buf = np.append(buf, tokens)
        if len(buf) >= BUFMAX:
            towrite = np.append(towrite, buf)
            buf = np.array([], dtype=np.uint16)
    if len(buf) > 0:
        towrite = np.append(towrite, buf)
    return towrite, index

def file_to_token_npz(tok: tokeniser.Tokeniser, input_p: os.PathLike | str, output_p: os.PathLike | str):
    towrite = np.array([], dtype=np.uint16)
    with open(INPUT, "rb") as f_in:
        boundaries = tokeniser.find_chunk_boundaries(f_in, NUM_PROCESSES, b"<|endoftext|>")
        startends = [ (start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]

    with futures.ProcessPoolExecutor() as executor:
        concurrents = [executor.submit(process_chunk, tok, input_p, start, end, i) for i, (start, end) in enumerate(startends)]
        results = {}
        with tqdm.tqdm(total=len(startends)) as pbar:
            for f in futures.as_completed(concurrents):
                tocombine, ind = f.result()
                results[ind] = tocombine
                pbar.update(1)

    for i in tqdm.tqdm(range(len(startends))):
        towrite = np.append(towrite, results[i])

    with open(output_p, "wb") as f:
        np.save(f, towrite)

if __name__ == "__main__":
    tok = tokeniser.Tokeniser.from_files("tinystories_voc", "tinystories_mer", ["<|endoftext|>"])
    #INPUT = "/home/wumeno/Documents/cs336-assignment1-soln/data/TinyStoriesV2-GPT4-train.txt"
    INPUT = "/home/wumeno/Documents/cs336-assignment1-soln/data/TinyStoriesV2-GPT4-valid.txt"
    ENCODED = "tinystories_encoded_valid"
    file_to_token_npz(tok, INPUT, ENCODED)
    """
    with open(ENCODED, "rb") as f:
        pp = np.load(f)
    out = tok.decode(pp.tolist())
    with open(INPUT, "r") as f:
        check = f.read()
    assert check == out
    """
