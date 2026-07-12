from typing import IO, BinaryIO
import tokeniser
import transformer
import training_utils
import torch
import torch.nn as nn
import os
import training
from einops import reduce, einsum

MODEL_FILENAME = training.BEST_MODEL_CHECKPOINT_FILENAME
DEVICE = torch.get_default_device()
TEMPERATURE = 10000

ENDOFTEXT = "<|endoftext|>"

def load_model(model: nn.Module, src : str | os.PathLike | BinaryIO | IO[bytes]):
    load_dict = torch.load(src)
    model.load_state_dict(load_dict[training_utils.MODEL_DICT])

def generate(model: nn.Module, tok: tokeniser.Tokeniser, stub: str, device: torch.device, max_tokens: int = 2048) -> str:
    model.eval()
    encoded = torch.IntTensor(tok.encode(stub))
    encoded = encoded.unsqueeze(0)
    encoded.to(device)
    endoftext_token = tok.opposite_vocab[ENDOFTEXT.encode("utf-8")]
    i = 0
    sample = []

    while i < max_tokens and endoftext_token not in sample:
        part = encoded[0][-512:].unsqueeze(0)

        logits = model(part)[0][-1]
        logits /= TEMPERATURE

        probs = transformer.softmax(logits, dim=-1)
        numerical_token = probs.argmax()
        token = torch.IntTensor([numerical_token]).unsqueeze(0)
        encoded = torch.cat((encoded, token), dim=-1)

        sample = encoded.squeeze().tolist()
        i += 1
        print(tok.decode(sample).encode("utf-8").decode("unicode_escape"))
        if numerical_token == endoftext_token:
            break

    sample = encoded.squeeze().tolist()
    # TODO: top p sampling
    return tok.decode(sample).encode("utf-8").decode("unicode_escape")

if __name__ == "__main__":
    assert os.path.exists(MODEL_FILENAME)
    model = transformer.TransformerLanguageModel(training.D_MODEL, training.NUM_HEADS, training.D_FF, training.ROPE_THETA, training.VOCAB_SIZE, training.CONTEXT_LENGTH, training.NUM_LAYERS, DEVICE)
    model.to(DEVICE)
    load_model(model, MODEL_FILENAME)
    print(f"NUM PARAMETERS: {model.get_parameter_count()}")
    tok = tokeniser.Tokeniser.from_files("tinystories_voc", "tinystories_mer", [])
    text = generate(model, tok, "There once was a ", DEVICE)
    print(text)
