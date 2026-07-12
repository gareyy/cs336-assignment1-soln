"""
training steps
- set torch and numpy random seeds
- load memory mapped datasets (data, validation)
- create model and move to target device
- create optimiser
- check existence of checkpoint, if so, resume from checkpoint
    - this means we get optimser, model, and iteration data
- for each iteration, starting from the current iteration to the max step count
    - update learning rate wrt schedule, and update it in the optimiser
    - sample a batch xb, yb of training data
    - forward pass
        - sample logits from passing xb through model
        - output is BATCH x CONTEXT_LENGTH x VOCAB_SIZE
    - calculate loss using cross entropy
    - set optimiser zero grad to true (clear gradients to prevent gradient accumulation)
    - backpropogation (loss.backward)
    - clip gradients for stability
    - do optmiser step (parameter update)
    - every so often, log training loss, learning rate
    - every so often, log validation loss, learning rate
        - do this by sampling batches xvb, yvb from validation data, run them through cross entropy, and get the loss from it
        - be sure to set model.eval() before validation forward passes, and model.train() afterwards
        - if this beats the current best valid loss, checkpoint it
    - every so often, save checkpoint
- save checkpoint
"""
import torch
import numpy as np
import transformer, training_utils
from einops import rearrange
import os
import tqdm

ENCODED_TRAIN = "tinystories_encoded_train"
ENCODED_VALID = "tinystories_encoded_valid"

D_MODEL = 512
NUM_HEADS = 16
ROPE_THETA = 10000
D_FF = 512
VOCAB_SIZE = 10000 # tinystories vocab
CONTEXT_LENGTH = 512
NUM_LAYERS = 3
DEVICE = torch.get_default_device()
MODEL_DTYPE = torch.float32

LR = 0.001
BETAS = (0.9, 0.999)
WEIGHT_DECAY = 0.01
EPS = 1e-8

MAX_ITERATIONS = 1000

LR_MIN = 1e-4
LR_MAX = 1e-3
WARMUP_ITERS = int(MAX_ITERATIONS * 0.2)
COSINE_CYCLE_ITERS = int(MAX_ITERATIONS * 0.8)

BATCH_SIZE = 16

GRAD_CLIP = 1.0
GRAD_EPS = 1e-6

RESUME_MODEL_CHECKPOINT_FILENAME = "resumemodel.ckpt"
BEST_MODEL_CHECKPOINT_FILENAME = "bestmodel.ckpt"

VALID_LOSS_EVERY = 50
TRAIN_LOSS_EVERY = 50
CHECKPOINT_EVERY = 25

def main():
    torch.manual_seed(67)
    np.random.seed(67)

    train_data = np.load(ENCODED_TRAIN, mmap_mode="r")
    valid_data = np.load(ENCODED_VALID, mmap_mode="r")

    model = transformer.TransformerLanguageModel(D_MODEL, NUM_HEADS, D_FF, ROPE_THETA, VOCAB_SIZE, CONTEXT_LENGTH, NUM_LAYERS, DEVICE, MODEL_DTYPE)
    model.to(DEVICE)
    print(f"NUM PARAMETERS: {model.get_parameter_count()}")
    optimiser = training_utils.AdamW(model.parameters(), LR, BETAS, WEIGHT_DECAY, EPS)

    start_iter = 0
    if os.path.exists(RESUME_MODEL_CHECKPOINT_FILENAME):
        start_iter = training_utils.load_checkpoint(model, optimiser, RESUME_MODEL_CHECKPOINT_FILENAME)
        print(f"RESUMING FROM ITER {start_iter+1}/{MAX_ITERATIONS}")
    else:
        print("STARTING NEW...")
    
    best_valid_loss = float("inf")
    for iteration in tqdm.tqdm(range(start_iter, MAX_ITERATIONS)):
        curr_lr = training_utils.learning_rate_schedule(iteration, LR_MAX, LR_MIN, WARMUP_ITERS, COSINE_CYCLE_ITERS)
        for group in optimiser.param_groups:
            group['lr'] = curr_lr

        input_batch, target_batch = training_utils.get_batch(train_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE.__str__())
        logits = model(input_batch)
        ce_target = rearrange(target_batch, "... batch context -> ... (batch context)")
        ce_logits = rearrange(logits, "... batch context vocab -> ... (batch context) vocab")
        loss = training_utils.cross_entropy_loss(ce_logits, ce_target)

        optimiser.zero_grad()
        loss.backward()

        if GRAD_CLIP > 0:
            training_utils.grad_clip(model.parameters(), GRAD_CLIP, GRAD_EPS)
        
        optimiser.step()
        if iteration % TRAIN_LOSS_EVERY == 0:
            print(f"Train Loss at iteration {iteration+1}: {loss.item()}")
        if iteration % VALID_LOSS_EVERY == 0:
            model.eval()
            input_batch, target_batch = training_utils.get_batch(valid_data, BATCH_SIZE, CONTEXT_LENGTH, DEVICE.__str__())
            logits = model(input_batch)
            ce_target = rearrange(target_batch, "... batch context -> ... (batch context)")
            ce_logits = rearrange(logits, "... batch context vocab -> ... (batch context) vocab")
            valid_loss = training_utils.cross_entropy_loss(ce_logits, ce_target)
            model.train()
            print(f"Validation Loss at iteration {iteration+1}: {valid_loss.item()}")
            if valid_loss.item() < best_valid_loss:
                best_valid_loss = valid_loss.item()
                training_utils.save_checkpoint(model, optimiser, iteration, BEST_MODEL_CHECKPOINT_FILENAME)
        if iteration % CHECKPOINT_EVERY == 0:
            print(f"Checkpointed at {iteration+1}/{MAX_ITERATIONS}")
            training_utils.save_checkpoint(model, optimiser, iteration, RESUME_MODEL_CHECKPOINT_FILENAME)

    training_utils.save_checkpoint(model, optimiser, MAX_ITERATIONS, RESUME_MODEL_CHECKPOINT_FILENAME)

if __name__ == "__main__":
    main()
