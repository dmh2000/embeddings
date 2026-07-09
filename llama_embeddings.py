#!/usr/bin/env python3
"""Generate embeddings from token IDs using the meta-llama/Llama-3.2-3B model."""

import os

import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-3B"
TOKEN_IDS = [128000, 519, 85342, 34500, 479, 8997, 2191, 374, 264, 1317, 3492]
MODEL_PATH = os.environ["MODEL_PATH"]

def main():
    print(f"Loading model: {MODEL_ID}: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    model.eval()

    embed_layer = model.model.embed_tokens

    token_tensor = torch.tensor([TOKEN_IDS], dtype=torch.long)

    with torch.no_grad():
        embeddings = embed_layer(token_tensor)

    # embeddings shape: [batch, seq_len, hidden_dim]
    seq_len = embeddings.shape[1]
    hidden_dim = embeddings.shape[2]

    print(f"\nInput token IDs : {TOKEN_IDS}")
    print(f"Number of tokens: {seq_len}")
    print(f"\nEmbedding structure:")
    print(f"  Shape           : {list(embeddings.shape)}  (batch=1, tokens={seq_len}, d_model={hidden_dim})")
    print(f"  Data type       : {embeddings.type()}")
    print(f"  Vocabulary size : {embed_layer.num_embeddings}")
    print(f"  Embedding dim   : {embed_layer.embedding_dim}")

    print(f"\nPer-token summary (min / mean / max):")
    for i, tid in enumerate(TOKEN_IDS):
        vec = embeddings[0, i]
        print(f"  token[{i}] id={tid:>7}  min={vec.min().item():+.4f}  mean={vec.mean().item():+.4f}  max={vec.max().item():+.4f}")


if __name__ == "__main__":
    main()
