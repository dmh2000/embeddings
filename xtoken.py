#!/usr/bin/env python
"""Tokenize a string using the meta-llama/Llama-3.2-3B tokenizer."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-3B"
MODEL_PATH = "./models/Llama-3.2-3B"
# CONTEXT = "hello world"
CONTEXT = "antidisestablishmentarianism"


def embeddings(context: str = CONTEXT):

    # ----------------
    # convert input string to token IDs
    print(f"\nInput string      : {context}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, clean_up_tokenization_spaces=False)
    token_ids = tokenizer.encode(context)
    print(f"Token IDs         : {token_ids}")
    print(f"Tokens (decoded)  : ", end="")
    for i, tid in enumerate(token_ids):
        token = tokenizer.convert_ids_to_tokens([tid])[0]
        print(f"{token} ", end="")
    print("")
    input("======================================================================")
    # ----------------

    # ----------------2
    print(f"\nLoading model: {MODEL_ID}: {MODEL_PATH}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    # switches your neural network from training mode to inference
    model.eval()
    input("======================================================================")
    # ----------------


    # ----------------
    print(f"\nExtracting embeddings for {len(token_ids)} tokens...")
    embed_layer = model.model.embed_tokens
    token_tensor = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        embeddings = embed_layer(token_tensor)
        print(f"{embeddings}")
    # extract some data about the embeddings
    seq_len = embeddings.shape[1]
    hidden_dim = embeddings.shape[2]
    input("======================================================================")
    # ----------------

    # ----------------
    print("\nSummarize The Results")
    print(f"Sequence length: {seq_len}")
    print(f"Embedding structure:")
    print(f"  Shape            : {list(embeddings.shape)}  (batch=1, tokens={seq_len}, d_model={hidden_dim})")
    print(f"  Data type        : {embeddings.type()}")
    print(f"  Vocabulary size  : {embed_layer.num_embeddings}")
    print(f"  Embedding dim    : {embed_layer.embedding_dim} (d_model)")
    print(f"\nPer-token summary ")
    for i, tid in enumerate(token_ids):
        vec = embeddings[0, i]
        token = tokenizer.convert_ids_to_tokens([tid])[0]
        print(f"  token[{i}] id={tid:>6}  token= {token:<17}  vectors={vec[0].item():+.3f} {vec[1].item():+.3f} ... {vec[-1].item():+.3f}")
    input("======================================================================")
    # ----------------
    print(f"\nDecode the token IDs back to a string")
    print(f"{tokenizer.decode(token_ids, skip_special_tokens=True)}")
    # ----------------

def main():
    embeddings()

if __name__ == "__main__":
    main()
