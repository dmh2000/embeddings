#!/usr/bin/env python3
"""Tokenize a string using the meta-llama/Llama-3.2-3B tokenizer."""

import os

from transformers import AutoTokenizer

MODEL_PATH = os.environ["MODEL_PATH"]
CONTEXT = os.environ["CONTEXT"]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, clean_up_tokenization_spaces=False)
    token_ids = tokenizer.encode(CONTEXT)
    print(token_ids)


if __name__ == "__main__":
    main()
