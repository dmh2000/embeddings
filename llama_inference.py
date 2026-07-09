#!/usr/bin/env python3
"""Full autoregressive inference pipeline using meta-llama/Llama-3.2-3B."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-3B"
MODEL_PATH = os.environ["MODEL_PATH"]
CONTEXT = os.environ["CONTEXT"]
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "200"))


# ── Step 1 ────────────────────────────────────────────────────────────────────
def load_tokenizer(model_path: str) -> AutoTokenizer:
    print(f"[1] Loading tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, clean_up_tokenization_spaces=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


# ── Step 2 ────────────────────────────────────────────────────────────────────
def load_model(model_path: str) -> tuple[AutoModelForCausalLM, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2] Loading model: {MODEL_ID}  device={device}")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.to(device)
    model.eval()
    return model, device


# ── Step 3 ────────────────────────────────────────────────────────────────────
def tokenize(tokenizer: AutoTokenizer, text: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    print(f"[3] Tokenizing: {text!r}")
    enc = tokenizer(text, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    print(f"    {input_ids.shape[1]} input tokens: {input_ids[0].tolist()}")
    return input_ids, attention_mask


# ── Step 4 ────────────────────────────────────────────────────────────────────
def generate(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    print(f"[4] Generating (max_new_tokens={max_new_tokens}, greedy decoding) …")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # greedy — deterministic
            repetition_penalty=1.1,
        )
    new_count = output_ids.shape[1] - input_ids.shape[1]
    print(f"    Generated {new_count} new tokens")
    return output_ids


# ── Step 5 ────────────────────────────────────────────────────────────────────
def decode(tokenizer: AutoTokenizer, output_ids: torch.Tensor, input_len: int) -> str:
    print("[5] Decoding output tokens …")
    new_ids = output_ids[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tokenizer = load_tokenizer(MODEL_PATH)
    # model, device = load_model(MODEL_PATH)
    input_ids, attention_mask = tokenize(tokenizer, CONTEXT, device)
    # output_ids = generate(model, input_ids, attention_mask, MAX_NEW_TOKENS)
    # result = decode(tokenizer, output_ids, input_ids.shape[1])
    # print(f"\n--- Generated Text ---\n{result}\n")


if __name__ == "__main__":
    main()
