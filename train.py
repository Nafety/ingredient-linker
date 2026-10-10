"""QLoRA fine-tuning of Qwen2.5-1.5B-Instruct on data/train.jsonl, then predictions on the test set.

    python train.py              # about 4 h on an 8 GB laptop GPU (RTX 4060)
    python train.py --resume     # restart from the last checkpoint in models/run/
Outputs: models/adapter/ (LoRA weights, about 50 MB) and data/predictions_test.jsonl.
Tracked in Weights & Biases when a key is configured (`wandb login`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DATA = Path("data")
MAX_LEN = 1024

SYSTEM = (
    "You link ingredient names found on food packaging to the Open Food Facts ingredients taxonomy. "
    "You get an unrecognized name, the declared language of the ingredient list (it may be wrong), "
    "a piece of the list for context, and candidate taxonomy entries. Answer with JSON only:\n"
    '- {"type": "existing", "id": "<candidate id>"} if a candidate is this ingredient (any language, typos, plurals)\n'
    '- {"type": "new", "parent": "<candidate id>"} if it is a real ingredient with no matching entry: give the closest parent\n'
    '- {"type": "not_ingredient"} for nutrition facts, packaging, addresses, legal mentions, fragments\n'
    '- {"type": "structure"} for list phrases like "contains 2% or less of", "may contain"'
)

# The base model is loaded in 4 bits (NF4): about 1 GB instead of 3 GB in bf16
BNB = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def to_chat(ex: dict) -> dict:
    return {
        "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["prompt"]}],
        "completion": [{"role": "assistant", "content": ex["completion"]}],
    }


def parse(text: str) -> dict:
    m = re.search(r"\{.*?\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else {"type": "not_ingredient"}
    except json.JSONDecodeError:
        return {"type": "not_ingredient"}


@torch.no_grad()
def predict(model, tok, items: list[dict], batch: int = 16) -> list[dict]:
    """items: {"i", "prompt"}. Greedy decoding, batched (left padding)."""
    model.eval()
    tok.padding_side = "left"
    out = []
    for k in range(0, len(items), batch):
        chunk = items[k : k + batch]
        texts = [
            tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": e["prompt"]}],
                                    tokenize=False, add_generation_prompt=True)
            for e in chunk
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model.device)
        gen = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
        for e, g in zip(chunk, gen):
            raw = tok.decode(g[enc["input_ids"].shape[1] :], skip_special_tokens=True)
            out.append({"i": e["i"], "raw": raw, "prediction": parse(raw)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n-train", type=int, default=30_000)
    ap.add_argument("--batch", type=int, default=2, help="examples per pass (effective batch: 16)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    import wandb

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=BNB, device_map="auto", dtype=torch.bfloat16)
    train, valid = read(DATA / "train.jsonl")[: args.n_train], read(DATA / "valid.jsonl")[:400]

    use_wandb = bool(os.environ.get("WANDB_API_KEY") or wandb.api.api_key)
    os.environ.setdefault("WANDB_PROJECT", "off-ingredient-linker")
    tok.padding_side = "right"
    config = SFTConfig(
        output_dir="models/run",
        num_train_epochs=1,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=16 // args.batch,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # 3 % of the steps (Transformers 5)
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,  # recompute activations instead of storing them
        max_length=MAX_LEN,
        completion_only_loss=True,  # the loss only covers the JSON answer, not the prompt
        report_to="wandb" if use_wandb else "none",
    )
    # LoRA: every linear layer gets two small trainable matrices (rank 16), the base weights stay frozen
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules="all-linear")
    trainer = SFTTrainer(
        model=model, args=config, processing_class=tok, peft_config=lora,
        train_dataset=Dataset.from_list([to_chat(e) for e in train]),
        eval_dataset=Dataset.from_list([to_chat(e) for e in valid]),
    )
    trainer.model.print_trainable_parameters()
    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.model.save_pretrained("models/adapter")
    tok.save_pretrained("models/adapter")

    preds = predict(trainer.model, tok, read(DATA / "test_prompts.jsonl"))
    with open(DATA / "predictions_test.jsonl", "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
