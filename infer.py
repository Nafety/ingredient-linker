"""Link unrecognized ingredient names with the fine-tuned model.

    python infer.py "paprikapulver" "extra virgin oil" --lang en
    python infer.py --file names.tsv      # one "lang<TAB>name<TAB>ingredient list (optional)" per line
Prints one JSON object per name: the model's decision, plus the label and allergens of the chosen entry.
Needs the taxonomy and the index in data/ (built by build_dataset.py) and the adapter in models/adapter/.
"""

from __future__ import annotations

import argparse
import json

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from build_dataset import Retriever, describe, prompt, tx
from train import BNB, predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--lang", default="en", help="declared language of the ingredient list")
    ap.add_argument("--context", default="", help="the ingredient list the name comes from")
    ap.add_argument("--file")
    ap.add_argument("--adapter", default="models/adapter", help="local folder or Hugging Face repo id")
    args = ap.parse_args()

    rows = [(args.lang, n, args.context or n) for n in args.names]
    if args.file:
        for line in open(args.file, encoding="utf-8"):
            lang, name, *ctx = line.rstrip("\n").split("\t")
            rows.append((lang, name, ctx[0] if ctx else name))

    # Base model (read from the adapter config) in 4 bits + LoRA adapter, as during training
    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter, quantization_config=BNB, device_map="auto",
                                                     dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.adapter)

    candidates = Retriever().search([name for _, name, _ in rows])
    items = [
        {"i": i, "prompt": prompt(name, lang, ctx, [(c[0], c[2]) for c in cands])}
        for i, ((lang, name, ctx), cands) in enumerate(zip(rows, candidates))
    ]
    for (lang, name, _), p in zip(rows, predict(model, tok, items)):
        decision = p["prediction"]
        ref = decision.get("id") or decision.get("parent")
        out = {"name": name, "lang": lang, **decision}
        if ref in tx.entries:
            out["entry"] = describe(ref)
            out["allergens"] = sorted(tx.allergens(ref))
        print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
