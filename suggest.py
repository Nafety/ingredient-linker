"""Suggestions for the most common unrecognized ingredients of one country, as a table for human review.

    python suggest.py --country en:united-kingdom --min-products 5
Reads the OFF export (data/food.parquet, see build_dataset.py) and the current taxonomies, downloaded
again into data/fresh/ so that names fixed since training are left out. Runs the model on the most
common unknown names and writes suggestions_<country>.xlsx: a work list of every unknown name, the most
useful first, with the suggested fix (the exact taxonomy line when there is one), the allergens it adds,
and Status / Applied fix columns for volunteers.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault("OFF_DATA_DIR", "data/fresh")  # before build_dataset is imported

import duckdb  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from peft import AutoPeftModelForCausalLM  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from build_dataset import FUNCTION_WORDS, SHORT_NOISE, Retriever, describe, normalize, prompt, tx  # noqa: E402
from train import BNB, MAX_LEN, SYSTEM, parse  # noqa: E402

SITE = {"en:united-kingdom": "uk", "en:france": "fr", "en:germany": "de", "en:spain": "es", "en:italy": "it",
        "en:united-states": "us", "en:belgium": "be", "en:switzerland": "ch", "en:canada": "ca"}


def unknown_names(export: Path, country: str) -> list[dict]:
    """Unknown ingredient ids of the country's products, most frequent first, with one context each."""
    rows = duckdb.connect().execute(
        "SELECT code, lang, ingredients, ingredients_text FROM read_parquet(?) "
        "WHERE list_contains(countries_tags, ?) AND unknown_ingredients_n > 0 AND ingredients IS NOT NULL",
        [export.as_posix(), country],
    )
    products: Counter[str] = Counter()
    seen: dict[str, dict] = {}
    while batch := rows.fetchmany(20_000):
        for code, lang, tree, texts in batch:
            found: dict[str, str] = {}
            stack = json.loads(tree)
            while stack:
                ing = stack.pop()
                if ing.get("is_in_taxonomy") == 0 and ing.get("id") and ing.get("text"):
                    found.setdefault(ing["id"], ing["text"].strip())
                stack.extend(ing.get("ingredients") or [])
            text = next((t["text"] for t in texts or [] if t["lang"] == lang), "") or ""
            for tag, written in found.items():
                products[tag] += 1
                entry = seen.setdefault(tag, {"id": tag, "name": written, "examples": [], "context": ""})
                if len(entry["examples"]) < 3:
                    entry["examples"].append(code)
                if not entry["context"] and text:
                    entry["context"] = excerpt(text, written)
    return [{**seen[tag], "products": n} for tag, n in products.most_common()]


def excerpt(text: str, name: str, width: int = 90) -> str:
    """The part of the ingredient list around the name."""
    text = re.sub(r"<[^>]+>", "", text)  # some lists keep HTML tags (allergens in bold)
    at = text.lower().find(name.lower())
    if at < 0:
        return text[: 2 * width]
    start, end = max(0, at - width), at + len(name) + width
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


@torch.no_grad()
def answer(model, tok, prompts: list[str], batch: int = 8) -> list[tuple[dict, float]]:
    """Model answer and confidence (lowest probability among the generated tokens)."""
    tok.padding_side = "left"
    out = []
    for k in range(0, len(prompts), batch):
        texts = [
            tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": p}],
                                    tokenize=False, add_generation_prompt=True)
            for p in prompts[k : k + batch]
        ]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(model.device)
        gen = model.generate(**enc, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id,
                             output_scores=True, return_dict_in_generate=True)
        logprobs = model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True)
        for seq, lp in zip(gen.sequences, logprobs):
            new = seq[enc["input_ids"].shape[1] :]
            keep = new != tok.eos_token_id
            confidence = float(lp[keep].exp().min()) if keep.any() else 0.0
            out.append((parse(tok.decode(new, skip_special_tokens=True)), confidence))
    return out


CATEGORIES = ["Taxonomy: synonym", "Taxonomy: new entry", "Product data: typo", "Product language",
              "List phrase", "Not an ingredient"]  # reviewers' priority order


def is_noise(name: str) -> bool:
    """Fragments and function words ("and", "per", "s"), by the training rules: no need for the model."""
    stripped = name.strip()
    return bool(SHORT_NOISE.match(stripped)) or stripped.lower() in FUNCTION_WORDS


def action(name: str, lang: str, decision: dict) -> tuple[str, str, str]:
    """(category, suggested action, literal taxonomy line) for one model answer."""
    kind = decision.get("type")
    name = name.lower()  # the taxonomy is written in lower case
    ref = decision.get("id") or decision.get("parent")
    if ref and ref not in tx.entries:
        ref = tx.resolve(ref)
    if kind == "existing" and ref:
        names = tx.entries[ref].names
        key = normalize(name)
        other = [lng for lng, ns in names.items() if lng != lang and any(normalize(n) == key for n in ns)]
        if other:
            return ("Product language", f"This is the {other[0]} name of {ref}: the product's ingredient list "
                    "is declared in the wrong language (no taxonomy change).", "")
        same = names.get(lang, [])
        plural = any(key in (normalize(n) + "s", normalize(n) + "es") or normalize(n) in (key + "s", key + "es")
                     for n in same)
        closest = max(same, key=lambda n: fuzz.ratio(key, normalize(n)), default="")
        if not plural and closest and fuzz.ratio(key, normalize(closest)) >= 90:
            return ("Product data: typo", f'Likely typo of "{closest}": fix the product\'s ingredient list '
                    "rather than adding a synonym.", "")
        return ("Taxonomy: synonym", f"Add a {lang} synonym to {ref}.",
                f"{lang}: {', '.join(same + [name]) if same else name}")
    if kind == "new" and ref:
        return "Taxonomy: new entry", f"New entry, child of {ref}.", f"< {ref}\n{lang}: {name}"
    if kind == "structure":
        return "List phrase", 'List phrase ("may contain"…): a parser case, not a taxonomy entry.', ""
    return "Not an ingredient", "Noise in the ingredient list (no taxonomy change).", ""


def build_prompts(asked: list[dict]) -> list[str]:
    """Candidate retrieval and prompts for the model."""
    candidates = Retriever().search([n["name"] for n in asked])
    return [prompt(n["name"], n["lang"], n["context"] or n["name"], [(c[0], c[2]) for c in cands])
            for n, cands in zip(asked, candidates)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="en:united-kingdom", help="OFF country tag")
    ap.add_argument("--min-products", type=int, default=5, help="names found in fewer products are left out")
    ap.add_argument("--export", default="data/food.parquet")
    ap.add_argument("--adapter", default="models/adapter")
    args = ap.parse_args()

    # Reading the 8 GB export is the heavy part: its result is cached (delete the file to refresh it)
    cache = Path(os.environ["OFF_DATA_DIR"]) / f"unknown_{args.country.split(':')[1]}.json"
    if not cache.exists():
        cache.write_text(json.dumps(unknown_names(Path(args.export), args.country)), encoding="utf-8")
    found = json.loads(cache.read_text(encoding="utf-8"))
    # Names the current taxonomy still does not know, found in enough products to be worth a fix
    names = [n for n in found if n["products"] >= args.min_products and tx.resolve(n["id"]) is None]
    for n in names:
        n["lang"] = n["id"].split(":")[0]

    asked = [n for n in names if not is_noise(n["name"])]
    # Retrieval runs in a child process: when it ends, all its memory goes back to the system before the LLM loads
    with ProcessPoolExecutor(max_workers=1) as pool:
        prompts = pool.submit(build_prompts, asked).result()
    del found
    gc.collect()
    model = AutoPeftModelForCausalLM.from_pretrained(args.adapter, quantization_config=BNB, device_map="auto",
                                                     dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.adapter)
    answers = dict(zip((n["id"] for n in asked), answer(model, tok, prompts)))

    site = SITE.get(args.country, "world")
    rows = []
    for n in names:
        decision, confidence = answers.get(n["id"], ({"type": "not_ingredient"}, 1.0))
        ref = decision.get("id") or decision.get("parent")
        if ref and ref not in tx.entries:
            ref = tx.resolve(ref)
        category, todo, line = action(n["name"], n["lang"], decision)
        allergens = sorted(tx.allergens(ref)) if ref in tx.entries else []
        rows.append([
            category, n["name"], n["lang"], n["products"],
            f"https://{site}.openfoodfacts.org/facets/ingredients/{quote(n['id'])}",
            " ".join(f"https://{site}.openfoodfacts.org/product/{c}" for c in n["examples"]),
            re.sub(r"\s+", " ", n["context"]),
            decision.get("type", ""), describe(ref) if ref in tx.entries else "",
            todo, line, ", ".join(a.split(":")[1] for a in allergens),
            "rule" if n["id"] not in answers else
            "high" if confidence >= 0.8 else "medium" if confidence >= 0.5 else "low",
            "", "",
        ])
    rows.sort(key=lambda r: (CATEGORIES.index(r[0]), -r[3]))  # what needs a decision first, then by products

    columns = ["Category", "Unknown name", "Language", "Products", "Ingredient page", "Example products",
               "Ingredient list excerpt", "Model answer", "Target or parent", "Suggested action", "Taxonomy line",
               "Allergens added", "Confidence", "Decision", "Comment"]
    out = Path(f"suggestions_{args.country.split(':')[1]}.xlsx")
    write_worklist(pd.DataFrame(rows, columns=columns), out, args.country, args.min_products)
    print(f"{len(names)} names -> {out}:", Counter(r[0] for r in rows).most_common())


WORKLIST = {  # column in the work list <- column of the full table
    "Unknown name": "Unknown name",
    "Language": "Language",
    "Fix type": "Category",
    "Target or parent": "Target or parent",
    "Taxonomy line": "Taxonomy line",
    "Allergens added": "Allergens added",
    "Confidence": "Confidence",
    "Ingredient list excerpt": "Ingredient list excerpt",
    "Ingredient page": "Ingredient page",
}


FIX_TYPES = {  # what the model thinks, as shown to volunteers
    "Taxonomy: synonym": "synonym",
    "Taxonomy: new entry": "new entry",
    "Product language": "product language",
    "Product data: typo": "typo",
    "List phrase": "list phrase",
    "Not an ingredient": "not an ingredient",
}


def write_worklist(table: pd.DataFrame, out: Path, country: str, min_products: int) -> None:
    """A work list of every unknown name, the most useful first: taxonomy fixes, then product data, then noise.
    Within each group: high confidence first (the quickest to check), then the names found in the most products."""
    rows = table.copy()
    rows["_group"] = rows["Category"].map(lambda c: 0 if c.startswith("Taxonomy") else 1 if c.startswith("Product") else 2)
    rows["_confidence"] = rows["Confidence"].map({"high": 0, "medium": 1, "low": 2, "rule": 3})
    rows = rows.sort_values(["_group", "_confidence", "Products"], ascending=[True, True, False])
    rows["Category"] = rows["Category"].map(FIX_TYPES)
    worklist = rows[list(WORKLIST.values())].set_axis(list(WORKLIST), axis=1)
    worklist["Status"] = ""  # Done / Corrected / Not taxonomy
    worklist["Applied fix"] = ""  # when Corrected: the line really added (also future training data)
    readme = README.format(country=country, n=len(worklist), m=min_products)
    with pd.ExcelWriter(out) as excel:
        pd.DataFrame({"": readme.splitlines()}).to_excel(excel, sheet_name="Read me", index=False, header=False)
        worklist.to_excel(excel, sheet_name="Unknown ingredients", index=False)
        sheet = excel.sheets["Unknown ingredients"]
        sheet.freeze_panes = "B2"
        sheet.auto_filter.ref = sheet.dimensions
        for column, width in zip("ABCDEFGHIJK", (30, 9, 10, 45, 45, 14, 11, 60, 30, 13, 40), strict=True):
            sheet.column_dimensions[column].width = width


README = """Unknown ingredients of {country} products (Open Food Facts), with suggestions from a fine-tuned LLM.
All {n} unknown names found in at least {m} products. The most useful first: suggested taxonomy fixes, then
product data issues, then noise; in each group, high model confidence first, then the most products.

How to use it: take the next row without a Status, check it (the ingredient page shows the products), then
- the suggestion is right: apply it, Status = Done;
- the suggestion is wrong: apply the right fix instead, Status = Corrected, and write it in "Applied fix";
- there is nothing to fix in the taxonomy (not an ingredient, an address, a cosmetic product, a name cut by
  the parser...): Status = Not taxonomy, with a short reason in "Applied fix".

Fix type = what the model thinks:
- synonym: add the name to the target entry (Taxonomy line = the target's line with the name added);
- new entry: a new entry under the parent (Taxonomy line = the block to add);
- product language: the name of an existing entry in another language: the product's ingredient list is
  declared in the wrong language (fix it on the products);
- typo: close to an existing name: fix the products' ingredient list;
- list phrase, not an ingredient: text the parser should not have read as an ingredient.
Allergens added: what products with this name will get once a taxonomy fix is in.

Confidence orders the rows, it does not replace the check: some high-confidence suggestions are wrong.

Model and code: https://huggingface.co/Nafety/off-ingredient-linker - https://github.com/Nafety/ingredient-linker"""


if __name__ == "__main__":
    main()
