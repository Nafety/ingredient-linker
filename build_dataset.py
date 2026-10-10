"""Build the training set: unrecognized ingredient name -> JSON decision.

    python build_dataset.py

Each step is skipped when its output already exists in data/:
  1. download the ingredient taxonomies (openfoodfacts-server) and the product export (Hugging Face)
  2. extract from the export the links made by the OFF parser and the names it did not recognize
  3. build a multilingual embedding index of every taxonomy name (candidate retrieval)
  4. generate the examples, from four sources:
       real       links actually made by the OFF parser
       corrupted  taxonomy names damaged on purpose (OCR errors, wrong declared language, percentages...)
       new        taxonomy leaves hidden from the index: the expected answer is their parent
       qualified  taxonomy names with a qualifier that does not change the ingredient ("fresh", "italian"...)
       noise / structure  rule-based picks among real unrecognized names
Outputs: data/train.jsonl, data/valid.jsonl ({"prompt", "completion", "source"} per line)
and data/test_prompts.jsonl (the annotated test set test_gold.jsonl, in the same format).
Names from test_gold.jsonl and test_issues.jsonl never appear in the training data.
"""

from __future__ import annotations

import json
import random
import re
import unicodedata
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
DATA = Path("data")
TEST_GOLD = Path("test_gold.jsonl")
TEST_ISSUES = Path("test_issues.jsonl")  # examples from openfoodfacts-server issues #14657 and #14526
K = 10  # candidates shown to the model
EMBEDDER = "intfloat/multilingual-e5-small"
SIZES = {"real": 12_000, "corrupted": 12_000, "new": 5_000, "qualified": 4_000, "noise": 4_000, "noise_short": 5_000,
         "structure": 2_000}


# ---------- 1. Taxonomy ----------
# Like the OFF parser, the ingredients taxonomy is merged with these ones
TAXONOMIES = {
    "ingredients": "food/ingredients",
    "additives": "additives",
    "additives_classes": "additives_classes",
    "vitamins": "vitamins",
    "minerals": "minerals",
    "amino_acids": "amino_acids",
    "nucleotides": "nucleotides",
    "other_nutritional_substances": "other_nutritional_substances",
}
TAXONOMY_URL = "https://raw.githubusercontent.com/openfoodfacts/openfoodfacts-server/main/taxonomies/{}.txt"

LANG_LINE = re.compile(r"^([a-z]{2,3}|xx):\s*(.+)$")
PROP_LINE = re.compile(r"^([a-z_]+):([a-z]{2,3}):\s*(.+)$")
PARENT_LINE = re.compile(r"^<\s*([a-z]{2,3}):\s*(.+)$")


def normalize(name: str) -> str:
    """Comparable form of a name, close to OFF's get_string_id_for_lang."""
    name = unicodedata.normalize("NFC", name.strip().lower())
    name = name.replace("œ", "oe").replace("æ", "ae")
    name = re.sub(r"[\s_'’`\-]+", "-", name)
    name = re.sub(r"[^\w\-]", "", name)
    return re.sub(r"-+", "-", name).strip("-")


def tag_id(lang: str, name: str) -> str:
    return f"{lang}:{normalize(name)}"


@dataclass
class Entry:
    id: str
    names: dict[str, list[str]] = field(default_factory=dict)  # lang -> names
    parent_refs: list[str] = field(default_factory=list)  # raw parent ids
    props: dict[str, str] = field(default_factory=dict)  # "allergens:en" -> "en:eggs"


class Taxonomy:
    def __init__(self, entries: dict[str, Entry]):
        self.entries = entries
        self.synonyms: dict[str, str] = {}  # normalized name with lang -> canonical id
        for e in entries.values():
            for lang, names in e.names.items():
                for n in names:
                    self.synonyms.setdefault(tag_id(lang, n), e.id)
        self.parents = {eid: [self.resolve(p) for p in e.parent_refs] for eid, e in entries.items()}
        self.parents = {k: [p for p in v if p] for k, v in self.parents.items()}

    def resolve(self, ref: str) -> str | None:
        """Canonical id of a name or id ("en:eggs" -> "en:egg")."""
        lang, _, name = ref.partition(":")
        return self.synonyms.get(tag_id(lang, name))

    def ancestors(self, eid: str) -> list[str]:
        seen, stack = [], list(self.parents.get(eid, []))
        while stack:
            p = stack.pop()
            if p not in seen:
                seen.append(p)
                stack.extend(self.parents.get(p, []))
        return seen

    def allergens(self, eid: str) -> set[str]:
        """Allergens of the entry, including those inherited from its ancestors."""
        found = set()
        for e in [eid, *self.ancestors(eid)]:
            value = self.entries[e].props.get("allergens:en") if e in self.entries else None
            if value:
                found.update(a.strip() for a in value.split(","))
        return found


def parse_taxonomy(path: Path) -> dict[str, Entry]:
    """One block per entry, separated by a blank line: "< en: parent", "en: name, synonym", "allergens:en: en:eggs"."""
    entries: dict[str, Entry] = {}
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
        parents, names, props = [], {}, {}
        for line in block.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if m := PARENT_LINE.match(line):
                parents.append(f"{m[1]}:{m[2].strip()}")
            elif m := PROP_LINE.match(line):
                props[f"{m[1]}:{m[2]}"] = m[3].strip()
            elif m := LANG_LINE.match(line):
                names[m[1]] = [n.strip() for n in m[2].split(",") if n.strip()]
        if not names:
            continue  # header block (stopwords, global synonyms...) or comment
        first_lang = next(iter(names))
        eid = tag_id(first_lang, names[first_lang][0])
        entries[eid] = Entry(eid, names, parents, props)
    return entries


def load_taxonomy() -> Taxonomy:
    DATA.mkdir(exist_ok=True)
    entries: dict[str, Entry] = {}
    for name, remote in TAXONOMIES.items():
        path = DATA / f"{name}.txt"
        if not path.exists():
            urllib.request.urlretrieve(TAXONOMY_URL.format(remote), path)
        for eid, e in parse_taxonomy(path).items():
            entries.setdefault(eid, e)  # on duplicates, the ingredients taxonomy wins
    return Taxonomy(entries)


tx = load_taxonomy()


# ---------- 2. What the OFF parser did on every product ----------
def extract_from_export() -> None:
    """Walks the ingredient tree of every product of the OFF export (about 4 GB) and writes:
    data/real_pairs.parquet     (lang, text as written, taxonomy id) -> number of products
    data/unknown_names.parquet  unrecognized "lang:name" -> number of products"""
    import duckdb
    from huggingface_hub import hf_hub_download

    export = hf_hub_download("openfoodfacts/product-database", "food.parquet", repo_type="dataset", local_dir=DATA)
    rows = duckdb.connect().execute(
        f"SELECT lang, ingredients FROM read_parquet('{Path(export).as_posix()}') WHERE ingredients IS NOT NULL"
    )
    pairs: Counter[tuple[str, str, str]] = Counter()
    unknown: Counter[str] = Counter()
    while batch := rows.fetchmany(50_000):
        for lang, tree in batch:
            seen_pairs, seen_unknown = set(), set()  # counted once per product
            stack = json.loads(tree)
            while stack:
                ing = stack.pop()
                if ing.get("is_in_taxonomy") == 1 and ing.get("text") and ing.get("id"):
                    seen_pairs.add((lang or "", ing["text"].strip(), ing["id"]))
                elif ing.get("is_in_taxonomy") == 0 and ing.get("id"):
                    seen_unknown.add(ing["id"])
                stack.extend(ing.get("ingredients") or [])
            pairs.update(seen_pairs)
            unknown.update(seen_unknown)

    real = pd.DataFrame([(*k, n) for k, n in pairs.items()], columns=["lang", "text", "id", "products"])
    real.sort_values("products", ascending=False).to_parquet(DATA / "real_pairs.parquet", index=False)
    unk = pd.DataFrame(unknown.items(), columns=["id", "products"]).sort_values("products", ascending=False)
    unk["lang"] = unk.id.str.split(":").str[0]
    unk.to_parquet(DATA / "unknown_names.parquet", index=False)
    print("Real links:", len(real), "| unrecognized names:", len(unk))


# ---------- 3. Candidate retrieval ----------
class Retriever:
    """Nearest taxonomy entries for a name, whatever their language: "salt" declared as French finds en:salt."""

    def __init__(self, exclude: set[str] | None = None):
        from sentence_transformers import SentenceTransformer

        if not (DATA / "index_names.npy").exists():
            build_index()
        self.model = SentenceTransformer(EMBEDDER, device="cpu")
        self.names = pd.read_parquet(DATA / "index_names.parquet")
        self.vectors = np.load(DATA / "index_names.npy").astype(np.float32)
        if exclude:  # entries removed, to simulate ingredients missing from the taxonomy
            keep = ~self.names.id.isin(exclude).to_numpy()
            self.names, self.vectors = self.names[keep].reset_index(drop=True), self.vectors[keep]

    def encode(self, queries: list[str]) -> np.ndarray:
        return self.model.encode([f"query: {q}" for q in queries], batch_size=256, normalize_embeddings=True)

    def entry_score(self, query: str, eid: str) -> float:
        """Similarity between a query and a given entry (its closest name)."""
        if not hasattr(self, "_rows"):
            self._rows = self.names.groupby("id").indices
        rows = self._rows.get(eid)
        return 0.0 if rows is None else float((self.vectors[rows] @ self.encode([query])[0]).max())

    def search(self, queries: list[str], k: int = K) -> list[list[tuple[str, str, float]]]:
        """For each query, the k closest distinct entries: (id, closest name, score)."""
        results = []
        for start in range(0, len(queries), 512):  # in chunks: 512 x 140,000 scores instead of several GB
            scores = self.encode(queries[start : start + 512]) @ self.vectors.T
            for row in scores:
                best: dict[str, tuple[str, float]] = {}
                for i in np.argsort(-row)[: k * 6]:  # several names per entry: look wide
                    eid = self.names.id.iat[i]
                    if eid not in best:
                        best[eid] = (self.names.name.iat[i], float(row[i]))
                    if len(best) == k:
                        break
                results.append([(eid, name, score) for eid, (name, score) in best.items()])
        return results


def build_index() -> None:
    from sentence_transformers import SentenceTransformer

    rows = [(eid, lang, n) for eid, e in tx.entries.items() for lang, names in e.names.items() for n in names]
    names = pd.DataFrame(rows, columns=["id", "lang", "name"]).drop_duplicates()
    vectors = SentenceTransformer(EMBEDDER, device="cpu").encode(
        [f"passage: {n}" for n in names.name], batch_size=256, normalize_embeddings=True, show_progress_bar=True
    )
    np.save(DATA / "index_names.npy", vectors.astype(np.float16))
    names.to_parquet(DATA / "index_names.parquet", index=False)
    print("Index:", len(names), "names for", names.id.nunique(), "entries")


# ---------- 4. Prompt format (shared with infer.py) ----------
def describe(eid: str, max_names: int = 4) -> str:
    """An entry as the model sees it: its id and a few names (English, French, then other languages)."""
    e = tx.entries[eid]
    names, seen = [], set()
    # up to 2 English names (for additives: the code then the chemical name), 1 French, then other languages
    picks = [(l, i) for l in ["en"] for i in range(2)] + [("fr", 0)]
    picks += [(l, 0) for l in e.names if l not in ("en", "fr")]
    for lang, i in picks:
        ns = e.names.get(lang, [])
        if i < len(ns) and ns[i].lower() not in seen:
            seen.add(ns[i].lower())
            names.append(ns[i])
        if len(names) >= max_names:
            break
    return f"{eid} — {' | '.join(names)}"


def prompt(name: str, lang: str, context: str, candidates: list[tuple[str, float]]) -> str:
    """candidates: (id, similarity with the name). Low similarity everywhere often betrays noise."""
    lines = "\n".join(
        f"{i + 1}. {eid} [{score:.2f}] — {describe(eid).split(' — ', 1)[1]}" for i, (eid, score) in enumerate(candidates)
    )
    return (
        f'Unrecognized ingredient: "{name}"\n'
        f"Declared list language: {lang}\n"
        f'Context: "{context[:300]}"\n'
        f"Candidates:\n{lines}"
    )


# ---------- 5. Examples ----------
# Common names per language, to build plausible ingredient lists around the name
_POOL: dict[str, list[str]] = {}
for _e in tx.entries.values():
    for _lang, _ns in _e.names.items():
        _POOL.setdefault(_lang, []).append(_ns[0])


def make_context(name: str, text_lang: str) -> str:
    """An ingredient list in the REAL language of the text, with the name inserted at random.
    When the declared language is wrong, the context gives it away, as on a real product."""
    pool = _POOL.get(text_lang) or _POOL["en"]
    others = random.sample(pool, k=min(len(pool), random.randint(3, 7)))
    others.insert(random.randrange(len(others) + 1), name)
    return ", ".join(others)


def weighted_sample(df: pd.DataFrame, n: int, weights: pd.Series, seed: int) -> pd.DataFrame:
    """Weighted sampling without replacement (Efraimidis-Spirakis): key u^(1/w), keep the n largest."""
    u = np.random.default_rng(seed).random(len(df))
    return df.iloc[np.argsort(-(u ** (1.0 / weights.to_numpy(dtype=float))))[:n]]


def example(name, lang, context, candidates, answer, source):
    return {"prompt": prompt(name, lang, context, candidates), "completion": json.dumps(answer), "source": source}


OCR_SWAPS = [("rn", "m"), ("m", "rn"), ("l", "1"), ("i", "l"), ("o", "0"), ("e", "c"), ("é", "e"), ("è", "e"), ("u", "ü")]


def corrupt(name: str) -> str:
    """Damage a name the way a bad photo reading would."""
    s = name
    for _ in range(random.choice([1, 1, 2])):
        kind = random.random()
        if kind < 0.35 and len(s) > 4:  # a letter lost or doubled
            i = random.randrange(1, len(s) - 1)
            s = s[:i] + s[i + 1 :] if random.random() < 0.5 else s[:i] + s[i] + s[i:]
        elif kind < 0.6:  # reading confusion
            a, b = random.choice(OCR_SWAPS)
            if a in s:
                s = s.replace(a, b, 1)
        elif kind < 0.75:
            s = s.upper()
        elif kind < 0.9:
            s = f"{s} {random.choice([2, 5, 12, 23.5, 40])}%"
        else:
            s = s.replace(" ", "", 1)
    return s


# Short noise: fragments, numbers, units, pieces of addresses or URLs, function words
SHORT_NOISE = re.compile(
    r"^(\W*\w{1,3}\W*|\d+([.,]\d+)?\s*(g|kg|mg|µg|ml|cl|l|kj|kcal|%|x)?|https?|www.*|.*\.(org|com|net)\b.*"
    r"|p\.?\s?o\.?( box)?|tel.*|fax.*|lot.*|n[°o]\.?\s?\d*)$",
    re.I,
)
FUNCTION_WORDS = {
    "also", "and", "or", "with", "the", "of", "from", "made", "per", "each", "only", "forever", "contains", "contain",
    "et", "ou", "avec", "soit", "dont", "du", "des", "de", "le", "la", "les", "pour", "sans", "issu", "issus",
    "und", "oder", "mit", "aus", "enthält", "enthalten", "ohne", "von", "y", "con", "sin", "contiene", "puede",
    "e", "o", "senza", "può", "en", "met", "zonder", "van", "bevat", "i", "lub", "z", "bez",
    "minim", "maxim", "sau", "cu", "fără", "och", "med", "utan", "og", "uden", "também", "sem", "com",
}
NOISE = re.compile(
    r"(\d+\s*(kj|kcal|cal)\b)|(\b(ingr[ée]dients?|zutaten|ingredientes|ingredienti|sk[łl]adniki|ingredi[ëe]nten)\b)"
    r"|(conserv|store |keep |aufbewahr|consumir|best before|à consommer|mindestens|haltbar|réfrigér)"
    r"|(energ|valeurs nutri|nutrition|n[äa]hrwert|matières grasses|kohlenhydrat|glucides|eiwei|protein \d)"
    r"|(www\.|\.com|\.fr|@|poids net|net weight|lot |exp )",
    re.I,
)
# Qualifiers that do not make a different ingredient: "fresh almonds" is still en:almond.
# Composed names that already exist in the taxonomy ("fresh cream"...) are skipped.
QUALIFIERS = {
    "en": ["fresh", "natural", "pure", "selected", "premium", "mini", "ripe", "imported", "italian", "french",
           "spanish", "greek", "belgian", "american"],
    "fr": ["frais", "naturel", "pur", "sélectionné", "extra", "italien", "français", "espagnol"],
    "es": ["fresco", "natural", "puro", "seleccionado", "extra", "italiano", "español"],
    "it": ["fresco", "naturale", "puro", "selezionato", "extra", "italiano"],
}
STRUCTURE = re.compile(
    r"(contains? (\d+ ?% )?(or|and) less|less than \d|may contain|peut contenir|traces? (éventuelles|de)|kann spuren"
    r"|puede contener|può contenere|of the following|to preserve freshness|as a preservative|added to)",
    re.I,
)


def main() -> None:
    if not (DATA / "real_pairs.parquet").exists() or not (DATA / "unknown_names.parquet").exists():
        extract_from_export()
    examples: list[dict] = []
    retriever = Retriever()
    known_names = {k.split(":", 1)[1] for k in tx.synonyms}

    def add_batch(items, source):
        """items: (name, lang, context, answer, id that must be among the candidates, or None)"""
        cands = retriever.search([n for n, *_ in items])
        for (name, lang, ctx, answer, must), cand in zip(items, cands):
            pairs = [(c[0], c[2]) for c in cand]
            if must and must not in [p[0] for p in pairs]:
                pairs[-1] = (must, retriever.entry_score(name, must))  # the right answer must be among the candidates...
            random.shuffle(pairs)  # ...but not always first
            examples.append(example(name, lang, ctx, pairs, answer, source))

    # Real links, weighted by frequency (capped, so that "salt" and "sugar" do not dominate)
    real = pd.read_parquet(DATA / "real_pairs.parquet")
    real = real[real.id.isin(tx.entries.keys())]
    real = weighted_sample(real, SIZES["real"], real.products.clip(upper=50), seed=1)
    add_batch([(r.text, r.lang, make_context(r.text, r.lang), {"type": "existing", "id": r.id}, r.id) for r in real.itertuples()], "real")

    # Damaged taxonomy names, sometimes declared in the wrong language
    names = [(eid, lang, n) for eid, e in tx.entries.items() for lang, ns in e.names.items() for n in ns]
    items = []
    for eid, lang, n in random.sample(names, SIZES["corrupted"]):
        declared = random.choice(["en", "fr", "de", "es", "it"]) if random.random() < 0.3 else lang
        bad = corrupt(n)
        items.append((bad, declared, make_context(bad, lang), {"type": "existing", "id": eid}, eid))
    add_batch(items, "corrupted")

    # Simulated new ingredients: taxonomy leaves removed from the index
    children = {p for ps in tx.parents.values() for p in ps}
    leaves = [eid for eid, ps in tx.parents.items() if ps and eid not in children]
    held_out = set(random.sample(leaves, min(len(leaves), SIZES["new"])))
    hidden = Retriever(exclude=held_out)
    items = []
    for eid in sorted(held_out):  # sorted: set order changes with every Python process
        lang, n = random.choice([(l, n) for l, ns in tx.entries[eid].names.items() for n in ns])
        items.append((n, lang, make_context(n, lang), {"type": "new", "parent": tx.parents[eid][0]}, tx.parents[eid][0]))
    for (name, lang, ctx, answer, parent), cand in zip(items, hidden.search([i[0] for i in items])):
        pairs = [(c[0], c[2]) for c in cand]
        if parent in [p[0] for p in pairs]:  # only when the parent can be found: otherwise nothing to learn
            random.shuffle(pairs)
            examples.append(example(name, lang, ctx, pairs, answer, "new"))

    # Qualified names: a qualifier that does not change the ingredient, so the answer is the entry itself
    names = [(eid, lang, n) for eid, e in tx.entries.items() for lang in QUALIFIERS for n in e.names.get(lang, [])
             if len(n.split()) <= 3]
    items = []
    for eid, lang, n in random.sample(names, len(names)):
        q = random.choice(QUALIFIERS[lang])
        name = f"{q} {n}" if lang == "en" else f"{n} {q}"
        if normalize(name) in known_names:
            continue
        items.append((name, lang, make_context(name, lang), {"type": "existing", "id": eid}, eid))
        if len(items) == SIZES["qualified"]:
            break
    add_batch(items, "qualified")

    # Noise and structure, among real unrecognized names
    unknown = pd.read_parquet(DATA / "unknown_names.parquet")
    unknown["name"] = unknown.id.str.split(":", n=1).str[1].str.replace("-", " ")
    unknown = unknown[~unknown.name.map(normalize).isin(known_names)]  # no real ingredient name taken for noise
    stripped = unknown.name.str.strip()
    is_short = stripped.str.match(SHORT_NOISE) | stripped.str.lower().isin(FUNCTION_WORDS)
    is_struct = unknown.name.str.contains(STRUCTURE)
    is_noise = unknown.name.str.contains(NOISE) & ~is_struct & ~is_short
    for mask, kind in [(is_noise, "noise"), (is_short & ~is_struct, "noise_short"), (is_struct, "structure")]:
        pool = unknown[mask]
        pick = weighted_sample(pool, SIZES[kind], pool.products.clip(upper=50), seed=2)
        answer = {"type": "structure"} if kind == "structure" else {"type": "not_ingredient"}
        add_batch([(r.name, r.lang, make_context(r.name, r.lang), answer, None) for r in pick.itertuples()], kind)

    # Never a test name in training, whatever the source
    test_names = {normalize(json.loads(l)["name"]) for path in (TEST_GOLD, TEST_ISSUES) for l in open(path, encoding="utf-8")}
    examples = [e for e in examples if normalize(re.search(r'^Unrecognized ingredient: "(.*)"$', e["prompt"], re.M)[1])
                not in test_names]
    random.shuffle(examples)
    n_valid = len(examples) // 20
    for path, part in [(DATA / "valid.jsonl", examples[:n_valid]), (DATA / "train.jsonl", examples[n_valid:])]:
        with open(path, "w", encoding="utf-8") as f:
            for ex in part:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("Examples:", len(examples), pd.Series([e["source"] for e in examples]).value_counts().to_dict())

    # The annotated test set, same format (same retrieval, same scores)
    test = [json.loads(l) for l in open(TEST_GOLD, encoding="utf-8")]
    names = [e["name"].replace("-", " ") for e in test]  # OFF ids replace spaces with hyphens
    with open(DATA / "test_prompts.jsonl", "w", encoding="utf-8") as f:
        for i, (e, name, cand) in enumerate(zip(test, names, retriever.search(names))):
            pairs = [(c[0], c[2]) for c in cand]
            f.write(json.dumps({"i": i, "prompt": prompt(name, e["lang"], e["context"], pairs)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
